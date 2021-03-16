import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"

import torch
from tqdm import tqdm

from config import *
from data import *
from model import *
from plot import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ------------------ Read data ------------------ #
if not os.path.exists(os.path.join(DATA_PATH,'X')):
	ct3D_to_np2D(DATA_PATH, 'imagesTr', 'X')
	ct3D_to_np2D(DATA_PATH, 'labelsTr', 'Y')
	ct3D_to_np2D(DATA_PATH, 'imagesTs', 'X_test')

def get_dataloader(path, k, K, batch_size, num_workers=1):
	"""
	Obtain dataloader (iterable with batches of data) for fold k out of a total of K folds
	The data is split up into K folds using the patient ID (such that patients are not multiple folds)
	A dataloader is used to preserve memory
	"""
	data = {'X_train':  CTDatasetSplit(os.path.join(path, 'X/'), split='train', k=k, K=K, normalize=True),
			'Y_train':	CTDatasetSplit(os.path.join(path, 'Y/'), split='train', k=k, K=K),
			'X_val':	CTDatasetSplit(os.path.join(path, 'X/'), split='val', k=k, K=K, normalize=True),
			'Y_val':	CTDatasetSplit(os.path.join(path, 'Y/'), split='val', k=k, K=K)}

	dataloader = {'train':	zip(DataLoader(data['X_train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
								DataLoader(data['Y_train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)),
				  'val':	zip(DataLoader(data['X_val'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
				  				DataLoader(data['Y_val'], batch_size=batch_size, shuffle=True, num_workers=num_workers))}
	return dataloader

# ------------------ Model ------------------ #
def fit(data, model, loss_fn, metrics, optimizer):
	model.train()

	# initialize logs
	logs = {'loss':0}
	logs.update({f:[] for f in metrics.keys()})

	for i, (inputs, target) in enumerate(tqdm(data)):
		inputs, target = inputs.to(device), target.to(device)
		
		# forward pass
		output = model(inputs)
		loss = loss_fn(output, target)
		
		# update epoch loss by weighted batch loss
		logs['loss'] += loss.item() * output.shape[0]
		
		# metrics
		for j, metric_fn in metrics.items():
			logs[j].extend(metric_fn(output, target).T[0].tolist())

		# backwards pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	for j in metrics.keys():
		logs[j] = np.mean(logs[j])
	return model, logs.values(), optimizer

def evaluate(data, model, loss_fn, metrics):
	model.eval()

	# initialize logs
	logs = {'loss':0}
	logs.update({f:[] for f in metrics.keys()})

	with torch.no_grad(): #set_grad_enabled(False)
		for i, (inputs, target) in enumerate(tqdm(data)):
			inputs, target = inputs.to(device), target.to(device)

			# forward pass
			output = model(inputs)
			loss = loss_fn(output, target)

			# update epoch loss by weighted batch loss
			logs['loss'] += loss.item() * output.shape[0]

			# metrics
			for j, metric_fn in metrics.items():
				logs[j].extend(metric_fn(output, target).T[0].tolist())
	for j in metrics.keys():
		logs[j] = np.mean(logs[j])
	return logs.values()

# hyperparameter optimization
for params in PARAM_GRID:
	print(params)
	if not os.path.exists(SAVE_PATH+str(params)):
		os.mkdir(SAVE_PATH+str(params))	

	loss_fn = FocalDiceLoss(**params, alpha=ALPHA, reduction='mean')
	metrics = { 'focal'	: FocalLoss(gamma=params['gamma'], alpha=ALPHA, reduction='none'),
				'dice'	: DiceScore(reduction='none'),
				'iou'	: IoUScore(reduction='none')}

	score = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
		np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=np.arange(EPOCHS))

	# K-fold cross-validation
	for k in range(K):
		print("Fold: ", k)

		model = UNET(FILTERS)
		model = nn.DataParallel(model) # use multiple gpus
		model.to(device)

		optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

		for epoch in tqdm(range(EPOCHS)):
			# get data
			dataloader = get_dataloader(DATA_PATH+'filter/', k, K, BATCH_SIZE)

			# fit model
			model, score.loc[epoch, ('train', k)], optimizer = fit(dataloader['train'],
				model, loss_fn, metrics, optimizer)
			
			# evaluate model
			score.loc[epoch, ('val', k)] = evaluate(dataloader['val'], model, loss_fn, metrics)

			print(score.xs(k, axis=1, level=1).loc[epoch])
			score.to_csv(os.path.join(SAVE_PATH, str(params), 'score.csv'))

			# save best model after each epoch
			if epoch > 0:
				if score.loc[epoch, ('val', k, 'loss')] < score.loc[:epoch-1, ('val', k, 'loss')].min():
					print("saving best model")
					torch.save({'params': params,
								'epoch': epoch,
								'model_state_dict': model.state_dict(),
								'optimizer_state_dict': optimizer.state_dict(),
								'metrics': score.to_dict()}, 
								os.path.join(SAVE_PATH, str(params), 'model%s'%k))
			
			# early stopping: break training loop if val loss increases for {PATIENCE} epochs
			if epoch > PATIENCE:
				if (score[('val', k, 'loss')].diff().loc[epoch-PATIENCE+1:epoch] > 0).sum() == PATIENCE:
					print("early stopping")
					break

		del model

# read model results
cv_score_mean = {} # mean of scores over k folds for each params
cv_score_std = {} # std of scores over k folds for each params
uncrt = {} # model robustness measure
n_epochs = {} # number of epochs with lowest validation loss
for params in PARAM_GRID:
	cv_score = pd.read_csv(os.path.join(SAVE_PATH, str(params), 'score.csv'), index_col=0, header=[0,1,2])
	
	# get score for epoch with lowest validation loss
	epoch_min = {} # epoch with lowest validation loss
	best_score = {} # score for epoch with lowest validation loss

	for k in range(K):
		epoch_min[k] = cv_score[('val', str(k), 'loss')].idxmin() 
		best_score[k] = cv_score.loc[epoch_min[k]].xs(str(k), level=1)

	best_score = pd.DataFrame.from_dict(best_score)

	cv_score_mean[params.values()] = best_score.mean(axis=1)
	cv_score_std[params.values()] = best_score.std(axis=1)

	# save number of epochs needed for training
	n_epochs[tuple(params.values())] = epoch_min

	# calculate mean uncertainty over all training epochs
	uncrt[tuple(params.values())] = cv_score.std(level=(0,2), axis=1).mean()

	# plot loss (+ all other metrics) during training on training and validation set
	for m in cv_score.columns.get_level_values(2).unique():
		history = pd.melt(cv_score.xs(m, level=2, axis=1), var_name=['split', 'k'], ignore_index=False)

		PlotResults(os.path.join(SAVE_PATH+str(params))).plot_history(history, m)

cv_score_mean = pd.DataFrame.from_dict(cv_score_mean).T
cv_score_std = pd.DataFrame.from_dict(cv_score_std).T

cv_score_mean.to_csv(SAVE_PATH+'cv_results_mean.csv')
cv_score_mean.to_latex(SAVE_PATH+'cv_results_mean.tex', float_format='%.3g')

cv_score_std.to_csv(SAVE_PATH+'cv_results_std.csv')
cv_score_std.to_latex(SAVE_PATH+'cv_results_std.tex', float_format='%.3g')

uncrt = pd.DataFrame.from_dict(uncrt).T
n_epochs = pd.DataFrame.from_dict(n_epochs).T

best_params = uncrt[('val', 'iou')].idxmin()
best_epochs = int(np.ceil(n_epochs.loc[best_params].mean()))

best_params = {list(PARAM_GRID[0].keys())[i]: param for i, param in enumerate(best_params)}

print("Best params found: ", best_params)

# ------------------ Evaluate ------------------ #

if not os.path.exists(SAVE_PATH+'best/'):
	os.mkdir(SAVE_PATH+'best/')
if not os.path.exists(SAVE_PATH+'best/img/'):
	os.mkdir(SAVE_PATH+'best/img/')
if not os.path.exists(SAVE_PATH+'best/img/X/'):
	os.mkdir(SAVE_PATH+'best/img/X/')
if not os.path.exists(SAVE_PATH+'best/img/X_test/'):
	os.mkdir(SAVE_PATH+'best/img/X_test/')

data = {'X': CTDataset(os.path.join(DATA_PATH, 'X/'), normalize=True),
		'Y': CTDataset(os.path.join(DATA_PATH, 'Y/')),
		'X_test': CTDataset3DList(os.path.join(DATA_PATH, 'X_test/'), normalize=True)}

loss_fn = FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean')
metrics = { 'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='none'),
			'dice'	: DiceScore(reduction='none'),
			'iou'	: IoUScore(reduction='none')}

score = pd.DataFrame(columns=['loss', 'focal', 'dice', 'iou'], index=np.arange(EPOCHS))

model = UNET(FILTERS)
model = nn.DataParallel(model) # use multiple gpus
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train
for epoch in tqdm(range(best_epochs)):
	# get data
	dataloader = zip(DataLoader(data['X'], batch_size=BATCH_SIZE, shuffle=True, num_workers=1),
					 DataLoader(data['Y'], batch_size=BATCH_SIZE, shuffle=True, num_workers=1))

	# fit model
	model, score.loc[epoch], optimizer = fit(dataloader, model, loss_fn, metrics, optimizer)
	
	torch.save({'params': best_params,
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}, 
				os.path.join(SAVE_PATH, 'best', 'model'))

checkpoint = torch.load(os.path.join(SAVE_PATH, 'best', 'model'))
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

with torch.no_grad(): #set_grad_enabled(False)
	for patient in data['X_test'].slicelist.patient.unique():

		output_3d = []
		for i in data['X_test'].slicelist[data['X_test'].slicelist.patient == patient].index:
	
			inputs = data['X_test'].__getitem__(i)
			inputs = inputs.reshape(1,*inputs.shape) # add shape of batch
			inputs = inputs.to(device)

			# forward pass
			output = model(inputs)

			# apply sigmoid
			output = torch.sigmoid(output)

			# clip values to 0,1 with threshold 0.5
			output = (output > 0.5).float()
			print(output.sum())

			output = output.reshape(*output.shape[-2:])

			output_3d.append(output)

		output_3d = torch.stack(output_3d, dim=2)
		torch.save(output_3d, SAVE_PATH+'best/img/X_test/colon_%s.pt'%patient)