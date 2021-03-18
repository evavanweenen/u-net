import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"

import torch
from tqdm import tqdm

from config import *
from data import *
from model import *
from loss import *
from plot import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ------------------ Read data ------------------ #
if not os.path.exists(os.path.join(DATA_PATH,'X')):
	ct3D_to_np2D(DATA_PATH, 'imagesTr', 'X')
	ct3D_to_np2D(DATA_PATH, 'labelsTr', 'Y')
	ct3D_to_np2D(DATA_PATH, 'imagesTs', 'X_test')

if not os.path.exists(os.path.join(DATA_PATH, 'filter')):
	os.mkdir(os.path.join(DATA_PATH, 'filter'))
	filter_empty(DATA_PATH, 'filter', 'imagesTr', 'labelsTr')

def get_dataloader(path, k, K, batch_size, num_workers=0, sort_train=False, sort_val=False):
	"""
	Obtain dataloader (iterable with batches of data) for fold k out of a total of K folds
	The data is split up into K folds using the patient ID (such that patients are not multiple folds)
	A dataloader is used to preserve memory
	"""
	data = {'X_train':  CTDatasetSplit(os.path.join(path, 'X/'), split='train', k=k, K=K, normalize=True, sort=sort_train),
			'Y_train':	CTDatasetSplit(os.path.join(path, 'Y/'), split='train', k=k, K=K, sort=sort_train),
			'X_val':	CTDatasetSplit(os.path.join(path, 'X/'), split='val', k=k, K=K, normalize=True, sort=sort_val),
			'Y_val':	CTDatasetSplit(os.path.join(path, 'Y/'), split='val', k=k, K=K, sort=sort_val)}

	dataloader = {'train':	zip(DataLoader(data['X_train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
								DataLoader(data['Y_train'], batch_size=batch_size, shuffle=True, num_workers=num_workers)),
				  'val':	zip(DataLoader(data['X_val'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
				  				DataLoader(data['Y_val'], batch_size=batch_size, shuffle=True, num_workers=num_workers))}
	return data, dataloader

# ------------------ Model ------------------ #
def fit(dataloader, model, loss_fn, metrics, optimizer):
	model.train()

	# initialize logs
	logs = {'loss':0}
	logs.update({f:[] for f in metrics.keys()})

	for i, (inputs, target) in enumerate(tqdm(dataloader)):
		inputs, target = inputs.to(device), target.to(device)

		optimizer.zero_grad() # zero gradient
		
		# forward pass
		output = model(inputs)
		loss = loss_fn(output, target)
		
		# update epoch loss by weighted batch loss
		logs['loss'] += loss.item() * output.shape[0]
		
		# metrics
		for j, metric_fn in metrics.items():
			logs[j].extend(metric_fn(output, target).T[0].tolist())

		# backwards pass
		loss.backward()
		optimizer.step()
	for j in metrics.keys():
		logs[j] = np.mean(logs[j])
	return model, logs.values(), optimizer

def evaluate(dataloader, model, loss_fn, metrics):
	model.eval()

	# initialize logs
	logs = {'loss':0}
	logs.update({f:[] for f in metrics.keys()})

	with torch.no_grad(): #set_grad_enabled(False)
		for i, (inputs, target) in enumerate(tqdm(dataloader)):
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

def evaluate_3d(data, split, model, metrics):
	model.eval()

	logs = {f:[] for f in metrics.keys()}

	with torch.no_grad():
		for patient in tqdm(data['X_'+split].index.patient.unique()):
			
			target_3d, output_3d = [], []
			
			for i in data['X_'+split].index[data['X_'+split].index.patient == patient].index:
				
				inputs = data['X_'+split].__getitem__(i)
				target = data['Y_'+split].__getitem__(i)

				inputs, target = inputs.to(device), target.to(device)

				# parse 2d image through model
				inputs = inputs.reshape(1,*inputs.shape) # add shape of batch
				output = model(inputs)
				output = output.reshape(target.shape) # remove shape of batch

				target_3d.append(target)
				output_3d.append(output)

			# stack all 2d images into 3d image
			target_3d = torch.stack(target_3d, dim=1)
			output_3d = torch.stack(output_3d, dim=1)

			# metrics
			for j, metric_fn in metrics.items():
				logs[j].append(metric_fn(output_3d, target_3d).tolist())

	logs = {j:np.mean(m) for j, m in logs.items()}
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

	metrics_3d = {	'loss'	: FocalDiceLoss(**params, alpha=ALPHA, reduction='mean', dim=3),
					'focal'	: FocalLoss(gamma=params['gamma'], alpha=ALPHA, reduction='mean'),
					'dice'	: DiceScore(reduction='none', dim=3),
					'iou'	: IoUScore(reduction='none', dim=3)}

	score = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
		np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=np.arange(EPOCHS))

	score_3d = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
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
			data, dataloader = get_dataloader(DATA_PATH+'filter/', k, K, BATCH_SIZE)

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

			# save last model in case we want to continue training afterwards
			print("saving last model")
			torch.save({'params': params,
						'epoch': epoch,
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),
						'metrics': score.to_dict()}, 
						os.path.join(SAVE_PATH, str(params), 'model%s_last'%k))

		del model

		# open model
		model = UNET(FILTERS)
		model = nn.DataParallel(model) # use multiple gpus
		model.to(device)

		optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
		
		# load model
		checkpoint = torch.load(os.path.join(SAVE_PATH, str(params), 'model%s'%k))
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		score = pd.DataFrame.from_dict(checkpoint['metrics'])

		# evaluate model in 3d
		data, dataloader = get_dataloader(DATA_PATH, k, K, BATCH_SIZE, sort_train=True, sort_val=True)

		score_3d.loc[epoch, ('train', k)] = evaluate_3d(data, 'train', model, metrics_3d)
		score_3d.loc[epoch, ('val', k)] = evaluate_3d(data, 'val', model, metrics_3d)
		
		print(score_3d.xs(k, axis=1, level=1).loc[epoch])
		score_3d.to_csv(os.path.join(SAVE_PATH, str(params), 'score_3d.csv'))

	for m in score.columns.get_level_values(2).unique():
		history = pd.melt(score.xs(m, level=2, axis=1), var_name=['split', 'k'], ignore_index=False)

		PlotResults(os.path.join(SAVE_PATH, str(params))).plot_history(history, m)