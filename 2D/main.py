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
# convert 3D CT scans to 2D numpy arrays
if not os.path.exists(os.path.join(DATA_PATH,'X')):
	ct3D_to_np2D(DATA_PATH, 'imagesTr', 'X')
	ct3D_to_np2D(DATA_PATH, 'labelsTr', 'Y')
	ct3D_to_np2D(DATA_PATH, 'imagesTs', 'X_test')

# filter out images without mask
if not os.path.exists(os.path.join(DATA_PATH, 'filter')):
	os.mkdir(os.path.join(DATA_PATH, 'filter'))
	filter_empty(DATA_PATH, 'filter', 'imagesTr', 'labelsTr')

def get_dataloader(path, k, K, batch_size, num_workers=0, sort_train=False, sort_val=False):
	"""
	Obtain dataloader (iterable with batches of data) for fold k out of a total of K folds
	The data is split up into K folds using the patient ID (such that patients are not multiple folds)
	A dataloader is used to preserve memory.

	sort_train and sort_val indicate whether the images that are loaded are sorted, 
	such that we can easily combine all 2D images of a patient into one 3D image
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

# ------------------ Model fit ------------------ #
def fit(dataloader, model, loss_fn, metrics, optimizer):
	""" 
	Fit model for one epoch.
	Optimize model and calculate loss (+ other metrics) over batches during one epoch of training
	"""
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

def evaluate(dataloader, model, loss_fn, metrics, do_plot=False, files=[], savedir=SAVE_PATH):
	""" 
	Evaluate model with 2D images
	If do_plot is true, we plot the images, targets and outputs, and we will need the variables 
		files (the fileslist in corresponding order of the dataloader), and
		savedir (where to save the images).
	"""
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
			
			# plot predicted masks
			if do_plot:
				PlotData(savedir).plot_scan(inputs.cpu(), files[i], 'inputs')
				PlotData(savedir).plot_scan(target.cpu(), files[i], 'target')
				PlotData(savedir).plot_scan(output.cpu().detach().numpy(), files[i], 'output')
	for j in metrics.keys():
		logs[j] = np.mean(logs[j])
	return logs.values()

def evaluate_3d(data, split, model, metrics, do_eval=True, do_save=False, savedir=SAVE_PATH):
	""" 
	Evaluate model with 3D images
	If do_eval is true, we calculate the loss function and metrics using target and output.
	If do_save is true, we save the predictions, and we will need the variable savedir.
	"""
	model.eval()

	logs = {f:[] for f in metrics.keys()}

	with torch.no_grad():
		for patient in tqdm(data['X_'+split].index.patient.unique()):
			
			target_3d, output_3d = [], []
			
			for i in data['X_'+split].index[data['X_'+split].index.patient == patient].index:
				
				inputs = data['X_'+split].__getitem__(i).to(device)
				target = data['Y_'+split].__getitem__(i).to(device) if do_eval else []

				# parse 2d image through model
				inputs = inputs.reshape(1,*inputs.shape) # add shape of batch
				output = model(inputs)
				output = output.reshape(*output.shape[1:]) # remove shape of batch

				target_3d.append(target)
				output_3d.append(output)

			# stack all 2d images into 3d image
			target_3d = torch.stack(target_3d, dim=1) if do_eval else []
			output_3d = torch.stack(output_3d, dim=1)

			# metrics
			if do_eval:
				for j, metric_fn in metrics.items():
					logs[j].append(metric_fn(output_3d, target_3d).tolist())

			# save prediction masks
			if do_save:
				output_3d = torch.sigmoid(output_3d)
				output_3d = (output_3d > THRESHOLD).float()
				output_3d = np.swapaxes(output_3d.reshape(*output_3d.shape[1:]), 0, 2) # inverse transform
				torch.save(output_3d, savedir+'colon_%s.pt'%patient)

	logs = {j:np.mean(m) for j, m in logs.items()}
	return logs.values()


# ------------------ Hyperparameter optimization ------------------ #
for params in PARAM_GRID:
	print(params)
	if not os.path.exists(SAVE_PATH+str(params)):
		os.mkdir(SAVE_PATH+str(params))	

	loss_fn = FocalDiceLoss(**params, alpha=ALPHA, reduction='mean', threshold=THRESHOLD)
	metrics = { 'focal'	: FocalLoss(gamma=params['gamma'], alpha=ALPHA, reduction='none'),
				'dice'	: DiceScore(reduction='none', threshold=THRESHOLD),
				'iou'	: IoUScore(reduction='none', threshold=THRESHOLD)}

	metrics_3d = {	'loss'	: FocalDiceLoss(**params, alpha=ALPHA, reduction='mean', dim=3, threshold=THRESHOLD),
					'focal'	: FocalLoss(gamma=params['gamma'], alpha=ALPHA, reduction='mean'),
					'dice'	: DiceScore(reduction='none', dim=3, threshold=THRESHOLD),
					'iou'	: IoUScore(reduction='none', dim=3, threshold=THRESHOLD)}

	score = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
		np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=np.arange(EPOCHS))

	score_3d = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
		np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=np.arange(EPOCHS))

	# K-fold cross-validation
	for k in range(1):# NOTE: we did not have time to run K-fold CV so normally this would be K
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

	# plot loss over epochs
	for m in score.columns.get_level_values(2).unique():
		history = pd.melt(score.xs(m, level=2, axis=1), var_name=['split', 'k'], ignore_index=False)
		history.value = history.value.astype(float)

		PlotResults(os.path.join(SAVE_PATH, str(params))).plot_history(history, m)

# identify best model
cv_score_mean = {} # mean of scores over k folds for each params
cv_score_std = {} # std of scores over k folds for each params
for params in PARAM_GRID:
	cv_score = pd.read_csv(os.path.join(SAVE_PATH, str(params), 'score.csv'), index_col=0, header=[0,1,2])
	
	# get score for epoch with lowest validation loss
	epoch_max = {} # epoch with lowest validation loss
	best_score = {} # score for epoch with lowest validation loss

	for k in range(1):# NOTE: we did not have time to run K-fold CV so normally this would be K
		epoch_max[k] = cv_score[('val', str(k), 'iou')].idxmax() 
		best_score[k] = cv_score.loc[epoch_max[k]].xs(str(k), level=1)

	best_score = pd.DataFrame.from_dict(best_score)

	cv_score_mean[params.values()] = best_score.mean(axis=1)
	cv_score_std[params.values()] = best_score.std(axis=1)

cv_score_mean = pd.DataFrame.from_dict(cv_score_mean).T
cv_score_std = pd.DataFrame.from_dict(cv_score_std).T

cv_score_mean.to_csv(SAVE_PATH+'cv_results_mean.csv')
cv_score_mean.to_latex(SAVE_PATH+'cv_results_mean.tex', float_format='%.3g')

cv_score_std.to_csv(SAVE_PATH+'cv_results_std.csv')
cv_score_std.to_latex(SAVE_PATH+'cv_results_std.tex', float_format='%.3g')

best_params = cv_score_mean[('val', 'iou')].idxmax()
best_params = list(best_params)
print("Best params found: ", best_params)


# ------------------ Retrain best model for longer time ------------------ #
best_params = {'gamma':best_params[0], 'weight':best_params[1]}

EPOCHS = EPOCHS_RETRAIN
PATIENCE = PATIENCE_RETRAIN

loss_fn = FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean', threshold=THRESHOLD)
metrics = { 'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='none'),
			'dice'	: DiceScore(reduction='none', threshold=THRESHOLD),
			'iou'	: IoUScore(reduction='none', threshold=THRESHOLD)}

metrics_3d = {	'loss'	: FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean', dim=3, threshold=THRESHOLD),
				'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='mean'),
				'dice'	: DiceScore(reduction='none', dim=3, threshold=THRESHOLD),
				'iou'	: IoUScore(reduction='none', dim=3, threshold=THRESHOLD)}

score = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
	np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=np.arange(EPOCHS))

score_3d = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
	np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=np.arange(EPOCHS))

# K-fold cross-validation
for k in range(1):# NOTE: we did not have time to run K-fold CV so normally this would be K
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
		score.to_csv(os.path.join(SAVE_PATH, 'best', 'score.csv'))

		# save best model after each epoch
		# NOTE: criterium changed to IOU
		if epoch > 0:
			if score.loc[epoch, ('val', k, 'iou')] > score.loc[:epoch-1, ('val', k, 'iou')].max():
				print("saving best model")
				torch.save({'params': best_params,
							'epoch': epoch,
							'model_state_dict': model.state_dict(),
							'optimizer_state_dict': optimizer.state_dict(),
							'metrics': score.to_dict()}, 
							os.path.join(SAVE_PATH, 'best', 'model%s'%k))

		# early stopping: break training loop if val loss increases for {PATIENCE} epochs
		if epoch > PATIENCE:
			if (score[('val', k, 'loss')].diff().loc[epoch-PATIENCE+1:epoch] > 0).sum() == PATIENCE:
				print("early stopping")
				break

		# save last model in case we want to continue training afterwards
		print("saving last model")
		torch.save({'params': best_params,
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'metrics': score.to_dict()}, 
					os.path.join(SAVE_PATH, 'best', 'model%s_last'%k))

	del model

	# open model
	model = UNET(FILTERS)
	model = nn.DataParallel(model) # use multiple gpus
	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	
	# load model
	checkpoint = torch.load(os.path.join(SAVE_PATH, 'best', 'model%s'%k))#, map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	best_epoch = checkpoint['epoch']
	
	score = pd.read_csv(os.path.join(SAVE_PATH, 'best', 'score.csv'), index_col=0, header=[0,1,2])
	print("Best score 2D: ", score.xs(str(k), level=1, axis=1).loc[best_epoch])


	# select best classification threshold for 2D images to first decimal
	thresh_list = np.arange(.1,1,.1)
	score_thresh = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
		np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=thresh_list, dtype=float)

	for t in thresh_list:
		loss_fn = FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean', threshold=t)
		metrics = { 'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='none'),
					'dice'	: DiceScore(reduction='none', threshold=t),
					'iou'	: IoUScore(reduction='none', threshold=t)}

		data, dataloader = get_dataloader(DATA_PATH+'filter/', k, K, 1, sort_train=True, sort_val=True)
		score_thresh.loc[t, ('train', k)] = evaluate(dataloader['train'], model, loss_fn, metrics)
		score_thresh.loc[t, ('val', k)] = evaluate(dataloader['val'], model, loss_fn, metrics)

	t_max = score_thresh[('train', k, 'iou')].idxmax()
	score_thresh.to_csv('score_2d_thresh.csv')

	# save 2D images of inputs, target and predicted values
	loss_fn = FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean', threshold=t_max)
	metrics = { 'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='none'),
				'dice'	: DiceScore(reduction='none', threshold=t_max),
				'iou'	: IoUScore(reduction='none', threshold=t_max)}

	data, dataloader = get_dataloader(DATA_PATH+'filter/', k, K, 1, sort_train=True, sort_val=True)
	evaluate(dataloader['train'], model, loss_fn, metrics, 
		do_plot=True, files=data['X_train'].files, savedir=SAVE_PATH+'best/img/2D/train/')
	evaluate(dataloader['val'], model, loss_fn, metrics, 
		do_plot=True, files=data['X_val'].files, savedir=SAVE_PATH+'best/img/2D/val/')


	# select best classification threshold for 3D images to first decimal
	thresh_list = np.arange(.1,1,.1)
	score_3d_thresh = pd.DataFrame(columns=pd.MultiIndex.from_product([['train', 'val'], 
		np.arange(K), ['loss', 'focal', 'dice', 'iou']]), index=thresh_list, dtype=float)

	for t in thresh_list:
		metrics_3d = {	'loss'	: FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean', dim=3, threshold=t),
						'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='mean'),
						'dice'	: DiceScore(reduction='none', dim=3, threshold=t),
						'iou'	: IoUScore(reduction='none', dim=3, threshold=t)}

		data, dataloader = get_dataloader(DATA_PATH, k, K, 1, sort_train=True, sort_val=True)
		score_3d_thresh.loc[t, ('train', k)] = evaluate_3d(data, 'train', model, metrics_3d)
		score_3d_thresh.loc[t, ('val', k)] = evaluate_3d(data, 'val', model, metrics_3d)

	t_max = score_3d_thresh[('train', k, 'iou')].idxmax()
	score_3d_thresh.to_csv('score_3d_thresh.csv')
	print("Classification threshold for 3D images: ", t_max)
	THRESHOLD = t_max

	# evaluate model in 3D + save predictions
	metrics_3d = {	'loss'	: FocalDiceLoss(**best_params, alpha=ALPHA, reduction='mean', dim=3, threshold=THRESHOLD),
					'focal'	: FocalLoss(gamma=best_params['gamma'], alpha=ALPHA, reduction='mean'),
					'dice'	: DiceScore(reduction='none', dim=3, threshold=THRESHOLD),
					'iou'	: IoUScore(reduction='none', dim=3, threshold=THRESHOLD)}

	data, dataloader = get_dataloader(DATA_PATH, k, K, 1, sort_train=True, sort_val=True)
	score_3d.loc[best_epoch, ('train', k)] = evaluate_3d(data, 'train', model, metrics_3d,
		do_save=True, savedir=SAVE_PATH+'best/img/3D/train/')
	score_3d.loc[best_epoch, ('val', k)] = evaluate_3d(data, 'val', model, metrics_3d,
		do_save=True, savedir=SAVE_PATH+'best/img/3D/val/')
	
	# save prediction for 3D test set
	data = {'X_test': CTDataset(os.path.join(DATA_PATH, 'X_test/'), normalize=True, sort=True)}
	evaluate_3d(data, 'test', model, metrics_3d,
		do_eval=False, do_save=True, savedir=SAVE_PATH+'best/img/3D/test/')
	
	print(score_3d.xs(k, axis=1, level=1).loc[best_epoch])
	score_3d.to_csv(os.path.join(SAVE_PATH, 'best', 'score_3d.csv'))

# plot loss over epochs
for m in score.columns.get_level_values(2).unique():
	history = pd.melt(score.xs(m, level=2, axis=1), var_name=['split', 'k'], ignore_index=False)
	history.value = history.value.astype(float)

	PlotResults(os.path.join(SAVE_PATH, 'best')).plot_history(history, m)

# CHECK if saving worked
iou_list = []
for file in os.listdir(SAVE_PATH+'best/img/3D/train/'):
	output = torch.load(SAVE_PATH+'best/img/3D/train/'+file)
	target = torch.tensor(nib.load(DATA_PATH+'labelsTr/'+file.rstrip('.pt')+'.nii').get_fdata())

	output = output.to(device)
	target = target.to(device)

	intersection = (output * target).sum()
	union = (output + target).sum() - intersection

	iou = (intersection + 1) / (union + 1)
	print(iou)

	iou_list.append(iou)
print(torch.tensor(iou_list).mean())