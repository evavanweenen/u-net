import os

import numpy as np
import pandas as pd
import nibabel as nib

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold

def ct3D_to_np2D(data_path, dir_in, dir_out):
	"""
	Convert 3D CT scans into 2D numpy arrays
	Read the 3D CT scans in from dir_in, and save the 2D numpy arrays in dir_out
	"""
	if not os.path.exists(os.path.join(path, dir_out)):
		os.mkdir(os.path.join(path, dir_out))

	for file in os.listdir(os.path.join(path, dir_in)):
		# read 3d ct scan
		data = nib.load(os.path.join(path, dir_in, file)).get_fdata()
		
		# split 3d image up in list of 2d images
		data = np.split(data, data.shape[-1], axis=-1)

		# save 2d image to npy file
		for i in range(len(data)):
			np.save(os.path.join(path, dir_out, file.rstrip('.nii')+'_'+str(i)), data[i])

def filter_empty(path, dir_out, dir_X='X', dir_Y='Y'):
	"""
	Move only masks (and corresponding images) that are non-empty from dir_in to dir_out
	These can then be used for training
	"""
	os.mkdir(os.path.join(path, dir_out, dir_X))
	os.mkdir(os.path.join(path, dir_out, dir_Y))	
	for file in os.listdir(os.path.join(path, dir_X)):
		X = np.load(os.path.join(path, dir_X, file))
		Y = np.load(os.path.join(path, dir_Y, file))
		if Y.sum() > 1:
			np.save(os.path.join(path, dir_out, dir_X, file), X)
			np.save(os.path.join(path, dir_out, dir_Y, file), Y)

class CTDataset(Dataset):
	"""
	Read 2D numpy arrays of CT scans from path
	Each individual image is transformed:
	- normalize indicates whether each image should be centered and standardized
	- it is reshaped to (channels x height x width)
	- it is converted to a pytorch Tensor
	"""
	def __init__(self, path, normalize=False, sort=False):
		self.path = path
		self.normalize = normalize
		self.files = np.array(os.listdir(self.path))

		# list of patient IDs for each image in self.path (useful for child classes)
		self.index = pd.DataFrame([f.lstrip('colon_').rstrip('.npy').split('_') for f in self.files], columns=['patient', 'slice'])

		if sort:
			# create sorted list of (patient, slice)
			self.index = self.index.astype({'slice':int})
			self.index = self.index.sort_values(['patient', 'slice']).reset_index(drop=True)

			# get filelist according to sorted index
			self.files = self.index.apply(lambda x: 'colon_%s_%s.npy'%(x[0], x[1]), axis=1)

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		data = np.load(os.path.join(self.path, self.files[idx]))

		# standardize
		if self.normalize:
			data = (data - np.mean(data)) / np.std(data)

		# torch wants data of shape (samples x channels x height x width)
		data = np.swapaxes(data, 0, 2)

		data = data.astype(np.float32)

		data = torch.from_numpy(data)
		return data

class CTDatasetSplit(CTDataset):
	"""
	Reads 2D numpy arrays for the training set or validation set (split) 
	of a given iteration (k) of (K)-fold cross-validation.

	The data is split by patient ID, such that one patient can only be in one fold.
	
	This class inherits functions and attributes from its parent CTDataset.
	It adjusts the file list (self.files), so that it only containsfiles
	 from the specific split (train or val) for the specific fold (k)
	"""
	def __init__(self, path, split:str, k, K=5, normalize=False, sort=False):
		super().__init__(path, normalize)

		self.split = 0 if split == 'train' else (1 if split == 'val' else 2)
		
		# list indices for the training and validation set for K folds
		self.folds = list(GroupKFold(K).split(self.files, groups=self.index.patient))

		# get fileslist for fold k and split
		self.files = self.files[self.folds[k][self.split]]
		self.index = self.index.loc[self.folds[k][self.split]]
		
		if sort:
			# create sorted list of (patient, slice)
			self.index = self.index.astype({'slice':int})
			self.index = self.index.sort_values(['patient', 'slice']).reset_index(drop=True)

			# get filelist according to sorted index
			self.files = self.index.apply(lambda x: 'colon_%s_%s.npy'%(x[0], x[1]), axis=1)