import os
from sklearn.model_selection import ParameterGrid

# Path parameters
DATA_PATH = "ml4h_proj1_colon_cancer_ct/"
SAVE_PATH = "results/"

if not os.path.exists(SAVE_PATH):
	os.mkdir(SAVE_PATH)
if not os.path.exists(SAVE_PATH+'best/'):
	os.mkdir(SAVE_PATH+'best/')
if not os.path.exists(SAVE_PATH+'best/img/'):
	os.mkdir(SAVE_PATH+'best/img/')
if not os.path.exists(SAVE_PATH+'best/img/2D/'):
	os.mkdir(SAVE_PATH+'best/img/2D/')
if not os.path.exists(SAVE_PATH+'best/img/2D/train/'):
	os.mkdir(SAVE_PATH+'best/img/2D/train/')
if not os.path.exists(SAVE_PATH+'best/img/2D/val/'):
	os.mkdir(SAVE_PATH+'best/img/2D/val/')
if not os.path.exists(SAVE_PATH+'best/img/3D/'):
	os.mkdir(SAVE_PATH+'best/img/3D/')
if not os.path.exists(SAVE_PATH+'best/img/3D/train/'):
	os.mkdir(SAVE_PATH+'best/img/3D/train/')
if not os.path.exists(SAVE_PATH+'best/img/3D/val/'):
	os.mkdir(SAVE_PATH+'best/img/3D/val/')
if not os.path.exists(SAVE_PATH+'best/img/3D/test/'):
	os.mkdir(SAVE_PATH+'best/img/3D/test/')

# Hyperparameters
SEED = 42

K = 5

BATCH_SIZE = 16

EPOCHS = 300
EPOCHS_RETRAIN = 1000

PATIENCE = 5
PATIENCE_RETRAIN = 50

LEARNING_RATE = 1e-3

ALPHA = .1 

FILTERS = [64, 128, 256, 512, 1024]

THRESHOLD = .5

# Optimizable hyperparameters
PARAM_GRID = list(ParameterGrid({'gamma': [1, 2, 3], 'weight': [0, .5, 1, 2]}))