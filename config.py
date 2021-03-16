import os
from sklearn.model_selection import ParameterGrid

DATA_PATH = "ml4h_proj1_colon_cancer_ct/"
SAVE_PATH = "results/"

if not os.path.exists(SAVE_PATH):
	os.mkdir(SAVE_PATH)

SEED = 42

K = 5

BATCH_SIZE = 16

EPOCHS = 20
PATIENCE = 5

LEARNING_RATE = 1e-3

ALPHA = .1 

FILTERS = [64, 128, 256, 512]

PARAM_GRID = list(ParameterGrid({'gamma': [1, 2, 3], 'weight': [0, .5, 1, 2]}))