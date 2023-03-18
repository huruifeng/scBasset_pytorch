import h5py
import numpy as np
import torch

from scBasset import scBasset
from util_class import Dataset
from utils import *

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

print(f"Running on {device}...")

# %%
## improt data
print("===========================")
print("Loading data...")
output_path = 'PBMC_example'
processed_data = h5py.File('%s/processed_data.h5' % output_path, 'r')
X_dataset = processed_data["X"]
Y_dataset = processed_data["Y"]

n_peaks = X_dataset.shape[0]
n_cells = Y_dataset.shape[1]
train_ids, test_ids, val_ids = split_train_test_val(np.arange(n_peaks),train_ratio=0.9)

# Datasets
train_X = X_dataset[train_ids] # IDs
train_Y = Y_dataset[train_ids]# Labels

val_X = X_dataset[val_ids] # IDs
val_Y = Y_dataset[val_ids]# Labels

# Generators
# Parameters
params = {'batch_size': 128,
          'shuffle': True}
training_set = Dataset(train_X, train_Y)
training_DataLoader = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(val_X, val_Y)
validation_training_DataLoader = torch.utils.data.DataLoader(validation_set, **params)

# %%
###########################
print("===========================")
print("Model training...")
# Loop over epochs
max_epochs = 20
learning_rate=0.005
model = scBasset(seq_len=1344, init_dim=288, bottleneck_size=32,cell_num=n_cells,device=device)
model.to(device)

# %%
hist = model.train_model(training_DataLoader, learning_rate, max_epochs, val_loader=False)
model.save(output_path)



