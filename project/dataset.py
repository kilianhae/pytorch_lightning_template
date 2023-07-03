"""
Define all the dataset related code.
Most importantly the datasets and the loading of data.
"""

import numpy as np
import torch

def load_dataset(**kwargs):
    # as a test we generate a dataset that is a simple non-linear function: a sinus wave
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(x)
    return np.concatenate([x[:,None], y[:,None]], axis=1)


class Dataset():
    def __init__(self, data, validation, cut=0.8, **kwargs):
        self.data = data
        self.validation = validation
        self.n = data.shape[0]

        self.train_size = int(round(self.n * 0.8))
        self.val_size = self.n - self.train_size
        self.permutation = np.random.RandomState().permutation(self.n)
        self.train_indices = self.permutation[:self.train_size]
        self.val_indices = self.permutation[self.train_size:]
    
    def __len__(self):
        if self.validation:
            return self.val_size
        else:
            return self.train_size
    
    def __getitem__(self, idx):
        if self.validation:
            idx = self.val_indices[idx]
        else:
            idx = self.train_indices[idx]
        
        return torch.tensor(self.data[idx,:1]).float(), torch.tensor(self.data[idx,1:]).float()
    
