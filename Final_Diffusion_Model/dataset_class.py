import os
import numpy as np
from torch.utils.data import Dataset
import torch

class NumpyDataset(Dataset):
    def __init__(self, npy_dir):
        self.npy_dir = npy_dir
        self.npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.npy_dir, self.npy_files[idx])
        # Load the numpy array
        npy_data = np.load(npy_path)

        # Ensure that the numpy array is in the shape (4, 64, 64)
        npy_data = np.reshape(npy_data, (4, 64, 64))

        # Convert numpy array to torch tensor
        tensor_data = torch.from_numpy(npy_data).float()  # Convert to float tensor

        return tensor_data
