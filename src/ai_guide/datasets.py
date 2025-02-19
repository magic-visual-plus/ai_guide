
import torch
import os
import pickle
from scipy.spatial.transform import Rotation as R
import random
import numpy as np


class PointNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, size=None):
        self.data_path = data_path
        self.filenames = os.listdir(data_path)
        self.filenames = [f for f in self.filenames if f.endswith('.pkl')]
        if size is not None:
            self.size = size
        else:
            self.size = len(self.filenames)
        pass

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.size != len(self.filenames):
            idx = random.randint(0, len(self.filenames) - 1)
            pass

        filename = self.filenames[idx]
        with open(os.path.join(self.data_path, filename), 'rb') as f:
            data = pickle.load(f)
            pass

        x, x_sampled, group_index, knn_index, point_indices = data

        # random rotate x
        if self.size != len(self.filenames):
            r = R.random().as_matrix()
            b = np.random.uniform(-1, 1, 3)
        else:
            r = np.eye(3)
            b = np.zeros(3)
            pass
        
        x[:, :3] = (r @ x[:, :3].T).T + b
        x_sampled[:, :3] = (r @ x_sampled[:, :3].T).T + b
        
        x = torch.from_numpy(x).float()
        x_sampled = torch.from_numpy(x_sampled).float()
        group_index = torch.from_numpy(group_index).long()
        knn_index = torch.from_numpy(knn_index).long()
        point_indices = torch.from_numpy(point_indices).long()
        return x, x_sampled, group_index, knn_index, point_indices
    pass


def collate_fn(batch):
    max_x_size = 0
    max_x_sampled_size = 0
    max_group_size = 0
    for x, x_sampled, group_index, knn_index, point_indices in batch:
        max_x_size = max(max_x_size, x.shape[0])
        max_x_sampled_size = max(max_x_sampled_size, x_sampled.shape[0])
        max_group_size = max(max_group_size, group_index.shape[1])
        pass

    x_batch = torch.zeros((len(batch), max_x_size, x.shape[1]))
    x_sampled_batch = torch.zeros((len(batch), max_x_sampled_size, x_sampled.shape[1]))
    group_index_batch = -torch.ones((len(batch), max_x_sampled_size, max_group_size), dtype=torch.long)
    knn_index_batch = torch.zeros((len(batch), max_x_size, knn_index.shape[1]), dtype=torch.long)
    labeled_batch = torch.zeros((len(batch), max_x_size))
    mask_batch = torch.zeros((len(batch), max_x_size))
    mask_sampled_batch = torch.zeros((len(batch), max_x_sampled_size))

    for i, (x, x_sampled, group_index, knn_index, point_indices) in enumerate(batch):
        x_batch[i, :x.shape[0]] = x
        x_sampled_batch[i, :x_sampled.shape[0]] = x_sampled
        group_index_batch[i, :group_index.shape[0], :group_index.shape[1]] = group_index
        knn_index_batch[i, :knn_index.shape[0], :] = knn_index
        labeled_batch[i, point_indices] = 1
        mask_sampled_batch[i, :x_sampled.shape[0]] = 1
        mask_batch[i, :x.shape[0]] = 1
        pass

    return x_batch, x_sampled_batch, group_index_batch, knn_index_batch, labeled_batch, mask_batch, mask_sampled_batch