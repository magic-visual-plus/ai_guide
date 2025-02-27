
import torch
import os
import pickle
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
import random
import numpy as np
from ai_guide import pcd_utils
import open3d as o3d
import os


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


class HierarchicalPointNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, size=None, is_test=False, layers=[0, 1]):
        self.data_path = data_path
        filenames = os.listdir(data_path)
        filenames = [f for f in filenames if f.endswith('.ply')]
        self.basenames = []
        
        for filename in filenames:
            basename = filename[:-4]
            label_filename = basename + '.txt'
            if not os.path.exists(os.path.join(data_path, label_filename)):
                continue
            self.basenames.append(basename)
            pass

        if size is None:
            self.size = 128
        else:
            self.size = size
            pass
        
        self.layers = set(layers)
        self.is_test = is_test 
        self.buffer = []
        self.fill_buffer()

        if is_test:
            # self.size = len(self.buffer)
            pass
        

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if len(self.buffer) == 0:
            self.fill_buffer()
            pass
        
        # if not self.is_test:
        if True:
            pcd, label = self.buffer[-1]
            self.buffer.pop(-1)
            pass
        else:
            pcd, label = self.buffer[idx]
            pass

        x, x_sampled, group_index, knn_index = pcd_utils.generate_model_data(pcd, sample_size=512)
        point_indices = np.where(label)[0]

        # random rotate x
        if not self.is_test:
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
    
    def fill_buffer(self, ):
        idx = random.randint(0, len(self.basenames) - 1)

        basename = self.basenames[idx]
        label_filename = basename + '.txt'
        pcd_filename = basename + '.ply'

        label_indices = []
        with open(os.path.join(self.data_path, label_filename), 'r') as f:
            for line in f:
                idx = int(line.strip())
                label_indices.append(idx)
                pass
            pass
        
        pcd = o3d.io.read_point_cloud(os.path.join(self.data_path, pcd_filename))
        labels = np.zeros(len(pcd.points), dtype=np.int32)
        labels[label_indices] = 1

        select_index = pcd_utils.select_points(pcd)

        pcd = pcd.select_by_index(select_index)
        labels = labels[select_index]

        points = np.asarray(pcd.points)
        points = (points - points.mean(axis=0, keepdims=True))
        r = R.random().as_matrix()
        b = np.random.uniform(-10, 10, 3)
        points = (r @ points.T).T + b
        pcd.points = o3d.utility.Vector3dVector(points)

        voxel_size = 50 + random.randint(-10, 10)
        pcd_down, trace = pcd_utils.split_pcd(pcd, sample_size=voxel_size)
        
        points_down = np.asarray(pcd_down.points)
        colors_down = np.asarray(pcd_down.colors)

        label_down = np.zeros(len(points_down), dtype=np.int32)
        for i, idx in enumerate(trace):
            if len(idx) == 0:
                continue
            label_down[i] = np.max(labels[idx])
            pass
        # print(np.sum(label_down))
        num_neg = np.sum(label_down == 0)
        num_pos = np.sum(label_down == 1)
        neg_proba = min(num_pos / num_neg, 1.0)

        if 0 in self.layers:
            for _ in range(10):
                self.buffer.append((pcd_down, label_down))
                pass
            pass

        tree = KDTree(points)
        if 1 in self.layers:
            for i, idx in enumerate(trace):
                if len(idx) == 0:
                    continue
                if label_down[i] == 1:
                    index = idx
                    # subpoints = points[idx]
                    # pdown = np.concatenate((points_down[:i], points_down[i+1:]))
                    # cdown = np.concatenate((colors_down[:i], colors_down[i+1:]))

                    # p = np.concatenate([subpoints, pdown])
                    # c = np.concatenate([colors[idx], cdown])
                    # sublabel = np.concatenate([labels[idx], np.zeros(len(pdown), dtype=np.int32)])

                    # subpcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(p))
                    # subpcd.colors = o3d.utility.Vector3dVector(c)
                    subpcd = pcd.select_by_index(index)
                    sublabel = labels[index]
                    self.buffer.append((subpcd, sublabel))
                    pass
                elif random.random() < neg_proba:
                    subpcd = pcd.select_by_index(idx)
                    sublabel = labels[idx]
                    # self.buffer.append((subpcd, sublabel))
                    pass
                pass
            pass
        pass
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