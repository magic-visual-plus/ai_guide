
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


def read_data(path, basename):
    pcd_filename = os.path.join(path, basename + '.ply')
    label_filename = os.path.join(path, basename + '.txt')
    with open(label_filename, 'r') as f:
        label_indices = [int(line.strip()) for line in f]
        pass
    pcd = o3d.io.read_point_cloud(pcd_filename)
    label = np.zeros(len(pcd.points), dtype=np.int32)
    label[label_indices] = 1
    return pcd, label


class PointNetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, size=64, is_test=False, voxel_size=0):
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
        self.is_test = is_test
        if is_test:
            self.size = len(self.basenames)
        else:
            self.size = size
            pass
        self.voxel_size = voxel_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.is_test:
            idx = random.randint(0, len(self.basenames) - 1)
            pass
        basename = self.basenames[idx]
        pcd, label = read_data(self.data_path, basename)
        select_index = pcd_utils.select_points(pcd)
        pcd = pcd.select_by_index(select_index)
        label = label[select_index]

        if self.voxel_size > 0:
            pcd, _, trace = pcd.voxel_down_sample_and_trace(
                voxel_size=self.voxel_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())
            
            label_down = np.zeros(len(pcd.points), dtype=np.int32)
            for i, id_list in enumerate(trace):
                label_down[i] = np.max(label[id_list])
                pass
            label = label_down
            pass

        if not self.is_test:
            pcd = pcd_utils.transform(pcd)
            pass
        x, x_sampled, group_index, knn_index = pcd_utils.generate_model_data(pcd, sample_size=512)
        point_indices = np.where(label)[0]

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


class PointNetDatasetPickled(torch.utils.data.Dataset):
    def __init__(self, data_path, size=64, is_test=False):
        self.data_path = data_path
        self.size = size
        self.is_test = is_test
        filenames = os.listdir(data_path)
        filenames = [f for f in filenames if f.endswith('.pkl')]
        self.filenames = filenames

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if True:
            idx = random.randint(0, len(self.filenames) - 1)
            pass
        filename = self.filenames[idx]
        with open(os.path.join(self.data_path, filename), 'rb') as f:
            data = pickle.load(f)
            pass
        x, x_sampled, group_index, knn_index, point_indices = data

        r = R.random().as_matrix()
        b = np.random.uniform(-1, 1, 3)
        x[:, :3] = (r @ x[:, :3].T).T + b
        x_sampled[:, :3] = (r @ x_sampled[:, :3].T).T + b

        std = np.std(x[:, :3], axis=0)
        x[:, :3] = x[:, :3] / std
        x_sampled[:, :3] = x_sampled[:, :3] / std
        
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



class PointTransformerDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, size=64, is_test=False, voxel_size=0):
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
        self.is_test = is_test
        if is_test:
            self.size = len(self.basenames)
        else:
            self.size = size
            pass
        self.voxel_size = voxel_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not self.is_test:
            idx = random.randint(0, len(self.basenames) - 1)
            pass
        basename = self.basenames[idx]
        pcd, label = read_data(self.data_path, basename)
        select_index = pcd_utils.select_points(pcd)
        pcd = pcd.select_by_index(select_index)
        label = label[select_index]

        if self.voxel_size > 0:
            pcd, _, trace = pcd.voxel_down_sample_and_trace(
                voxel_size=self.voxel_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())
            
            label_down = np.zeros(len(pcd.points), dtype=np.int32)
            for i, id_list in enumerate(trace):
                label_down[i] = np.max(label[id_list])
                pass
            label = label_down
            pass

        if not self.is_test:
            pcd = pcd_utils.transform(pcd)
            pass

        x, feat = pcd_utils.generate_model_data2(pcd)
        point_indices = np.where(label)[0]

        x = torch.from_numpy(x).float()
        feat = torch.from_numpy(feat).float()

        return x, feat, point_indices
    pass


def collate_fn_pt(batch):

    xs = []
    feats = []
    labels = []
    offsets = []
    offset = 0
    for x, feat, point_indices in batch:
        label = torch.zeros(x.size(0))
        label[point_indices] = 1

        xs.append(x)
        feats.append(feat)
        labels.append(label)
        offset += x.size(0)
        offsets.append(offset)
        pass

    x = torch.cat(xs, dim=0)
    feat = torch.cat(feats, dim=0)
    label = torch.cat(labels, dim=0)
    offset = torch.tensor(offsets)

    return x, feat, label, offset

