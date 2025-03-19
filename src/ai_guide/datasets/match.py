import torch
import os
import random
import numpy as np
import open3d as o3d
from .. import pcd_utils
import copy


class PointMatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, size=64, is_test=False, voxel_size=0):
        self.data_path = data_path
        filenames = os.listdir(data_path)
        filenames = [f for f in filenames if f.endswith('.ply')]
        self.filenames = filenames
        self.is_test = is_test
        if is_test:
            self.size = len(self.filenames)
        else:
            self.size = size
            pass
        self.voxel_size = voxel_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # torch.manual_seed(0)
        # np.random.seed(0)
        # random.seed(0)
        if not self.is_test:
            idx = random.randint(0, len(self.filenames) - 1)
            pass
        filename = self.filenames[idx]
        pcd = o3d.io.read_point_cloud(os.path.join(self.data_path, filename))

        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pass

        if not self.is_test:
            pcd = pcd_utils.transform(pcd)
            pass

        pcd = pcd_utils.move_to_center(pcd)
        # generate transform matrix
        r = np.random.rand(3) * 2 * np.pi
        t = (np.random.rand(3) - 0.5) * 10
        T = np.eye(4)
        T[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(r)
        T[:3, 3] = t

        pcd_transformed = copy.deepcopy(pcd).transform(T)


        x0 = np.asarray(pcd.points)
        feat0 = np.asarray(pcd.colors)
        x1 = np.asarray(pcd_transformed.points)
        feat1 = np.asarray(pcd_transformed.colors)

        # print(x0)
        # print(x1)
        shuffle_index = np.random.permutation(x1.shape[0])
        x1 = x1[shuffle_index]
        feat1 = feat1[shuffle_index]
        
        x0 = torch.from_numpy(x0).float()
        x1 = torch.from_numpy(x1).float()
        feat0 = torch.from_numpy(feat0).float()
        feat1 = torch.from_numpy(feat1).float()
        T = torch.from_numpy(T).float()

        return x0, feat0, x1, feat1, T
    pass


def collate_fn(batch):

    xs0 = []
    feats0 = []
    xs1 = []
    feats1 = []
    Ts = []
    offsets0 = []
    offsets1 = []
    offset0 = 0
    offset1 = 0
    for x0, feat0, x1, feat1, T in batch:
        xs0.append(x0)
        feats0.append(feat0)
        xs1.append(x1)
        feats1.append(feat1)
        Ts.append(T)
        offset0 += x0.size(0)
        offset1 += x1.size(0)
        offsets0.append(offset0)
        offsets1.append(offset1)
        pass

    x0 = torch.cat(xs0, dim=0)
    x1 = torch.cat(xs1, dim=0)
    feat0 = torch.cat(feats0, dim=0)
    feat1 = torch.cat(feats1, dim=0)
    T = torch.stack(Ts, dim=0)
    offset0 = torch.tensor(offsets0)
    offset1 = torch.tensor(offsets1)

    return x0, feat0, offset0, x1, feat1, offset1, T