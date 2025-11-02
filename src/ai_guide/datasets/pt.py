import torch
import os
import random
import open3d as o3d
import numpy as np
from .. import pcd_utils


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
            if not self.is_test:
                voxel_size = self.voxel_size + random.uniform(-1.0, 1.0)
                pass
            else:
                voxel_size = self.voxel_size
                pass
            pcd, _, trace = pcd.voxel_down_sample_and_trace(
                voxel_size=voxel_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())
            
            label_down = np.zeros(len(pcd.points), dtype=np.int32)
            for i, id_list in enumerate(trace):
                label_down[i] = np.max(label[id_list])
                pass
            label = label_down
            pass

        if not self.is_test:
            pcd = pcd_utils.transform(pcd)

            # if random.random() < 0.9:
            #     # random drop background
            #     selected_index = np.where(label == 0)[0]
            #     num = len(selected_index) * np.random.uniform(0.5, 1.0)
            #     if num == 0:
            #         num = 1
            #         pass

            #     selected_index = np.random.choice(selected_index, int(num), replace=False)
            #     selected_index = np.unique(np.concatenate((selected_index, np.where(label == 1)[0])))
            #     pcd = pcd.select_by_index(selected_index)
            #     label = label[selected_index]
            #     pass
            pass

        x, feat = pcd_utils.generate_model_data2(pcd)
        point_indices = np.where(label)[0]
        # print(x.shape)
        x = torch.from_numpy(x).float()
        feat = torch.from_numpy(feat).float()

        return x, feat, point_indices
    pass


def collate_fn(batch):

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
