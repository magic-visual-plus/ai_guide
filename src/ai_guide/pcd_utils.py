import numpy as np
from scipy.spatial.distance import cdist
import open3d as o3d
import fpsample
import time
import scipy.spatial


def group(x, x_sampled):
    # x: (N, D)
    # x_sampled: (M, D)
    # output: (M, G)

    batch_size = 64

    nn_index = []
    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i:i + batch_size]
        dist = cdist(x_batch, x_sampled)
        index = np.argmin(dist, axis=1)
        nn_index.append(index)
        pass
    nn_index = np.concatenate(nn_index, axis=0)
    ind, counts = np.unique(nn_index, return_counts=True)
    group_index = -np.ones((x_sampled.shape[0], counts.max()), dtype=np.int32)
    cols = np.zeros(x_sampled.shape[0], dtype=np.int32)

    for i, n in enumerate(nn_index):
        group_index[n, cols[n]] = i
        cols[n] += 1
        pass

    return group_index


def group2(x, x_sampled):
    batch_size = 64
    bucket_size = 64

    groups = [
        [] for _ in range(x_sampled.shape[0])
    ]

    for i in range(0, x.shape[0], batch_size):
        x_batch = x[i:i + batch_size]
        dist = cdist(x_batch, x_sampled)
        index = np.argmin(dist, axis=1)
        for j, n in enumerate(index):
            groups[n].append(i + j)
            pass
        pass
    
    group_index = np.zeros((x_sampled.shape[0], bucket_size), dtype=np.int32)
    for ig, g in enumerate(groups):
        if len(g) > bucket_size:
            x_group = x[g]
            index_sampled = fpsample.fps_sampling(x_group, bucket_size)
            group_index[ig, :] = np.asarray(g)[index_sampled]
            pass
        elif len(g) < bucket_size:
            dist = cdist(x_sampled[ig:ig+1], x)
            index = np.argsort(dist[0])

            gs = set(g)
            i = 0
            while len(gs) < bucket_size:
                gs.add(index[i])
                i += 1
                pass
            group_index[ig, :] = list(gs)
            pass
        else:
            group_index[ig, :] = g
            pass

        pass

    return group_index


def find_nearest(x, group_index):
    group_index = group_index.reshape(-1)
    uindex = np.unique(group_index)
    if uindex[0] == -1:
        uindex = uindex[1:]
        pass
    x_group = x[uindex]
    tree = scipy.spatial.cKDTree(x_group)

    batch_size = 64
    k = 4

    nn_index = np.zeros((x.shape[0], 4), dtype=np.int32)

    for i in range(0, x.shape[0], batch_size):
        # x_batch = x[i:i + batch_size]
        # dist = cdist(x_batch, x_group)
        # index = np.argsort(dist, axis=1)
        index = tree.query(x[i:i + batch_size], k=k)[1]
        nn_index[i:i + batch_size] = uindex[index[:, :k]]
        pass

    return nn_index
    pass

def select_points(pcd):
    x = np.asarray(pcd.points).astype(np.float32)
    mask_nonzero = np.abs(x).sum(axis=1) > 0
    index_nonzero = np.where(mask_nonzero)[0]
    return index_nonzero

def generate_model_data(pcd, sample_size=256):
    x = np.asarray(pcd.points).astype(np.float32)
    x_color = np.asarray(pcd.colors).astype(np.float32)
    
    index_sampled = fpsample.fps_sampling(x, sample_size)
    x_sampled = x[index_sampled]
    x_sampled_color = x_color[index_sampled]

    group_index = group2(x, x_sampled)

    start = time.time()
    knn_index = find_nearest(x, group_index)
    print("Time taken for find_nearest: ", time.time() - start)

    x = (x - x.mean(axis=0, keepdims=True)) / 10
    x_sampled = (x_sampled - x_sampled.mean(axis=0, keepdims=True)) / 10

    x = np.concatenate([x, x_color], axis=1)
    x_sampled = np.concatenate([x_sampled, x_sampled_color], axis=1)

    return x, x_sampled, group_index, knn_index
    pass


def replace_nan(pcd, v):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    nan_mask = np.isnan(points).any(axis=1)
    points[nan_mask] = v
    colors[nan_mask] = 0

    pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
