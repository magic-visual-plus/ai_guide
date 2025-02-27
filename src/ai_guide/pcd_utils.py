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
    bucket_size = min(32, x.shape[0])

    groups = [
        [] for _ in range(x_sampled.shape[0])
    ]

    tree = scipy.spatial.cKDTree(x_sampled)

    x_idx = tree.query(x, k=1)[1]
    for i, n in enumerate(x_idx):
        groups[n].append(i)
        pass
    
    group_index = np.zeros((x_sampled.shape[0], bucket_size), dtype=np.int32)

    tree = scipy.spatial.cKDTree(x)
    for ig, g in enumerate(groups):
        if len(g) > bucket_size:
            x_group = x[g]
            index_sampled = fpsample.fps_sampling(x_group, bucket_size)
            group_index[ig, :] = np.asarray(g)[index_sampled]
            pass
        elif len(g) < bucket_size:
            index = tree.query(x_sampled[ig], k=bucket_size)[1]

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

    k = 4

    nn_index = np.zeros((x.shape[0], 4), dtype=np.int32)

    index = tree.query(x, k=k)[1]
    index[index == tree.n] = 0
    nn_index = uindex[index]

    return nn_index
    pass

def select_points(pcd):
    x = np.asarray(pcd.points).astype(np.float32)
    x = np.nan_to_num(x, nan=0)

    mask_nonzero = np.abs(x).sum(axis=1) > 0
    index_nonzero = np.where(mask_nonzero)[0]
    return index_nonzero

def generate_model_data(pcd, sample_size=512):
    x = np.asarray(pcd.points).astype(np.float32)
    x_color = np.asarray(pcd.colors).astype(np.float32)
    
    if x.shape[0] < sample_size:
        sample_size = x.shape[0]
        pass
    # index_sampled = fpsample.fps_sampling(x, sample_size)
    index_sampled = fpsample.fps_npdu_sampling(x, sample_size)

    x_sampled = x[index_sampled]
    x_sampled_color = x_color[index_sampled]

    group_index = group2(x, x_sampled)

    start = time.time()
    knn_index = find_nearest(x, group_index)
    # print("Time taken for find_nearest: ", time.time() - start)

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

def filter_nan_and_zero(pcd):
    points = np.asarray(pcd.points)

    nan_mask = np.isnan(points).any(axis=1)
    points[nan_mask] = 0
    zero_mask = np.all(points == 0, axis=1)
    zero_index = np.where(~zero_mask)[0]

    return pcd.select_by_index(zero_index)

def split_pcd(pcd, sample_size=50):
    pcd_down, _, trace = pcd.voxel_down_sample_and_trace(
        voxel_size=sample_size, min_bound=pcd.get_min_bound(), max_bound=pcd.get_max_bound())
    # pcd_down = pcd.voxel_down_sample(voxel_size=sample_size)
    points_down = np.asarray(pcd_down.points)
    points = np.asarray(pcd.points)
    tree = scipy.spatial.cKDTree(points)
    for i in range(len(points_down)):
        idx = trace[i]
        
        if len(idx) < 10000:
            mean = np.mean(points[idx], axis=0)
            index = tree.query(mean, k=10000)[1]
            idx_set = set(idx)
            for j in index:
                if j not in idx_set:
                    idx.append(j)
                    pass
                if len(idx) >= 10000:
                    break
                pass
            pass
        else:
            index = idx
            pass
        trace[i] = np.unique(index)
        pass

    return pcd_down, trace


def remove_background_plane(pcd):
    pcd_down = pcd.voxel_down_sample(voxel_size=10)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10))
    # pcd_down = pcd.farthest_point_down_sample(10000)
    points_down = np.asarray(pcd_down.points)
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # # find plane with max z
    # indices = np.argpartition(points_down[:, 2], -10)[-10:]
    # plane_points = points_down[indices]
    # num_normals = 100
    # normal_index =  []
    # while len(normal_index) < num_normals:
    #     index = np.random.choice(10, 3, replace=False)
    #     if index[0] == index[1] or index[1] == index[2] or index[0] == index[2]:
    #         continue
    #     normal_index.append(index)
    #     pass
    
    # normal_index = np.array(normal_index)
    # v1 = plane_points[normal_index[:, 1]] - plane_points[normal_index[:, 0]]
    # # v1: (100, 3)
    # v2 = plane_points[normal_index[:, 2]] - plane_points[normal_index[:, 0]]
    # # v2: (100, 3)
    # normals = np.cross(v1, v2)
    # # normal: (100, 3)
    # normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # # normal: (100, 3)
    # normal = np.mean(normals, axis=0, keepdims=True)
    # # find distance from plane
    # diff =points - plane_points[np.random.randint(0, 10, size=len(points))]
    # # diff: (N, 3)
    # dist = np.abs(np.sum(diff * normal, axis=1))
    # mask = dist >= 5

    plane_model, inliers = pcd_down.segment_plane(
        distance_threshold=1,
        ransac_n=3,
        num_iterations=1000)
    
    a, b, c, d = plane_model

    plane_normal = np.array([a, b, c])
    dist = np.abs(a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    cos = np.abs(np.sum(normals * plane_normal, axis=1))

    # mask = (dist >= 10.0) | (cos > 0.1)
    mask = dist > 5
    # mask = np.ones(len(points), dtype=bool)
    # mask[inliers] = False

    return pcd.select_by_index(np.where(mask)[0])
