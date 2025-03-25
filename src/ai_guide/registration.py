import open3d as o3d
import torch
import itertools
import scipy.spatial
import numpy as np
import time
import torch.random
import pytorch3d.ops
from . import pcd_utils


def register_point_cloud(src, dst):
    voxel_size = 5.0
    radius_feature = voxel_size * 5
    pcd1_small = src.voxel_down_sample(voxel_size=voxel_size)
    pcd1_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd1_small,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    pcd2_small = dst.voxel_down_sample(voxel_size=voxel_size)
    pcd2_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd2_small,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd1_small, pcd2_small, pcd1_fpfh, pcd2_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    distance_threshold = 0.1
    result = o3d.pipelines.registration.registration_icp(
        src, dst, distance_threshold, result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    r = result.transformation[:3, :3]
    t = result.transformation[:3, 3]

    return r, t


def icp_registration_with_torch2(src, dst, dst_tree, R, t, device, num_sample):
    src = torch.from_numpy(src).to(device).float()
    dst = torch.from_numpy(dst).to(device).float()

    R = torch.from_numpy(R).to(device).float()
    t = torch.from_numpy(t).to(device).float()
    s = torch.ones(t.shape[0]).to(device).float()

    print(R.shape, t.shape, s.shape)
    init = pytorch3d.ops.points_alignment.SimilarityTransform(R, t, s)
    result = pytorch3d.ops.iterative_closest_point(src, dst, init, estimate_scale=False)

    R, t, s = result.RTs
    loss = result.rmse
    R = R.transpose(1, 2)
    return loss.cpu().numpy(), R.squeeze(0).cpu().numpy(), t.cpu().numpy()

def icp_registration_with_torch(src, dst, dst_tree, R, t, device, num_sample):

    R = torch.nn.Parameter(
        torch.from_numpy(R).to(device).float(), requires_grad=True)
    t = torch.nn.Parameter(
        torch.from_numpy(t).to(device).float(), requires_grad=True)

    src = torch.from_numpy(src).to(device).float()
    dst = torch.from_numpy(dst).to(device).float()

    optimizer = torch.optim.SGD([R, t], lr=1e-4)

    last_loss = 1e10
    for iter in range(1000):
        start = time.time()
        sampled_indices = torch.randperm(len(src))[:num_sample]
        src_sample = src[sampled_indices]
        src_ = (R @ src_sample.T).T + t
        
        # src_np = src_.detach().cpu().numpy()
        # _, indices = dst_tree.query(src_np, 1)
        index = []
        for i in range(0, len(src_), 128):
            src_batch = src_[i:i+128]
            dists = torch.cdist(src_batch, dst)
            _, indices = torch.min(dists, dim=1)
            indices = indices.squeeze()
            index.append(indices)
        indices = torch.cat(index, dim=0)
        # print("time: ", time.time() - start)
        dst_comp = dst[indices]
        # dst_comp = torch.from_numpy(dst_comp).to(device).float()
        loss = torch.nn.functional.mse_loss(src_, dst_comp)
        dist = torch.norm(src_ - dst_comp, dim=1).mean()
        print(dist)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if abs(last_loss - loss.item()) < 1e-4:
            break

        last_loss = loss.item()
        pass

    return loss, R.detach().cpu().numpy(), t.detach().cpu().numpy()
    pass

def remove_background(pcd):
    index = pcd_utils.select_points(pcd)
    pcd = pcd.select_by_index(index)

    # find plane
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.5,
        ransac_n=3,
        num_iterations=1000)
    [a, b, c, d] = plane_model
    # remove the plane
    points = np.asarray(pcd.points)
    distance = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    index = np.where((distance <= -3.0) & (points[:, 0] < 300) & (points[:, 0] > -300))[0]
    pcd = pcd.select_by_index(index)
    pcd, _ = pcd.remove_radius_outlier(nb_points=30, radius=10)
    
    return pcd

def icp_registration(src, dst):
    # move to center
    center_src = np.mean(src, axis=0, keepdims=True)
    center_dst = np.mean(dst, axis=0, keepdims=True)
    num_sample = 5000

    src = src - center_src
    dst = dst - center_dst

    try_x_angles = [0]
    try_y_angles = [0]
    try_z_angles = [0, np.pi]

    try_angles = list(itertools.product(try_x_angles, try_y_angles, try_z_angles))

    dst_tree = scipy.spatial.cKDTree(dst)
    src_sampled = src[np.random.choice(len(src), num_sample)]

    Rs = []
    ts = []
    for angle in try_angles:
        R = o3d.geometry.get_rotation_matrix_from_xyz(angle)
        t = np.zeros((3))

        Rs.append(R)
        ts.append(t)
        pass

    Rs = np.stack(Rs, axis=0)
    ts = np.stack(ts, axis=0)
    src = np.tile(src_sampled[None, ...], (len(try_angles), 1, 1))
    dst = np.tile(dst[None, ...], (len(try_angles), 1, 1))

    loss, R, t = icp_registration_with_torch2(src, dst, dst_tree, Rs, ts, "cuda:0", num_sample)
    print(loss)
    index = np.argmin(loss)
    R = R[index]
    t = t[index]
    loss = loss[index]

    return loss, R, t - (R @ center_src.T).T + center_dst
    pass

def point_cloud_registration(pcd_src, pcd_dst):
    # (R @ pcd_src.T).T + t = pcd_dst

    pcd_src = remove_background(pcd_src)
    pcd_dst = remove_background(pcd_dst)

    src = np.asarray(pcd_src.points)
    dst = np.asarray(pcd_dst.points)

    loss, R, t = icp_registration(src, dst)

    return loss, R, t
    pass