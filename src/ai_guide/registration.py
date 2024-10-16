import open3d as o3d

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