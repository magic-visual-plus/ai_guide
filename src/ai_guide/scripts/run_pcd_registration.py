from ai_guide import registration, pcd_utils
import sys
import open3d as o3d
import numpy as np
import scipy.spatial
import time
from ai_guide import foreground_extractor


if __name__ == '__main__':
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
    model_path = sys.argv[3]
    output_path = sys.argv[4]

    src_pcd = o3d.io.read_point_cloud(src_path)
    dst_pcd = o3d.io.read_point_cloud(dst_path)

    foreground_extractor = foreground_extractor.ForegroundExtractor(model_path, volume_size=4, threshold=0.5)
    # src_pcd = registration.remove_background(src_pcd, distance_threshold=-50)
    # for i in range(5):
    #     dst_pcd_ = registration.remove_background(dst_pcd, distance_threshold=-5)
    #     start = time.time()
    #     loss, R, t = registration.point_cloud_registration(src_pcd, dst_pcd_, loss_max=5.0, retry=2, volume_size=4)
    #     print(f'time cost: {time.time() - start}, rmse: {loss}')
    #     if loss > 5:
    #         continue
    #     break

    dst_pcd_ = foreground_extractor.extract(dst_pcd)
    start = time.time()
    loss, R, t = registration.point_cloud_registration(src_pcd, dst_pcd_, loss_max=5.0, retry=2, volume_size=4)
    print(f'time cost: {time.time() - start}, rmse: {loss}')

    src_points = np.asarray(src_pcd.points)
    dst_points = np.asarray(dst_pcd.points)

    np.nan_to_num(src_points, copy=False)
    np.nan_to_num(dst_points, copy=False)

    src_points = (R @ src_points.T).T + t
    
    # write out
    all_points = np.concatenate([src_points, dst_points], axis=0)
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
    colors = np.zeros((len(all_points), 3))
    colors[len(src_points):, 0] = 1
    colors[:len(src_points), 1] = 1
    all_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path, all_pcd)


    pass