import open3d as o3d


def create_surface_from_cloud_points(pcd: o3d.geometry.PointCloud, min_radius) -> o3d.geometry.TriangleMesh:
    # check existence of normals
    if not pcd.has_normals():
        pcd.estimate_normals()
        pass

    radii = [min_radius, min_radius * 2, min_radius * 4]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    return mesh