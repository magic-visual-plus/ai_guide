
import unittest
import ai_guide.surface
import open3d as o3d
import ai_guide.segmentation
import numpy as np

class TestSegmentation(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test
        pass

    def test_remove_background(self):
        pcd = o3d.io.read_point_cloud("data/test2.ply")
        pcd = ai_guide.segmentation.remove_background(pcd)
        o3d.io.write_point_cloud("data/test2_no_background.ply", pcd)
        pass

    def test_find_keypoint(self):
        pcd = o3d.io.read_point_cloud("data/test_no_background.ply")
        index = ai_guide.segmentation.find_keypoint(pcd)

        # paint color by index
        colors = np.asarray(pcd.colors)
        colors[index] = [1, 0, 0]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud("data/test_keypoint.ply", pcd)

        pass

    def test_segment_plane(self):
        mesh = o3d.io.read_triangle_mesh("data/test2_mesh.ply")
        points = np.asarray(mesh.vertices)
        keypoint = ai_guide.segmentation.find_keypoint(points)
        index = ai_guide.segmentation.segment_plane(mesh, keypoint)
        seg = mesh.select_by_index(index)
        # paint color by index
        o3d.io.write_triangle_mesh("data/test2_seg.ply", seg)

        pass

    def tearDown(self):
        # Cleanup code to run after each test
        pass

if __name__ == '__main__':
    unittest.main()