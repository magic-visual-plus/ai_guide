import unittest
import ai_guide.surface
import open3d as o3d

class TestSurface(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test
        pass

    def test_create_surface(self):
        pcd = o3d.io.read_point_cloud("data/test2_no_background.ply")
        mesh = ai_guide.surface.create_surface_from_cloud_points(
            pcd, 0.5
        )

        o3d.io.write_triangle_mesh("data/test2_mesh.ply", mesh)
        pass

    def tearDown(self):
        # Cleanup code to run after each test
        pass

if __name__ == '__main__':
    unittest.main()