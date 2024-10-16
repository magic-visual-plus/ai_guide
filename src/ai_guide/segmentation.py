import open3d as o3d
import numpy as np
import scipy.sparse

def compute_adj_mat(triangles):
    adj_list = []
    # a->b
    adj_list.append(triangles[:, [0, 1]])
    adj_list.append(triangles[:, [1, 0]])
    # b->c
    adj_list.append(triangles[:, [1, 2]])
    adj_list.append(triangles[:, [2, 1]])
    # c->a
    adj_list.append(triangles[:, [2, 0]])
    adj_list.append(triangles[:, [0, 2]])
    adj_list = np.concatenate(adj_list, axis=0)
    # reverse

    adj_list = np.unique(adj_list, axis=0)
    return adj_list

def remove_background(pcd: o3d.geometry.PointCloud, distance_threshold=20) -> o3d.geometry.PointCloud:
    points = np.asarray(pcd.points)
    
    max_z = np.max(points[:, 2])

    max_z_below = max_z - 2.0

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    index = np.where(points[:, 2] >= max_z_below)[0]

    plane_normal = np.mean(normals[index], axis=0)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    key_point = points[index[2]]
    # calculate the distance between the plane and points
    p = points - key_point[None, ...]
    distance = np.dot(p, plane_normal)

    # remove the points that are near the plane
    index = np.where(abs(distance) > distance_threshold)[0]

    pcd = pcd.select_by_index(
        index
    )

    return pcd


def find_keypoint(points: np.ndarray) -> np.ndarray:
    center= np.mean(points, axis=0, keepdims=True)


    distances = np.linalg.norm(points - center, axis=1)

    # find nearest 100 points
    index = np.argsort(distances)[:100]

    return index


def segment_plane(mesh: o3d.geometry.TriangleMesh, keypoint: np.ndarray, similarity_threshold=0.95):
    vertices = np.array(mesh.vertices)
    vertex_normals = np.array(mesh.vertex_normals)

    triangles = np.array(mesh.triangles)
    
    # compute adjacent matrix
    adj_matrix = compute_adj_mat(triangles)

    pos_idx = keypoint

    vertices_pos = vertices[pos_idx]
    vertex_normals_pos = vertex_normals[pos_idx]
    mean_normal_pos = np.mean(vertex_normals_pos, axis=0)

    edge_normals = 0.5 * (vertex_normals[adj_matrix[:, 1]] + vertex_normals[adj_matrix[:, 0]])
    edge_features = np.sum(
        mean_normal_pos[None, :] * edge_normals,
        axis=-1)

    edge_features[edge_features>similarity_threshold] = 99999
    edge_features[edge_features<=similarity_threshold] *= 10
    edge_features = edge_features.astype("int64")

    num_vertices = len(vertices)
    idx_source = num_vertices
    idx_sink = num_vertices + 1
    added_edge_source = np.stack(
        (np.ones(num_vertices, dtype="int64") * idx_source, np.arange(num_vertices)),
        axis=-1)
    added_edge_sink = np.stack(
        (np.arange(num_vertices), np.ones(num_vertices, dtype="int64")*idx_sink),
        axis=-1)
    num_select = 10000
    idx_select_start = num_vertices // 2 - num_select
    idx_select_end = num_vertices // 2 + num_select
    pos_weight = np.ones(num_vertices, dtype="int64")
    neg_weight = np.ones(num_vertices, dtype="int64")

    pos_weight[pos_idx] = 99999999
    neg_weight[:] = 10
    
    adj_matrix = np.concatenate([adj_matrix, added_edge_source, added_edge_sink], axis=0)
    print(adj_matrix.dtype)
    edge_features = np.concatenate([edge_features, pos_weight, neg_weight], axis=0)

    csr_matrix = scipy.sparse.csr_matrix(
        (edge_features, (adj_matrix[:, 0], adj_matrix[:, 1])),
        shape=(num_vertices+2, num_vertices+2))
    
    result = scipy.sparse.csgraph.maximum_flow(csr_matrix, idx_source, idx_sink)
    
    source_nodes = scipy.sparse.csgraph.depth_first_order(csr_matrix - result.flow, idx_source)[0]
    source_nodes[source_nodes >= num_vertices] = num_vertices - 1
    
    print(source_nodes)
    print(np.max(source_nodes), np.min(source_nodes))

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals

    seg = mesh.select_by_index(source_nodes[1:])
    return seg
    pass


