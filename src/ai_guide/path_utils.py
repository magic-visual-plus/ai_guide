import scipy
from scipy.sparse import csgraph
from scipy.spatial.distance import cdist
import numpy as np
import time


def find_path_most_points(parents, children, current):
    if len(children[current]) == 0:
        return [current], [current]
    else:
        # check if including current node is better
        max_path_one_way1 = []
        max_path_one_way2 = []
        max_path_two_way = []
        for c in children[current]:
            path_one_way, path_two_way = find_path_most_points(parents, children, c)
            # including current node path will plus 1
            if len(path_one_way) > len(max_path_one_way1):
                max_path_one_way2 = max_path_one_way1
                max_path_one_way1 = path_one_way
            elif len(path_one_way) > len(max_path_one_way2):
                max_path_one_way2 = path_one_way
            else:
                pass
            if len(path_two_way) > len(max_path_two_way):
                max_path_two_way = path_two_way
                pass
            pass
        if len(max_path_one_way1) + len(max_path_one_way2) + 1 > len(max_path_two_way):
            return max_path_one_way1 + [current], max_path_one_way1 + [current] + max_path_one_way2
        else:
            return max_path_one_way1 + [current], max_path_two_way
        pass
    pass

def find_path_most_points_from_root(parents, children, root):
    return find_path_most_points(parents, children, root)[1]

def find_path(points):
    # points: [N, 3]
    # return: [K]

    # find a path that connects as many points as possible

    # first find minimum spanning tree
    k = 2000
    dists = cdist(points, points)
    # make down diag zero
    # dists = np.triu(dists, 1)

    dists_order = np.argpartition(dists, k, axis=1)[:, :k]
    rows = np.tile(np.arange(len(points)), (k, 1)).T
    kth = dists[rows, dists_order].max(axis=1, keepdims=True)
    # dists[dists >= kth] = 0
    # dists_ = np.zeros_like(dists)
    # dists_[rows, dists_order] = dists[rows, dists_order]
    # dists = dists_
    
    dists = scipy.sparse.csr_matrix(dists)
    start = time.time()
    spann_mat = csgraph.minimum_spanning_tree(dists)
    print("Time taken for mst: ", time.time() - start)
    spann_mat = spann_mat + spann_mat.T

    # build tree from spann_mat
    N = len(points)
    parents = -np.ones(N, dtype=int)
    children = [[] for _ in range(N)]
    # root = np.random.randint(N)
    root = 0
    parents[root] = -2
    leaves = [root]


    while len(leaves) > 0:
        l = leaves.pop(0)
        for j in range(spann_mat.indptr[l], spann_mat.indptr[l + 1]):
            c = spann_mat.indices[j]
            if c == l:
                continue
            if parents[c] != -1:
                continue
            parents[c] = l
            children[l].append(c)
            leaves.append(c)
            pass
        pass

    # find path that connects most points

    ## find root
    path = find_path_most_points_from_root(parents, children, root)
    return path
