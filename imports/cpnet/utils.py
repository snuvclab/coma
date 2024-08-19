import os
import numpy as np
import torch
import h5py
import multiprocessing
import pickle
from sklearn import neighbors
from sklearn.neighbors import KDTree
from sklearn.utils.graph import graph_shortest_path
from tqdm import tqdm

BASEDIR = os.path.dirname(os.path.abspath(__file__))


class ModelWrapper(torch.nn.Module):
    def __init__(self, model_impl) -> None:
        super().__init__()
        self.model_impl = model_impl

    def forward(self, data):
        pc = data
        if isinstance(pc, np.ndarray):
            pc = torch.from_numpy(pc).float()
        res = self.model_impl(pc.transpose(1, 2).cuda())
        return res


def point_augment(pcd, num_point=3000):
    if pcd.shape[0] < num_point:
        while num_point - pcd.shape[0] > pcd.shape[0]:
            pcd = np.concatenate([pcd, pcd])
        num_aug = num_point - pcd.shape[0]
        pcd_aug = pcd[:num_aug]
        pcd = np.concatenate([pcd, pcd_aug])
        return pcd
    else:
        return pcd[:num_point]


def normalize_adj(adj):
    rowsum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def normalized_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    norm_laplacian = np.eye(adj.shape[0]) - adj_normalized
    return norm_laplacian


def pc2lap(pcd, knn=20):
    graph = neighbors.kneighbors_graph(pcd, knn, mode="distance", include_self=False)
    graph = graph.toarray()
    conns = np.sum(graph > 0, axis=-1)
    graph = np.exp(-(graph**2) / (np.sum(graph, axis=-1, keepdims=True) / conns[:, None]) ** 2) * (graph > 0).astype(np.float32)
    return normalized_laplacian(graph)


def normalize_adj(adj):
    rowsum = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt


def normalized_laplacian(adj):
    adj_normalized = normalize_adj(adj)
    norm_laplacian = np.eye(adj.shape[0]) - adj_normalized
    return norm_laplacian


def pc2lap(pcd, knn=20):
    graph = neighbors.kneighbors_graph(pcd, knn, mode="distance", include_self=False)
    graph = graph.toarray()
    conns = np.sum(graph > 0, axis=-1)
    graph = np.exp(-(graph**2) / (np.sum(graph, axis=-1, keepdims=True) / conns[:, None]) ** 2) * (graph > 0).astype(np.float32)
    return normalized_laplacian(graph)


def jitter_point_cloud(pcd, sigma=0.01, clip=0.05):
    N, C = pcd.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += pcd
    return jittered_data


def rotate_point_cloud(pcd, degree=np.pi / 60):
    rotation_angle = np.random.uniform() * degree
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    rotated_data = np.dot(pcd, rotation_matrix)
    return rotated_data


def normalize_point_cloud(pts):
    norm = pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2
    norm = torch.sqrt(norm).reshape(-1, 1)
    pts = pts / norm
    return pts


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode="distance", include_self=False)
    return graph_shortest_path(graph, directed=False)


def gen_geo_dists_wrapper(args):
    pc_name, pc = args
    return (pc_name, gen_geo_dists(pc))


def load_geodesics(dataset, split):
    fn = os.path.join(BASEDIR, "cache", "{}_geodists_{}.pkl".format(dataset.catg, split))
    # need a large amount of memory to load geodesic distances!!!
    if not os.path.exists(os.path.join(BASEDIR, "cache")):
        os.makedirs(os.path.join(BASEDIR, "cache"))
    if os.path.exists(fn):
        print("Found geodesic cache...")
        geo_dists = pickle.load(open(fn, "rb"))
    else:
        print("Generating geodesics, this may take some time...")
        geo_dists = []
        with multiprocessing.Pool(processes=os.cpu_count() // 2) as pool:
            for res in tqdm(pool.imap_unordered(gen_geo_dists_wrapper, [(dataset.mesh_names[i], dataset.pcds[i]) for i in range(len(dataset))]), total=len(dataset)):
                geo_dists.append(res)
        geo_dists = dict(geo_dists)
        pickle.dump(geo_dists, open(fn, "wb"))
    return geo_dists


def geo_error_per_cp(pcds, embeddings, kp_indices, dist_mats=None):
    """
    pcds: [num_data, 2048, 3]
    embeddings: [num_data, 2048, dim_feat]
    kp_indices: [num_data, ]
    """
    pcds.shape[0]
    valid_inds = (kp_indices >= 0).nonzero()
    pcds = pcds[valid_inds]
    embeddings = embeddings[valid_inds]
    kp_indices = kp_indices[valid_inds]
    if dist_mats is not None:
        dist_mats = dist_mats[valid_inds]
    kdtrees_embeddings = []
    num_data = pcds.shape[0]

    for i in range(num_data):
        kdtrees_embeddings.append(KDTree(embeddings[i], leaf_size=2))

    error = 0
    cnt = 0
    for i in range(num_data):
        kp_idx = kp_indices[i]
        kp_embedding = embeddings[i][kp_idx]
        for j in range(num_data):
            if i == j:
                continue
            min_dist_feat, idx = kdtrees_embeddings[j].query(kp_embedding.reshape(1, -1), k=1)
            idx = idx[0][0]
            p_nearest = pcds[j][idx]
            idx_gt = kp_indices[j]
            p_gt = pcds[j][idx_gt]
            if dist_mats is None:  # l2 distance
                dist = np.linalg.norm(p_nearest - p_gt)
            else:  # geodesic
                dist = dist_mats[j][idx, idx_gt]
            error += dist
            cnt += 1
    valid = True
    if cnt == 0:
        # print("only one model was annotated with this point, clean this later")
        error_avg = 0
        valid = False
    else:
        error_avg = error / cnt
    return error_avg, valid
