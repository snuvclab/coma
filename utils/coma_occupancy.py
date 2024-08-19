from copy import deepcopy
from tqdm import tqdm
import pickle

import numpy as np
import open3d as o3d

import numpy as np
import torch

from utils.transformations import normalize_vectors_torch
from utils.misc import to_np_torch_recursive


def get_uniform_points_on_sphere(num_points=1000):
    indices = np.arange(0, num_points, dtype=float) + 0.5

    phi = np.arccos(1 - 2 * indices / num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)

    return x, y, z  # each length of 'num_points'


def simplify_mesh_and_get_indices(
    mesh,
    number_of_points: int,
    simplify_method="poisson_disk",
    mesh_index_find_method="distance-based",
    debug=False,
):
    ## simplify the mesh to point-cloud (default: poisson-disk method)
    if simplify_method == "poisson_disk":  # for equidistance point sampling
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    elif simplify_method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    else:
        raise NotImplementedError

    ## get the mesh indices that corresponds to the sampled pcd points
    if mesh_index_find_method == "raytracing-based":
        # perturb the point-cloud with very small noise along the pcd normal direction
        points = np.asarray(pcd.points)  # Nx3
        normals = np.asarray(pcd.normals)  # Nx3
        epsilon = 1e-6
        ray_ds = np.hstack((points + epsilon * normals, -normals))
        rays = o3d.core.Tensor([ray_ds], dtype=o3d.core.Dtype.Float32)

        # do ray casting with the ray to the original mesh
        scene = o3d.t.geometry.RaycastingScene()
        legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(legacy_mesh)
        ray_casting_results = scene.cast_rays(rays)

        # triangle ids
        coma_ids = ray_casting_results["coma_ids"].numpy()  # 1xN
        coma_ids = coma_ids[0]  # N

        # triangle uvs (barycentric)
        coma_uvs = ray_casting_results["coma_uvs"].numpy()  # 1xNx2
        coma_uvs = coma_uvs[0]  # Nx2

        # get downsampled mesh
        mesh_vertices = np.asarray(mesh.vertices)  # Vx3
        mesh_vertex_normals = np.asarray(mesh.vertex_normals)  # Vx3
        mesh_faces = np.asarray(mesh.triangles)  # Fx3
        selected_vertex_indices = mesh_faces[coma_ids, :]  # the first vertex index of each selected triangles

    elif mesh_index_find_method == "distance-based":
        points = np.asarray(pcd.points)  # Nx3
        mesh_verts = np.asarray(mesh.vertices)  # Vx3
        squared_dists = np.sum(np.square(points[None, :, :] - mesh_verts[:, None, :]), axis=-1)  # VxN
        selected_vertex_indices = np.argmin(squared_dists, axis=0)
    else:
        raise NotImplementedError

    selected_vertex_indices = list(set(selected_vertex_indices))

    return selected_vertex_indices, pcd


## warning: this code is not generalized for any shape of relative normals
def geodesic_gaussian_scores(
    normal_grid: torch.Tensor,
    canon_normals: torch.Tensor,
    normal_gaussian_sigma: float,
    eps: float,
):
    cossims = torch.sum(normal_grid[None, None, :, :] * canon_normals[:, :, None, :], dim=-1)  # H x O x N
    geodesic = torch.arccos(torch.clip(cossims, min=-1.0 + eps, max=1.0 - eps))  # H x O x N
    gaussian_scores = 1.0 / torch.exp(geodesic**2 / normal_gaussian_sigma**2)  # H x O x N

    return gaussian_scores


## function used for computing proximity scores
def negative_exp(x, spatial_grid_size, spatial_grid_thres, **kwargs):
    # x = torch.where(x > spatial_grid_thres, torch.inf, x)
    x = torch.exp(-x / spatial_grid_size)
    return x


## function used to transform 'normal a' when 'normal b' canonicalizes to 'normal p'
def canonicalize_a_wrt_b_to_p(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, sub_p: torch.Tensor, eps: float = 1e-8, normalize_first: bool = True):
    # a: Ax3 / b: Bx3 / prin: 3
    # we get ? where [b -> prin, a -> ?]
    # ? --> AxBx3

    ## normalize first
    if normalize_first:
        a = normalize_vectors_torch(a, eps)  # Ax3
        b = normalize_vectors_torch(b, eps)  # Bx3
        p = normalize_vectors_torch(p[None, :], eps)[0]  # 3
        sub_p = normalize_vectors_torch(sub_p[None, :], eps)[0]  # 3

    ## dot products
    b_dot_p = torch.sum(b * p[None, :], dim=-1)[None, :]  # 1,B
    a_dot_b = torch.sum(a[:, None, :] * b[None, :, :], dim=-1)  # A,B
    a_dot_p = torch.sum(a * p[None, :], dim=-1)[:, None]  # A,1
    a_dot_sub_p = torch.sum(a * sub_p[None, :], dim=-1)[:, None]  # A,1
    assert np.allclose(torch.sum(p * sub_p).item(), 0)

    ## exceptions (exactly opposite of p)
    indices2replace = (1 + b_dot_p) < eps  # boolean
    indices2replace = indices2replace[:, :, None]  # 1,B,1
    replacer = 2 * a_dot_sub_p[:, :, None] * sub_p[None, None, :] - a[:, None, :]  # A,1,1

    ## cross products (b & p)
    # b as cross product matrices (Bx3x3)
    b_cross = torch.zeros([b.shape[0], 3, 3], dtype=b.dtype, device=b.device)  # Bx3x3
    b_cross[:, 0, 1] = -b[:, 2]
    b_cross[:, 0, 2] = b[:, 1]
    b_cross[:, 1, 0] = b[:, 2]
    b_cross[:, 1, 2] = -b[:, 0]
    b_cross[:, 2, 0] = -b[:, 1]
    b_cross[:, 0, 0] = b[:, 0]

    # (b x p), a dot (b x p)
    b_cross_p = torch.einsum("bij,j->bi", b_cross, p)  # Bx3
    a_dot_b_cross_p = torch.sum(a[:, None, :] * b_cross_p[None, :, :], dim=-1)  # A,B

    ## get final canonicalized vector for a
    final = b_cross_p[None, :, :] * a_dot_b_cross_p[:, :, None]  # AxBx3
    final = torch.where(indices2replace, 0, final / (1 + b_dot_p[:, :, None]))  # AxBx3
    final += b_dot_p[:, :, None] * a[:, None, :]  # AxBx3
    final += a_dot_b[:, :, None] * p[None, None, :]  # AxBx3
    final -= a_dot_p[:, :, None] * b[None, :, :]  # AxBx3

    ## finally, replace the exceptions (angle between b and p is 180 degrees)
    final = torch.where(indices2replace, replacer, final)
    final = final / torch.sqrt(torch.sum(torch.square(final), dim=-1, keepdim=True))  # normalize again

    return final  # AxBx3


from utils.misc import get_3d_indexgrid_ijk


def load_voxelgrid(gridsize=3.0, resolution=24, center=[0, 0, 0]):
    length_x = length_y = length_z = gridsize
    N_x = N_y = N_z = resolution
    voxel_size = gridsize / resolution
    center = np.array(center)
    start_point = center - np.array([length_x / 2, length_y / 2, length_z / 2])

    # indexgrid
    indexgrid = get_3d_indexgrid_ijk(N_x, N_y, N_z)  # by querying with [:, i,j,k], you can get index [i,j,k]

    # canon-space grid. By querying with [:,i,j,k], you can get the canon-world-coordinate values of that grid index.
    canon_grid = start_point.reshape(3, 1, 1, 1) + voxel_size * indexgrid.astype(np.float32) + voxel_size / 2

    grid_metadata = dict(
        length_x=length_x,
        length_y=length_y,
        length_z=length_z,
        N_x=N_x,
        N_y=N_y,
        N_z=N_z,
        start_point=start_point,
        voxel_size=voxel_size,
    )
    return canon_grid, indexgrid, grid_metadata


## class holding comprehensive information for affordance
class ComA_Occupancy:
    selected_obj_idxs = [0]

    def __init__(
        self,
        scale_tolerance: float,  # 2.0?
        human_res: int,
        obj_res: int,
        normal_res: int,
        spatial_res: int,  # --> if 0, save as discrete distributions
        proximity_settings=dict(),
        principle_vec=[0, 0, 1],
        sub_principle_vec=[0, 1, 0],
        rel_dist_method: str = "dist",  # 'dist' or 'sdf'
        normal_gaussian_sigma: float = 0.1,
        selected_obj_idx: int = None,
        eps: float = 1e-8,
        device: str = "cuda",
    ):
        super().__init__()
        ## device
        self.device = device

        ## human & object resolution
        self.human_res = human_res
        self.obj_res = obj_res

        ## normal resolution
        self.normal_res = normal_res  # N
        self.spatial_res = spatial_res  # S
        assert normal_res == 0, "In this version, normal res is 0."

        ## the spatial grid (each shape of 3xNxNxN)
        self.spatial_grid, self.spatial_indexgrid, self.spatial_grid_metadata = load_voxelgrid(gridsize=2.4, resolution=self.spatial_res, center=[0, 0, 0])
        self.N_x = self.spatial_grid_metadata["N_x"]
        self.N_y = self.spatial_grid_metadata["N_y"]
        self.N_z = self.spatial_grid_metadata["N_z"]
        self.spatial_grid = torch.from_numpy(self.spatial_grid).to(device)

        # spatial prob placeholders: H x N x N x N
        self.spatial_occupancy_grids = torch.zeros([self.human_res, self.N_x, self.N_y, self.N_z], dtype=torch.float32).to(device)

        ## cache for saving before aggregation: saves 'normal_rel', 'distance_rel', 'position_rel (optional)'
        self.cache_count = 0
        self.used_count = 0
        self.cache = dict()
        self.used = dict()  # for saving used ones, mainly for visualization

        ## principle vector for normal canonicalization
        self.principle_vec = torch.tensor(principle_vec, dtype=torch.float32).to(device)
        self.sub_principle_vec = torch.tensor(sub_principle_vec, dtype=torch.float32).to(device)

        ## relative distance calculation methods
        assert rel_dist_method in ["dist", "sdf"], f"rel_dist_method: '{rel_dist_method}' not allowed"
        self.rel_dist_method = rel_dist_method
        self.rel_dist_thres = self.spatial_grid_metadata["voxel_size"] * scale_tolerance

        ## gaussian kernal sigma / epsilon for robust calculation
        self.normal_gaussian_sigma = normal_gaussian_sigma
        self.eps = eps

        self.debug_obj_vert = None
        self.debug_obj_normal = None

    def register_sample_to_cache(self, **kwargs):
        self.cache[f"{self.cache_count:05}"] = kwargs
        self.cache_count = len(self.cache.keys())

    def aggregate_all_samples(self):
        for cache_id in tqdm(self.cache.keys(), desc="Aggregating Samples"):
            # aggregate to single sample
            self.aggregate_single_sample(**self.cache[cache_id])

            # add to used
            self.used[f"{self.used_count:05}"] = self.cache[cache_id]
            self.used_count = len(self.used.keys())

        # reboot cache
        self.cache = {}
        self.cache_count = 0

    def aggregate_single_sample(self, **kwargs):
        # this is the occupancy version.
        self.aggregate_single_sample_for_occupancy(**kwargs)

    def aggregate_single_sample_for_occupancy(self, human_verts, human_normals, obj_verts, obj_normals, **kwargs):
        # get the obj vertex
        for obj_idx in self.selected_obj_idxs:
            obj_vert = obj_verts[obj_idx]  # 3,
            obj_normal = obj_normals[obj_idx]  # 3,
            if self.debug_obj_vert is None:
                self.debug_obj_vert = obj_vert
            else:
                assert np.allclose(self.debug_obj_vert, obj_vert)
            if self.debug_obj_normal is None:
                self.debug_obj_normal = obj_normal
            else:
                assert np.allclose(self.debug_obj_normal, obj_normal)

            # move human verts
            human_verts_canon = human_verts - obj_vert[None]  # H x 3
            human_verts_canon = to_np_torch_recursive(human_verts_canon, use_torch=True, device=self.device)
            assert human_verts_canon.shape[0] == self.human_res

            # check the distance between voxelgrid point and human vertex # rel_dists are H x N x N x N
            rel_dists = (self.spatial_grid[None, :, :, :, :] - human_verts_canon[:, :, None, None, None]).square().sum(dim=1).sqrt()
            score_add = rel_dists < self.rel_dist_thres  # H x N x N x N, boolean
            # score_add = torch.exp(-rel_dists / self.rel_dist_thres)
            self.spatial_occupancy_grids += score_add.type(torch.float32)  # H x N x N x N

    def normalize_prob_grid_for_spatials(self):
        self.spatial_occupancy_grids = self.spatial_occupancy_grids.reshape(self.human_res, -1)  # H x N^3
        self.spatial_occupancy_grids = self.spatial_occupancy_grids / self.spatial_occupancy_grids.sum(dim=-1, keepdim=True)  # H x N^3
        self.spatial_occupancy_grids = self.spatial_occupancy_grids.reshape(self.human_res, self.N_x, self.N_y, self.N_z)  # H x N x N x N

    def normalize_prob_grid_for_spatials_v2(self):
        self.spatial_occupancy_grids = self.spatial_occupancy_grids / self.used_count  # H x N^3

    def return_aggregated_spatial_grids(self, human_indices=None):
        self.normalize_prob_grid_for_spatials()

        if human_indices is None:
            human_indices = list(range(self.human_res))

        aggregated_score_grid = self.spatial_occupancy_grids[human_indices].max(dim=0)  # N x N x N
        return aggregated_score_grid.values

    ## export results
    def export(self, save_pth=None):
        # prepare export files
        to_export = deepcopy(vars(self))

        # remove the cache
        del to_export["cache"]
        del to_export["used"]

        # send to numpy ndarray
        to_export = to_np_torch_recursive(to_export, use_torch=False, device="cpu")

        if save_pth is None:
            return to_export

        with open(save_pth, "wb") as handle:
            pickle.dump(to_export, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ## load results
    def load(self, load_pth):
        # load the file from pth
        with open(load_pth, "rb") as handle:
            loadables = pickle.load(handle)

        # send the numpy arrays to torch
        loadables = to_np_torch_recursive(loadables, use_torch=True, device=self.device)

        # set the variables
        for k, v in loadables.items():
            setattr(self, k, v)
