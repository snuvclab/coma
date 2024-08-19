from copy import deepcopy
from functools import partial
from tqdm import tqdm
import pickle

import numpy as np
import open3d as o3d

import math
import numpy as np
import torch

from utils.load_3d import load_obj_as_o3d_preserving_face_order
from utils.transformations import normalize_vectors_torch, normalize_vectors_np
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
        from IPython import embed

        embed()

        # get downsampled mesh
        mesh_vertices = np.asarray(mesh.vertices)  # Vx3
        mesh_vertex_normals = np.asarray(mesh.vertex_normals)  # Vx3
        mesh_faces = np.asarray(mesh.triangles)  # Fx3
        selected_vertex_indices = mesh_faces[coma_ids, :]  # the first vertex index of each selected triangles

        if debug:
            from IPython import embed

            embed()
            debug_pcd = o3d.geometry.PointCloud()
            debug_pcd.points = o3d.utility.Vector3dVector(mesh_vertices[selected_vertex_indices, :])  # V_selected x 3
            debug_pcd.normals = o3d.utility.Vector3dVector(mesh_vertex_normals[selected_vertex_indices, :])  # V_selected x 3
            o3d.visualization.draw_geometries([debug_pcd, pcd])
            o3d.visualization.draw_geometries([mesh, pcd])
            o3d.visualization.draw_geometries([mesh, debug_pcd])

    elif mesh_index_find_method == "distance-based":
        points = np.asarray(pcd.points)  # Nx3
        mesh_verts = np.asarray(mesh.vertices)  # Vx3
        squared_dists = np.sum(np.square(points[None, :, :] - mesh_verts[:, None, :]), axis=-1)  # VxN
        selected_vertex_indices = np.argmin(squared_dists, axis=0)

    else:
        raise NotImplementedError

    selected_vertex_indices = list(selected_vertex_indices)

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


## class holding coma information for affordance
class ComA:
    def __init__(
        self,
        human_res: int,
        obj_res: int,
        normal_res: int,
        spatial_res: int,  # --> if 0, save as discrete distributions
        proximity_settings=dict(),
        principle_vec=[0, 0, 1],
        sub_principle_vec=[0, 1, 0],
        rel_dist_method: str = "dist",  # 'dist' or 'sdf'
        normal_gaussian_sigma: float = 0.1,
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

        ## the principal normals
        x, y, z = get_uniform_points_on_sphere(num_points=normal_res)
        self.canon_normal_grid = torch.tensor(np.stack([x, y, z], axis=-1)).to(device)  # Nx3

        ## probability placeholders --> domain size: [HxOx(N+1)] or [HxOxNxS^3 (else)]
        ## the probability placeholders
        if self.spatial_res == 0:  # this version used for contact derivation only
            self.prob_grid_canon_human_wrt_obj = torch.zeros([self.human_res, self.obj_res, self.normal_res], dtype=torch.float32).to(device)
            self.prob_grid_canon_obj_wrt_human = torch.zeros([self.human_res, self.obj_res, self.normal_res], dtype=torch.float32).to(device)

            # contact distribution expectation pre-calculated
            self.contact_dist_expectation_grid_nom = torch.zeros([self.human_res, self.obj_res], dtype=torch.float32).to(device)
            self.contact_dist_expectation_grid_denom = torch.zeros([self.human_res, self.obj_res], dtype=torch.float32).to(device)

            # the count for how many times (human-vertex, obj-vertex) pair have significant contact (close enough)
            self.significant_contact_count = torch.zeros([self.human_res, self.obj_res]).to(device)
        else:
            print("Please implement the spatial grid")
            raise NotImplementedError

        ## contact-derivative placeholders
        self.proximity_settings = proximity_settings
        self.contact_dist_func = partial(negative_exp, **proximity_settings)
        self.cross_contact_scores_nom = torch.zeros([self.human_res, self.obj_res], dtype=torch.float32).to(device)
        self.cross_contact_scores_denom = torch.zeros([self.human_res, self.obj_res], dtype=torch.float32).to(device)

        ## nonphysical-affordance-deerivative placeholders
        pass  # --> Implement Later!

        ## human-occupancy
        pass  # --> Implement Later!

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

        ## gaussian kernal sigma / epsilon for robust calculation
        self.normal_gaussian_sigma = normal_gaussian_sigma
        self.eps = eps

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
        # cross relative distance
        if self.spatial_res == 0:  # -> In this case, we directly compute & save the expectation
            self.assert_inputs(**kwargs)
            self.aggregate_single_sample_for_contact(**to_np_torch_recursive(kwargs, use_torch=True, device=self.device))
        else:
            print("Please implement the spatial grid and aggregation in spatial grid")
            raise NotImplementedError

    def aggregate_single_sample_for_contact(self, human_verts, human_normals, obj_verts, obj_normals, **kwargs):
        ## in this case, we only consider the distance for relative distance
        if self.rel_dist_method == "dist":
            ## relative location
            # compute the distances between human-verts and obj-verts
            rel_dists = torch.sqrt(torch.sum(torch.square(human_verts[:, None, :] - obj_verts[None, :, :]), dim=-1))  # H x O

            # count significant contact
            self.significant_contact_count += (rel_dists < self.proximity_settings["spatial_grid_thres"]).type(torch.int64)

            # save dist as expectation
            self.contact_dist_expectation_grid_nom += self.contact_dist_func(rel_dists)
            self.contact_dist_expectation_grid_denom += 1.0

            ## relative normal
            # canonicalize the human normals and get the n_rel (wrt human, wrt obj) --> Shape: H x O x 3
            canon_human_normals_wrt_obj = canonicalize_a_wrt_b_to_p(
                a=human_normals,
                b=obj_normals,
                p=self.principle_vec,
                sub_p=self.sub_principle_vec,
                eps=self.eps,
            )
            canon_obj_normals_wrt_human = canonicalize_a_wrt_b_to_p(
                a=obj_normals,
                b=human_normals,
                p=self.principle_vec,
                sub_p=self.sub_principle_vec,
                eps=self.eps,
            )
            canon_obj_normals_wrt_human = torch.permute(canon_obj_normals_wrt_human, dims=(1, 0, 2))

            ## register the relative normals to the distribution
            self.prob_grid_canon_human_wrt_obj += geodesic_gaussian_scores(
                normal_grid=self.canon_normal_grid,
                canon_normals=canon_human_normals_wrt_obj,
                normal_gaussian_sigma=self.normal_gaussian_sigma,
                eps=self.eps,
            )
            self.prob_grid_canon_obj_wrt_human += geodesic_gaussian_scores(
                normal_grid=self.canon_normal_grid,
                canon_normals=canon_obj_normals_wrt_human,
                normal_gaussian_sigma=self.normal_gaussian_sigma,
                eps=self.eps,
            )

        if self.rel_dist_method == "sdf":
            raise NotImplementedError

    def normalize_prob_grid_for_normals(self):
        self.prob_grid_canon_human_wrt_obj /= torch.sum(self.prob_grid_canon_human_wrt_obj, dim=-1, keepdim=True) + self.eps  # H x O x N
        self.prob_grid_canon_obj_wrt_human /= torch.sum(self.prob_grid_canon_obj_wrt_human, dim=-1, keepdim=True) + self.eps  # H x O x N

    ## returns per vertex-pair contact mappings (as dictionary, human-contact map & obj-contact map)
    def compute_contact_map(self, contact_map_type: str, as_numpy: bool = True):
        # assert inputs
        self.assert_inputs(contact_map_type=contact_map_type)

        ## prepare before calculating contact map
        # normalize probability grid first
        self.normalize_prob_grid_for_normals()

        # compute expected proximity & dot products first
        normal_dot_products = torch.sum(self.principle_vec[None, :] * self.canon_normal_grid, dim=-1)[None, None, :]  # 1 x 1 xN
        expected_proximity = self.contact_dist_expectation_grid_nom / self.contact_dist_expectation_grid_denom  # H x O

        ## compute contact map
        per_vertex_contact_on_human = None
        per_vertex_contact_on_obj = None
        # contact map on human (per obj point)
        if contact_map_type in ["human", "both"]:
            per_vertex_contact_on_human = torch.sum(self.prob_grid_canon_human_wrt_obj * ((1.0 - normal_dot_products) / 2.0), dim=-1)  # H x O
            per_vertex_contact_on_human *= expected_proximity  # H x O

        # contact map on obj (per human point)
        if contact_map_type in ["obj", "both"]:
            per_vertex_contact_on_obj = torch.sum(self.prob_grid_canon_obj_wrt_human * ((1.0 - normal_dot_products) / 2.0), dim=-1)  # H x O -->?
            per_vertex_contact_on_obj *= expected_proximity  # H x O

        ## return results
        contact_map_dict = {
            "human": per_vertex_contact_on_human,
            "obj": per_vertex_contact_on_obj,
        }
        if as_numpy:
            return to_np_torch_recursive(contact_map_dict, use_torch=False, device="cpu")
        else:
            return contact_map_dict

    ## get all contact pairs with 'enough' significant contact count
    def significant_contact_pairs(
        self,
        significant_contact_ratio: float,
        as_numpy: bool = True,
    ):

        # how many times are the object considered 'significant contact' along the data sequence?
        significant_contact_num = significant_contact_ratio * self.used_count
        significant_contact_pairs = self.significant_contact_count >= significant_contact_num  # H x O, boolean

        if as_numpy:
            return to_np_torch_recursive(significant_contact_pairs, use_torch=False, device="cpu")
        else:
            return significant_contact_pairs

    ## computes the 'aggregated' human contact map, for all the obj vertices with enough significant contact with human
    def aggregate_contact_for_significant_pairs(
        self,
        contact_map_dict: dict,
        contact_map_type: str,  # 'human': returns human contact map / 'obj': returns obj contact map
        significant_contact_ratio: float,
        as_numpy: bool = True,
    ):
        ## assert inputs
        self.assert_inputs(contact_map_type=contact_map_type)

        ## get the significant contact pairs
        significant_contact_pairs = self.significant_contact_pairs(significant_contact_ratio=significant_contact_ratio, as_numpy=False)  # H x O, boolean

        ## aggregate human contact map
        aggregated_human_contact_map = None
        aggregated_obj_contact_map = None
        # for human
        if contact_map_type in ["human", "both"]:
            # human contact map (per vertex) must be given
            assert contact_map_dict["human"] is not None, "If 'contact_map_type' is 'human' or 'both', contact_map_dict['human'] must not be None"

            # obj vertices with at least one significant contact (with human vertex)
            obj_verts_w_significant_contact = significant_contact_pairs.any(dim=0)  # O, boolean
            if not obj_verts_w_significant_contact.any():
                aggregated_human_contact_map = torch.zeros([self.human_res], dtype=torch.float32, device=self.device)  # H
            else:
                # aggregate the contact maps for selected obj vertices
                aggregated_human_contact_map = contact_map_dict["human"][:, obj_verts_w_significant_contact]  # H x O_select
                aggregated_human_contact_map = aggregated_human_contact_map.max(dim=-1).values  # H

        # for obj
        if contact_map_type in ["obj", "both"]:
            # obj contact map (per vertex) must be given
            assert contact_map_dict["obj"] is not None, "If 'contact_map_type' is 'obj' or 'both', contact_map_dict['obj'] must not be None"

            # human vertices with at least one significant contact (with object vertex)
            human_verts_w_significant_contact = significant_contact_pairs.any(dim=1)  # H, boolean
            if not human_verts_w_significant_contact.any():
                aggregated_obj_contact_map = torch.zeros([self.obj_res], dtype=torch.float32, device=self.device)  # O
            else:
                # aggregate the contact maps for selected human vertices
                aggregated_obj_contact_map = contact_map_dict["obj"][human_verts_w_significant_contact, :]  # H_select x O
                aggregated_obj_contact_map = aggregated_obj_contact_map.max(dim=0).values  # O, boolean

        ## return results
        aggregated_contact_map_dict = {
            "human": aggregated_human_contact_map,
            "obj": aggregated_obj_contact_map,
            "significant_contact_pairs": significant_contact_pairs,
        }
        if as_numpy:
            return to_np_torch_recursive(aggregated_contact_map_dict, use_torch=False, device="cpu")
        else:
            return aggregated_contact_map_dict

    ## computes nonphysical response per vertex-pairs standardized between 0~1 (for distribution in full sphere)
    def compute_nonphysical_response_sphere(self, n_bin: int, nonphysical_type: str, as_numpy: bool = True):
        ## assert inputs
        self.assert_inputs(nonphysical_type=nonphysical_type)

        ## prepare before calculating nonphysical response
        # normalize probability grid first
        self.normalize_prob_grid_for_normals()

        ## calculate negated normalized shannon entropy as nonphysical score
        nonphysical_score_human_wrt_obj = None
        nonphysical_score_obj_wrt_human = None
        # on human (per obj point)
        if nonphysical_type in ["human", "both"]:
            # discretize the probability with the given bin size
            discretized_prob_grid_canon_human_wrt_obj = torch.round(self.prob_grid_canon_human_wrt_obj * n_bin) / n_bin  # H x O x N
            # calculate negated normalized shannon entropy
            nonphysical_score_human_wrt_obj = torch.where(
                discretized_prob_grid_canon_human_wrt_obj == 0, 0, discretized_prob_grid_canon_human_wrt_obj * torch.log(discretized_prob_grid_canon_human_wrt_obj)
            ).sum(
                dim=-1
            )  # H x O
            nonphysical_score_human_wrt_obj /= math.log(n_bin)  # H x O
            nonphysical_score_human_wrt_obj += 1.0  # to standardize between 0.~1. / H x O

        # on obj (per human point)
        if nonphysical_type in ["obj", "both"]:
            # discretize the probability with the given bin size
            discretized_prob_grid_canon_obj_wrt_human = torch.round(self.prob_grid_canon_obj_wrt_human * n_bin) / n_bin  # H x O x N
            # calculate negated normalized shannon entropy
            nonphysical_score_obj_wrt_human = torch.where(
                discretized_prob_grid_canon_obj_wrt_human == 0, 0, discretized_prob_grid_canon_obj_wrt_human * torch.log(discretized_prob_grid_canon_obj_wrt_human)
            ).sum(
                dim=-1
            )  # H x O
            nonphysical_score_obj_wrt_human /= math.log(n_bin)  # H x O
            nonphysical_score_obj_wrt_human += 1.0  # to standardize between 0.~1. / H x O

        ## return results
        nonphysical_score_dict = {
            "human": nonphysical_score_human_wrt_obj,  # H x O
            "obj": nonphysical_score_obj_wrt_human,  # H x O
            "n_bin": n_bin,
        }
        if as_numpy:
            return to_np_torch_recursive(nonphysical_score_dict, use_torch=False, device="cpu")
        else:
            return nonphysical_score_dict

    def assert_inputs(self, **kwargs):
        # human-verts
        if "human_verts" in kwargs.keys():
            human_verts = kwargs["human_verts"]
            assert human_verts.ndim == 2
            assert human_verts.shape[-1] == 3
            assert len(human_verts) == self.human_res

        # human-vertex-normals
        if "human_normals" in kwargs.keys():
            human_normals = kwargs["human_normals"]
            assert human_normals.ndim == 2
            assert human_normals.shape[-1] == 3
            assert len(human_normals) == self.human_res

        # obj-verts
        if "obj_verts" in kwargs.keys():
            obj_verts = kwargs["obj_verts"]
            assert obj_verts.ndim == 2
            assert obj_verts.shape[-1] == 3
            assert len(obj_verts) == self.obj_res

        # obj-vertex-normals
        if "obj_normals" in kwargs.keys():
            obj_normals = kwargs["obj_normals"]
            assert obj_normals.ndim == 2
            assert obj_normals.shape[-1] == 3
            assert len(obj_normals) == self.obj_res

        # contact_map_type
        if "contact_map_type" in kwargs.keys():
            contact_map_type = kwargs["contact_map_type"]
            assert contact_map_type in ["human", "obj", "both"], "Only ['human'/'obj'/'both'] allowed for Argument: 'contact_map_type'"

        # nonphysical_type
        if "nonphysical_type" in kwargs.keys():
            nonphysical_type = kwargs["nonphysical_type"]
            assert nonphysical_type in ["human", "obj", "both"], "Only ['human'/'obj'/'both'] allowed for Argument: 'nonphysical_type'"

    ## computes nonphysical response per vertex-pairs standardized between 0~1 (for distribution in full sphere)
    def compute_nonphysical_response_sphere_v2(self, n_bin: int, nonphysical_type: str, as_numpy: bool = True):
        ## assert inputs
        self.assert_inputs(nonphysical_type=nonphysical_type)

        ## prepare before calculating nonphysical response
        # normalize probability grid first
        self.normalize_prob_grid_for_normals()

        ## calculate negated normalized shannon entropy as nonphysical score
        nonphysical_score_human_wrt_obj = None
        nonphysical_score_obj_wrt_human = None
        # on human (per obj point)
        if nonphysical_type in ["human", "both"]:
            # compute the normal alignments with principle vector
            principle_alignment_human_wrt_obj = (self.canon_normal_grid * self.principle_vec[None]).sum(dim=-1)  # N
            # discretize the probability with the given bin size
            discretized_prob_grid_canon_human_wrt_obj = torch.round(self.prob_grid_canon_human_wrt_obj * n_bin) / n_bin  # H x O x N
            # calculate 'normal-aligned' self informations
            normalized_p_log_p_human_wrt_obj = (
                torch.where(discretized_prob_grid_canon_human_wrt_obj == 0, 0, discretized_prob_grid_canon_human_wrt_obj * torch.log(discretized_prob_grid_canon_human_wrt_obj))
                .divide(math.log(n_bin))
                .add(1.0)
            )  # H x O x N
            # calculate the average score
            nonphysical_score_human_wrt_obj = torch.sum(normalized_p_log_p_human_wrt_obj * principle_alignment_human_wrt_obj[None, None, :], dim=-1)  # H x O

        # on obj (per human point)
        if nonphysical_type in ["obj", "both"]:
            # compute the normal alignments with principle vector
            principle_alignment_obj_wrt_human = (self.canon_normal_grid * self.principle_vec[None]).sum(dim=-1)  # N
            # discretize the probability with the given bin size
            discretized_prob_grid_canon_obj_wrt_human = torch.round(self.prob_grid_canon_obj_wrt_human * n_bin) / n_bin  # H x O x N
            # calculate 'normal-aligned' self informations
            normalized_p_log_p_obj_wrt_human = (
                torch.where(discretized_prob_grid_canon_obj_wrt_human == 0, 0, discretized_prob_grid_canon_obj_wrt_human * torch.log(discretized_prob_grid_canon_obj_wrt_human))
                .divide(math.log(n_bin))
                .add(1.0)
            )  # H x O x N
            # calculate the average score
            nonphysical_score_obj_wrt_human = torch.sum(normalized_p_log_p_obj_wrt_human * principle_alignment_obj_wrt_human[None, None, :], dim=-1)  # H x O

        ## return results
        nonphysical_score_dict = {
            "human": nonphysical_score_human_wrt_obj,  # H x O
            "obj": nonphysical_score_obj_wrt_human,  # H x O
            "n_bin": n_bin,
        }
        if as_numpy:
            return to_np_torch_recursive(nonphysical_score_dict, use_torch=False, device="cpu")
        else:
            return nonphysical_score_dict

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


### Aggregated Contact ###
def get_aggregated_contact(
    coma: ComA,
    contact_map_type: str,  # 'human' or 'obj'
    significant_contact_ratio: float,  # for determining which vertices to use for 'contact aggregation'
):
    # assert inputs
    assert contact_map_type in ["human", "obj"]

    # compute per-vertex-pair contact map
    contact_map_dict = coma.compute_contact_map(
        contact_map_type=contact_map_type,
        as_numpy=False,
    )

    # aggregate the human contact map
    aggregated_contact_dict = coma.aggregate_contact_for_significant_pairs(
        contact_map_dict=contact_map_dict, contact_map_type=contact_map_type, significant_contact_ratio=significant_contact_ratio, as_numpy=True
    )

    # detach the contact map dict
    aggregated_contact = aggregated_contact_dict[contact_map_type]  # H (if human), or O (if object)

    # vertices with significant contact
    significant_contact_pairs = aggregated_contact_dict["significant_contact_pairs"]  # H x O
    boolean_indicator_for_verts_w_significant_contact = np.any(significant_contact_pairs, axis=0 if contact_map_type == "human" else 1)  # H (if human), or O (if object)
    significant_contact_vertex_indices = np.argwhere(boolean_indicator_for_verts_w_significant_contact)[:, 0]  # H_select (if human), or O_select (if object)

    return aggregated_contact, significant_contact_vertex_indices


### non physical affordance score
def get_nonphysical_score(coma: ComA, nonphysical_type: str):
    return coma.compute_nonphysical_response_sphere(n_bin=1e6, nonphysical_type=nonphysical_type, as_numpy=True)[nonphysical_type]  # H or O


def prepare_affordance_extraction_inputs(
    human_mesh_pth,
    human_mesh_pth_type,
    human_downsample_metadata,
    object_downsample_metadata,
    human_use_downsample_pcd_raw: bool,
    object_use_downsample_pcd_raw: bool,
    eps,
    standardize_human_scale: bool,
    scaler_range,
    camera_pth,
    human_params_pth,  # from step6, to get the scale of human
    object_mesh_for_check_pth=None,
    interactive=False,
):
    ### Load Human Mesh and Object Mesh ###
    # load original human vertices/faces/vertex-normals
    if human_mesh_pth_type == "obj":
        human_mesh = load_obj_as_o3d_preserving_face_order(human_mesh_pth)
        human_mesh.compute_vertex_normals()
        human_verts_orig = np.asarray(human_mesh.vertices)
        human_faces_orig = np.asarray(human_mesh.triangles)

    if human_mesh_pth_type == "pickle":
        with open(human_mesh_pth, "rb") as handle:
            human_data = pickle.load(handle)
        human_verts_orig = human_data["verts"]
        human_faces_orig = human_data["faces"]
        human_mesh = o3d.geometry.TriangleMesh()
        human_mesh.vertices = o3d.utility.Vector3dVector(human_verts_orig)
        human_mesh.triangles = o3d.utility.Vector3iVector(human_faces_orig)
        human_mesh.compute_vertex_normals()

    ## get vertex normals
    human_vertex_normals_orig = normalize_vectors_np(
        np.asarray(human_mesh.vertex_normals),
        eps=eps,
    )

    # load original object vertices/faces/vertex-normals
    obj_verts_orig = object_downsample_metadata["obj_vertices_original"]
    obj_faces_orig = object_downsample_metadata["obj_faces_original"]
    obj_vertex_normals_orig = normalize_vectors_np(object_downsample_metadata["obj_vertex_normals_original"])

    # double-check
    if object_mesh_for_check_pth is not None:
        obj_mesh_for_check = load_obj_as_o3d_preserving_face_order(object_mesh_for_check_pth)
        obj_verts_orig_for_check = np.asarray(obj_mesh_for_check.vertices)
        obj_faces_orig_for_check = np.asarray(obj_mesh_for_check.triangles)
        assert np.allclose(obj_verts_orig, obj_verts_orig_for_check)
        assert np.allclose(obj_faces_orig, obj_faces_orig_for_check)

    ### Downsample Human Mesh and Object Mesh ###
    human_downsample_indices = human_downsample_metadata["downsample_indices"]  # N_human
    object_downsample_indices = object_downsample_metadata["downsample_indices"]  # N_obj

    # downsample vertices
    if human_use_downsample_pcd_raw:
        assert False, "Human must use 'mesh' for Representation. You'll know why"
        human_verts = human_downsample_metadata["downsampled_pcd_points_raw"]
        human_vertex_normals = human_downsample_metadata["downsampled_pcd_normal_raw"]
        assert len(human_verts) == human_downsample_metadata["N_raw"]
    else:
        human_verts = human_verts_orig.copy()[human_downsample_indices]  # N_humanx3
        human_vertex_normals = human_vertex_normals_orig.copy()[human_downsample_indices]  # N_humanx3
        assert len(human_verts) == human_downsample_metadata["N"]

    if object_use_downsample_pcd_raw:
        obj_verts = object_downsample_metadata["downsampled_pcd_points_raw"]  # N_objx3
        obj_vertex_normals = object_downsample_metadata["downsampled_pcd_normal_raw"]  # N_objx3
        assert len(obj_verts) == object_downsample_metadata["N_raw"]
    else:
        obj_verts = obj_verts_orig.copy()[object_downsample_indices]  # N_objx3
        obj_vertex_normals = obj_vertex_normals_orig.copy()[object_downsample_indices]  # N_objx3
        assert len(obj_verts) == object_downsample_metadata["N"]

    if interactive:
        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_orig)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces_orig)
        obj_mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([human_mesh, obj_mesh])

        human_pcd = o3d.geometry.PointCloud()
        obj_pcd = o3d.geometry.PointCloud()
        human_pcd.points = o3d.utility.Vector3dVector(human_verts)
        obj_pcd.points = o3d.utility.Vector3dVector(obj_verts)
        geometries = [human_pcd, obj_pcd]

        ## visualize the normals, too!
        from visualization.visualize_open3d import get_arrow

        # object
        for i in range(len(obj_verts)):
            start = obj_verts[i]
            assert obj_vertex_normals[i].sum() != 0, "Should have been Already Processed in 'test_behave_downsample.py'"
            end = obj_verts[i] + obj_vertex_normals[i] * 0.05
            geometries.append(get_arrow(end=end, origin=start))

        # gravity vector
        start = np.array([0, 0, 0], dtype=np.float32)
        end = start + np.array([0, 1.0, 0], dtype=np.float32)
        gravity_dir = get_arrow(end=end, origin=start)
        gravity_dir.paint_uniform_color([0, 0, 1])
        geometries.append(gravity_dir)
        o3d.visualization.draw_geometries(geometries)

    if standardize_human_scale:
        with open(camera_pth, "rb") as handle:
            camera = pickle.load(handle)
        cam_scale = camera["scale"]

        with open(human_params_pth, "rb") as handle:
            human_params = pickle.load(handle)

        scaler = (512 / cam_scale) * (human_params["convert_data"]["z_mean"] / human_params["convert_data"]["focals"][0])

        if scaler_range is None:
            pass
        else:
            min_scaler, max_scaler = scaler_range
            if scaler < min_scaler or scaler > max_scaler:
                return None
        ## rescale the human meshes ##

    ### Return Inputs for Affordance Extraction ###
    affordance_extraction_inputs = dict(
        human_verts_orig=human_verts_orig,
        human_faces_orig=human_faces_orig,
        human_vertex_normals_orig=human_vertex_normals_orig,
        obj_verts_orig=obj_verts_orig,
        obj_faces_orig=obj_faces_orig,
        obj_vertex_normals_orig=obj_vertex_normals_orig,
        human_downsample_indices=human_downsample_indices,
        object_downsample_indices=object_downsample_indices,
        human_verts=human_verts,
        human_vertex_normals=human_vertex_normals,
        obj_verts=obj_verts,
        obj_vertex_normals=obj_vertex_normals,
    )

    return affordance_extraction_inputs
