import os
import argparse
import pickle
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import torch
import open3d as o3d

from utils.reproducibility import seed_everything
from utils.load_3d import load_obj_as_o3d_preserving_face_order
from utils.coma import ComA, prepare_affordance_extraction_inputs, get_aggregated_contact
from utils.coma_occupancy import ComA_Occupancy
from utils.visualization.colormap import MplColorHelper

from constants.coma.coma_basic_settings import AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT
from constants.coma.qual import QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT
from constants.metadata import DEFAULT_SEED

mayavi_scaler = lambda x: x * 16.0 + 24.0

colormap = MplColorHelper("jet", 0.0, 1.0)


def inference(supercategory, category, coma_path, smplx_downsample_pth, asset_downsample_pth, visualize_type, hyperparams_key, hyperparams, output_dir):
    hyperparams["human_res"]
    human_use_downsample_pcd_raw = hyperparams["human_use_downsample_pcd_raw"]  # if true, then uses pcd
    hyperparams["object_res"]
    object_use_downsample_pcd_raw = hyperparams["object_use_downsample_pcd_raw"]  # if true, then uses pcd
    principle_vec = hyperparams["principle_vec"]
    sub_principle_vec = hyperparams["sub_principle_vec"]
    rel_dist_method = hyperparams["rel_dist_method"]
    spatial_grid_size = hyperparams["spatial_grid_size"]
    spatial_grid_thres = hyperparams["spatial_grid_thres"]
    normal_gaussian_sigma = hyperparams["normal_gaussian_sigma"]
    normal_res = hyperparams["normal_res"]
    spatial_res = hyperparams["spatial_res"]
    eps = hyperparams["eps"]
    significant_contact_ratio = hyperparams["significant_contact_ratio"]
    hyperparams["enable_postfilter"]
    hyperparams["standardize_human_scale"]
    hyperparams["scaler_range"]

    hyperparams["vis_example_num"] if "vis_example_num" in hyperparams else vis_example_num
    visualize_type = hyperparams["visualize_type"]

    with open(smplx_downsample_pth, "rb") as handle:
        human_downsample_metadata = pickle.load(handle)

    with open(asset_downsample_pth, "rb") as handle:
        object_downsample_metadata = deepcopy(pickle.load(handle))

    H = human_downsample_metadata["N_raw"] if human_use_downsample_pcd_raw else human_downsample_metadata["N"]
    O = object_downsample_metadata["N_raw"] if object_use_downsample_pcd_raw else object_downsample_metadata["N"]

    if visualize_type == "occupancy":
        coma = ComA_Occupancy(
            scale_tolerance=3.0,
            human_res=H,
            obj_res=O,
            normal_res=normal_res,
            spatial_res=spatial_res,
            proximity_settings=dict(
                spatial_grid_size=spatial_grid_size,
                spatial_grid_thres=spatial_grid_thres,
            ),
            principle_vec=principle_vec,
            sub_principle_vec=sub_principle_vec,
            rel_dist_method=rel_dist_method,
            normal_gaussian_sigma=normal_gaussian_sigma,
            eps=eps,
            device="cuda",
        )
    else:
        coma = ComA(
            human_res=H,
            obj_res=O,
            normal_res=normal_res,
            spatial_res=spatial_res,
            proximity_settings=dict(
                spatial_grid_size=spatial_grid_size,
                spatial_grid_thres=spatial_grid_thres,
            ),
            principle_vec=principle_vec,
            sub_principle_vec=sub_principle_vec,
            rel_dist_method=rel_dist_method,
            normal_gaussian_sigma=normal_gaussian_sigma,
            eps=eps,
            device="cuda",
        )

    coma.load(coma_path)

    if visualize_type == "aggr-human-contact":
        aggregated_contact, significant_contact_vertex_indices = get_aggregated_contact(
            coma=coma,
            contact_map_type="human",
            significant_contact_ratio=significant_contact_ratio,
        )

        os.makedirs(f"{output_dir}/{supercategory}/{category}/", exist_ok=True)
        np.save(f"{output_dir}/{supercategory}/{category}/human_contact.npy", aggregated_contact / aggregated_contact.max())

    elif visualize_type == "aggr-object-contact":
        aggregated_contact, significant_contact_vertex_indices = get_aggregated_contact(
            coma=coma,
            contact_map_type="obj",
            significant_contact_ratio=significant_contact_ratio,
        )

        obj_pcd_points = (object_downsample_metadata["downsampled_pcd_points_raw"],)
        obj_pcd_normals = (object_downsample_metadata["downsampled_pcd_normal_raw"],)

        score_on_geo = o3d.geometry.PointCloud()
        score_on_geo.points = o3d.utility.Vector3dVector(obj_pcd_points[0])
        score_on_geo.normals = o3d.utility.Vector3dVector(obj_pcd_normals[0])

        score = aggregated_contact / aggregated_contact.max()
        colors = colormap.get_rgb(score)[:, :3]
        score_on_geo.colors = o3d.utility.Vector3dVector(colors)

        os.makedirs(f"{output_dir}/{supercategory}/{category}/", exist_ok=True)
        o3d.io.write_point_cloud(f"{output_dir}/{supercategory}/{category}/object_contact.ply", score_on_geo)

    elif visualize_type == "orientation":
        nonphysical_scores = coma.compute_nonphysical_response_sphere(n_bin=1e6, nonphysical_type="human", as_numpy=True,)[
            "human"
        ]  # H x O

        obj_i = 0
        nonphysical_score = nonphysical_scores[:, obj_i]  # H

        os.makedirs(f"{output_dir}/{supercategory}/{category}/", exist_ok=True)
        np.save(f"{output_dir}/{supercategory}/{category}/orientational_tendency.npy", (nonphysical_score - nonphysical_score.min()) / (nonphysical_score.max() - nonphysical_score.min()))

    elif visualize_type == "occupancy":
        prob_field = coma.return_aggregated_spatial_grids(human_indices=None).cpu().numpy()  # N x N x N

        prob_field /= prob_field.max()
        prob_field = 0.7 * prob_field

        spatial_grid_metadata = coma.spatial_grid_metadata

        occupancy_info = dict(prob_field=prob_field, spatial_grid_metadata=spatial_grid_metadata)
        os.makedirs(f"{output_dir}/{supercategory}/{category}/", exist_ok=True)
        np.save(f"{output_dir}/{supercategory}/{category}/occupancy.npy", occupancy_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategory", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--coma_path", type=str)
    parser.add_argument("--visualize_type", type=str, choices=["aggr-human-contact", "aggr-object-contact", "orientation", "occupancy"])
    parser.add_argument("--smplx_downsample_pth", type=str)
    parser.add_argument("--asset_downsample_pth", type=str)
    parser.add_argument("--hyperparams_key", type=str)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    # seed for reproducible generation
    seed_everything(args.seed)

    assert args.hyperparams_key is not None, "You must Specify the 'args.hypeparams_key'"
    if "qual:" in args.hyperparams_key:
        hyperparams = QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[args.hyperparams_key]
    else:
        hyperparams = AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[args.hyperparams_key]

    inference(
        supercategory=args.supercategory,
        category=args.category,
        coma_path=args.coma_path,
        visualize_type=args.visualize_type,
        smplx_downsample_pth=args.smplx_downsample_pth,
        asset_downsample_pth=args.asset_downsample_pth,
        hyperparams_key=args.hyperparams_key,
        hyperparams=hyperparams,
        output_dir=args.output_dir,
    )
