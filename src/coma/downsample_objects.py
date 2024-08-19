import os
import argparse
import shutil
import json
from tqdm import tqdm
import pickle

import numpy as np

from utils.reproducibility import seed_everything
from utils.load_3d import load_obj_as_o3d_preserving_face_order
from utils.coma import simplify_mesh_and_get_indices

from constants.generation.assets import DATASET_PTHS, DATASET_TYPE2CATEGORIES, CATEGORY2ASSET


def run_downsampling(supercategory, category, asset_id, obj_pth, number_of_points, simplify_method, debug=False):
    ## load object mesh
    obj_mesh = load_obj_as_o3d_preserving_face_order(obj_pth)
    obj_vertices = np.asarray(obj_mesh.vertices)
    obj_faces = np.asarray(obj_mesh.triangles)
    obj_vertex_normals = np.asarray(obj_mesh.vertex_normals)

    V = obj_vertices.shape[0]
    F = obj_faces.shape[0]

    # 'downsample_indices': (N,) / 'downsampled_pcd': o3d.geometry.PointCloud
    downsample_indices, downsampled_pcd = simplify_mesh_and_get_indices(obj_mesh, number_of_points=number_of_points, simplify_method=simplify_method, debug=debug)

    downsampled_pcd_normal_raw = np.asarray(downsampled_pcd.normals)
    final_downsample_indices = []
    for d_index in range(len(downsampled_pcd_normal_raw)):
        if downsampled_pcd_normal_raw[d_index].sum() == 0:
            continue
        else:
            final_downsample_indices.append(d_index)
    final_downsample_indices = np.array(final_downsample_indices).astype(np.int64)

    downsampled_pcd_points_raw = np.asarray(downsampled_pcd.points)[final_downsample_indices]
    downsampled_pcd_normal_raw = downsampled_pcd_normal_raw[final_downsample_indices]

    assert len(downsampled_pcd_points_raw) == len(downsampled_pcd_normal_raw)
    for d_index in range(len(downsampled_pcd_points_raw)):
        assert downsampled_pcd_normal_raw[d_index].sum() != 0

    to_save = {
        "supercategory": supercategory,
        "category": category,
        "asset_id": asset_id,
        "V": V,
        "F": F,
        "N": len(downsample_indices),
        "N_raw": len(downsampled_pcd_points_raw),
        "downsample_indices": downsample_indices,
        "downsampled_pcd_points_raw": downsampled_pcd_points_raw,
        "downsampled_pcd_normal_raw": downsampled_pcd_normal_raw,
        "obj_vertices_original": obj_vertices,
        "obj_faces_original": obj_faces,
        "obj_vertex_normals_original": obj_vertex_normals,
    }

    return to_save


def main(args):
    # run object downsampling if not BEHAVE
    scs = DATASET_TYPE2CATEGORIES[args.dataset_type]

    if args.supercategories is not None:
        scs = [sc for sc in scs if sc[0].lower() in args.supercategories]
    if args.categories is not None:
        scs = [sc for sc in scs if sc[1].lower() in args.categories]

    ## iterate for (supercategory / category) pairs
    for supercategory, category in tqdm(scs):
        ## dataset-pth
        dataset_dir = DATASET_PTHS[args.dataset_type]

        ## supercategory / category as string
        supercategory_str = supercategory.replace("/", ":")
        category_str = category.replace("/", ":")

        ## for all assets
        for asset_id in CATEGORY2ASSET[supercategory][category]:
            # save path
            os.makedirs(f"{args.save_dir}/{supercategory_str}/{category_str}", exist_ok=True)
            save_pth = f"{args.save_dir}/{supercategory_str}/{category_str}/{asset_id}_{args.number_of_points}.pickle"
            mesh_copy_pth = f"{args.save_dir}/{supercategory_str}/{category_str}/{asset_id}.obj"

            if args.skip_done and os.path.exists(save_pth):
                continue

            ## BEHAVE
            if args.dataset_type == "BEHAVE":
                assert supercategory.lower() == "behave"

                # object mesh path
                obj_pth = f"{dataset_dir}/{asset_id}/raw_model.obj"
                if args.disable_lowres_switch_for_behave:
                    obj_pth = f"{dataset_dir}/objects/{category}/{category}.obj"
                else:
                    obj_pth = f"{dataset_dir}/objects/{category}/{category}_canon_lowres_in_gen_coord.obj"

            ## INTERCAP
            if args.dataset_type == "INTERCAP":
                assert supercategory.lower() == "intercap"

                # object mesh path
                obj_pth = f"{dataset_dir}/objects/{category}/mesh.obj"

            ## 3D-FUTURE
            if args.dataset_type == "SHAPENET":
                with open(f"{dataset_dir}/taxonomy.json", "r") as file:
                    categories = json.load(file)

                selected_category_info, *_ = [category_info for category_info in categories if category_info["name"] == category]
                obj_pth = f"{dataset_dir}/{selected_category_info['synsetId']}/{asset_id}/models/model_normalized.obj"

            ## SHAPENET
            if args.dataset_type == "3D-FUTURE":
                # object mesh path
                obj_pth = f"{dataset_dir}/{asset_id}/raw_model.obj"

            ## SKETCHFAB
            if args.dataset_type == "SKETCHFAB":
                obj_pth = f"{dataset_dir}/{supercategory}/{asset_id}/model.obj"

            ## SAPIEN
            if args.dataset_type == "SAPIEN":
                obj_pth = f"{dataset_dir}/{supercategory}/{asset_id}/model.obj"

            downsample_metadata = run_downsampling(
                supercategory=supercategory,
                category=category,
                asset_id=asset_id,
                obj_pth=obj_pth,
                number_of_points=args.number_of_points,
                simplify_method=args.simplify_method,
            )

            with open(save_pth, "wb") as handle:
                pickle.dump(downsample_metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            shutil.copy(src=obj_pth, dst=mesh_copy_pth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")

    parser.add_argument("--save_dir", type=str, default="results/coma/asset_downsample")

    parser.add_argument("--simplify_method", choices=["poisson_disk", "uniform"], default="poisson_disk")
    parser.add_argument("--dataset_type", type=str, choices=DATASET_PTHS.keys())
    parser.add_argument("--disable_lowres_switch_for_behave", action="store_true")  # keep as false

    parser.add_argument("--number_of_points", type=int)  # may not be exact since (1) duplicating vertices / (2) no vertex-normal points
    parser.add_argument("--use_watertight", action="store_true")  # keep as False

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.use_watertight:
        raise NotImplementedError

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]

    # seed for reproducible generation
    seed_everything(args.seed)

    main(args)
