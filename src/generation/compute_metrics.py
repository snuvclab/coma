import blenderproc as bproc
import bpy

import argparse
from glob import glob
import pickle
import os
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
import trimesh
from trimesh.boolean import intersection
import cv2

import sys

sys.path.append(os.getcwd())

from constants.metadata import DEFAULT_SEED
from constants.generation.assets import DATASET_PTHS, CATEGORY2DATASET_TYPE
from constants.generation.visualizers import COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER

from utils.blenderproc import initialize_scene, add_light, add_camera, add_assets, set_camera_config, set_render_config
from utils.reproducibility import seed_everything

ASSET_INFO = dict()


def get_gt_human_segmap(human_pred_pth):
    with open(human_pred_pth, "rb") as handle:
        human_params = pickle.load(handle)

    segmentation_human_gt = human_params["kps_aux"]["mask_person_list"][0]  # HxW

    return (np.array(Image.fromarray(segmentation_human_gt).convert("L")) / 255).astype(np.uint8)


def get_rendered_human_segmap(supercategory, category, asset_id, human_verts, human_faces, camera_data, disable_lowres_switch_for_behave):
    initialize_scene(reset=True)
    add_light()
    camera = add_camera(resolution=camera_data["resolution"], name="CAMERA")

    # add asset for rendering
    asset_object, _, __ = add_assets(supercategory, category, asset_id, disable_lowres_switch_for_behave=disable_lowres_switch_for_behave, place_on_floor=False)
    asset_object.rotation_euler = camera_data["obj_euler"]
    asset_object.location = camera_data["obj_location"]

    camera_matrix = np.vstack([np.hstack([camera_data["R"], camera_data["t"].reshape(-1, 1)]), np.array([0, 0, 0, 1.0])])  # 4 x 4
    bproc.camera.add_camera_pose(camera_matrix)

    set_camera_config(scale=camera_data["scale"])
    set_render_config(resolution=camera_data["resolution"])

    # add human
    collection = bpy.data.collections.new("SMPL Meshes")
    bpy.context.scene.collection.children.link(collection)

    mesh = bpy.data.meshes.new(name="SMPL")
    mesh.from_pydata(human_verts, [], human_faces)
    mesh.update()

    mesh_object = bpy.data.objects.new("SMPL", mesh)
    mesh_object_name = mesh_object.name
    collection.objects.link(mesh_object)

    # render segmentation
    data = bproc.renderer.render_segmap(map_by=["instance", "name"])

    # make human segmentation
    segmap_index = [instance["idx"] for instance in data["instance_attribute_maps"][0] if instance["name"] == mesh_object_name]
    if len(segmap_index) == 0:
        return None

    seg_idx, *_ = segmap_index
    segmentation = np.array(data["instance_segmaps"][0])

    human_segmentation = segmentation.copy()
    human_segmentation[human_segmentation != seg_idx] = 0
    human_segmentation[human_segmentation == seg_idx] = 255

    return (np.array(Image.fromarray(human_segmentation).convert("L")) / 255).astype(np.uint8)


def compute_metrics(camera_data, human_pred_pth, supercategory, category, human_verts, human_faces, asset_id, asset_verts, asset_faces, disable_lowres_switch_for_behave):
    def compute_instersection_ratio(vertsA, facesA, vertsB, facesB):
        """
        middle priority metric
        vertsA: 1 x N x 3, pytorch tensor
        vertsB: 1 x M x 3, pytorch tensor
        facesA: 1 x P x 3, pytorch tensor
        facesB: 1 x Q x 3, pytorch tensor
        """
        meshA = trimesh.Trimesh(vertices=vertsA, faces=facesA)
        meshB = trimesh.Trimesh(vertices=vertsB, faces=facesB)

        intersection_volume_ratio = np.abs(intersection([meshA, meshB], engine="blender").volume / meshA.volume)

        return intersection_volume_ratio

    def compute_IoU(camera_data, human_pred_pth, supercategory, category, human_verts, human_faces, asset_id, disable_lowres_switch_for_behave):
        gt_numpy = get_gt_human_segmap(human_pred_pth)
        human_numpy = get_rendered_human_segmap(supercategory, category, asset_id, human_verts, human_faces, camera_data, disable_lowres_switch_for_behave)

        if human_numpy is None:
            IoU = 0.0
        else:
            intersection = cv2.countNonZero(cv2.bitwise_and(gt_numpy, human_numpy))
            union = cv2.countNonZero(cv2.bitwise_or(gt_numpy, human_numpy))
            IoU = intersection / union

        return IoU

    interscetion_ratio = compute_instersection_ratio(human_verts, human_faces, asset_verts, asset_faces)
    IoU = compute_IoU(camera_data, human_pred_pth, supercategory, category, human_verts, human_faces, asset_id, disable_lowres_switch_for_behave)

    metrics = dict(interscetion_ratio=interscetion_ratio, IoU=IoU)

    return metrics


def get_asset_info(supercategory, category, asset_id, view_id, camera_data, disable_lowres_switch_for_behave):
    if ASSET_INFO.get(f"{supercategory}_{category}_{asset_id}_{view_id}", None) is None:

        dataset_type = CATEGORY2DATASET_TYPE[(supercategory, category)]
        dataset_pth = DATASET_PTHS[dataset_type]

        if dataset_type == "3D-FUTURE":
            obj_pth = f"{dataset_pth}/{asset_id}/raw_model.obj"

        elif dataset_type == "SHAPENET":
            with open(f"{dataset_pth}/taxonomy.json", "r") as file:
                categories = json.load(file)

            selected_category_info, *_ = [category_info for category_info in categories if category_info["name"] == category]
            obj_pth = f"{dataset_pth}/{selected_category_info['synsetId']}/{asset_id}/models/model_normalized.obj"

        elif dataset_type == "SKETCHFAB":
            obj_pth = f"{dataset_pth}/{supercategory}/{asset_id}/model.obj"

        elif dataset_type == "BEHAVE":
            if disable_lowres_switch_for_behave:
                obj_pth = f"{dataset_pth}/objects/{category}/{category}.obj"
            else:
                obj_pth = f"{dataset_pth}/objects/{category}/{category}_canon_lowres_in_gen_coord.obj"

        elif dataset_type == "INTERCAP":
            obj_pth = f"{dataset_pth}/objects/{category}/mesh.obj"
        else:
            raise NotImplementedError

        asset_mesh = trimesh.load(obj_pth, force="mesh", process=False)

        asset_verts = np.array(asset_mesh.vertices) @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER
        asset_faces = np.array(asset_mesh.faces)

        z_min = asset_verts[:, 2].min()

        asset_obj_R = camera_data["obj_R"]
        asset_obj_t = camera_data["obj_t"].reshape((1, 3))

        asset_verts = asset_verts @ asset_obj_R.T + asset_obj_t

        if dataset_type in ["SHAPENET", "SKETCHFAB", "INTERCAP", "BEHAVE"]:
            asset_verts -= [0.0, 0.0, z_min]

        ASSET_INFO[f"{supercategory}_{category}_{asset_id}_{view_id}"] = dict(verts=asset_verts, faces=asset_faces, z_min=z_min)

    return ASSET_INFO[f"{supercategory}_{category}_{asset_id}_{view_id}"]


def save_human(
    supercategories,
    categories,
    prompts,
    human_after_opt_dir,
    human_pred_dir,
    camera_dir,
    save_dir,
    enable_aggregate_total_prompts,
    disable_lowres_switch_for_behave,
    skip_done,
    parallel_idx,
    parallel_num,
):
    if enable_aggregate_total_prompts:
        human_pths = sorted(list(glob(f"{human_after_opt_dir}/*/*/*/*/*/total*/*.pickle")))
    else:
        human_pths = sorted(list(glob(f"{human_after_opt_dir}/*/*/*/*/*/[!total]*/*.pickle")))

    filtered_human_pths = [
        human_pth
        for human_pth in human_pths
        if (not supercategories or human_pth.split("/")[-7].lower() in supercategories)
        and (not categories or human_pth.split("/")[-6].lower() in categories)
        and (not prompts or human_pth.split("/")[-2].lower() in prompts)
    ]

    # parallel execution setting
    sub_length = (len(filtered_human_pths) // parallel_num) + 1
    start_idx = (parallel_idx) * sub_length
    end_idx = (parallel_idx + 1) * sub_length
    filtered_human_pths = sorted(filtered_human_pths)

    for human_pth in tqdm(filtered_human_pths[start_idx:end_idx]):
        supercategory, category, asset_id, view_id, asset_mask_id, prompt, inpaint_id_with_ext = human_pth.split("/")[-7:]
        inpaint_id, ext = inpaint_id_with_ext.split(".")
        prompt_without_total = prompt.split("total:")[-1]

        save_pth = f"{save_dir}/{supercategory}/{category}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id}.pickle"
        save_directory = f"{save_dir}/{supercategory}/{category}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/"
        camera_pth = f"{camera_dir}/{supercategory}/{category}/{asset_id}/{view_id}.pickle"
        human_pred_pth = f"{human_pred_dir}/{supercategory}/{category}/{asset_id}/{view_id}/{asset_mask_id}/{prompt_without_total}/{inpaint_id}.pickle"

        if skip_done and os.path.exists(save_pth):
            continue

        with open(human_pth, "rb") as handle:
            human_mesh = pickle.load(handle)

        os.makedirs(save_directory, exist_ok=True)
        if type(human_mesh) == str:
            with open(save_pth, "wb") as handle:
                pickle.dump(human_mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(camera_pth, "rb") as handle:
                camera_data = pickle.load(handle)

            asset_info = get_asset_info(supercategory, category, asset_id, view_id, camera_data, disable_lowres_switch_for_behave)
            z_min = asset_info["z_min"]

            asset_verts = asset_info["verts"]
            asset_faces = asset_info["faces"]
            human_verts = human_mesh["verts"]
            human_faces = human_mesh["faces"]

            metrics = compute_metrics(camera_data, human_pred_pth, supercategory, category, human_verts, human_faces, asset_id, asset_verts, asset_faces, disable_lowres_switch_for_behave)
            human_mesh.update(metrics)

            verts_blender = (human_mesh["verts"] + [0.0, 0.0, z_min] - camera_data["obj_t"].T) @ camera_data["obj_R"] - [0.0, 0.0, z_min]  # TO BLENDER
            verts_trimesh = verts_blender @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER.T + np.array([0.0, 0.0, z_min]) @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER.T

            human_mesh["verts"] = verts_trimesh
            human_mesh["z_min"] = z_min

            with open(save_pth, "wb") as handle:
                pickle.dump(human_mesh, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--human_after_opt_dir", type=str, default="results/generation/human_after_opt")
    parser.add_argument("--human_pred_dir", type=str, default="results/generation/human_preds")
    parser.add_argument("--save_dir", type=str, default="results/generation/human_sample")

    parser.add_argument("--enable_aggregate_total_prompts", action="store_true")
    parser.add_argument("--disable_lowres_switch_for_behave", default=True)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)

    args = parser.parse_args()

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]
    if args.prompts is not None:
        args.prompts = [prompt.lower() for prompt in args.prompts]

    # seed for reproducible generation
    seed_everything(args.seed)

    ## run depth optimization
    save_human(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        human_after_opt_dir=args.human_after_opt_dir,
        human_pred_dir=args.human_pred_dir,
        camera_dir=args.camera_dir,
        save_dir=args.save_dir,
        enable_aggregate_total_prompts=args.enable_aggregate_total_prompts,
        disable_lowres_switch_for_behave=args.disable_lowres_switch_for_behave,
        skip_done=args.skip_done,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
    )
