import blenderproc as bproc
import bpy

from PIL import Image
import numpy as np
import cv2

from trimesh.boolean import intersection
import trimesh
import torch
from mathutils import Matrix

from tqdm import tqdm
from time import time
import argparse
import pickle
import os
import sys

sys.path.append(os.getcwd())

from constants.generation.visualizers import COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER
from constants.generation.assets import CATEGORY2DATASET_TYPE
from constants.metadata import DEFAULT_SEED

from utils.blenderproc import initialize_scene, add_light, add_camera, add_assets, set_camera_config, set_render_config
from utils.prepare_renders import prepare_inpainting_pths
from utils.reproducibility import seed_everything


def compute_directional_size(mesh_verts, direction):
    direction = direction / np.linalg.norm(direction)  # 3 x 1, normalize direction vector
    projections = np.dot(mesh_verts, direction)  # N x 1

    min_projection, max_projection = np.min(projections), np.max(projections)
    directional_size = max_projection - min_projection

    return directional_size


def compute_nearest_point(asset_verts, point, direction):
    direction = (direction / np.linalg.norm(direction)).reshape((1, 3))  # normalize cam front vector
    point = point.copy().reshape((1, 3))

    # compute minimum distance index
    displacement = point - asset_verts  # N x 3
    perpendicular_vectors = displacement - (displacement @ direction.reshape((3, 1))) * direction  # N x 3
    distances = np.linalg.norm(perpendicular_vectors, axis=1)
    minimum_index = np.argmin(distances)

    # compute nearest point
    selected_verts = asset_verts[minimum_index]

    print(f"selected_verts: {selected_verts}")
    print(f"point: {point}")
    displacement = point - selected_verts  # 1 x 3
    nearest_point = (point - (displacement @ direction.reshape((3, 1))) * direction).reshape((3, 1))  # 3 x 1
    distance_from_point = -float((displacement @ direction.reshape((3, 1)))[0])

    print(f"displacement: {displacement}")

    return nearest_point, distance_from_point


def compute_instersection_ratio(vertsA, facesA, vertsB, facesB):
    meshA = trimesh.Trimesh(vertices=vertsA, faces=facesA)
    meshB = trimesh.Trimesh(vertices=vertsB, faces=facesB)

    intersection_volume_ratio = np.abs(intersection([meshA, meshB], engine="blender").volume / meshA.volume)

    return intersection_volume_ratio


def compute_collision(verts, faces, max_collisions):
    verts_torch = torch.tensor(verts, dtype=torch.float32, device=torch.device("cuda"))
    faces_torch = torch.tensor(faces.astype(np.int64), dtype=torch.long, device=torch.device("cuda"))

    triangles = verts_torch[faces_torch].unsqueeze(dim=0)

    m = BVH(max_collisions=max_collisions).to(torch.device("cuda"))
    torch.cuda.synchronize()
    outputs = m(triangles)
    torch.cuda.synchronize()

    outputs = outputs.detach().cpu().numpy().squeeze()
    collision = outputs[outputs[:, 0] >= 0, :]
    collision = collision.shape[0]

    return collision


def extract_candidates(human_verts, human_faces, asset_verts, asset_faces, displacements, direction, kernel_size, max_collisions, filter_out=False):
    direction = direction.reshape((1, 3))

    candidates = []
    if filter_out:
        start = time()

        x = compute_collision(human_verts, human_faces, max_collisions)
        y = compute_collision(asset_verts, asset_faces, max_collisions)
        default_collision = x + y

        collisions = []
        for idx, displacement in enumerate(displacements):
            shifted_human_verts = human_verts + displacement * direction
            shifted_human_faces = human_faces.copy() + asset_verts.shape[0]

            fused_verts = np.concatenate([asset_verts, shifted_human_verts], axis=0)
            fused_faces = np.concatenate([asset_faces, shifted_human_faces], axis=0)

            collision = compute_collision(fused_verts, fused_faces, max_collisions) - default_collision
            collisions.append(collision)

        end = time()

        print(f"elapsed time: {end - start}s")

        kernel_size_half = kernel_size // 2
        for idx, intersection_ratio in enumerate(collisions[kernel_size_half:-kernel_size_half]):
            if collisions[idx + (kernel_size_half - 1)] == 0.0 and collisions[idx + (kernel_size_half + 1)] == 0.0:
                continue

            surrounding_collisions = collisions[idx : idx + (kernel_size_half - 1)] + collisions[idx + (kernel_size_half + 1) : idx + kernel_size_half * 2]
            if intersection_ratio <= min(surrounding_collisions):
                candidates.append(dict(verts=human_verts + displacements[idx] * direction, faces=human_faces, displacement=displacement * direction))
    else:
        for displacement in displacements:
            shifted_human_verts = human_verts + displacement * direction
            candidates.append(dict(verts=shifted_human_verts, faces=human_faces, displacement=displacement * direction))

    return candidates


def select_human(candidate_lists, camera_data, segmentation_human_gt):
    candidates = []
    for idx, candidate in enumerate(candidate_lists):
        human_verts, human_faces = candidate["verts"], candidate["faces"]

        # set camera
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
            continue

        seg_idx, *_ = segmap_index
        segmentation = np.array(data["instance_segmaps"][0])
        human_segmentation = segmentation.copy()
        human_segmentation[human_segmentation != seg_idx] = 0
        human_segmentation[human_segmentation == seg_idx] = 255

        gt_numpy = (np.array(Image.fromarray(segmentation_human_gt).convert("L")) / 255).astype(np.uint8)
        human_numpy = (np.array(Image.fromarray(human_segmentation).convert("L")) / 255).astype(np.uint8)

        intersection = cv2.countNonZero(cv2.bitwise_and(gt_numpy, human_numpy))
        union = cv2.countNonZero(cv2.bitwise_or(gt_numpy, human_numpy))

        IoU = intersection / union

        candidates.append(
            dict(
                idx=idx,
                verts=human_verts,
                faces=human_faces,
                IoU=IoU,
                human_segmentation=human_segmentation,
                interval_from_center=np.abs(idx - len(candidate_lists)),
                displacement=candidate["displacement"],
            )
        )

        # remove human
        mesh_objects = bpy.data.objects
        mesh_objects.remove(mesh_objects[mesh_object_name], do_unlink=True)

    # select human with minimum segmentation error
    if len(candidates) == 0:
        return None
    else:
        selected_human = max(candidates, key=lambda candidate: (candidate["IoU"], -candidate["interval_from_center"]))
        return selected_human


def initialize_depth(
    supercategories,
    categories,
    prompts,
    inpaint_dir,
    camera_dir,
    human_pred_dir,
    human_prefilter_dir,
    save_dir,
    interval_ratio,
    retrieval_range,
    kernel_size,
    max_collisions,
    parallel_num,
    parallel_idx,
    disable_lowres_switch_for_behave,
    no_initialize,
    skip_done,
    verbose,
):
    inpaint_pths = sorted(prepare_inpainting_pths(inpaint_dir, supercategories, categories, prompts))

    selected_meshes = []
    ## for all inpaintings, prepare input & metadata for depth initialization
    for inpaint_pth in tqdm(inpaint_pths, "Preparing Inputs for Depth Initialization..."):
        # metadata
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = inpaint_pth.split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        inpaint_id, ext = inpaint_id_ext.split(".")
        assert ext == "png", "Inpainting must have '.png' extension"

        ## camera path
        camera_path = f"{camera_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}.pickle"
        with open(camera_path, "rb") as handle:
            camera_data = pickle.load(handle)
        cam_front = camera_data["R"][:, 2].reshape((3, 1))  # 3 x 1
        cam_resolution = camera_data.get("resolution", Image.open(inpaint_pth).size)

        # human-pred path
        human_pred_pth = f"{human_pred_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id}.pickle"

        # save path
        save_directory = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}"
        save_path = f"{save_directory}/{inpaint_id}.pickle"

        # filter out no human detected
        os.makedirs(save_directory, exist_ok=True)
        with open(human_pred_pth, "rb") as handle:
            mesh_pred = pickle.load(handle)
            if mesh_pred == "NO HUMANS":
                if verbose:
                    print(f"NO HUMANS for: {human_pred_pth}")
                with open(save_path, "wb") as handle:
                    pickle.dump("NO HUMANS", handle, protocol=pickle.HIGHEST_PROTOCOL)
                continue
            if mesh_pred == "MORE THAN 2 HUMANS":
                if verbose:
                    print(f"MORE THAN 2 HUMANS for: {human_pred_pth}")
                with open(save_path, "wb") as handle:
                    pickle.dump("MORE THAN 2 HUMANS", handle, protocol=pickle.HIGHEST_PROTOCOL)
                continue

        selected_meshes.append(
            dict(
                inpaint_pth=inpaint_pth,
                mesh_pred=mesh_pred,
                camera_data=camera_data,
                save_directory=save_directory,
                human_pred_pth=human_pred_pth,
                save_path=save_path,
                pbar_desc=f"Running '{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id}'",
            )
        )

    # parallel execution setting
    sub_length = (len(selected_meshes) // parallel_num) + 1
    start_idx = (parallel_idx) * sub_length
    end_idx = (parallel_idx + 1) * sub_length

    selected_meshes = sorted(selected_meshes, key=lambda x: x["save_path"])
    pbar = tqdm(selected_meshes[start_idx:end_idx])

    for mesh_data in pbar:
        # set pbar description
        pbar.set_description(desc=mesh_data["pbar_desc"])

        # metadata
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = mesh_data["inpaint_pth"].split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        inpaint_id, ext = inpaint_id_ext.split(".")

        mesh_pred = mesh_data["mesh_pred"]
        camera_data = mesh_data["camera_data"]
        save_directory = mesh_data["save_directory"]
        save_path = mesh_data["save_path"]

        if skip_done and os.path.exists(save_path):
            continue

        # camera specific data
        cam_front = camera_data["R"][:, 2].reshape((3, 1))  # 3 x 1
        cam_resolution = camera_data.get("resolution", Image.open(mesh_data["inpaint_pth"]).size)

        human_verts, human_faces, pelvis = mesh_pred["verts"], mesh_pred["faces"], mesh_pred["pelvis"]

        # transform from [pixel-scale -> camera-world scale]
        human_verts[:, 0] = (human_verts[:, 0] - cam_resolution[0] / 2) / max(cam_resolution) * camera_data["scale"]
        human_verts[:, 1] = (human_verts[:, 1] - cam_resolution[1] / 2) / max(cam_resolution) * camera_data["scale"]
        human_verts[:, 2] = (human_verts[:, 2] + 0) / max(cam_resolution) * camera_data["scale"]
        human_verts = human_verts @ (COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER @ camera_data["R"].T) + camera_data["t"]
        pelvis[0] = (pelvis[0] - cam_resolution[0] / 2) / max(cam_resolution) * camera_data["scale"]
        pelvis[1] = (pelvis[1] - cam_resolution[1] / 2) / max(cam_resolution) * camera_data["scale"]
        pelvis[2] = (pelvis[2] + 0) / max(cam_resolution) * camera_data["scale"]
        pelvis = pelvis @ (COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER @ camera_data["R"].T) + camera_data["t"]

        # human segmentation ground truth (obtained by human segmentation for inpainted image)
        with open(mesh_data["human_pred_pth"], "rb") as handle:
            human_params = pickle.load(handle)

        segmentation_human_gt = human_params["kps_aux"]["mask_person_list"][0]  # HxW

        # initialize blender scene
        initialize_scene(reset=True)
        add_light()
        camera = add_camera(resolution=cam_resolution, name="CAMERA")

        # add asset for rendering
        asset_object, _, __ = add_assets(supercategory, category, asset_id, disable_lowres_switch_for_behave=disable_lowres_switch_for_behave, place_on_floor=False)
        asset_object.rotation_euler = camera_data["obj_euler"]
        asset_object.location = camera_data["obj_location"]

        # compute asset verts and asset faces considering perturbation
        asset_verts = np.array([vertex.co for vertex in asset_object.data.vertices])  # N x 3
        asset_faces = np.array([[vertex_id for vertex_id in polygon.vertices] for polygon in asset_object.data.polygons])  # M x 3
        asset_verts = asset_verts @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER
        z_min = asset_verts[:, 2].min()
        asset_verts = asset_verts @ camera_data["obj_R"].T + camera_data["obj_t"].T

        if CATEGORY2DATASET_TYPE[(supercategory, category)] in ["SHAPENET", "SKETCHFAB", "INTERCAP", "BEHAVE"]:
            asset_verts -= [0.0, 0.0, z_min]

        set_camera_config(scale=camera_data["scale"], rotation=Matrix(camera_data["R"]).to_euler("XYZ"), location=camera_data["t"])

        if no_initialize:
            selected_human = dict(idx=None, verts=human_verts, faces=human_faces, IoU=None, human_segmentation=None, interval_from_center=None, displacement=None)
        else:
            retrieval_interval = compute_directional_size(mesh_verts=human_verts, direction=cam_front) * interval_ratio
            nearest_point, distance_from_point = compute_nearest_point(asset_verts=asset_verts, point=pelvis, direction=cam_front)  # 3 x 1

            retrieval_disaplacements = [distance_from_point + (index - retrieval_range) * retrieval_interval for index in range(retrieval_range * 2 + 1)]
            candidates = extract_candidates(
                human_verts, human_faces, asset_verts, asset_faces, displacements=retrieval_disaplacements, direction=cam_front, kernel_size=kernel_size, max_collisions=max_collisions
            )
            selected_human = select_human(candidates, camera_data, segmentation_human_gt)

            if selected_human is None:
                with open(save_path, "wb") as handle:
                    pickle.dump("ERRONEOUS SAMPLE DUE TO TOO SMALL HUMAN", handle, protocol=pickle.HIGHEST_PROTOCOL)
                continue

        # make save directory
        os.makedirs(save_directory, exist_ok=True)

        with open(save_path, "wb") as handle:
            pickle.dump(selected_human, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--inpaint_dir", type=str, default="results/generation/inpaintings")
    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--human_pred_dir", type=str, default="results/generation/human_preds")
    parser.add_argument("--human_prefilter_dir", type=str, default="results/generation/human_prefilterings")
    parser.add_argument("--save_dir", type=str, default="results/generation/human_before_opt")

    parser.add_argument("--interval_ratio", type=float, default=0.3)
    parser.add_argument("--retrieval_range", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--max_collisions", type=int, default=1000)

    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)

    parser.add_argument("--disable_lowres_switch_for_behave", action="store_true")
    parser.add_argument("--no_initialize", action="store_true")

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]
    if args.prompts is not None:
        args.prompts = [prompt.lower() for prompt in args.prompts]
    if args.no_initialize:
        args.save_dir = f"{args.save_dir}_no_initialize"
        assert args.save_dir != "results/generation/human_before_opt"

    # seed for reproducible generation
    seed_everything(args.seed)

    ## predict 3d human
    initialize_depth(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        inpaint_dir=args.inpaint_dir,
        camera_dir=args.camera_dir,
        human_pred_dir=args.human_pred_dir,
        human_prefilter_dir=args.human_prefilter_dir,
        save_dir=args.save_dir,
        interval_ratio=args.interval_ratio,
        retrieval_range=args.retrieval_range,
        kernel_size=args.kernel_size,
        max_collisions=args.max_collisions,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
        disable_lowres_switch_for_behave=args.disable_lowres_switch_for_behave,
        no_initialize=args.no_initialize,
        skip_done=args.skip_done,
        verbose=args.verbose,
    )
