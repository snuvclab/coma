import blenderproc as bproc
import bpy

import os
import sys

sys.path.append(os.getcwd())

import json
import pickle
import argparse
from glob import glob
from tqdm import tqdm

from mathutils import Matrix
import numpy as np
import cv2

from utils.transformations import deg2rad
from utils.blenderproc import initialize_scene, add_plane, add_light, add_camera, set_camera_config, render_points, render, segmentation_handler
from utils.reproducibility import seed_everything

from constants.metadata import DEFAULT_SEED
from constants.generation.assets import CATEGORY2CAMERA_CONFIG, CATEGORY2PERTURB_CONFIG, CATEGORY2ASSET, DATASET_PTHS


def render_from_asset_info(
    assets_info_list,
    supercategories,
    categories,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):
    # only run for specified asset_ids under (supercategory/category)
    assets_info_list = [asset_info for asset_info in assets_info_list if asset_info["super-category"] in CATEGORY2ASSET.keys()]
    assets_info_list = [asset_info for asset_info in assets_info_list if asset_info["category"] in CATEGORY2ASSET[asset_info["super-category"]].keys()]
    assets_info_list = [asset_info for asset_info in assets_info_list if asset_info["model_id"] in CATEGORY2ASSET[asset_info["super-category"]][asset_info["category"]]]

    # only run for specified supercategory/category
    if supercategories is not None:
        assets_info_list = [asset_info for asset_info in assets_info_list if asset_info["super-category"].lower() in supercategories]
    if categories is not None:
        assets_info_list = [asset_info for asset_info in assets_info_list if asset_info["category"].lower() in categories]

    # blender settings
    plane = add_plane()
    light = add_light()

    resolution = default_resolution  ## change here if wanted
    camera = add_camera(default_resolution, "INTERCAP-CAMERA")

    for asset_info in tqdm(assets_info_list):
        # asset mesh path
        obj_pth = asset_info["obj_pth"]

        # metadata
        supercategory_str = asset_info["super-category"].replace("/", ":")
        category_str = asset_info["category"].replace("/", ":")
        asset_id = asset_info["model_id"]

        # saving paths
        asset_render_save_dir = f"{asset_render_dir}/{supercategory_str}/{category_str}/{asset_id}"
        asset_mask_save_dir = f"{asset_mask_dir}/{supercategory_str}/{category_str}/{asset_id}"
        asset_seg_save_dir = f"{asset_seg_dir}/{supercategory_str}/{category_str}/{asset_id}"
        camera_save_dir = f"{camera_dir}/{supercategory_str}/{category_str}/{asset_id}"
        os.makedirs(asset_render_save_dir, exist_ok=True)
        os.makedirs(asset_mask_save_dir, exist_ok=True)
        os.makedirs(asset_seg_save_dir, exist_ok=True)
        os.makedirs(camera_save_dir, exist_ok=True)

        # prepare weak-perspective camera
        camera_config = CATEGORY2CAMERA_CONFIG[asset_info["super-category"]][asset_info["category"]]
        if "asset_specific_config" in camera_config:
            if asset_id in camera_config["asset_specific_config"].keys():
                camera_config = camera_config["asset_specific_config"][asset_id]

        # prepare asset
        bpy.ops.import_scene.obj(filepath=obj_pth)
        asset_data = bpy.context.selected_objects[0].data
        vertices = np.array([vertex.co for vertex in asset_data.vertices])

        # asset size (x,y,z)
        x_min = vertices[:, 0].min()
        x_max = vertices[:, 0].max()
        y_min = vertices[:, 2].min()  # For Blender Coordinate Direction Correction
        y_max = vertices[:, 2].max()  # For Blender Coordinate Direction Correction
        z_min = vertices[:, 1].min()  # For Blender Coordinate Direction Correction
        z_max = vertices[:, 1].max()  # For Blender Coordinate Direction Correction
        length_x = x_max - x_min
        length_y = y_max - y_min
        length_z = z_max - z_min
        maximum_length = max(length_x, length_y, length_z)
        scale = maximum_length * camera_config["ortho_scale"] * 2

        # put object on the xy-plane
        bpy_object = bpy.context.selected_objects[0]
        bpy_object.location.z -= z_min

        # default asset transformations
        default_euler_rotation = bpy_object.rotation_euler
        default_rotation_matrix = np.array(default_euler_rotation.to_matrix())
        default_location = np.array(bpy_object.location).reshape((3, 1))

        # setting camera configuration
        radius = 10  # efficiently large value, doesn't matter since the rendering is orthographic (2023.09.30 some objects are exception; too big -> raw model size is not accurate!)

        # weak-perspective camera settings
        elevation = deg2rad(camera_config.get("elevation", default_elevation))
        azimuth = deg2rad(camera_config.get("azimuth", default_azimuth))
        view_num = camera_config.get("view_num", default_view_num)
        perturb_sample_num = camera_config.get("perturb_sample_num", default_perturb_sample_num)

        cameras = [
            dict(
                location=(
                    radius * np.cos(elevation) * np.cos(azimuth + (2 * np.pi / view_num) * view_idx),
                    radius * np.cos(elevation) * np.sin(azimuth + (2 * np.pi / view_num) * view_idx),
                    radius * np.sin(elevation) + length_z * camera_config["z_scale"],
                ),
                rotation=(np.pi / 2 - elevation, 0, np.pi / 2 + azimuth + (2 * np.pi / view_num) * view_idx),
            )
            for view_idx in range(view_num)
        ]

        # perturb config
        perturb_config = CATEGORY2PERTURB_CONFIG[asset_info["super-category"]][asset_info["category"]]
        if "asset_specific_config" in perturb_config:
            if asset_id in perturb_config["asset_specific_config"].keys():
                perturb_config = perturb_config["asset_specific_config"][asset_id]

        if not perturb_config["need_perturb"]:
            pertub_list = [
                dict(
                    rotation_matrix_x=np.eye(3),
                    rotation_matrix_y=np.eye(3),
                    rotation_matrix_z=np.eye(3),
                    displacement_x=0.0,
                    displacement_y=0.0,
                    displacement_z=0.0,
                )
            ]
        else:
            # perturb config for "rotation_x"
            if perturb_config.get("rotation_x", None) is not None:
                x_intervals = perturb_config["rotation_x"]
                prob = np.array([start - end for start, end in x_intervals])
                prob = prob / prob.sum()
                sampled_x_angles = [np.random.choice([np.random.uniform(*interval) for interval in x_intervals], p=prob) for _ in range(perturb_sample_num)]
                rotation_matrix_x_list = [
                    np.array(
                        [
                            [1.0, 0.0, 0.0],
                            [0.0, np.cos(deg2rad(sampled_x_angle)), -np.sin(deg2rad(sampled_x_angle))],
                            [0.0, np.sin(deg2rad(sampled_x_angle)), np.cos(deg2rad(sampled_x_angle))],
                        ]
                    )
                    for sampled_x_angle in sampled_x_angles
                ]
            else:
                rotation_matrix_x_list = [np.eye(3) for _ in range(perturb_sample_num)]

            # perturb config for "rotation_y"
            if perturb_config.get("rotation_y", None) is not None:
                y_intervals = perturb_config["rotation_y"]
                prob = np.array([start - end for start, end in y_intervals])
                prob = prob / prob.sum()
                sampled_y_angles = [np.random.choice([np.random.uniform(*interval) for interval in y_intervals], p=prob) for _ in range(perturb_sample_num)]
                rotation_matrix_y_list = [
                    np.array(
                        [
                            [np.cos(deg2rad(sampled_y_angle)), 0.0, -np.sin(deg2rad(sampled_y_angle))],
                            [0.0, 1.0, 0.0],
                            [np.sin(deg2rad(sampled_y_angle)), 0.0, np.cos(deg2rad(sampled_y_angle))],
                        ]
                    )
                    for sampled_y_angle in sampled_y_angles
                ]
            else:
                rotation_matrix_y_list = [np.eye(3) for _ in range(perturb_sample_num)]

            # perturb config for "rotation_z"
            if perturb_config.get("rotation_z", None) is not None:
                z_intervals = perturb_config["rotation_z"]
                prob = np.array([start - end for start, end in z_intervals])
                prob = prob / prob.sum()
                sampled_z_angles = [np.random.choice([np.random.uniform(*interval) for interval in z_intervals], p=prob) for _ in range(perturb_sample_num)]
                rotation_matrix_z_list = [
                    np.array(
                        [
                            [np.cos(deg2rad(sampled_z_angle)), -np.sin(deg2rad(sampled_z_angle)), 0.0],
                            [np.sin(deg2rad(sampled_z_angle)), np.cos(deg2rad(sampled_z_angle)), 0.0],
                            [0.0, 0.0, 1.0],
                        ]
                    )
                    for sampled_z_angle in sampled_z_angles
                ]
            else:
                rotation_matrix_z_list = [np.eye(3) for _ in range(perturb_sample_num)]

            # perturb config for locations
            displacement_list = []
            for displacement_direction in ["displacement_x", "displacement_y", "displacement_z"]:
                if perturb_config.get(displacement_direction, None) is not None:
                    x_intervals = perturb_config[displacement_direction]
                    prob = np.array([start - end for start, end in x_intervals])
                    prob = prob / prob.sum()
                    displacement_list.append([np.random.choice([np.random.uniform(*interval) for interval in x_intervals], p=prob) for _ in range(perturb_sample_num)])
                else:
                    displacement_list.append([0.0 for _ in range(perturb_sample_num)])
            displacement_x_list, displacement_y_list, displacement_z_list = displacement_list

            pertub_list = [
                dict(
                    rotation_matrix_x=rotation_x,
                    rotation_matrix_y=rotation_y,
                    rotation_matrix_z=rotation_z,
                    displacement_x=displacement_x,
                    displacement_y=displacement_y,
                    displacement_z=displacement_z,
                )
                for rotation_x, rotation_y, rotation_z, displacement_x, displacement_y, displacement_z in list(
                    zip(rotation_matrix_x_list, rotation_matrix_y_list, rotation_matrix_z_list, displacement_x_list, displacement_y_list, displacement_z_list)
                )
            ]

        for perturb_idx, perturb_info in enumerate(pertub_list):
            # fetch perturb information
            rotation_matrix_x = perturb_info["rotation_matrix_x"]
            rotation_matrix_y = perturb_info["rotation_matrix_y"]
            rotation_matrix_z = perturb_info["rotation_matrix_z"]
            displacement_x = perturb_info["displacement_x"]
            displacement_y = perturb_info["displacement_y"]
            displacement_z = perturb_info["displacement_z"]

            # perturb rotation
            rotation_matrix = rotation_matrix_x @ rotation_matrix_y @ rotation_matrix_z
            rotation = rotation_matrix @ default_rotation_matrix.copy()
            euler_angle = Matrix(rotation).to_euler("XYZ")
            bpy_object.rotation_euler = euler_angle

            # perturb location
            displacements = [displacement_x * length_x, displacement_y * length_y, displacement_z * length_z]
            bpy_object.location.x = default_location.copy()[0] + displacements[0]
            bpy_object.location.y = default_location.copy()[1] + displacements[1]
            bpy_object.location.z = default_location.copy()[2] + displacements[2]

            # run for rendering cameras
            for idx, render_camera in enumerate(cameras):
                set_camera_config(scale, location=render_camera["location"], rotation=render_camera["rotation"])

                # setting mask render configuration
                angle_z = render_camera["rotation"][-1]
                axis_x = np.array([np.cos(angle_z), np.sin(angle_z), 0])
                np.array([np.sin(angle_z), -np.cos(angle_z), 0])
                axis_z = np.array([0, 0, 1])

                bbox_x, bbox_y, bbox_z = camera_config["bbox_size"] if camera_config.get("bbox_size", None) is not None else default_bbox_size

                stride_x = camera_config.get("stride_x", default_stride_x)
                stride_y = camera_config.get("stride_y", default_stride_y)

                x_coordinates = np.arange(np.ceil((x_min - bbox_x) / stride_x) * stride_x, x_max + bbox_x, stride_x).reshape((-1, 1))
                y_coordinates = np.arange(np.ceil((y_min - bbox_y) / stride_y) * stride_y, y_max + bbox_y, stride_y).reshape((-1, 1))

                x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
                xyz_coordinates = np.column_stack((x_grid.ravel(), y_grid.ravel(), np.zeros(x_grid.ravel().shape)))  # N x 3

                four_points = np.stack(
                    [
                        xyz_coordinates + axis_x * bbox_y,
                        xyz_coordinates + axis_x * bbox_y + axis_z * bbox_z * 2,
                        xyz_coordinates - axis_x * bbox_y + axis_z * bbox_z * 2,
                        xyz_coordinates - axis_x * bbox_y,
                    ],
                    axis=1,
                )  # N x 4 x 3
                four_points_homogeneous = np.concatenate([four_points, np.ones((four_points.shape[0], four_points.shape[1], 1))], axis=2)  # N x 4 x 4

                rendered_points = render_points(four_points_homogeneous)

                # render mask and save
                for mask_idx, four_point in enumerate(rendered_points):
                    os.makedirs(f"{asset_mask_save_dir}/view:{perturb_idx * view_num + idx:05d}", exist_ok=True)
                    asset_mask_save_pth = f"{asset_mask_save_dir}/view:{perturb_idx * view_num + idx:05d}/{mask_idx:05d}.png"

                    xy_coordinates = np.array(four_point, dtype=np.int32)
                    hull = cv2.convexHull(xy_coordinates)
                    mask = np.zeros(resolution, dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull, 255)

                    cv2.imwrite(asset_mask_save_pth, mask)

                # render asset and save
                hide_objects = [object for object in bpy.data.objects if object.name not in [asset_data.name, plane.name, light.name]]
                asset_render_save_pth = f"{asset_render_save_dir}/view:{perturb_idx * view_num + idx:05d}.png"
                if not os.path.exists(asset_render_save_pth) or not skip_done:
                    render(asset_render_save_pth, hide_objects=hide_objects, use_gpu=True, desc="multiview object rendering")
                elif verbose:
                    print(f"{asset_render_save_pth} already processed!!")

                # render segmentation and save
                hide_objects = [object for object in bpy.data.objects if object.name != asset_data.name]
                asset_seg_save_pth = f"{asset_seg_save_dir}/view:{perturb_idx * view_num + idx:05d}.png"
                handler = segmentation_handler(asset_seg_save_pth)
                if not os.path.exists(asset_seg_save_pth) or not skip_done:
                    render(asset_seg_save_pth, hide_objects=hide_objects, use_gpu=True, desc="multiview segmentation rendering", handler=handler)
                elif verbose:
                    print(f"{asset_seg_save_pth} already processed!!")

                # save camera
                scene = bpy.context.scene
                camera = scene.camera
                camera_matrix_world = camera.matrix_world

                R = np.array(camera_matrix_world)[:3, :3]  # 3 x 3
                t = np.array(camera_matrix_world)[:3, 3]  # 3 x 1

                camera_save_pth = f"{camera_save_dir}/view:{perturb_idx * view_num + idx:05d}.pickle"
                if not os.path.exists(camera_save_pth) or not skip_done:
                    with open(camera_save_pth, "wb") as handle:
                        pickle.dump(
                            dict(
                                R=R,
                                t=t,
                                scale=scale,
                                resolution=resolution,
                                obj_rotation=np.array(rotation).reshape((3, 3)),  # 3 x 3
                                obj_R=np.array(rotation_matrix).reshape((3, 3)),  # 3 x 3
                                obj_euler=np.array(euler_angle).reshape((3, 1)),  # 3 x 1
                                obj_location=np.array(bpy_object.location).reshape((3, 1)),  # 3 x 1
                                obj_t=np.array(displacements).reshape((3, 1)),  # 3 x 1
                            ),
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                elif verbose:
                    print(f"{camera_save_pth} already processed!!")


def render_3d_future(
    supercategories,
    categories,
    dataset_pth,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):
    ## list that contains all info about supercategories & categories
    supercategories_info_list = __import__(f"{dataset_pth.replace('/', '.')}.categories", fromlist=["_SUPER_CATEGORIES_3D"])._SUPER_CATEGORIES_3D
    categories_info_list = __import__(f"{dataset_pth.replace('/', '.')}.categories", fromlist=["_CATEGORIES_3D"])._CATEGORIES_3D

    ## categories to id
    supercategories2id = {supercategory_info["category"]: supercategory_info["id"] for supercategory_info in supercategories_info_list}
    categories2id = {category_info["category"]: category_info["id"] for category_info in categories_info_list}  # examples: "Lounge Chair / Cafe Chair / Office Chair: 19", "armchair: 27"

    ## asset info
    with open(f"{dataset_pth}/model_info.json", "r") as file:
        assets_info_list = json.load(file)

    ## append necessary metadata
    for idx in range(len(assets_info_list)):
        assets_info_list[idx]["category_id"] = None  # not needed if 3d-future
        assets_info_list[idx]["obj_pth"] = f"{dataset_pth}/{assets_info_list[idx]['model_id']}/raw_model.obj"
        assets_info_list[idx]["dataset_pth"] = dataset_pth

    render_from_asset_info(
        assets_info_list,
        supercategories,
        categories,
        asset_render_dir,
        asset_mask_dir,
        asset_seg_dir,
        camera_dir,
        default_stride_x,
        default_stride_y,
        default_resolution,
        default_elevation,
        default_azimuth,
        default_view_num,
        default_bbox_size,
        default_perturb_sample_num,
        skip_done,
        verbose,
    )


def render_shapenet(
    supercategories,
    categories,
    dataset_pth,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):
    with open(f"{dataset_pth}/taxonomy.json", "r") as file:
        supercategories_info_list = json.load(file)
    with open(f"{dataset_pth}/taxonomy.json", "r") as file:
        categories_info_list = json.load(file)

    supercategories2id = {supercategory_info["name"]: supercategory_info["synsetId"] for supercategory_info in supercategories_info_list}
    categories2id = {category_info["name"]: category_info["synsetId"] for category_info in categories_info_list}  # examples: 'motorcycle,bike': '03790512'

    ## asset info
    assets_info_list = []
    ## append necessary metadata
    for category in categories2id.keys():
        # category id
        category_id = categories2id[category]

        asset_with_texture_dirs = sorted(list(set(list(glob(f"{dataset_pth}/{category_id}/*/*")))))  # only deals with model with textures
        for asset_with_texture_dir in asset_with_texture_dirs:
            # model id
            model_id = asset_with_texture_dir.split("/")[-2]

            assets_info_list.append(
                {
                    "super-category": category,
                    "category": category,
                    "category_id": category_id,
                    "model_id": model_id,
                    "obj_pth": f"{dataset_pth}/{category_id}/{model_id}/models/model_normalized.obj",
                    "dataset_pth": dataset_pth,
                }
            )

    render_from_asset_info(
        assets_info_list,
        supercategories,
        categories,
        asset_render_dir,
        asset_mask_dir,
        asset_seg_dir,
        camera_dir,
        default_stride_x,
        default_stride_y,
        default_resolution,
        default_elevation,
        default_azimuth,
        default_view_num,
        default_bbox_size,
        default_perturb_sample_num,
        skip_done,
        verbose,
    )


def render_sketchfab(
    supercategories,
    categories,
    dataset_pth,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):
    ## list that contains all info about supercategories & categories
    categories_info_list = __import__(f"{dataset_pth.replace('/', '.')}.categories", fromlist=["_CATEGORIES_3D"])._CATEGORIES_3D

    assets_info_list = []
    for category_info in categories_info_list:
        super_category = category_info["super-category"]
        category = category_info["category"]

        model_obj_paths = list(glob(f"{dataset_pth}/{super_category}/*/model.obj"))
        for model_obj_path in model_obj_paths:
            assets_info_list.append(
                {
                    "category_id": None,
                    "model_id": model_obj_path.split("/")[-2],
                    "obj_pth": model_obj_path,
                    "super-category": super_category,
                    "category": category,
                }
            )

    render_from_asset_info(
        assets_info_list,
        supercategories,
        categories,
        asset_render_dir,
        asset_mask_dir,
        asset_seg_dir,
        camera_dir,
        default_stride_x,
        default_stride_y,
        default_resolution,
        default_elevation,
        default_azimuth,
        default_view_num,
        default_bbox_size,
        default_perturb_sample_num,
        skip_done,
        verbose,
    )


def render_3d_behave(
    supercategories,
    categories,
    dataset_pth,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):

    assets_info_list = []
    all_category_dirs = sorted(list(glob(f"{dataset_pth}/objects/*")))

    for category_dir in all_category_dirs:
        category = category_dir.split("/")[-1]
        super_category = "BEHAVE"
        model_obj_path = f"{dataset_pth}/objects/{category}/{category}.obj"
        assets_info_list.append(
            {
                "category_id": None,
                "model_id": "behave_asset",
                "obj_pth": model_obj_path,
                "super-category": super_category,
                "category": category,
                "dataset_pth": dataset_pth,
            }
        )

    render_from_asset_info(
        assets_info_list,
        supercategories,
        categories,
        asset_render_dir,
        asset_mask_dir,
        asset_seg_dir,
        camera_dir,
        default_stride_x,
        default_stride_y,
        default_resolution,
        default_elevation,
        default_azimuth,
        default_view_num,
        default_bbox_size,
        default_perturb_sample_num,
        skip_done,
        verbose,
    )


def render_3d_intercap(
    supercategories,
    categories,
    dataset_pth,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):

    assets_info_list = []
    all_category_dirs = sorted(list(glob(f"{dataset_pth}/objects/*")))

    for category_dir in all_category_dirs:
        category = category_dir.split("/")[-1]
        super_category = "INTERCAP"
        model_obj_path = f"{dataset_pth}/objects/{category}/mesh.obj"
        assets_info_list.append(
            {
                "category_id": None,
                "model_id": "intercap_asset",
                "obj_pth": model_obj_path,
                "super-category": super_category,
                "category": category,
                "dataset_pth": dataset_pth,
            }
        )

    render_from_asset_info(
        assets_info_list,
        supercategories,
        categories,
        asset_render_dir,
        asset_mask_dir,
        asset_seg_dir,
        camera_dir,
        default_stride_x,
        default_stride_y,
        default_resolution,
        default_elevation,
        default_azimuth,
        default_view_num,
        default_bbox_size,
        default_perturb_sample_num,
        skip_done,
        verbose,
    )


def render_sapien(
    supercategories,
    categories,
    dataset_pth,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    camera_dir,
    default_stride_x,
    default_stride_y,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_bbox_size,
    default_perturb_sample_num,
    skip_done,
    verbose,
):
    ## list that contains all info about supercategories & categories
    categories_info_list = __import__(f"{dataset_pth.replace('/', '.')}.categories", fromlist=["_CATEGORIES_3D"])._CATEGORIES_3D

    assets_info_list = []
    for category_info in categories_info_list:
        super_category = category_info["super-category"]
        category = category_info["category"]

        model_obj_paths = list(glob(f"{dataset_pth}/{super_category}/*/model.obj"))
        for model_obj_path in model_obj_paths:
            assets_info_list.append(
                {
                    "category_id": None,
                    "model_id": model_obj_path.split("/")[-2],
                    "obj_pth": model_obj_path,
                    "super-category": super_category,
                    "category": category,
                }
            )

    render_from_asset_info(
        assets_info_list,
        supercategories,
        categories,
        asset_render_dir,
        asset_mask_dir,
        asset_seg_dir,
        camera_dir,
        default_stride_x,
        default_stride_y,
        default_resolution,
        default_elevation,
        default_azimuth,
        default_view_num,
        default_bbox_size,
        default_perturb_sample_num,
        skip_done,
        verbose,
    )


def render_asset(args):
    ## initialize blenderproc scene
    initialize_scene()

    for dataset_type in args.dataset_types:
        if dataset_type == "3D-FUTURE":
            render_3d_future(
                supercategories=args.supercategories,
                categories=args.categories,
                dataset_pth=DATASET_PTHS[dataset_type],
                asset_render_dir=args.asset_render_dir,
                asset_mask_dir=args.asset_mask_dir,
                asset_seg_dir=args.asset_seg_dir,
                camera_dir=args.camera_dir,
                default_stride_x=args.default_stride_x,
                default_stride_y=args.default_stride_y,
                default_resolution=(args.default_resolution_x, args.default_resolution_y),
                default_elevation=args.default_elevation,
                default_azimuth=args.default_azimuth,
                default_view_num=args.default_view_num,
                default_bbox_size=(args.default_bbox_x, args.default_bbox_y, args.default_bbox_z),
                default_perturb_sample_num=args.default_perturb_sample_num,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
        elif dataset_type == "SHAPENET":
            render_shapenet(
                supercategories=args.supercategories,
                categories=args.categories,
                dataset_pth=DATASET_PTHS[dataset_type],
                asset_render_dir=args.asset_render_dir,
                asset_mask_dir=args.asset_mask_dir,
                asset_seg_dir=args.asset_seg_dir,
                camera_dir=args.camera_dir,
                default_stride_x=args.default_stride_x,
                default_stride_y=args.default_stride_y,
                default_resolution=(args.default_resolution_x, args.default_resolution_y),
                default_elevation=args.default_elevation,
                default_azimuth=args.default_azimuth,
                default_view_num=args.default_view_num,
                default_bbox_size=(args.default_bbox_x, args.default_bbox_y, args.default_bbox_z),
                default_perturb_sample_num=args.default_perturb_sample_num,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
        elif dataset_type == "SKETCHFAB":
            render_sketchfab(
                supercategories=args.supercategories,
                categories=args.categories,
                dataset_pth=DATASET_PTHS[dataset_type],
                asset_render_dir=args.asset_render_dir,
                asset_mask_dir=args.asset_mask_dir,
                asset_seg_dir=args.asset_seg_dir,
                camera_dir=args.camera_dir,
                default_stride_x=args.default_stride_x,
                default_stride_y=args.default_stride_y,
                default_resolution=(args.default_resolution_x, args.default_resolution_y),
                default_elevation=args.default_elevation,
                default_azimuth=args.default_azimuth,
                default_view_num=args.default_view_num,
                default_bbox_size=(args.default_bbox_x, args.default_bbox_y, args.default_bbox_z),
                default_perturb_sample_num=args.default_perturb_sample_num,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
        elif dataset_type == "BEHAVE":
            render_3d_behave(
                supercategories=args.supercategories,
                categories=args.categories,
                dataset_pth=DATASET_PTHS[dataset_type],
                asset_render_dir=args.asset_render_dir,
                asset_mask_dir=args.asset_mask_dir,
                asset_seg_dir=args.asset_seg_dir,
                camera_dir=args.camera_dir,
                default_stride_x=args.default_stride_x,
                default_stride_y=args.default_stride_y,
                default_resolution=(args.default_resolution_x, args.default_resolution_y),
                default_elevation=args.default_elevation,
                default_azimuth=args.default_azimuth,
                default_view_num=args.default_view_num,
                default_bbox_size=(args.default_bbox_x, args.default_bbox_y, args.default_bbox_z),
                default_perturb_sample_num=args.default_perturb_sample_num,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
        elif dataset_type == "INTERCAP":
            render_3d_intercap(
                supercategories=args.supercategories,
                categories=args.categories,
                dataset_pth=DATASET_PTHS[dataset_type],
                asset_render_dir=args.asset_render_dir,
                asset_mask_dir=args.asset_mask_dir,
                asset_seg_dir=args.asset_seg_dir,
                camera_dir=args.camera_dir,
                default_stride_x=args.default_stride_x,
                default_stride_y=args.default_stride_y,
                default_resolution=(args.default_resolution_x, args.default_resolution_y),
                default_elevation=args.default_elevation,
                default_azimuth=args.default_azimuth,
                default_view_num=args.default_view_num,
                default_bbox_size=(args.default_bbox_x, args.default_bbox_y, args.default_bbox_z),
                default_perturb_sample_num=args.default_perturb_sample_num,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
        elif dataset_type == "SAPIEN":
            render_sapien(
                supercategories=args.supercategories,
                categories=args.categories,
                dataset_pth=DATASET_PTHS[dataset_type],
                asset_render_dir=args.asset_render_dir,
                asset_mask_dir=args.asset_mask_dir,
                asset_seg_dir=args.asset_seg_dir,
                camera_dir=args.camera_dir,
                default_stride_x=args.default_stride_x,
                default_stride_y=args.default_stride_y,
                default_resolution=(args.default_resolution_x, args.default_resolution_y),
                default_elevation=args.default_elevation,
                default_azimuth=args.default_azimuth,
                default_view_num=args.default_view_num,
                default_bbox_size=(args.default_bbox_x, args.default_bbox_y, args.default_bbox_z),
                default_perturb_sample_num=args.default_perturb_sample_num,
                skip_done=args.skip_done,
                verbose=args.verbose,
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # supercategory / category
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")

    ## save directories
    parser.add_argument("--asset_render_dir", type=str, default="results/generation/asset_renders")
    parser.add_argument("--asset_mask_dir", type=str, default="results/generation/asset_masks")
    parser.add_argument("--asset_seg_dir", type=str, default="results/generation/asset_segs")
    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")

    ## input directories
    parser.add_argument("--dataset_types", type=str, nargs="+", choices=list(DATASET_PTHS.keys()), default=sorted(list(DATASET_PTHS.keys())))

    parser.add_argument("--default_stride_x", type=float, default=0.1)
    parser.add_argument("--default_stride_y", type=float, default=0.1)
    parser.add_argument("--default_resolution_x", type=int, default=512)
    parser.add_argument("--default_resolution_y", type=int, default=512)
    parser.add_argument("--default_elevation", type=float, default=30)
    parser.add_argument("--default_azimuth", type=float, default=45)
    parser.add_argument("--default_view_num", type=int, default=8)

    parser.add_argument("--default_bbox_x", type=float, default=0.5)
    parser.add_argument("--default_bbox_y", type=float, default=0.5)
    parser.add_argument("--default_bbox_z", type=float, default=0.85)

    # pertubation setting
    parser.add_argument("--default_perturb_sample_num", type=int, default=10)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", default=DEFAULT_SEED)

    args = parser.parse_args()

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]

    # seed for reproducible generation
    seed_everything(args.seed)

    ## render the assets in dataset
    render_asset(args)
