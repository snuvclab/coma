from pytorch3d.io import load_obj

import os
import argparse
from glob import glob
from tqdm import tqdm
import pickle
import json

from trimesh.boolean import intersection
from PIL import Image
import numpy as np
import trimesh
import torch
import smplx
from smplx.utils import SMPLXOutput

from imports.coap import attach_coap
from utils.reproducibility import seed_everything
from utils.prepare_renders import prepare_inpainting_pths
from utils.smpl import smpl_to_openpose

from constants.metadata import DEFAULT_SEED
from constants.generation.assets import DATASET_PTHS, CATEGORY2DATASET_TYPE, CATEGORY2CAMERA_CONFIG, CATEGORY2PERTURB_CONFIG, CATEGORY2ASSET
from constants.generation.visualizers import COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER
from constants.generation.mocap import BODY_MOCAP_PATH


def minimum_distance(vertsA, vertsB, num_vertices=100):
    batch_size = 1 + vertsB.shape[1] // 10000
    vertsB_batches = vertsB.split(batch_size)

    distances_A_to_B = []
    for vertsB_batch in vertsB_batches:
        distances = torch.cdist(vertsA.float(), vertsB_batch.float())
        distances_A_to_B.append(distances)

    distances_A_to_B = torch.cat(distances_A_to_B, dim=1)
    min_distance_A_to_B = torch.min(distances_A_to_B, dim=1).values
    sorted_min_distance_A_to_B, _ = torch.sort(min_distance_A_to_B)
    clip_sorted_min_distance_A_to_B = sorted_min_distance_A_to_B[:num_vertices]
    distance = torch.mean(clip_sorted_min_distance_A_to_B)

    return distance


def compute_metrics(human_verts, human_faces, asset_verts, asset_faces):
    def compute_instersection_ratio(vertsA, facesA, vertsB, facesB):
        # [!important] This might be deprecated when COAP collision loss is implemented on optimization pipeline
        """
        middle priority metric

        vertsA: 1 x N x 3, pytorch tensor
        vertsB: 1 x M x 3, pytorch tensor
        facesA: 1 x P x 3, pytorch tensor
        facesB: 1 x Q x 3, pytorch tensor
        """
        vertsA_numpy = vertsA.squeeze(0).detach().cpu().numpy()
        vertsB_numpy = vertsB.squeeze(0).detach().cpu().numpy()
        facesA_numpy = facesA.squeeze(0).detach().cpu().numpy()
        facesB_numpy = facesB.squeeze(0).detach().cpu().numpy()

        meshA = trimesh.Trimesh(vertices=vertsA_numpy, faces=facesA_numpy)
        meshB = trimesh.Trimesh(vertices=vertsB_numpy, faces=facesB_numpy)

        intersection_volume_ratio = np.abs(intersection([meshA, meshB], engine="blender").volume / meshA.volume)

        return intersection_volume_ratio

    intersection_ratio = compute_instersection_ratio(human_verts, human_faces, asset_verts, asset_faces)

    metrics = dict(
        intersection_ratio=intersection_ratio,
    )

    return metrics


def convert_cam2real(verts, transl, cam_resolution, camera_data, convert_data):
    verts = verts + transl[:, None, :]  # for rendering

    focals = convert_data["focals"]
    princpt = convert_data["princpt"]
    z_mean = convert_data["z_mean"]

    verts[:, :, 0] *= focals[0] / z_mean
    verts[:, :, 1] *= focals[1] / z_mean
    verts[:, :, 2] *= ((focals[0] + focals[1]) / 2.0) / z_mean

    z_mean_img = verts[:, :, 2].mean()
    verts[:, :, 0] += princpt[0]
    verts[:, :, 1] += princpt[1]
    verts[:, :, 2] += 500.0 - z_mean_img

    verts[:, :, 0] = (verts[:, :, 0] - cam_resolution[0] / 2) / max(cam_resolution) * camera_data["scale"]
    verts[:, :, 1] = (verts[:, :, 1] - cam_resolution[1] / 2) / max(cam_resolution) * camera_data["scale"]
    verts[:, :, 2] = (verts[:, :, 2] + 0) / max(cam_resolution) * camera_data["scale"]

    verts = verts @ (torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float() @ torch.transpose(camera_data["R"], 0, 1)) + camera_data["t"]

    return verts


@torch.no_grad()
def sample_scene_points(model, smpl_output, scene_vertices, scene_normals=None, n_upsample=2, max_queries=10000):
    points = scene_vertices.clone()
    # remove points that are well outside the SMPL bounding box
    bb_min = smpl_output.vertices.min(1).values.reshape(1, 3)
    bb_max = smpl_output.vertices.max(1).values.reshape(1, 3)

    inds = (scene_vertices >= bb_min).all(-1) & (scene_vertices <= bb_max).all(-1)
    if not inds.any():
        return None
    points = scene_vertices[inds, :]
    model.coap.eval()
    colliding_inds = (model.coap.query(points.reshape((1, -1, 3)), smpl_output) > 0.01).reshape(-1)
    model.coap.detach_cache()  # detach variables to enable differentiable pass in the opt. loop
    if not colliding_inds.any():
        return None
    points = points[colliding_inds.reshape(-1)]

    if scene_normals is not None and points.size(0) > 0:  # sample extra points if normals are available
        norms = scene_normals[inds][colliding_inds]

        offsets = 0.5 * torch.normal(0.05, 0.05, size=(points.shape[0] * n_upsample, 1), device=norms.device).abs()
        verts, norms = points.repeat(n_upsample, 1), norms.repeat(n_upsample, 1)
        points = torch.cat((points, (verts - offsets * norms).reshape(-1, 3)), dim=0)

    if points.shape[0] > max_queries:
        points = points[torch.randperm(points.size(0), device=points.device)[:max_queries]]

    return points.float().reshape(1, -1, 3)  # add batch dimension


def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device).float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device).float()
    return x


def compute_ransac_inclusives_with_triangulation(
    joints_proj, inpaint_pth, human_preds_dir, camera_dir, maximum_candidates, ransac_threshold=200, triangulation_threshold=10, enable_aggregate_total_prompts=False, allowed_viewpoint_prompts=None
):
    """
    joints: 137 x 3, numpy array
    cam_front_vector: 1 x 3, numpy array
    """
    body_hand_indices = smpl_to_openpose(model_type="smplx", use_hands=True, use_face=False, use_face_contour=False)
    supercategory_str, category_str, asset_id, view_id, _, prompt, __ = inpaint_pth.split("/")[-7:]

    view2camera_config = dict()
    view2projection_matrix = dict()
    view2joints_render = dict()

    def get_camera_config(camera_view_id):
        if view2camera_config.get(camera_view_id, None) is None:
            camera_pth = f"{camera_dir}/{supercategory_str}/{category_str}/{asset_id}/{camera_view_id}.pickle"
            with open(camera_pth, "rb") as handle:
                camera_data = pickle.load(handle)
            view2camera_config[camera_view_id] = dict(R=camera_data["R"], t=camera_data["t"], resolution=camera_data["resolution"], scale=camera_data["scale"])
        return view2camera_config[camera_view_id]

    def get_projection_matrix(camera_view_id):
        if view2projection_matrix.get(camera_view_id, None) is None:
            camera_config = get_camera_config(camera_view_id)
            cam_resolution = camera_config["resolution"]
            cam_scale = camera_config["scale"]
            cam_R = camera_config["R"]
            cam_t = camera_config["t"].reshape((1, 3))

            # rotation = cam_R.T @ COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER # 3 x 3
            # translation = -cam_R.T @ COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER @ cam_t.T # 3 x 1

            rotation = (COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER @ cam_R.T) / cam_scale * max(cam_resolution)  # 3 x 3
            translation = (-COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER @ cam_R.T @ cam_t.T) / cam_scale * max(cam_resolution)  # 3 x 1

            projection_matrix = np.concatenate([rotation, translation], axis=1)  # 3 x 4 (pixel scale, but origin 0, 0)
            view2projection_matrix[camera_view_id] = dict(projection_matrix=projection_matrix, rotation=rotation, translation=translation)  # 3 x 4

        return view2projection_matrix[camera_view_id]  # 3 x 4

    def get_view2joints_render(input_joints, camera_view_id):
        camera_config = get_camera_config(camera_view_id)
        cam_resolution = camera_config["resolution"]
        cam_scale = camera_config["scale"]
        cam_R = camera_config["R"]
        cam_t = camera_config["t"]

        joints_cam = input_joints @ (cam_R @ COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER) - cam_t.reshape((1, 3)) @ (cam_R @ COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER)
        joints_cam[:, 0] = joints_cam[:, 0] / cam_scale * max(cam_resolution) + cam_resolution[0] / 2
        joints_cam[:, 1] = joints_cam[:, 1] / cam_scale * max(cam_resolution) + cam_resolution[1] / 2
        joints_cam[:, 2] = joints_cam[:, 2] / cam_scale * max(cam_resolution)

        joints_proj = joints_cam[:, :2]  # 137 x 2

        return joints_proj

    def solve_DLT(ref_joints_proj, ref_view_id, another_joints_proj, another_view_id):
        ref_camera_config = get_camera_config(ref_view_id)
        another_camera_config = get_camera_config(another_view_id)

        # pixel scale, but origin in (0, 0)
        ref_joints_proj_origin = ref_joints_proj - np.array(ref_camera_config["resolution"]).reshape((1, 2)) / 2
        another_joints_proj_origin = another_joints_proj - np.array(another_camera_config["resolution"]).reshape((1, 2)) / 2

        ref_projection = get_projection_matrix(ref_view_id)
        another_projection = get_projection_matrix(another_view_id)

        ref_rotation = ref_projection["rotation"]
        ref_translation = ref_projection["translation"]
        another_rotation = another_projection["rotation"]
        another_translation = another_projection["translation"]

        full_joints = []
        for joints_ref, joints_another in list(zip(ref_joints_proj_origin, another_joints_proj_origin)):
            joints_ref_x, joints_ref_y = joints_ref
            joints_another_x, joints_another_y = joints_another

            A = np.vstack(
                [
                    ref_rotation[0, :],
                    ref_rotation[1, :],
                    another_rotation[0, :],
                    another_rotation[1, :],
                ]
            )
            b = np.array([[joints_ref_x - ref_translation[0], joints_ref_y - ref_translation[1], joints_another_x - another_translation[0], joints_another_y - another_translation[1]]]).T

            A_pseudo_inverse = np.linalg.pinv(A)
            joints = (A_pseudo_inverse @ b).reshape((3, 1))

            full_joints.append(joints)

        algebraic_triangulation_joints = np.array(full_joints).reshape((-1, 3))  # 25 x 3

        return algebraic_triangulation_joints

    prompt.split(", ")[0]

    """ ADDED """
    prompt_split = prompt.split(",")
    if len(prompt_split) == 1:
        mainprompt = prompt_split[0]
        thisprompts_viewprompt = "original"
    else:
        mainprompt = prompt_split[0]
        thisprompts_viewprompt = prompt_split[-1].strip().lower()
    assert thisprompts_viewprompt in allowed_viewpoint_prompts

    if enable_aggregate_total_prompts:
        human_pred_pths = []
        for allowed_viewprompt in allowed_viewpoint_prompts:
            if allowed_viewprompt == "original":
                human_pred_pths += [pth for pth in list(glob(f"{human_preds_dir}/{supercategory_str}/{category_str}/{asset_id}/*[!{view_id}]*/*/*/*.pickle")) if "," not in pth.split("/")[-2]]
            else:
                human_pred_pths += list(glob(f"{human_preds_dir}/{supercategory_str}/{category_str}/{asset_id}/*[!{view_id}]*/*/*{allowed_viewprompt}*/*.pickle"))

    else:
        human_pred_pths = []
        for allowed_viewprompt in allowed_viewpoint_prompts:
            if allowed_viewprompt == "original":
                human_pred_pths += list(glob(f"{human_preds_dir}/{supercategory_str}/{category_str}/{asset_id}/*[!{view_id}]*/*/{mainprompt}/*.pickle"))
            else:
                human_pred_pths += list(glob(f"{human_preds_dir}/{supercategory_str}/{category_str}/{asset_id}/*[!{view_id}]*/*/*{mainprompt}*{allowed_viewprompt}*/*.pickle"))
        human_pred_pths = list(set(human_pred_pths))

    if CATEGORY2PERTURB_CONFIG[supercategory_str][category_str]["need_perturb"]:
        view_num = CATEGORY2CAMERA_CONFIG[supercategory_str][category_str]["view_num"]
        view_group = int(view_id.split(":")[-1]) // view_num

        human_pred_pths = [human_pred_pth for human_pred_pth in human_pred_pths if int(human_pred_pth.split("/")[-4].split(":")[-1]) // view_num == view_group]

    candidates = dict()
    total_humans = []
    for human_pred_pth in tqdm(human_pred_pths):
        another_view_id = human_pred_pth.split("/")[-4]

        with open(human_pred_pth, "rb") as handle:
            human_pred = pickle.load(handle)
        if type(human_pred) == str:
            continue

        triangulation_joints = solve_DLT(
            ref_joints_proj=joints_proj[body_hand_indices], ref_view_id=view_id, another_joints_proj=human_pred["joints_proj"][body_hand_indices], another_view_id=another_view_id
        )

        ref_joints_MSE = np.mean(np.sum((get_view2joints_render(triangulation_joints, view_id) - joints_proj[body_hand_indices]) ** 2, axis=1))
        another_joints_MSE = np.mean(np.sum((get_view2joints_render(triangulation_joints, another_view_id) - human_pred["joints_proj"][body_hand_indices]) ** 2, axis=1))
        total_joints_MSE = ref_joints_MSE + another_joints_MSE

        if candidates.get(another_view_id, None) is None:
            candidates[another_view_id] = []

        candidates[another_view_id].append(
            dict(
                human_pred_pth=human_pred_pth,
                camera_config=get_camera_config(another_view_id),
                view_id=another_view_id,
                another_joints_proj=human_pred["joints_proj"],
                triangulation_joints=triangulation_joints,
                ref_joints_MSE=ref_joints_MSE,
                another_joints_MSE=another_joints_MSE,
                joints_MSE=total_joints_MSE,
            )
        )

        total_humans.append(
            dict(
                human_pred_pth=human_pred_pth,
                camera_config=get_camera_config(another_view_id),
                view_id=another_view_id,
                another_joints_proj=human_pred["joints_proj"],
                triangulation_joints=triangulation_joints,
                ref_joints_MSE=ref_joints_MSE,
                another_joints_MSE=another_joints_MSE,
                joints_MSE=total_joints_MSE,
            )
        )

    # select k-nearest candidates for RANSAC (to speed up, cannot apply RANSAC for thousands x thousands number computation)
    best_candidates = []
    for view_id, candidate_list in candidates.items():
        best_candidates.extend(candidate_list)

    best_candidates = sorted([candidate for candidate in best_candidates if candidate["ref_joints_MSE"] < triangulation_threshold], key=lambda item: item["joints_MSE"])[:maximum_candidates]

    max_inclusive = 0
    selected_inclusives = []
    for best_candidate in best_candidates:
        triangulation_joints = best_candidate["triangulation_joints"]  # 25 x 3
        ref_joints_MSE = best_candidate["ref_joints_MSE"]  # 25 x 3

        inclusives = []
        view2joints_render = dict()
        # use optimal of "best candidate", reproject it to "another best candidate" to find error
        for another_best_candidate in best_candidates:
            another_view_id = another_best_candidate["view_id"]

            another_view_joints_2d = another_best_candidate["another_joints_proj"]  # 25 x 2
            if view2joints_render.get(another_view_id, None) is None:
                another_mesh_render_joints_2d = get_view2joints_render(triangulation_joints, another_view_id)  # 25 x 2
                view2joints_render[another_view_id] = another_mesh_render_joints_2d  # 25 x 2
            else:
                another_mesh_render_joints_2d = view2joints_render[another_view_id]  # 25 x 2

            joints_MSE = np.mean(np.sum((another_view_joints_2d[body_hand_indices] - another_mesh_render_joints_2d) ** 2, axis=1))

            if joints_MSE < ransac_threshold:
                inclusives.append(
                    dict(
                        human_pred_pth=another_best_candidate["human_pred_pth"],
                        view_id=another_view_id,
                        camera_config={key: to_tensor(val, "cuda") for key, val in get_camera_config(another_view_id).items()},
                        joints_proj=to_tensor(another_view_joints_2d, "cuda").unsqueeze(0),
                        joints_MSE=joints_MSE,
                    )
                )

        if len(inclusives) > max_inclusive:
            selected_inclusives = inclusives
            max_inclusive = len(selected_inclusives)
            best_candidate["human_pred_pth"]

    selected_inclusives = sorted(selected_inclusives, key=lambda item: item["joints_MSE"])

    return selected_inclusives


def multiview_joint_loss(joints, ransac_inclusives):
    """
    joints: 137 x 2, torch tensor
    ransac_inclusives: camera_config, joints_proj, joints_error_2d
    """
    body_hand_indices = smpl_to_openpose(model_type="smplx", use_hands=False, use_face=False, use_face_contour=False)

    loss = 0.0
    for ransac_inclusive in ransac_inclusives:
        camera_config = ransac_inclusive["camera_config"]
        joints_proj = ransac_inclusive["joints_proj"]

        cam_resolution = camera_config["resolution"]
        cam_scale = camera_config["scale"]
        cam_R = camera_config["R"]
        cam_t = camera_config["t"]

        joints_cam = joints @ (cam_R @ torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float()) - cam_t.reshape((1, 3)) @ (
            cam_R @ torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float()
        )
        joints_cam[:, :, 0] = joints_cam[:, :, 0] / cam_scale * max(cam_resolution) + cam_resolution[0] / 2
        joints_cam[:, :, 1] = joints_cam[:, :, 1] / cam_scale * max(cam_resolution) + cam_resolution[1] / 2
        joints_cam[:, :, 2] = joints_cam[:, :, 2] / cam_scale * max(cam_resolution)

        joints_mse_loss = torch.mean(torch.sum((joints_proj[:, body_hand_indices] - joints_cam[:, :, :2][:, body_hand_indices]) ** 2, axis=1))
        loss += joints_mse_loss

    loss /= len(ransac_inclusives)

    return loss


def reference_view_joint_loss(joints, initial_joints, cam_resolution, cam_scale, cam_R, cam_t):
    """
    joints: 137 x 2, torch tensor
    initial_joints: 137 x 2, torch tensor
    """
    body_hand_indices = smpl_to_openpose(model_type="smplx", use_hands=True, use_face=False, use_face_contour=False)

    cam_R = torch.tensor(cam_R, device="cuda").float()
    cam_t = torch.tensor(cam_t, device="cuda").float()

    joints_cam = joints @ (cam_R @ torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float()) - cam_t.reshape((1, 3)) @ (
        cam_R @ torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float()
    )
    joints_cam[:, :, 0] = joints_cam[:, :, 0] / cam_scale * max(cam_resolution) + cam_resolution[0] / 2
    joints_cam[:, :, 1] = joints_cam[:, :, 1] / cam_scale * max(cam_resolution) + cam_resolution[1] / 2
    joints_cam[:, :, 2] = joints_cam[:, :, 2] / cam_scale * max(cam_resolution)

    initial_joints_cam = initial_joints @ (cam_R @ torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float()) - cam_t.reshape((1, 3)) @ (
        cam_R @ torch.tensor(COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, device="cuda").float()
    )
    initial_joints_cam[:, :, 0] = initial_joints_cam[:, :, 0] / cam_scale * max(cam_resolution) + cam_resolution[0] / 2
    initial_joints_cam[:, :, 1] = initial_joints_cam[:, :, 1] / cam_scale * max(cam_resolution) + cam_resolution[1] / 2
    initial_joints_cam[:, :, 2] = initial_joints_cam[:, :, 2] / cam_scale * max(cam_resolution)

    joints_mse_loss = torch.mean(torch.sum((torch.square((joints_cam - initial_joints_cam)[:, :, :2][:, body_hand_indices])), axis=1))

    return joints_mse_loss


def run_depth_optimization(
    supercategories,
    categories,
    prompts,
    inpaint_dir,
    asset_seg_dir,
    human_initial_dir,
    human_preds_dir,
    camera_dir,
    save_dir,
    smplx_path,
    maximum_candidates,
    ransac_threshold,
    triangulation_threshold,
    num_epoch,
    minimum_inliers,
    lr,
    w_collision,
    w_multiview,
    w_refview,
    enable_aggregate_total_prompts,
    allowed_viewpoint_prompts,
    disable_lowres_switch_for_behave,
    skip_done,
    verbose,
    parallel_num,
    parallel_idx,
):
    ## prepare inpainting paths
    inpaint_pths = prepare_inpainting_pths(inpaint_dir, supercategories, categories, prompts)

    ## for all inpainted images, prepare input & metadata for depth optimization (remove bad inputs)
    depthopt_inputs = []
    for inpaint_pth in tqdm(inpaint_pths, desc="Preparing Inputs..."):
        # metadata
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = inpaint_pth.split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        inpaint_id_str, ext = inpaint_id_ext.split(".")
        inpaint_id = int(inpaint_id_str)
        assert ext == "png", "Inpainting must have '.png' extension"

        # load input pths
        asset_seg_pth = f"{asset_seg_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}.pickle"
        human_initial_pth = f"{human_initial_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id:06}.pickle"
        human_preds_pth = f"{human_preds_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id:06}.pickle"
        camera_pth = f"{camera_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}.pickle"
        if not os.path.exists(human_initial_pth):
            continue

        # result-save directory & path
        if enable_aggregate_total_prompts:
            result_save_dir = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/total:{prompt}"
        else:
            result_save_dir = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}"
        result_save_pth = f"{result_save_dir}/{inpaint_id:06}.pickle"
        pbar_desc = f"Running '{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id:06}'"

        if os.path.exists(result_save_pth) and skip_done:
            if verbose:
                print(f"Continueing {result_save_pth} Since Already Done!")
            continue

        with open(human_initial_pth, "rb") as handle:
            human_initial_mesh = pickle.load(handle)

        if human_initial_mesh == "NO HUMANS":
            os.makedirs(result_save_dir, exist_ok=True)
            with open(result_save_pth, "wb") as handle:
                pickle.dump("NO HUMANS", handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print("here1")
            continue
        if human_initial_mesh == "MORE THAN 2 HUMANS":
            os.makedirs(result_save_dir, exist_ok=True)
            with open(result_save_pth, "wb") as handle:
                pickle.dump("MORE THAN 2 HUMANS", handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("here2")
            continue
        if human_initial_mesh == "LARGELY PENETRATED HUMAN":
            os.makedirs(result_save_dir, exist_ok=True)
            with open(result_save_pth, "wb") as handle:
                pickle.dump("LARGELY PENETRATED HUMAN", handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("here3")
            continue
        if human_initial_mesh == "ERRONEOUS SAMPLE DUE TO TOO SMALL HUMAN":
            os.makedirs(result_save_dir, exist_ok=True)
            with open(result_save_pth, "wb") as handle:
                pickle.dump("ERRONEOUS SAMPLE DUE TO TOO SMALL HUMAN", handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("here4")
            continue

        prompt_split = prompt.split(",")
        if len(prompt_split) == 1:
            viewprompt = "original"
        else:
            viewprompt = prompt_split[-1]

        if viewprompt.strip().lower() in allowed_viewpoint_prompts:
            pass
        else:
            os.makedirs(result_save_dir, exist_ok=True)
            with open(result_save_pth, "wb") as handle:
                pickle.dump("NOT ALLOWED VIEWPOINT PROMPTS", handle, protocol=pickle.HIGHEST_PROTOCOL)
            continue

        # input hyperparams
        num_epoch = num_epoch

        # path to asset mesh (in dataset)
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

        ## add to depth-opt inputs
        depthopt_inputs.append(
            {
                "inpaint_pth": inpaint_pth,
                "human_initial_pth": human_initial_pth,
                "human_preds_pth": human_preds_pth,
                "camera_pth": camera_pth,
                "result_save_dir": result_save_dir,
                "result_save_pth": result_save_pth,
                "pbar_desc": pbar_desc,
                "supercategory": supercategory,
                "category": category,
                "asset_id": asset_id,
                "asset_seg_pth": asset_seg_pth,
                "inpaint_id": inpaint_id,
                "num_epoch": num_epoch,
                "dataset_type": dataset_type,
                "dataset_pth": dataset_pth,
                "obj_pth": obj_pth,  # asset obj file
            }
        )

    # parallel execution setting
    sub_length = (len(depthopt_inputs) // parallel_num) + 1
    start_idx = (parallel_idx) * sub_length
    end_idx = (parallel_idx + 1) * sub_length

    ## iterate for all inpaintings
    depthopt_inputs = sorted(depthopt_inputs, key=lambda x: x["result_save_pth"])
    pbar = tqdm(depthopt_inputs[start_idx:end_idx])
    for depthopt_input in pbar:
        # set pbar description
        pbar.set_description(desc=depthopt_input["pbar_desc"])

        # retrieve paths
        human_initial_pth = depthopt_input["human_initial_pth"]
        human_preds_pth = depthopt_input["human_preds_pth"]
        camera_pth = depthopt_input["camera_pth"]

        # retrieve save paths
        result_save_dir = depthopt_input["result_save_dir"]

        result_save_pth = depthopt_input["result_save_pth"]
        if os.path.exists(result_save_pth) and skip_done:
            if verbose:
                print(f"Continueing {result_save_pth} Since Already Done!")
            continue

        # input human pred (initial human pred)
        os.makedirs(result_save_dir, exist_ok=True)
        with open(human_initial_pth, "rb") as handle:
            human_initial_mesh = pickle.load(handle)

        with open(human_preds_pth, "rb") as handle:
            human_preds_data = pickle.load(handle)

        smplx_data = human_preds_data["smplx_data"]
        joints_proj = human_preds_data["joints_proj"]

        # input camera
        with open(camera_pth, "rb") as handle:
            camera_data = pickle.load(handle)
        cam_R = camera_data["R"]
        cam_front_vector = cam_R[:, 2]
        cam_t = camera_data["t"]
        cam_resolution = camera_data.get("resolution", Image.open(inpaint_pth).size)
        cam_scale = camera_data["scale"]

        # load asset obj file
        obj_pth = depthopt_input["obj_pth"]
        dataset_type = depthopt_input["dataset_type"]
        asset_verts, asset_faces, aux = load_obj(obj_pth)
        asset_verts = asset_verts @ torch.tensor(COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER).float()  # compatibility

        x_min = asset_verts.numpy()[:, 0].min()
        x_max = asset_verts.numpy()[:, 0].max()
        y_min = asset_verts.numpy()[:, 1].min()
        y_max = asset_verts.numpy()[:, 1].max()
        z_min = asset_verts.numpy()[:, 2].min()
        z_max = asset_verts.numpy()[:, 2].max()
        x_max - x_min
        y_max - y_min
        z_max - z_min

        ## asset transformation (if perturbed)
        asset_obj_R = torch.tensor(camera_data["obj_R"]).float()  # 3x3
        asset_obj_t = torch.tensor(camera_data["obj_t"].reshape((1, 3))).float()  # 1 x 3

        asset_verts = asset_verts @ torch.transpose(asset_obj_R, 0, 1) + asset_obj_t

        if dataset_type in ["SHAPENET", "SKETCHFAB", "INTERCAP", "BEHAVE"]:
            asset_verts -= torch.tensor([0.0, 0.0, z_min])

        asset_verts = asset_verts.to("cuda").unsqueeze(0)
        asset_faces = asset_faces.verts_idx.to("cuda").unsqueeze(0)

        ## load human mesh
        human_faces = human_initial_mesh["faces"]
        human_faces = torch.tensor(human_faces, device="cuda").unsqueeze(0).float()
        cam_front_vector = torch.tensor(cam_front_vector, device="cuda").float()

        ## create smpl
        human = smplx.create(model_path=smplx_path, model_type="smplx", num_pca_comps=45)
        human = attach_coap(human, pretrained=True, device="cuda")

        smplx_data_torch = {key: to_tensor(val, "cuda") for key, val in smplx_data.items() if key not in ["transl", "body_pose", "betas", "global_orient"]}
        transl_torch = to_tensor(smplx_data["transl"], "cuda")
        displacement_torch = to_tensor(human_initial_mesh.get("displacement", np.array([[0.0, 0.0, 0.0]])), "cuda")
        camera_data_torch = {key: to_tensor(val, "cuda") for key, val in camera_data.items() if key in ["R", "t", "scale"]}

        if displacement_torch is None:
            displacement_torch = to_tensor(np.array([[0.0, 0.0, 0.0]]), "cuda")

        initial_body_pose = to_tensor(smplx_data["body_pose"], "cuda")
        initial_betas = to_tensor(smplx_data["betas"], "cuda")
        initial_global_orient = to_tensor(smplx_data["global_orient"], "cuda")
        initial_human_joints = human(**smplx_data_torch, betas=initial_betas, body_pose=initial_body_pose, global_orient=initial_global_orient, return_verts=True, return_full_pose=True).joints
        initial_human_joints_real = (
            convert_cam2real(initial_human_joints, transl_torch, cam_resolution, camera_data_torch, human_preds_data["convert_data"]) + displacement_torch.unsqueeze(0)
        ).detach()

        ## set up parameter and optimizer
        displacement = torch.nn.Parameter(torch.tensor([0.0], device="cuda"), requires_grad=True)
        pose_residual = torch.nn.Parameter(torch.zeros((1, 63), device="cuda"), requires_grad=True)
        betas_residual = torch.nn.Parameter(torch.zeros((1, 10), device="cuda"), requires_grad=True)
        global_orient_residual = torch.nn.Parameter(torch.zeros((1, 3), device="cuda"), requires_grad=True)

        optimizer = torch.optim.Adam([displacement], lr=lr)

        ransac_inclusives = compute_ransac_inclusives_with_triangulation(
            joints_proj,
            depthopt_input["inpaint_pth"],
            human_preds_dir,
            camera_dir,
            maximum_candidates,
            ransac_threshold,
            triangulation_threshold,
            enable_aggregate_total_prompts,
            allowed_viewpoint_prompts,
        )
        print(f"inliers number: {len(ransac_inclusives)}")

        if len(ransac_inclusives) < minimum_inliers:
            with open(result_save_pth, "wb") as handle:
                pickle.dump("TOO LITTLE INLIERS", handle, protocol=pickle.HIGHEST_PROTOCOL)
            continue

        for iter in tqdm(range(depthopt_input["num_epoch"])):
            print(f"epoch: {iter} | t: {displacement}")
            optimizer.zero_grad()

            smplx_cam_output = human(
                **smplx_data_torch,
                body_pose=initial_body_pose + pose_residual,
                betas=initial_betas + betas_residual,
                global_orient=initial_global_orient + global_orient_residual,
                return_verts=True,
                return_full_pose=True,
            )
            smplx_real_vertices = (
                convert_cam2real(smplx_cam_output.vertices, transl_torch, cam_resolution, camera_data_torch, human_preds_data["convert_data"])
                + displacement_torch.unsqueeze(0)
                + displacement * cam_front_vector
            )
            smplx_real_joints = (
                convert_cam2real(smplx_cam_output.joints, transl_torch, cam_resolution, camera_data_torch, human_preds_data["convert_data"])
                + displacement_torch.unsqueeze(0)
                + displacement * cam_front_vector
            )

            smplx_real_output = SMPLXOutput(
                vertices=smplx_real_vertices,
                joints=smplx_real_joints,
                betas=smplx_cam_output.betas.clone().detach(),
                expression=smplx_cam_output.expression.clone().detach(),
                global_orient=smplx_cam_output.global_orient.clone().detach(),
                body_pose=smplx_cam_output.body_pose.clone().detach(),
                left_hand_pose=smplx_cam_output.left_hand_pose.clone().detach(),
                right_hand_pose=smplx_cam_output.right_hand_pose.clone().detach(),
                jaw_pose=smplx_cam_output.jaw_pose.clone().detach(),
                v_shaped=smplx_cam_output.v_shaped.clone().detach(),
                full_pose=smplx_cam_output.full_pose.clone().detach(),
            )

            asset_points = sample_scene_points(human, smplx_real_output, asset_verts)
            collision_loss = w_collision * (human.coap.collision_loss(asset_points, smplx_real_output)[0] if asset_points is not None else 0.0)
            multiview_loss = w_multiview * multiview_joint_loss(smplx_real_output.joints, ransac_inclusives)
            reference_view_loss = w_refview * reference_view_joint_loss(smplx_real_output.joints, initial_human_joints_real, cam_resolution, cam_scale, cam_R, cam_t)

            loss = multiview_loss + collision_loss
            print(f"collision_loss: {collision_loss} | multiview_loss: {multiview_loss} | reference_view_loss: {reference_view_loss}")

            loss.backward(retain_graph=False)

            optimizer.step()

        smplx_output = human(
            **smplx_data_torch,
            body_pose=initial_body_pose + pose_residual,
            betas=initial_betas + betas_residual,
            global_orient=initial_global_orient + global_orient_residual,
            return_verts=True,
            return_full_pose=True,
        )
        default_vertices = convert_cam2real(smplx_output.vertices, transl_torch, cam_resolution, camera_data_torch, human_preds_data["convert_data"]) + displacement_torch.unsqueeze(0)
        optimized_human_verts = default_vertices + displacement * cam_front_vector  # 1 x N x 3

        optimized_results = dict(verts=optimized_human_verts.squeeze(0).detach().cpu().numpy(), faces=human_faces.squeeze(0).detach().cpu().numpy().astype(np.uint32))
        optimized_results.update(dict(num_inliers=len(ransac_inclusives)))

        # save result
        with open(depthopt_input["result_save_pth"], "wb") as handle:
            pickle.dump(optimized_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--inpaint_dir", type=str, default="results/generation/inpaintings")
    parser.add_argument("--asset_seg_dir", type=str, default="results/generation/asset_segs")
    parser.add_argument("--human_initial_dir", type=str, default="results/generation/human_before_opt")
    parser.add_argument("--human_preds_dir", type=str, default="results/generation/human_preds")
    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--save_dir", type=str, default="results/generation/human_after_opt")

    parser.add_argument("--smplx_path", type=str, default=BODY_MOCAP_PATH)
    parser.add_argument("--maximum_candidates", type=int, default=400)
    parser.add_argument("--ransac_threshold", type=int, default=200)
    parser.add_argument("--triangulation_threshold", type=int, default=100)
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--minimum_inliers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--w_collision", type=float, default=0.4)
    parser.add_argument("--w_multiview", type=float, default=1e-3)
    parser.add_argument("--w_refview", type=float, default=0.0)

    parser.add_argument("--disable_lowres_switch_for_behave", action="store_true")
    parser.add_argument("--enable_aggregate_total_prompts", action="store_true")
    parser.add_argument("--allowed_viewpoint_prompts", nargs="+", default=["original", "full body"])

    parser.add_argument("--no_initialize", action="store_true")
    parser.add_argument("--no_collision", action="store_true")

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

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
    if args.allowed_viewpoint_prompts is not None:
        args.allowed_viewpoint_prompts = [viewprompt.lower() for viewprompt in args.allowed_viewpoint_prompts]
    if args.no_initialize:
        args.human_initial_dir = f"{args.human_initial_dir}_no_initialize"
        args.save_dir = f"{args.save_dir}_no_initialize"
        assert args.human_initial_dir != "results/generation/human_before_opt"
        assert args.save_dir != "results/generation/human_after_opt"
    if args.no_collision:
        args.save_dir = f"{args.save_dir}_no_collision"
        args.w_collision = 0.0
        assert args.save_dir != "results/generation/human_after_opt"

    # seed for reproducible generation
    seed_everything(args.seed)

    ## run depth optimization
    run_depth_optimization(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        inpaint_dir=args.inpaint_dir,
        asset_seg_dir=args.asset_seg_dir,
        human_initial_dir=args.human_initial_dir,
        human_preds_dir=args.human_preds_dir,
        camera_dir=args.camera_dir,
        save_dir=args.save_dir,
        smplx_path=args.smplx_path,
        maximum_candidates=args.maximum_candidates,
        ransac_threshold=args.ransac_threshold,
        triangulation_threshold=args.triangulation_threshold,
        num_epoch=args.num_epoch,
        minimum_inliers=args.minimum_inliers,
        lr=args.lr,
        w_collision=args.w_collision,
        w_multiview=args.w_multiview,
        w_refview=args.w_refview,
        enable_aggregate_total_prompts=args.enable_aggregate_total_prompts,
        allowed_viewpoint_prompts=args.allowed_viewpoint_prompts,
        disable_lowres_switch_for_behave=args.disable_lowres_switch_for_behave,
        skip_done=args.skip_done,
        verbose=args.verbose,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
    )
