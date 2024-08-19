from copy import deepcopy

import numpy as np
import torch
from torch.nn.parallel.data_parallel import DataParallel
import torchvision

import sys

sys.path.append("./imports/hand4whole/main")
sys.path.append("./imports/hand4whole/common")
from imports.hand4whole.main.model import get_model
from imports.hand4whole.main.config import cfg
from imports.hand4whole.common.utils_hand4whole.preprocessing import process_bbox, generate_patch_image
from imports.hand4whole.common.utils_hand4whole.human_models import smpl_x
from imports.hand4whole.common.utils_hand4whole.vis import render_mesh


# [hand4whole] mocap for smplx
def prepare_hand4whole_regressor():
    assert torch.cuda.is_available()

    # Set mocap regressor
    class Hand4Whole_Regressor:
        def __init__(
            self,
            model_pth="./imports/hand4whole/snapshot_6.pth.tar",
        ):
            ## model
            self.model = get_model("test")
            self.model = DataParallel(self.model).cuda()
            ckpt = torch.load(model_pth)
            self.model.load_state_dict(ckpt["network"], strict=False)
            self.model.eval()

            ## transform to tensor
            self.transform = torchvision.transforms.ToTensor()

        def regress(self, image_bgr, body_bbox_list, debug=False, visualize=False):
            # make image rgb
            image_rgb = image_bgr[:, :, ::-1].astype(np.float32)
            H, W, C = image_rgb.shape

            # iterate for body_bbox_list: list of 'xywh' bboxes
            mocap_output_list = []
            visualize_image_list = []
            for bbox in body_bbox_list:
                # process bbox
                bbox = process_bbox(bbox, W, H)
                img, img2bb_trans, bb2img_trans = generate_patch_image(image_rgb, bbox, 1.0, 0.0, False, cfg.input_img_shape)
                img = self.transform(img.astype(np.float32)) / 255
                img = img.cuda()[None, :, :, :]

                # regress human mesh
                inputs = {"img": img}
                targets = {}
                meta_info = {}
                with torch.no_grad():
                    """
                    Keys:
                        'img': cropped image of (H: 512 / W: 384)
                        'joint_img':
                    """
                    out = self.model(inputs, targets, meta_info, "test")

                # export smplx data
                smplx_data = dict(
                    betas=out["smplx_shape"].detach().cpu(),
                    global_orient=out["smplx_root_pose"].detach().cpu(),
                    transl=out["cam_trans"].detach().cpu(),
                    left_hand_pose=out["smplx_lhand_pose"].detach().cpu(),
                    right_hand_pose=out["smplx_rhand_pose"].detach().cpu(),
                    jaw_pose=out["smplx_jaw_pose"].detach().cpu(),
                    body_pose=out["smplx_body_pose"].detach().cpu(),
                    expression=out["smplx_expr"].detach().cpu(),
                    leye_pose=torch.zeros((1, 3)).float().repeat(out["img"].shape[0], 1),
                    reye_pose=torch.zeros((1, 3)).float().repeat(out["img"].shape[0], 1),
                )

                mesh_cam = out["smplx_mesh_cam"].detach().cpu().numpy()[0]
                cam = out["cam_trans"].detach().cpu().numpy()[0]
                pelvis_xyz = out["joint_cam_for_rendering"].squeeze()[0].detach().cpu().numpy()
                joints_proj = out["joint_cam_for_rendering"].squeeze(0)[:, :2].detach().cpu().numpy()
                pred_joints_img = out["mocap_output"]["preds_joints_img"].detach().cpu().numpy()

                trans = cam[:2]  # shape: 2,
                scale = cam[2]  # shape: 1,

                # convert mesh from "SMPL 3D" -> "Bbox Coordinates"
                # mesh_bbox = self.convert_smpl_to_bbox(mesh_cam, scale=scale, trans=trans)
                focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
                eps = 1e-2
                assert focal[0] - focal[1] < eps and focal[0] - focal[1] > -eps, "By internal algorithm they should be almost same"
                princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]

                # get mean of z
                z_mean = mesh_cam[:, 2].mean()

                # with camera center (origin) as center of similarity, expand the mesh using 'focal'
                mesh_img = np.copy(mesh_cam)
                mesh_img[:, 0] *= focal[0] / z_mean
                mesh_img[:, 1] *= focal[1] / z_mean
                mesh_img[:, 2] *= ((focal[0] + focal[1]) / 2.0) / z_mean
                pelvis_xyz[0] *= focal[0] / z_mean
                pelvis_xyz[1] *= focal[1] / z_mean
                pelvis_xyz[2] *= ((focal[0] + focal[1]) / 2.0) / z_mean
                joints_proj[:, 0] *= focal[0] / z_mean
                joints_proj[:, 1] *= focal[1] / z_mean
                pred_joints_img[:, 0] *= focal[0] / z_mean
                pred_joints_img[:, 1] *= focal[1] / z_mean

                # move the mesh with 'princpt'
                z_mean_img = mesh_img[:, 2].mean()
                mesh_img[:, 0] += princpt[0]
                mesh_img[:, 1] += princpt[1]
                mesh_img[:, 2] += 500.0 - z_mean_img
                pelvis_xyz[0] += princpt[0]
                pelvis_xyz[1] += princpt[1]
                pelvis_xyz[2] += 500.0 - z_mean_img
                joints_proj[:, 0] += princpt[0]
                joints_proj[:, 1] += princpt[1]
                pred_joints_img[:, 0] += princpt[0]
                pred_joints_img[:, 1] += princpt[1]

                # save as 'mocap_output'
                mocap_output = dict(
                    pred_vertices_img=mesh_img,
                    pelvis_xyz=pelvis_xyz,
                    faces=deepcopy(smpl_x.face).astype(np.int64),
                    smplx_data=smplx_data,
                    joints_proj=joints_proj,
                    convert_data=dict(
                        focals=focal,
                        princpt=princpt,
                        z_mean=z_mean,
                    ),
                    # COMPAT ISSUE
                    pred_joints_img=pred_joints_img,
                    pred_joints_smpl_zerorot_zerobeta=out["mocap_output"]["pred_joints_smpl_zerorot_zerobeta"],
                    pred_vertices_smpl_zerorot_zerobeta=out["mocap_output"]["pred_vertices_smpl_zerorot_zerobeta"],
                    pred_vertices_smpl_zerobeta=out["mocap_output"]["pred_vertices_smpl_zerobeta"],
                    pred_camera=out["mocap_output"]["pred_camera"],
                    pred_body_pose=out["mocap_output"]["pred_body_pose"],
                )
                mocap_output_list.append(mocap_output)

                # visualize
                if visualize:
                    # import matplotlib.pyplot as plt
                    vis_img = image_rgb.copy()
                    rendered_img = render_mesh(vis_img, mesh_cam, smpl_x.face, {"focal": focal, "princpt": princpt})
                    # plt.figure(); plt.imshow(rendered_img); plt.show()
                    visualize_image_list.append(rendered_img)

            return mocap_output_list, visualize_image_list

    return Hand4Whole_Regressor()


def prepare_bodymocap(mode):
    assert mode in ["hand4whole"], f"mode '{mode}' not supported"

    if mode == "hand4whole":
        return prepare_hand4whole_regressor()
    else:
        raise NotImplementedError
