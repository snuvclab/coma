import argparse
import pickle

import open3d as o3d

import numpy as np
import torch
import os

from utils.coma import simplify_mesh_and_get_indices
from utils.reproducibility import seed_everything
from constants.generation.mocap import SMPLX_MOCAP_PATH

from smplx.body_models import SMPLX


def downsample_smplx(args):
    smplxmodel = SMPLX(model_path=SMPLX_MOCAP_PATH)

    NUM_BODY_JOINTS = smplxmodel.NUM_BODY_JOINTS
    body_pose = torch.zeros(NUM_BODY_JOINTS * 3)
    body_pose[2] = +torch.pi / 6
    body_pose[5] = -torch.pi / 6
    body_pose = body_pose.unsqueeze(0)

    vertices = smplxmodel(body_pose=body_pose).vertices.squeeze().detach().cpu().numpy()
    max_num_verts = len(vertices)
    faces = smplxmodel.faces.astype(np.int64)

    smplxmesh = o3d.geometry.TriangleMesh()
    smplxmesh.vertices = o3d.utility.Vector3dVector(vertices)
    smplxmesh.triangles = o3d.utility.Vector3iVector(faces)
    smplxmesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh("./constants/mesh/smplx_star.obj", smplxmesh)
    with open("./constants/mesh/smplx_star.pickle", "wb") as handle:
        pickle.dump(
            {"vertices": vertices, "faces": faces},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    if args.num_human_downsample_points < max_num_verts:
        downsample_indices, downsampled_pcd = simplify_mesh_and_get_indices(
            smplxmesh,
            number_of_points=args.num_human_downsample_points,
            simplify_method=args.simplify_method,
            debug=args.debug,
        )

    else:
        downsampled_pcd = o3d.geometry.PointCloud()
        downsampled_pcd.points = o3d.utility.Vector3dVector(vertices)
        downsampled_pcd.normals = o3d.utility.Vector3dVector(np.asarray(smplxmesh.vertex_normals))
        downsample_indices = sorted(list(range(max_num_verts)))

    # finally, check for erroneous indices with no vertex normals, and skip if so
    original_vertex_normals = np.asarray(smplxmesh.vertex_normals)
    final_downsample_indices = []
    for d_index in downsample_indices:
        if original_vertex_normals[d_index].sum() == 0:
            continue
        else:
            final_downsample_indices.append(d_index)
    downsample_indices = final_downsample_indices

    to_save = {
        "vertices": vertices,
        "faces": faces,
        "V": vertices.shape[0],
        "F": faces.shape[0],
        "N": len(downsample_indices),
        "N_raw": len(np.asarray(downsampled_pcd.points)),
        "downsample_indices": downsample_indices,
        "downsampled_pcd_points_raw": np.asarray(downsampled_pcd.points),
        "downsampled_pcd_normal_raw": np.asarray(downsampled_pcd.normals),
    }

    if args.num_human_downsample_points < max_num_verts:
        save_pth = f"./constants/mesh/smplx_star_downsampled_{args.num_human_downsample_points}.pickle"
    else:
        save_pth = f"./constants/mesh/smplx_star_downsampled_FULL.pickle"

    if not args.skip_done or not os.path.exists(save_pth):
        with open(save_pth, "wb") as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplify_method", choices=["poisson_disk", "uniform"], default="poisson_disk")

    parser.add_argument(
        "--num_human_downsample_points_list",
        type=int,
        nargs="+",
        default=[1000, 1500, 2000, 2048, 20000],
    )
    parser.add_argument("--use_watertight", action="store_true")  # keep as False

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    for num_human_downsample_points in args.num_human_downsample_points_list:
        args.num_human_downsample_points = num_human_downsample_points
        args.dont_do_object = num_human_downsample_points <= 10475
        downsample_smplx(args)
