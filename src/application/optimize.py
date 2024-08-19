import argparse
from tqdm import tqdm
import pickle
from copy import deepcopy

import trimesh
import open3d as o3d
from pytorch3d.structures.meshes import Meshes

import numpy as np
import torch

from utils.primitives import simplify_mesh_and_get_indices
from utils.reproducibility import seed_everything
from utils.load_3d import load_obj_as_o3d_preserving_face_order
from utils.transformations import normalize_vectors_torch, normalize_vectors_np
from utils.vposer.model_loader import load_vposer
from utils.vposer.prior import create_prior

from smplx.body_models import SMPLX, SMPLH, SMPL
import smplx
from constants.generation.mocap import BODY_MOCAP_PATH
from imports.coap import attach_coap

from constants.affordance_extraction.assets import CATEGORY2CAMERA_CONFIG
from constants.affordance_extraction.visualizers import COMPATIBILITY_MATRIX_OPENGL_TO_BLENDER, COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER


def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device).float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device).float()
    return x


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
    colliding_inds = (model.coap.query(points.reshape((1, -1, 3)), smpl_output) > 0.5).reshape(-1)
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


## function used to transform 'normal a' when 'normal b' canonicalizes to 'normal p'
def canonicalize_a_wrt_b_to_p(a: torch.Tensor, b: torch.Tensor, p: torch.Tensor, sub_p: torch.Tensor, eps: float = 1e-8, normalize_first: bool = True):

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


def compute_vertex_normals(meshes):
    faces_packed = meshes.faces_packed()
    verts_packed = meshes.verts_packed()
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 1],
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 2],
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
            dim=1,
        ),
    )
    verts_normals = verts_normals.index_add(
        0,
        faces_packed[:, 0],
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
            dim=1,
        ),
    )

    return torch.nn.functional.normalize(verts_normals, eps=1e-6, dim=1)


def chamfer_distance(point_cloud_A, point_cloud_B):
    dist_A_to_B = torch.cdist(point_cloud_A, point_cloud_B)
    dist_B_to_A = torch.cdist(point_cloud_B, point_cloud_A)

    min_dist_A_to_B, _ = torch.min(dist_A_to_B, dim=1)
    min_dist_B_to_A, _ = torch.min(dist_B_to_A, dim=1)

    chamfer_dist = torch.mean(min_dist_A_to_B) + torch.mean(min_dist_B_to_A)

    return chamfer_dist


def optimize_smpl(
    supercategory,
    category,
    coma_path,
    asset_downsample_pth,
    eps,
    principle_vec,
    sub_principle_vec,
    reference_object_vertex_index,
    lr,
    body_pose_weight,
    bending_prior_weight,
    pprior_weight,
    orientation_weight,
    contact_weight,
    contact_threshold,
    scale_factor,
    use_collision,
    save_dir="",
):
    with open(coma_path, "rb") as handle:
        affordance_info = pickle.load(handle)

    grid_prob = affordance_info["prob_grid_canon_human_wrt_obj"][:, reference_object_vertex_index, :]  # 10475 x 250
    max_prob_indices = np.argmax(grid_prob, axis=1)  # 10475 x 1
    relative_orientation_GT = np.array([affordance_info["canon_normal_grid"][max_prob_index].reshape((3,)) for max_prob_index in max_prob_indices])  # 10475 x 3

    # Get Contact GT
    selected_human_indices = np.nonzero(np.max(affordance_info["contact_dist_expectation_grid_nom"] / affordance_info["contact_dist_expectation_grid_denom"], axis=1) > contact_threshold)
    corresponding_object_indices = np.argmax(affordance_info["contact_dist_expectation_grid_nom"][selected_human_indices], axis=1)

    ########################## can override scale factor ##########################
    # with open([HUMAN_PRED_PATH], "rb") as handle:
    #     human_pred_info = pickle.load(handle)
    # with open([CAMERA_PATH], "rb") as handle:
    #     cam_data = pickle.load(handle)
    # initial_betas = human_pred_info['smplx_data']['betas']
    # focal_length = human_pred_info['convert_data']['focals'][0]
    # z_mean = human_pred_info['convert_data']['z_mean']
    # cam_resolution = 512
    # cam_scale = cam_data['scale']
    # scale_factor = focal_length * cam_scale / (z_mean * cam_resolution)
    ########################## can override scale factor ##########################

    # Get object information
    with open(asset_downsample_pth, "rb") as handle:
        object_downsample_metadata = deepcopy(pickle.load(handle))

    obj_verts = object_downsample_metadata["downsampled_pcd_points_raw"]  # N_objx3
    obj_normals = obj_vertex_normals = object_downsample_metadata["downsampled_pcd_normal_raw"]  # N_objx3
    obj_vert = obj_verts[reference_object_vertex_index]

    # Change constants device to cuda
    obj_normals = to_tensor(obj_normals, "cuda")
    obj_verts = to_tensor(obj_verts, "cuda")
    obj_vert = to_tensor(obj_vert, "cuda")
    principle_vec = to_tensor(np.array(principle_vec).reshape((3,)), "cuda")
    sub_principle_vec = to_tensor(np.array(sub_principle_vec).reshape((3,)), "cuda")
    grid_prob = to_tensor(grid_prob, "cuda")  # 10475 x 250
    relative_orientation_GT = to_tensor(relative_orientation_GT, "cuda")  # 10475 x 3

    smplxmodel = smplx.create(model_path=BODY_MOCAP_PATH, model_type="smplx", num_pca_comps=45).cuda()
    smplxmodel = attach_coap(smplxmodel, pretrained=True, device="cuda")

    vposer = load_vposer("imports/vposer", vp_model="snapshot")
    vposer = vposer.to(device="cuda")
    vposer.eval()
    angle_prior = create_prior(prior_type="angle").cuda()

    t_pose = np.zeros((1, 63))
    t_pose_torch = torch.from_numpy(t_pose).type(torch.float).to("cuda")
    t_pose_embedding = vposer.encode(t_pose_torch).mean
    pose_embedding = t_pose_embedding.clone().detach().requires_grad_(True)

    betas = to_tensor(np.array([[-0.00982137, 0.03693837, 0.0949352, -0.01299302, 0.00492086, -0.04505398, -0.0008909, -0.00054313, 0.03646483, -0.00803524]]), "cuda")  # default beta
    expression = to_tensor(np.zeros((1, 10)), "cuda")
    leye_pose = to_tensor(np.zeros((1, 3)), "cuda")
    reye_pose = to_tensor(np.zeros((1, 3)), "cuda")
    global_orient = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 0.0]], device="cuda"), requires_grad=True)
    transl = torch.nn.Parameter(torch.tensor([[3.0, 1.0, 0.0]], device="cuda"), requires_grad=True)
    left_hand_pose = torch.nn.Parameter(torch.zeros((1, 45), device="cuda"), requires_grad=True)
    right_hand_pose = torch.nn.Parameter(torch.zeros((1, 45), device="cuda"), requires_grad=True)
    jaw_pose = torch.nn.Parameter(torch.zeros((1, 3), device="cuda"))
    optimizer = torch.optim.Adam([global_orient, transl, left_hand_pose, right_hand_pose, pose_embedding], lr=lr)

    for iter in tqdm(range(2000)):
        print(f"epoch: {iter} | t: {global_orient}")
        optimizer.zero_grad()
        vposer_body_pose = vposer.decode(pose_embedding, output_type="aa").view(1, -1)

        smplxmodel_output = smplxmodel(
            betas=betas,
            global_orient=global_orient,
            body_pose=vposer_body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            transl=transl,
            expression=expression,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            return_verts=True,
            return_full_pose=True,
        )
        vertices = smplxmodel_output.vertices * scale_factor
        faces = to_tensor(smplxmodel.faces.astype(np.int64), "cuda")

        human_mesh = Meshes(verts=vertices, faces=faces.unsqueeze(0))
        human_normals = compute_vertex_normals(human_mesh)

        human_normals = normalize_vectors_torch(
            human_normals,
            eps=eps,
        )

        canon_human_normals_wrt_obj = canonicalize_a_wrt_b_to_p(
            a=human_normals,
            b=obj_normals,
            p=principle_vec,
            sub_p=sub_principle_vec,
            eps=eps,
        )
        relative_normal_for_reference_object_index = canon_human_normals_wrt_obj[:, reference_object_vertex_index, :]
        vertices.reshape(-1, 3) - obj_vert

        pprior_loss = (pose_embedding.pow(2).sum() * body_pose_weight**2) * pprior_weight
        angle_prior_loss = torch.sum(angle_prior(vposer_body_pose)) * bending_prior_weight

        orientation_loss = torch.mean(torch.nan_to_num(1 - (torch.bmm(relative_orientation_GT.view(-1, 1, 3), relative_normal_for_reference_object_index.view(-1, 3, 1)).squeeze() + 1) / 2)) * orientation_weight
        contact_loss = chamfer_distance(vertices.reshape(-1, 3)[selected_human_indices], obj_verts.reshape(-1, 3)[corresponding_object_indices]) * contact_weight

        loss = pprior_loss + angle_prior_loss + contact_loss + orientation_loss

        if use_collision:
            asset_points = sample_scene_points(smplxmodel, smplxmodel_output, obj_verts.reshape(-1, 3))
            collision_loss = 1e9 * smplxmodel.coap.collision_loss(asset_points, smplxmodel_output)[0] if asset_points is not None else 0.0
            loss += collision_loss

        loss.backward()

        optimizer.step()

    save_vertices = vertices.squeeze().detach().cpu().numpy()
    save_faces = smplxmodel.faces.astype(np.int64)

    smplmesh = o3d.geometry.TriangleMesh()
    smplmesh.vertices = o3d.utility.Vector3dVector(save_vertices)
    smplmesh.triangles = o3d.utility.Vector3iVector(save_faces)
    smplmesh.compute_vertex_normals()

    o3d.io.write_triangle_mesh(f"{save_dir}/{supercategory}/{category}/optimized.obj", smplmesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategory", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--coma_path", type=str)
    parser.add_argument("--save_dir", type=str, default="output/")
    parser.add_argument("--asset_downsample_pth", type=str)

    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--body_pose_weight", type=float, default=10000)
    parser.add_argument("--bending_prior_weight", type=float, default=31700)
    parser.add_argument("--pprior_weight", type=float, default=1e-6)
    parser.add_argument("--orientation_weight", type=float, default=1e12)
    parser.add_argument("--contact_weight", type=float, default=2.6e11)

    parser.add_argument("--contact_threshold", type=float, default=0.3)
    parser.add_argument("--scale_factor", type=float, default=0.84)
    parser.add_argument("--use_collision", action="store_true")
    args = parser.parse_args()

    optimize_smpl(
        supercategory=args.supercategory,
        category=args.category,
        coma_path=args.coma_path,
        asset_downsample_pth=args.asset_downsample_pth,
        eps=args.eps,
        principle_vec=[0, 0, 1],
        sub_principle_vec=[0, 1, 0],
        reference_object_vertex_index=0,
        lr=args.lr,
        body_pose_weight=args.body_pose_weight,
        bending_prior_weight=args.bending_prior_weight,
        pprior_weight=args.pprior_weight,
        orientation_weight=args.orientation_weight,
        contact_weight=args.contact_weight,
        contact_threshold=args.contact_threshold,
        scale_factor=args.scale_factor,
        use_collision=args.use_collision,
        save_dir=args.save_dir,
    )
