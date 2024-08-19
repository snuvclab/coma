import torch
import numpy as np
import scipy
from config import cfg
from torch.nn import functional as F

import torch
import torch.nn as nn

""" TORCHGEOMETRY """
__all__ = [
    # functional api
    "pi",
    "rad2deg",
    "deg2rad",
    "convert_points_from_homogeneous",
    "convert_points_to_homogeneous",
    "angle_axis_to_rotation_matrix",
    "rotation_matrix_to_angle_axis",
    "rotation_matrix_to_quaternion",
    "quaternion_to_angle_axis",
    "angle_axis_to_quaternion",
    "rtvec_to_pose",
    # layer api
    "RadToDeg",
    "DegToRad",
    "ConvertPointsFromHomogeneous",
    "ConvertPointsToHomogeneous",
]


"""Constant with number pi
"""
pi = torch.Tensor([3.14159265358979323846])


def rad2deg(tensor):
    r"""Function that converts angles from radians to degrees.

    See :class:`~torchgeometry.RadToDeg` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Example:
        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.rad2deg(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return 180.0 * tensor / pi.to(tensor.device).type(tensor.dtype)


def deg2rad(tensor):
    r"""Function that converts angles from degrees to radians.

    See :class:`~torchgeometry.DegToRad` for details.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.deg2rad(input)
    """
    if not torch.is_tensor(tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def convert_points_from_homogeneous(points):
    r"""Function that converts points from homogeneous to Euclidean space.

    See :class:`~torchgeometry.ConvertPointsFromHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_from_homogeneous(input)  # BxNx2
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(points.shape))

    return points[..., :-1] / points[..., -1:]


def convert_points_to_homogeneous(points):
    r"""Function that converts points from Euclidean to homogeneous space.

    See :class:`~torchgeometry.ConvertPointsToHomogeneous` for details.

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> output = tgm.convert_points_to_homogeneous(input)  # BxNx4
    """
    if not torch.is_tensor(points):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(points)))
    if len(points.shape) < 2:
        raise ValueError("Input must be at least a 2D tensor. Got {}".format(points.shape))

    return nn.functional.pad(points, (0, 1), "constant", 1.0)


def angle_axis_to_rotation_matrix(angle_axis):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """

    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat([k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h

    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    return rotation_matrix  # Nx4x4


def rtvec_to_pose(rtvec):
    """
    Convert axis-angle rotation and translation vector to 4x4 pose matrix

    Args:
        rtvec (Tensor): Rodrigues vector transformations

    Returns:
        Tensor: transformation matrices

    Shape:
        - Input: :math:`(N, 6)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(3, 6)  # Nx6
        >>> output = tgm.rtvec_to_pose(input)  # Nx4x4
    """
    assert rtvec.shape[-1] == 6, "rtvec=[rx, ry, rz, tx, ty, tz]"
    pose = angle_axis_to_rotation_matrix(rtvec[..., :3])
    pose[..., :3, 3] = rtvec[..., 3:]
    return pose


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError("Input size must be a three dimensional tensor. Got {}".format(rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError("Input size must be a N x 3 x 4  tensor. Got {}".format(rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1], t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] + rmat_t[:, 1, 0], t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0], rmat_t[:, 2, 0] + rmat_t[:, 0, 2], rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1], rmat_t[:, 2, 0] - rmat_t[:, 0, 2], rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2.float() * mask_d0_d1.float()
    mask_c1 = mask_d2.float() * (1 - mask_d0_d1.float())
    mask_c2 = (1 - mask_d2.float()) * mask_d0_nd1.float()
    mask_c3 = (1 - mask_d2.float()) * (1 - mask_d0_nd1.float())
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 + t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(cos_theta < 0.0, torch.atan2(-sin_theta, -cos_theta), torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


# based on:
# https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py#L138


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert an angle axis to a quaternion.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis (torch.Tensor): tensor with angle axis.

    Return:
        torch.Tensor: tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = torch.rand(2, 4)  # Nx4
        >>> quaternion = tgm.angle_axis_to_quaternion(angle_axis)  # Nx3
    """
    if not torch.is_tensor(angle_axis):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(angle_axis)))

    if not angle_axis.shape[-1] == 3:
        raise ValueError("Input must be a tensor of shape Nx3 or 3. Got {}".format(angle_axis.shape))
    # unpack input and compute conversion
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2

    theta: torch.Tensor = torch.sqrt(theta_squared)
    half_theta: torch.Tensor = theta * 0.5

    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)

    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = torch.sin(half_theta) / theta
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)

    quaternion: torch.Tensor = torch.zeros_like(angle_axis)
    quaternion[..., 0:1] += a0 * k
    quaternion[..., 1:2] += a1 * k
    quaternion[..., 2:3] += a2 * k
    return torch.cat([w, quaternion], dim=-1)


# TODO: add below funtionalities
#  - pose_to_rtvec


# layer api


class RadToDeg(nn.Module):
    r"""Creates an object that converts angles from radians to degrees.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = tgm.pi * torch.rand(1, 3, 3)
        >>> output = tgm.RadToDeg()(input)
    """

    def __init__(self):
        super(RadToDeg, self).__init__()

    def forward(self, input):
        return rad2deg(input)


class DegToRad(nn.Module):
    r"""Function that converts angles from degrees to radians.

    Args:
        tensor (Tensor): Tensor of arbitrary shape.

    Returns:
        Tensor: Tensor with same shape as input.

    Examples::

        >>> input = 360. * torch.rand(1, 3, 3)
        >>> output = tgm.DegToRad()(input)
    """

    def __init__(self):
        super(DegToRad, self).__init__()

    def forward(self, input):
        return deg2rad(input)


class ConvertPointsFromHomogeneous(nn.Module):
    r"""Creates a transformation that converts points from homogeneous to
    Euclidean space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N-1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = tgm.ConvertPointsFromHomogeneous()
        >>> output = transform(input)  # BxNx2
    """

    def __init__(self):
        super(ConvertPointsFromHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_from_homogeneous(input)


class ConvertPointsToHomogeneous(nn.Module):
    r"""Creates a transformation to convert points from Euclidean to
    homogeneous space.

    Args:
        points (Tensor): tensor of N-dimensional points.

    Returns:
        Tensor: tensor of N+1-dimensional points.

    Shape:
        - Input: :math:`(B, D, N)` or :math:`(D, N)`
        - Output: :math:`(B, D, N + 1)` or :math:`(D, N + 1)`

    Examples::

        >>> input = torch.rand(2, 4, 3)  # BxNx3
        >>> transform = tgm.ConvertPointsToHomogeneous()
        >>> output = transform(input)  # BxNx4
    """

    def __init__(self):
        super(ConvertPointsToHomogeneous, self).__init__()

    def forward(self, input):
        return convert_points_to_homogeneous(input)


""" TORCHGEOMETRY """


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    return np.stack((x, y, z), 1)


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)).transpose(1, 0)
    return world_coord


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1 / varP * np.sum(s)

    t = -np.dot(c * R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c * R, np.transpose(A))) + t
    return A2


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


def rot6d_to_axis_angle(x):
    batch_size = x.shape[0]

    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    rot_mat = torch.cat([rot_mat, torch.zeros((batch_size, 3, 1)).cuda().float()], 2)  # 3x4 rotation matrix
    axis_angle = rotation_matrix_to_angle_axis(rot_mat).reshape(-1, 3)  # axis-angle
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle


def sample_joint_features(img_feat, joint_xy):
    height, width = img_feat.shape[2:]
    x = joint_xy[:, :, 0] / (width - 1) * 2 - 1
    y = joint_xy[:, :, 1] / (height - 1) * 2 - 1
    grid = torch.stack((x, y), 2)[:, :, None, :]
    img_feat = F.grid_sample(img_feat, grid, align_corners=True)[:, :, :, 0]  # batch_size, channel_dim, joint_num
    img_feat = img_feat.permute(0, 2, 1).contiguous()  # batch_size, joint_num, channel_dim
    return img_feat


def soft_argmax_2d(heatmap2d):
    batch_size = heatmap2d.shape[0]
    height, width = heatmap2d.shape[2:]
    heatmap2d = heatmap2d.reshape((batch_size, -1, height * width))
    heatmap2d = F.softmax(heatmap2d, 2)
    heatmap2d = heatmap2d.reshape((batch_size, -1, height, width))

    accu_x = heatmap2d.sum(dim=(2))
    accu_y = heatmap2d.sum(dim=(3))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y), dim=2)
    return coord_out


def soft_argmax_3d(heatmap3d):
    batch_size = heatmap3d.shape[0]
    depth, height, width = heatmap3d.shape[2:]
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth * height * width))
    heatmap3d = F.softmax(heatmap3d, 2)
    heatmap3d = heatmap3d.reshape((batch_size, -1, depth, height, width))

    accu_x = heatmap3d.sum(dim=(2, 3))
    accu_y = heatmap3d.sum(dim=(2, 4))
    accu_z = heatmap3d.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(width).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(height).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(depth).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)
    return coord_out


def restore_bbox(bbox_center, bbox_size, aspect_ratio, extension_ratio):
    bbox = bbox_center.view(-1, 1, 2) + torch.cat((-bbox_size.view(-1, 1, 2) / 2.0, bbox_size.view(-1, 1, 2) / 2.0), 1)  # xyxy in (cfg.output_hm_shape[2], cfg.output_hm_shape[1]) space
    bbox[:, :, 0] = bbox[:, :, 0] / cfg.output_hm_shape[2] * cfg.input_body_shape[1]
    bbox[:, :, 1] = bbox[:, :, 1] / cfg.output_hm_shape[1] * cfg.input_body_shape[0]
    bbox = bbox.view(-1, 4)

    # xyxy -> xywh
    bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

    # aspect ratio preserving bbox
    w = bbox[:, 2]
    h = bbox[:, 3]
    c_x = bbox[:, 0] + w / 2.0
    c_y = bbox[:, 1] + h / 2.0

    mask1 = w > (aspect_ratio * h)
    mask2 = w < (aspect_ratio * h)
    h[mask1] = w[mask1] / aspect_ratio
    w[mask2] = h[mask2] * aspect_ratio

    bbox[:, 2] = w * extension_ratio
    bbox[:, 3] = h * extension_ratio
    bbox[:, 0] = c_x - bbox[:, 2] / 2.0
    bbox[:, 1] = c_y - bbox[:, 3] / 2.0

    # xywh -> xyxy
    bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
    bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
    return bbox
