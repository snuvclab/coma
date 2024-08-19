"""
Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
from argparse import ArgumentParser
import sys, os

sys.path.append(os.getcwd())
from os.path import join, isfile, basename, dirname, isdir
import pickle
from psbody.mesh import Mesh
import json
import cv2
import numpy as np
import pickle as pkl
from PIL import Image
from sklearn.neighbors import KDTree
from scipy import interpolate
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
import open3d as o3d
from glob import glob

from utils.transformations import normalize_vectors_np, batch_rodrigues
from utils.load_3d import load_obj_as_o3d_preserving_face_order

SMPL_NAME_BEHAVE = "fit02"
OBJ_NAME_BEHAVE = "fit01"


class KinectCalib:
    def __init__(self, calibration, pc_table):
        """
        the localize file is the transformation matrix from this kinect RGB to kinect zero RGB
        """
        self.pc_table_ext = np.dstack([pc_table, np.ones(pc_table.shape[:2] + (1,), dtype=pc_table.dtype)])
        color2depth_R = np.array(calibration["color_to_depth"]["rotation"]).reshape(3, 3)
        color2depth_t = np.array(calibration["color_to_depth"]["translation"])
        depth2color_R = np.array(calibration["depth_to_color"]["rotation"]).reshape(3, 3)
        depth2color_t = np.array(calibration["depth_to_color"]["translation"])

        self.color2depth_R = color2depth_R
        self.color2depth_t = color2depth_t
        self.depth2color_R = depth2color_R
        self.depth2color_t = depth2color_t

        color_calib = calibration["color"]
        self.image_size = (color_calib["width"], color_calib["height"])
        self.focal_dist = (color_calib["fx"], color_calib["fy"])
        self.center = (color_calib["cx"], color_calib["cy"])
        self.calibration_matrix = np.eye(3)
        self.calibration_matrix[0, 0], self.calibration_matrix[1, 1] = self.focal_dist
        self.calibration_matrix[:2, 2] = self.center
        self.dist_coeffs = np.array(color_calib["opencv"][4:])

        # depth intrinsic
        depth_calib = calibration["depth"]
        self.depth_size = (depth_calib["width"], depth_calib["height"])
        self.depth_center = (depth_calib["cx"], depth_calib["cy"])
        self.depth_focal = (depth_calib["fx"], depth_calib["fy"])
        self.depth_matrix = np.eye(3)
        self.depth_matrix[0, 0], self.depth_matrix[1, 1] = self.depth_focal
        self.depth_matrix[:2, 2] = self.depth_center
        self.depth_distcoeffs = np.array(depth_calib["opencv"][4:])

        # additional parameters for distortion
        if "codx" in color_calib and "codx" in depth_calib:
            self.depth_codx = depth_calib["codx"]
            self.depth_cody = depth_calib["cody"]
            self.depth_metric_radius = depth_calib["metric_radius"]
            self.color_codx = color_calib["codx"]
            self.color_cody = color_calib["cody"]
            self.color_metric_radius = color_calib["metric_radius"]
        else:
            # for backward compatibility
            self.depth_codx = 0
            self.depth_cody = 0
            self.depth_metric_radius = np.nan
            self.color_codx = 0
            self.color_cody = 0
            self.color_metric_radius = np.nan

    def undistort(self, img):
        return cv2.undistort(img, self.calibration_matrix, self.dist_coeffs)

    def project_points(self, points):
        """
        given points in the color camera coordinate, project it into color image
        return: (N, 2)
        """
        return cv2.projectPoints(points[..., np.newaxis], np.zeros(3), np.zeros(3), self.calibration_matrix, self.dist_coeffs)[0].reshape(-1, 2)

    def dmap2pc(self, depth, return_mask=False):
        """
        use precomputed table to convert depth map to point cloud
        """
        nanmask = depth == 0
        d = depth.copy().astype(np.float) / 1000.0
        d[nanmask] = np.nan
        pc = self.pc_table_ext * d[..., np.newaxis]
        validmask = np.isfinite(pc[:, :, 0])
        pc = pc[validmask]
        if return_mask:
            return pc, validmask
        return pc

    def interpolate_depth(self, depth_im):
        "borrowed from PROX"
        # fill depth holes to avoid black spots in aligned rgb image
        zero_mask = np.array(depth_im == 0.0).ravel()
        depth_im_flat = depth_im.ravel()
        depth_im_flat[zero_mask] = np.interp(np.flatnonzero(zero_mask), np.flatnonzero(~zero_mask), depth_im_flat[~zero_mask])
        depth_im = depth_im_flat.reshape(depth_im.shape)
        return depth_im

    def pc2color(self, pointcloud):
        """
        given point cloud, return its pixel coordinate in RGB image
        """
        # project the point cloud in depth camera to RGB camera
        pointcloud_color = np.matmul(pointcloud, self.depth2color_R.T) + self.depth2color_t
        projected_color_pc = self.project_points(pointcloud_color)
        return projected_color_pc

    def pc2color_valid(self, pointcloud):
        """
        given point cloud in depth camera,
        return its pixel coordinate in RGB image, invalid pixels (out of range are removed)
        """
        # project the point cloud in depth camera to RGB camera
        pointcloud_color = np.matmul(pointcloud, self.depth2color_R.T) + self.depth2color_t
        projected_color_pc = self.project_points(pointcloud_color)
        mask = self.valid_pixmask(projected_color_pc)
        return projected_color_pc[mask, :], pointcloud[mask, :]

    def valid_pixmask(self, color_pixels):
        w, h = self.image_size
        valid_x = np.logical_and(color_pixels[:, 0] < w, color_pixels[:, 0] >= 0)
        valid_y = np.logical_and(color_pixels[:, 1] < h, color_pixels[:, 1] >= 0)
        valid = np.logical_and(valid_y, valid_x)
        return valid

    def color_to_pc(self, colorpts, pc_depth, projected_color_pc=None, k=4, std=1.0):
        """
        project point clouds to RGB image, use KDTree to query each color point's closest point cloud
        """
        def weight_func(x, std=1.0):
            return np.exp(-x / (2 * std**2))

        if projected_color_pc is None:
            projected_color_pc = self.pc2color(pc_depth)
        tree = KDTree(projected_color_pc)
        dists, inds = tree.query(colorpts, k=k)  # return the closest distance of each colorpts to the tree
        weights = weight_func(dists, std=std)
        weights_sum = weights.sum(axis=1)
        w = weights / weights_sum[:, np.newaxis]
        pts_world = (pc_depth[inds.flatten(), :].reshape(-1, k, 3) * w[:, :, np.newaxis]).sum(axis=1)
        return pts_world

    def get_pc_colors(self, pointcloud, color_frame, projected_color_pc=None):
        """
        given point cloud and color frame, return the colors for the point cloud
        """
        if projected_color_pc is None:
            projected_color_pc = self.pc2color(pointcloud)
        pc_colors = np.ones_like(pointcloud)
        for i in range(3):
            # the project pixel coordinate in color frame is non-integer, interpolate the color mesh to get best result
            spline = RectBivariateSpline(np.arange(color_frame.shape[0]), np.arange(color_frame.shape[1]), color_frame[:, :, i])

            pc_colors[:, i] = spline(projected_color_pc[:, 1], projected_color_pc[:, 0], grid=False)
        pc_colors /= 255.0
        pc_colors = np.clip(pc_colors, 0, 1)
        return pc_colors

    def pc2dmap(self, points):
        """
        reproject pc to image plane, find closes grid for each one
        """
        p2d = self.project_points(points)
        cw, ch = self.image_size
        px, py = np.meshgrid(np.linspace(0, cw - 1, cw), np.linspace(0, ch - 1, ch))
        depth = interpolate.griddata(p2d, points[:, 2], (px, py), method="nearest")
        dmap = np.zeros((ch, cw))
        dmap[py.astype(int), px.astype(int)] = depth
        return dmap

    def dmap2colorpc(self, color, depth):
        "convert depth in color camera to pc"
        pc, mask = self.dmap2pc(depth, return_mask=True)
        pc_colors = color[mask]
        return pc, pc_colors


def rotate_yaxis(R, t):
    "rotate the transformation matrix around z-axis by 180 degree ==>> let y-axis point up"
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    global_trans = np.eye(4)
    global_trans[0, 0] = global_trans[1, 1] = -1  # rotate around z-axis by 180
    rotated = np.matmul(global_trans, transform)
    return rotated[:3, :3], rotated[:3, 3]


def load_intrinsics(intrinsic_folder, kids):
    """
    kids: list of kinect id that should be loaded
    """
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]

    return kinects


def load_kinect_poses(config_folder, kids):
    pose_calibs = [json.load(open(join(config_folder, f"{x}/config.json"))) for x in kids]
    rotations = [np.array(pose_calibs[x]["rotation"]).reshape((3, 3)) for x in kids]
    translations = [np.array(pose_calibs[x]["translation"]) for x in kids]
    return rotations, translations


def load_kinects(intrinsic_folder, config_folder, kids):
    intrinsic_calibs = [json.load(open(join(intrinsic_folder, f"{x}/calibration.json"))) for x in kids]
    pc_tables = [np.load(join(intrinsic_folder, f"{x}/pointcloud_table.npy")) for x in kids]
    [join(config_folder, f"{x}/config.json") for x in kids]
    kinects = [KinectCalib(cal, pc) for cal, pc in zip(intrinsic_calibs, pc_tables)]
    return kinects


def load_kinect_poses_back(config_folder, kids, rotate=False):
    """
    backward transform
    rotate: kinect y-axis pointing down, if rotate, then return a transform that make y-axis pointing up
    """
    rotations, translations = load_kinect_poses(config_folder, kids)
    rotations_back = []
    translations_back = []
    for r, t in zip(rotations, translations):
        trans = np.eye(4)
        trans[:3, :3] = r
        trans[:3, 3] = t

        trans_back = np.linalg.inv(trans)  # now the y-axis point down

        r_back = trans_back[:3, :3]
        t_back = trans_back[:3, 3]
        if rotate:
            r_back, t_back = rotate_yaxis(r_back, t_back)

        rotations_back.append(r_back)
        translations_back.append(t_back)
    return rotations_back, translations_back


def availabe_kindata(input_video, kinect_count=3):
    # all available kinect videos in this folder, return the list of kinect id, and str representation
    fname_split = os.path.basename(input_video).split(".")
    idx = int(fname_split[1])
    kids = []
    comb = ""
    for k in range(kinect_count):
        file = input_video.replace(f".{idx}.", f".{k}.")
        if os.path.exists(file):
            kids.append(k)
            comb = comb + str(k)
        else:
            print("Warning: {} does not exist in this folder!".format(file))
    return kids, comb


def save_color_depth(out_dir, color, depth, kid, color_only=False, ext="jpg"):
    color_file = join(out_dir, f"k{kid}.color.{ext}")
    # cv2.imwrite(color_file, color[:, :, ::-1])
    Image.fromarray(color).save(color_file)
    if not color_only:
        depth_file = join(out_dir, f"k{kid}.depth.png")
        cv2.imwrite(depth_file, depth)


# path to the simplified mesh used for registration
_mesh_template = {
    "backpack": "backpack/backpack_f1000.ply",
    "basketball": "basketball/basketball_f1000.ply",
    "boxlarge": "boxlarge/boxlarge_f1000.ply",
    "boxtiny": "boxtiny/boxtiny_f1000.ply",
    "boxlong": "boxlong/boxlong_f1000.ply",
    "boxsmall": "boxsmall/boxsmall_f1000.ply",
    "boxmedium": "boxmedium/boxmedium_f1000.ply",
    "chairblack": "chairblack/chairblack_f2500.ply",
    "chairwood": "chairwood/chairwood_f2500.ply",
    "monitor": "monitor/monitor_closed_f1000.ply",
    "keyboard": "keyboard/keyboard_f1000.ply",
    "plasticcontainer": "plasticcontainer/plasticcontainer_f1000.ply",
    "stool": "stool/stool_f1000.ply",
    "tablesquare": "tablesquare/tablesquare_f2000.ply",
    "toolbox": "toolbox/toolbox_f1000.ply",
    "suitcase": "suitcase/suitcase_f1000.ply",
    "tablesmall": "tablesmall/tablesmall_f1000.ply",
    "yogamat": "yogamat/yogamat_f1000.ply",
    "yogaball": "yogaball/yogaball_f1000.ply",
    "trashbin": "trashbin/trashbin_f1000.ply",
}


def get_template_path(behave_path, obj_name):
    path = join(behave_path, "objects", _mesh_template[obj_name])
    if not isfile(path):
        print(path, "does not exist, please check input parameters!")
        raise ValueError()
    return path


def load_scan_centered(scan_path, cent=True):
    """load a scan and centered it around origin"""
    scan = Mesh()
    # print(scan_path)
    scan.load_from_file(scan_path)
    if cent:
        center = np.mean(scan.v, axis=0)
        verts_centerd = scan.v - center
        scan.v = verts_centerd

    return scan


def load_template(obj_name, cent=True, dataset_path=None):
    assert dataset_path is not None, "please specify BEHAVE dataset path!"
    temp_path = get_template_path(dataset_path, obj_name)
    return load_scan_centered(temp_path, cent)


def write_pointcloud(filename, xyz_points, rgb_points=None):
    """
    updated on March 22, use trimesh for writing
    """
    import trimesh

    assert xyz_points.shape[1] == 3, "Input XYZ points should be Nx3 float array"
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8) * 255
    assert xyz_points.shape == rgb_points.shape, "Input RGB colors should be Nx3 float array and have same size as input XYZ points"
    outfolder = dirname(filename)
    os.makedirs(outfolder, exist_ok=True)
    pc = trimesh.points.PointCloud(xyz_points, rgb_points)
    pc.export(filename)


class KinectTransform:
    "transform between different kinect cameras, sequence specific"

    def __init__(self, seq, kinect_count=4):
        self.seq_info = SeqInfo(seq)
        self.kids = [x for x in range(self.seq_info.kinect_count())]
        self.intrinsics = load_intrinsics(self.seq_info.get_intrinsic(), self.kids)
        rot, trans = load_kinect_poses(self.seq_info.get_config(), self.kids)
        self.local2world_R, self.local2world_t = rot, trans
        rot, trans = load_kinect_poses_back(self.seq_info.get_config(), self.kids)
        self.world2local_R, self.world2local_t = rot, trans

    def world2color_mesh(self, mesh: Mesh, kid):
        "world coordinate to local color coordinate, assume mesh world coordinate is in k1 color camera coordinate"
        m = self.copy_mesh(mesh)
        m.v = np.matmul(mesh.v, self.world2local_R[kid].T) + self.world2local_t[kid]
        return m

    def flip_mesh(self, mesh: Mesh):
        "flip the mesh along x axis"
        m = self.copy_mesh(mesh)
        m.v[:, 0] = -m.v[:, 0]
        return m

    def copy_mesh(self, mesh: Mesh):
        m = Mesh(v=mesh.v)
        if hasattr(mesh, "f"):
            m.f = mesh.f.copy()
        if hasattr(mesh, "vc"):
            m.vc = np.array(mesh.vc)
        return m

    def world2local_meshes(self, meshes, kid):
        transformed = []
        for m in meshes:
            transformed.append(self.world2color_mesh(m, kid))
        return transformed

    def local2world_mesh(self, mesh, kid):
        m = self.copy_mesh(mesh)
        m.v = self.local2world(m.v, kid)
        return m

    def world2local(self, points, kid):
        return np.matmul(points, self.world2local_R[kid].T) + self.world2local_t[kid]

    def project2color(self, p3d, kid):
        "project 3d points to local color image plane"
        p2d = self.intrinsics[kid].project_points(self.world2local(p3d, kid))
        return p2d

    def kpts2center(self, kpts, depth: np.ndarray, kid):
        "kpts: (N, 2), x y format"
        kinect = self.intrinsics[kid]
        pc = kinect.pc_table_ext * depth[..., np.newaxis]
        kpts_3d = pc[kpts[:, 1], kpts[:, 0]]
        return kpts_3d

    def local2world(self, points, kid):
        R, t = self.local2world_R[kid], self.local2world_t[kid]
        return np.matmul(points, R.T) + t

    def dmap2pc(self, depth, kid):
        kinect = self.intrinsics[kid]
        pc = kinect.dmap2pc(depth)
        return pc


class SeqInfo:
    "a simple class to handle information of a sequence"

    def __init__(self, seq_path):
        self.info = self.get_seq_info_data(seq_path)

    def get_obj_name(self, convert=False):
        if convert:  # for using detectron
            if "chair" in self.info["cat"]:
                return "chair"
            if "ball" in self.info["cat"]:
                return "sports ball"
        return self.info["cat"]

    def get_gender(self):
        return self.info["gender"]

    def get_config(self):
        return self.info["config"]

    def get_intrinsic(self):
        return self.info["intrinsic"]

    def get_empty_dir(self):
        return self.info["empty"]

    def beta_init(self):
        return self.info["beta"]

    def kinect_count(self):
        if "kinects" in self.info:
            return len(self.info["kinects"])
        else:
            return 3

    @property
    def kids(self):
        count = self.kinect_count()
        return [i for i in range(count)]

    def get_seq_info_data(self, seq):
        info_file = join(seq, "info.json")
        data = json.load(open(info_file))
        # all paths are relative to the sequence path
        path_names = ["config", "empty", "intrinsic"]
        for name in path_names:
            if data[name] is not None:
                data[name] = join(seq, data[name])
        return data


def save_seq_info(seq_folder, config, intrinsic, cat, gender, empty, beta, kids=[0, 1, 2, 3]):
    # from data.utils import load_kinect_poses
    outfile = join(seq_folder, "info.json")
    info = {"config": config, "intrinsic": intrinsic, "cat": cat, "gender": gender, "empty": empty, "kinects": kids, "beta": beta}
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("{} saved.".format(outfile))
    print("{}: {}, {}, {}, {}, {}".format(seq_folder, config, intrinsic, cat, beta, gender))


class KinectFrameReader:
    def __init__(self, seq, empty=None, kinect_count=4, ext="jpg", check_image=True):
        # prepare depth and color paths
        if seq.endswith("/"):
            seq = seq[:-1]
        self.seq_path = seq
        self.ext = ext  # file extension for color image
        self.kinect_count = kinect_count
        self.frames = self.prepare_paths(check_image=check_image)
        self.empty = empty
        self.bkgs = self.prepare_bkgs()
        self.seq_name = basename(seq)
        self.kids = [i for i in range(kinect_count)]

    def prepare_paths(self, check_image=True):
        "find out which frames contain complete color and depth images"
        frames = sorted(os.listdir(self.seq_path))
        valid_frames = []
        for frame in frames:
            frame_folder = join(self.seq_path, frame)
            if check_image:
                if self.check_frames(frame_folder):
                    valid_frames.append(frame)
                # else:
                #     print("frame {} not complete".format(frame))
            else:
                if isdir(frame_folder):
                    valid_frames.append(frame)

        """ ADDED """
        valid_frames = sorted(valid_frames, key=lambda x: (int(x.split(".")[0].replace("t", "")), int(x.split(".")[1])))
        """ ADDED """

        return valid_frames

    def check_frames(self, frame_folder):
        """DEPRECATED"""
        # print(self.kinect_count)
        # for k in range(self.kinect_count):
        # color_file = join(frame_folder, 'k{}.color.{}'.format(k, self.ext))
        # depth_file = join(frame_folder, 'k{}.depth.png'.format(k))
        # if not isfile(color_file) or not isfile(depth_file):
        #     return False
        """ DEPRECATED """

        if os.path.isdir(frame_folder):
            return True
        else:
            return False

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx):
        "return the i-th frame of color and depth images"
        frame_folder = join(self.seq_path, self.frames[idx])
        color_files = [join(frame_folder, f"k{k}.color.{self.ext}") for k in range(self.kinect_count)]
        depth_files = [join(frame_folder, f"k{k}.depth.png") for k in range(self.kinect_count)]

        colors = [Image.open(c).convert("RGB") for c in color_files]
        colors = [np.array(c) for c in colors]
        depths = [cv2.imread(d, cv2.IMREAD_ANYDEPTH) for d in depth_files]

        if self.bkgs is not None:
            depths_filtered = []
            for d, bkg in zip(depths, self.bkgs):
                df = remove_background(d, bkg, tol=30)
                depths_filtered.append(df)
            depths = depths_filtered

        return colors, depths

    def get_color_images(self, idx, kids, bgr=False):
        color_files = self.get_color_files(idx, kids)
        if bgr:
            colors = [cv2.imread(x) for x in color_files]
        else:
            colors = [Image.open(c).convert("RGB") for c in color_files]
            colors = [np.array(c) for c in colors]
        return colors

    def get_color_files(self, idx, kids):
        # frame_folder = join(self.seq_path, self.frames[idx])
        frame_folder = self.get_frame_folder(idx)
        color_files = [join(frame_folder, f"k{k}.color.{self.ext}") for k in kids]
        return color_files

    def get_depth_images(self, idx, kids):
        frame_folder = join(self.seq_path, self.frames[idx])
        depth_files = [join(frame_folder, f"k{k}.depth.png") for k in kids]
        depths = [cv2.imread(d, cv2.IMREAD_ANYDEPTH) for d in depth_files]
        return depths

    def get_frame_folder(self, idx):
        if isinstance(idx, int):
            assert idx < len(self)
            return join(self.seq_path, self.frames[idx])
        elif isinstance(idx, str):
            return join(self.seq_path, idx)
        else:
            raise NotImplemented

    def prepare_bkgs(self):
        if self.empty is None:
            return None
        else:
            bkgs = [get_seq_bkg(self.empty, x) for x in range(self.kinect_count)]
            return bkgs

    def remove_background(self, depth, bkg, tol=100):
        diff = np.abs(depth - bkg)
        mask = ~(diff >= tol)
        depth[mask] = 0
        return depth

    def frame_time(self, idx):
        return self.frames[idx]

    def get_timestamps(self):
        "timestamp list for all frames"
        times = [float(x[1:]) for x in self.frames]
        return times

    def get_frame_idx(self, timestr):
        try:
            idx = self.frames.index(timestr)
            return idx
        except ValueError:
            return -1


def get_seq_bkg(seq, kid, start=0):
    "get the average depth for this sequence of one kinect"
    frames = sorted(os.listdir(seq))
    depths = []
    # d_size = (576, 640)
    for frame in frames[start:]:
        depth_file = join(seq, frame, f"k{kid}.depth.png")
        depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            depths.append(depth)
    avg = np.stack(depths, axis=-1).mean(axis=-1)
    return avg


def remove_background(depth, bkg, tol=100):
    diff = np.abs(depth - bkg)
    mask = ~(diff >= tol)
    depth[mask] = 0
    return depth


class FrameDataReader(KinectFrameReader):
    "read more data: pc, mocap, fitted smpl, obj etc"

    def __init__(self, seq, empty=None, ext="jpg", check_image=True):
        seq_info = SeqInfo(seq)
        super(FrameDataReader, self).__init__(seq, empty, seq_info.kinect_count(), ext, check_image=check_image)
        self.seq_info = seq_info
        self.kids = self.seq_info.kids

    def get_pc(self, idx, cat="person", convert=False):
        pcfile = self.get_pcpath(idx, cat, convert)
        if not isfile(pcfile):
            pcfile = self.get_pcpath(idx, cat, not convert)
        return self.load_mesh(pcfile)

    def get_pcpath(self, idx, cat, convert=False):
        if cat == "person":
            name = "person"
        else:
            name = self.seq_info.get_obj_name(convert)
        frame_folder = self.get_frame_folder(idx)
        pcfile = join(frame_folder, f"{name}/{name}.ply")
        return pcfile

    def load_mesh(self, pcfile):
        if not isfile(pcfile):
            return None
        m = Mesh()
        m.load_from_file(pcfile)
        return m

    def get_J3d(self, idx):
        frame_folder = self.get_frame_folder(idx)
        pcfile = join(frame_folder, "person/person_J.ply")
        return self.load_mesh(pcfile)

    def get_mocap_mesh(self, idx, kid=1):
        mocap_file = self.get_mocap_meshfile(idx, kid)
        return self.load_mesh(mocap_file)

    def get_mocap_meshfile(self, idx, kid=1):
        frame_folder = self.get_frame_folder(idx)
        mocap_file = join(frame_folder, f"k{kid}.mocap.ply")
        return mocap_file

    def get_mocap_pose(self, idx, kid=1):
        jsonfile = join(self.get_frame_folder(idx), "k{}.mocap.json".format(kid))
        if not isfile(jsonfile):
            return None
        params = json.load(open(jsonfile))
        return np.array(params["pose"])

    def get_mocap_beta(self, idx, kid=1):
        jsonfile = join(self.get_frame_folder(idx), "k{}.mocap.json".format(kid))
        if not isfile(jsonfile):
            return None
        params = json.load(open(jsonfile))
        return np.array(params["betas"])

    def get_smplfit(self, idx, save_name, ext="ply"):
        if save_name is None:
            return None
        mesh_file = self.smplfit_meshfile(idx, save_name, ext)
        return self.load_mesh(mesh_file)

    def get_smplfit_as_smplx(self, idx, save_name, ext="obj"):
        if save_name is None:
            return None
        mesh_file = self.smplfit_meshfile_as_smplx(idx, save_name, ext)
        return self.load_mesh(mesh_file)

    def smplfit_meshfile(self, idx, save_name, ext="obj"):
        mesh_file = join(self.get_frame_folder(idx), "person", save_name, f"person_fit.{ext}")
        return mesh_file

    def smplfit_meshfile_as_smplx(self, idx, save_name, ext="ply"):
        mesh_file = join(self.get_frame_folder(idx), "person", save_name, f"person_fit_smplx.{ext}")
        return mesh_file

    def objfit_meshfile(self, idx, save_name, ext="ply", convert=True):
        name = self.seq_info.get_obj_name(convert=convert)
        mesh_file = join(self.get_frame_folder(idx), name, save_name, f"{name}_fit.{ext}")
        if not isfile(mesh_file):
            name = self.seq_info.get_obj_name()
            mesh_file = join(self.get_frame_folder(idx), name, save_name, f"{name}_fit.{ext}")
        return mesh_file

    def get_objfit(self, idx, save_name, ext="ply"):
        if save_name is None:
            return None
        mesh_file = self.objfit_meshfile(idx, save_name, ext)
        return self.load_mesh(mesh_file)

    def objfit_param_file(self, idx, save_name):
        name = self.seq_info.get_obj_name(convert=True)
        pkl_file = join(self.get_frame_folder(idx), name, save_name, f"{name}_fit.pkl")
        return pkl_file

    def category(self, idx, save_name):
        category = self.seq_info.get_obj_name(convert=True)
        return category

    def get_objfit_params(self, idx, save_name):
        "return angle and translation"
        if save_name is None:
            return None, None
        pkl_file = self.objfit_param_file(idx, save_name)
        if not isfile(pkl_file):
            return None, None
        fit = pkl.load(open(pkl_file, "rb"))
        return fit["angle"], fit["trans"]

    def get_smplfit_params(self, idx, save_name):
        "return pose, beta, translation"
        if save_name is None:
            return None, None, None
        pkl_file = self.smplfit_param_file(idx, save_name)
        if not isfile(pkl_file):
            return None, None, None
        fit = pkl.load(open(pkl_file, "rb"))
        return fit["pose"], fit["betas"], fit["trans"]

    def smplfit_param_file(self, idx, save_name):
        return join(self.get_frame_folder(idx), "person", save_name, "person_fit.pkl")

    def times2indices(self, frame_times):
        "convert frame time str to indices"
        indices = [self.get_frame_idx(f) for f in frame_times]
        return indices

    def get_body_j3d(self, idx):
        file = self.body_j3d_file(idx)
        if not isfile(file):
            return None
        data = json.load(open(file))
        J3d = np.array(data["body_joints3d"]).reshape((-1, 4))  # the forth column is the score
        return J3d

    def body_j3d_file(self, idx):
        pcfile = self.get_pcpath(idx, "person")
        return pcfile.replace(".ply", "_J3d.json")

    def get_body_kpts(self, idx, kid, tol=0.5):
        J2d_file = join(self.get_frame_folder(idx), f"k{kid}.color.json")
        if not isfile(J2d_file):
            return None
        data = json.load(open(J2d_file))
        J2d = np.array(data["body_joints"]).reshape((-1, 3))
        J2d[:, 2][J2d[:, 2] < tol] = 0
        return J2d

    def get_mask(self, idx, kid, cat="person", ret_bool=True):
        file = self.get_mask_file(idx, kid, cat)
        if not isfile(file):
            return None
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"mask file: {file} invalid! removing it...")
            os.system(f"rm {file}")
            raise ValueError()
        if ret_bool:
            mask = mask > 127
        return mask

    def get_mask_file(self, idx, kid, cat):
        if cat == "person":
            file = join(self.get_frame_folder(idx), f"k{kid}.person_mask.png")
            if not isfile(file):
                file = join(self.get_frame_folder(idx), f"k{kid}.person_mask.jpg")
            # file = join(self.get_frame_folder(idx), f'k{kid}.person_mask.{self.ext}')
            # print(file)
        elif cat == "obj":
            exts = ["png", "jpg"]
            for ext in exts:
                file = join(self.get_frame_folder(idx), f"k{kid}.obj_rend_mask.{ext}")
                if not isfile(file):
                    file = join(self.get_frame_folder(idx), f"k{kid}.obj_mask.{ext}")
                if isfile(file):
                    break
        else:
            raise NotImplemented
        return file

    def get_person_mask(self, idx, kids, ret_bool=True):
        frame_folder = join(self.seq_path, self.frames[idx])
        mask_files = [join(frame_folder, f"k{k}.person_mask.{self.ext}") for k in kids]
        # print(mask_files)
        masks = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in mask_files]
        if ret_bool:
            masks = [x > 127 if x is not None else None for x in masks]
        return masks

    def get_pcfiles(self, frames, cat, convert=False):
        # if cat !='person':
        #     name = self.seq_info.get_obj_name(convert)
        # else:
        #     name = 'person'
        pcfiles = [self.get_pcpath(x, cat, convert) for x in frames]
        return pcfiles

    def pc_exists(self, idx, cat, convert=False):
        pcfile = self.get_pcpath(idx, cat, convert)
        return isfile(pcfile)

    def cvt_end(self, end):
        batch_end = len(self) if end is None else end
        if batch_end > len(self):
            batch_end = len(self)
        return batch_end


"""
a simple wrapper for pytorch3d rendering
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import numpy as np
import torch
import pytorch3d
from copy import deepcopy

# Data structures and functions for rendering
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
import trimesh

from psbody.mesh import Mesh
from psbody.mesh.sphere import Sphere


SMPL_OBJ_COLOR_LIST = [
    [0.65098039, 0.74117647, 0.85882353],  # SMPL
    [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
]

part_colors = (
    np.array(
        [
            44,
            160,
            44,
            31,
            119,
            180,
            255,
            127,
            14,
            214,
            39,
            40,
            148,
            103,
            189,
            140,
            86,
            75,
            227,
            119,
            194,
            127,
            127,
            127,
            189,
            189,
            34,
            255,
            152,
            150,
            23,
            190,
            207,
            174,
            199,
            232,
            255,
            187,
            120,
            152,
            223,
            138,
        ]
    ).reshape((-1, 3))
    / 255.0
)
color_reorder = [1, 1, 3, 3, 4, 5, 6, 10, 10, 10, 7, 11, 12, 13, 14]


class ContactVisualizer:
    def __init__(self, thres=0.04, radius=0.06, color=(0.12156863, 0.46666667, 0.70588235)):
        self.part_labels = self.load_part_labels()
        self.part_colors = self.load_part_colors()
        self.thres = thres
        self.radius = radius
        self.color = color  # sphere color

    def load_part_labels(self):
        part_labels = pkl.load(open("./imports/behave-dataset/data/smpl_parts_dense.pkl", "rb"))
        labels = np.zeros((6890,), dtype="int32")
        for n, k in enumerate(part_labels):
            labels[part_labels[k]] = n  # in range [0, 13]
        return labels

    def load_part_colors(self):
        colors = np.zeros((14, 3))
        for i in range(len(colors)):
            colors[i] = part_colors[color_reorder[i]]
        return colors

    def get_contact_spheres(self, smpl: Mesh, obj: Mesh):
        kdtree = KDTree(smpl.v)
        obj_tri = trimesh.Trimesh(obj.v, obj.f, process=False)
        points = obj_tri.sample(10000)
        dist, idx = kdtree.query(points)  # query each object vertex's nearest neighbour
        contact_mask = dist < self.thres
        if np.sum(contact_mask) == 0:
            return {}
        contact_labels = self.part_labels[idx][contact_mask]
        contact_verts = points[contact_mask[:, 0]]

        contact_regions = {}
        for i in range(14):
            parts_i = contact_labels == i
            if np.sum(parts_i) > 0:
                color = self.part_colors[i]
                contact_i = contact_verts[parts_i]
                center_i = np.mean(contact_i, 0)
                contact_sphere = Sphere(center_i, self.radius).to_mesh()
                contact_regions[i] = (color, contact_sphere)

        return contact_regions


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"

    def __init__(self, image_size=1200, faces_per_pixel=1, device="cuda:0", blur_radius=0, lights=None, materials=None, max_faces_per_bin=50000):
        self.image_size = image_size
        self.faces_per_pixel = faces_per_pixel
        self.max_faces_per_bin = max_faces_per_bin  # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights = lights if lights is not None else PointLights(((0.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),), ((0.05, 0.05, 0.05),), ((0, -2, 0),), device)
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin,
        )
        shader = SoftPhongShader(device=self.device, lights=self.lights, materials=self.materials)
        renderer = MeshRenderer(rasterizer=MeshRasterizer(raster_settings=raster_settings), shader=shader)
        return renderer

    def render(self, meshes, cameras, ret_mask=False):
        images = self.renderer(meshes, cameras=cameras)
        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapper:
    def __init__(self, image_size, device="cuda:0", colors=SMPL_OBJ_COLOR_LIST):
        self.renderer = MeshRendererWrapper(image_size, device=device)
        self.front_camera = self.get_kinect_camera(device)
        self.colors = deepcopy(colors)
        self.device = device
        self.contact_vizer = ContactVisualizer()

    @staticmethod
    def get_kinect_camera(device="cuda:0"):
        R, T = torch.eye(3), torch.zeros(3)
        R[0, 0] = R[1, 1] = -1  # pytorch3d y-axis up, need to rotate to kinect coordinate
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)
        fx, fy = 979.7844, 979.840  # focal length
        cx, cy = 1018.952, 779.486  # camera centers
        color_w, color_h = 2048, 1536  # kinect color image size
        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)

        pyt3d_version = pytorch3d.__version__
        if pyt3d_version >= "0.6.0":
            cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center, image_size=((color_w, color_h),), device=device, R=R, T=T, in_ndc=False)
        else:
            cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center, image_size=((color_w, color_h),), device=device, R=R, T=T)
        return cam

    def render_meshes(self, meshes, viz_contact=False):
        """
        render a list of meshes
        :param meshes: a list of psbody meshes
        :return: rendered image
        """
        colors = deepcopy(self.colors)
        if viz_contact:
            contact_regions = self.contact_vizer.get_contact_spheres(meshes[0], meshes[1])
            for k, v in contact_regions.items():
                color, sphere = v
                meshes.append(sphere)
                colors.append(color)
        pyt3d_mesh = self.prepare_render(meshes, colors)
        rend = self.renderer.render(pyt3d_mesh, self.front_camera)
        return rend

    def prepare_render(self, meshes, colors):
        py3d_meshes = []
        for mesh, color in zip(meshes, colors):
            vc = np.zeros_like(mesh.v)
            vc[:, :] = color
            text = TexturesVertex([torch.from_numpy(vc).float().to(self.device)])
            py3d_mesh = Meshes([torch.from_numpy(mesh.v).float().to(self.device)], [torch.from_numpy(mesh.f.astype(int)).long().to(self.device)], text)
            py3d_meshes.append(py3d_mesh)
        joined = join_meshes_as_scene(py3d_meshes)
        return joined
    

def prepare_affordance_extraction_inputs_for_behave(
    human_mesh_pth,
    human_mesh_pth_type,
    human_downsample_metadata,
    object_downsample_metadata,
    human_use_downsample_pcd_raw: bool,
    object_use_downsample_pcd_raw: bool,
    lowres_center_pth,
    raw2normal_pth,
    obj_param_pth,
    eps,
    object_mesh_for_check_pth=None,
    interactive=False,
):
    ### Load Human Mesh and Object Mesh ###
    # load original human vertices/faces/vertex-normals
    if human_mesh_pth_type == "obj":
        human_mesh = load_obj_as_o3d_preserving_face_order(human_mesh_pth)
        human_mesh.compute_vertex_normals()
        human_verts_orig = np.asarray(human_mesh.vertices)
        human_faces_orig = np.asarray(human_mesh.triangles)

    if human_mesh_pth_type == "pickle":
        with open(human_mesh_pth, "rb") as handle:
            human_data = pickle.load(handle)
        human_verts_orig = human_data["verts"]
        human_faces_orig = human_data["faces"]
        human_mesh = o3d.geometry.TriangleMesh()
        human_mesh.vertices = o3d.utility.Vector3dVector(human_verts_orig)
        human_mesh.triangles = o3d.utility.Vector3iVector(human_faces_orig)
        human_mesh.compute_vertex_normals()

    ## get vertex normals
    human_vertex_normals_orig = normalize_vectors_np(
        np.asarray(human_mesh.vertex_normals),
        eps=eps,
    )

    # load original object vertices/faces/vertex-normals
    obj_verts_orig = object_downsample_metadata["obj_vertices_original"]
    obj_faces_orig = object_downsample_metadata["obj_faces_original"]
    obj_vertex_normals_orig = normalize_vectors_np(object_downsample_metadata["obj_vertex_normals_original"])

    # double-check
    if object_mesh_for_check_pth is not None:
        obj_mesh_for_check = load_obj_as_o3d_preserving_face_order(object_mesh_for_check_pth)
        obj_verts_orig_for_check = np.asarray(obj_mesh_for_check.vertices)
        obj_faces_orig_for_check = np.asarray(obj_mesh_for_check.vertices)
        assert np.allclose(obj_verts_orig, obj_verts_orig_for_check)
        assert np.allclose(obj_faces_orig, obj_faces_orig_for_check)

    ### Downsample Human Mesh and Object Mesh ###
    human_downsample_indices = human_downsample_metadata["downsample_indices"]  # N_human
    object_downsample_indices = object_downsample_metadata["downsample_indices"]  # N_obj

    # downsample vertices
    if human_use_downsample_pcd_raw:
        assert False, "Human must use 'mesh' for Representation. You'll know why"
        human_verts = human_downsample_metadata["downsampled_pcd_points_raw"]
        human_vertex_normals = human_downsample_metadata["downsampled_pcd_normal_raw"]
        assert len(human_verts) == human_downsample_metadata["N_raw"]
    else:
        human_verts = human_verts_orig.copy()[human_downsample_indices]  # N_humanx3
        human_vertex_normals = human_vertex_normals_orig.copy()[human_downsample_indices]  # N_humanx3
        assert len(human_verts) == human_downsample_metadata["N"]

    if object_use_downsample_pcd_raw:
        obj_verts = object_downsample_metadata["downsampled_pcd_points_raw"]  # N_objx3
        obj_vertex_normals = object_downsample_metadata["downsampled_pcd_normal_raw"]  # N_objx3
        assert len(obj_verts) == object_downsample_metadata["N_raw"]
    else:
        obj_verts = obj_verts_orig.copy()[object_downsample_indices]  # N_objx3
        obj_vertex_normals = obj_vertex_normals_orig.copy()[object_downsample_indices]  # N_objx3
        assert len(obj_verts) == object_downsample_metadata["N"]

    #### SPECIFIC FOR BEHAVE ####
    with open(lowres_center_pth, "rb") as handle:
        lowres_center_data = pickle.load(handle)
    obj_lowres_center = lowres_center_data["obj_lowres_center"]  # 3

    with open(raw2normal_pth, "rb") as handle:
        raw2normal_data = pickle.load(handle)
    R_raw2normal = raw2normal_data["R_raw2normal"]  # 3 x 3
    t_raw2normal = raw2normal_data["t_raw2normal"]  # 3

    # transform the obj verts & obj vertex normals from 'normal->raw'
    obj_verts = (obj_verts - t_raw2normal[None]) @ R_raw2normal
    obj_verts_orig = (obj_verts_orig - t_raw2normal[None]) @ R_raw2normal
    obj_verts = obj_verts - obj_lowres_center[None]
    obj_verts_orig = obj_verts_orig - obj_lowres_center[None]

    # obj_vertex_normals_orig = obj_vertex_normals_orig @ R_raw2normal
    obj_vertex_normals = obj_vertex_normals @ R_raw2normal
    obj_vertex_normals_orig = obj_vertex_normals_orig @ R_raw2normal

    # load rotation & translation
    with open(obj_param_pth, "rb") as handle:
        obj_param_data = pickle.load(handle)
    obj_angle = obj_param_data["angle"]
    obj_trans = obj_param_data["trans"]
    obj_rotmat = batch_rodrigues(torch.from_numpy(obj_angle).unsqueeze(0))  # 1x3 -> 1x3x3
    obj_rotmat = obj_rotmat.squeeze().detach().cpu().numpy()  # 3x3
    obj_trans = np.expand_dims(obj_trans, axis=0)  # 1x3

    # rotate & translate
    obj_verts = obj_verts @ obj_rotmat.T + obj_trans
    obj_verts_orig = obj_verts_orig @ obj_rotmat.T + obj_trans
    obj_vertex_normals = normalize_vectors_np(obj_vertex_normals @ obj_rotmat.T, eps=eps)
    obj_vertex_normals_orig = normalize_vectors_np(obj_vertex_normals_orig @ obj_rotmat.T, eps=eps)

    if interactive:
        obj_mesh = o3d.geometry.TriangleMesh()
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_orig)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces_orig)
        obj_mesh.compute_vertex_normals()

        o3d.visualization.draw_geometries([human_mesh, obj_mesh])

        human_pcd = o3d.geometry.PointCloud()
        obj_pcd = o3d.geometry.PointCloud()
        human_pcd.points = o3d.utility.Vector3dVector(human_verts)
        obj_pcd.points = o3d.utility.Vector3dVector(obj_verts)
        geometries = [human_pcd, obj_pcd]

        # o3d.visualization.draw_geometries(geometries)

        ## visualize the normals, too!
        from visualization.visualize_open3d import get_arrow

        # object
        for i in range(len(obj_verts)):
            start = obj_verts[i]
            assert obj_vertex_normals[i].sum() != 0, "Should have been Already Processed in 'test_behave_downsample.py'"
            end = obj_verts[i] + obj_vertex_normals[i] * 0.05
            geometries.append(get_arrow(end=end, origin=start))
        o3d.visualization.draw_geometries(geometries)

    ### Return Inputs for Affordance Extraction ###
    affordance_extraction_inputs = dict(
        human_verts_orig=human_verts_orig,
        human_faces_orig=human_faces_orig,
        human_vertex_normals_orig=human_vertex_normals_orig,
        obj_verts_orig=obj_verts_orig,
        obj_faces_orig=obj_faces_orig,
        obj_vertex_normals_orig=obj_vertex_normals_orig,
        human_downsample_indices=human_downsample_indices,
        object_downsample_indices=object_downsample_indices,
        human_verts=human_verts,
        human_vertex_normals=human_vertex_normals,
        obj_verts=obj_verts,
        obj_vertex_normals=obj_vertex_normals,
    )

    return affordance_extraction_inputs


def load_behave_inputs_legacy(
    args,
    category,
    obj_canon_pth,
    human_downsample_metadata,
    object_downsample_metadata,
    reader: FrameDataReader,
    frame_idx: int,
):
    ###### BEHAVE SPECIFIC ######
    # load human mesh
    if args.disable_smplx:
        human_fit = reader.get_smplfit(frame_idx, SMPL_NAME_BEHAVE)
        raise NotImplementedError
    else:
        human_fit = reader.get_smplfit_as_smplx(frame_idx, SMPL_NAME_BEHAVE)

    ## human vertices & vertex normals
    # load original vertices and faces
    _human_verts_ = human_fit.v
    _human_faces_ = human_fit.f
    _human_mesh_ = o3d.geometry.TriangleMesh()
    o3d.geometry.TriangleMesh()
    _human_mesh_.vertices = o3d.utility.Vector3dVector(_human_verts_)
    _human_mesh_.triangles = o3d.utility.Vector3iVector(_human_faces_)
    _human_mesh_.compute_vertex_normals()
    _human_vertex_normals_ = np.asarray(_human_mesh_.vertex_normals)
    _human_vertex_normals_ = normalize_vectors_np(vecs=_human_vertex_normals_, eps=args.eps)

    ## bj vertices & vertex normals
    # load canon object mesh data
    with open(obj_canon_pth, "rb") as handle:
        obj_canon_mesh_data = pickle.load(handle)

    # get vertices, vertex-normals, faces
    _obj_verts_ = obj_canon_mesh_data["vertices"]  # Vx3
    _obj_faces_ = obj_canon_mesh_data["faces"]  # Fx3
    _obj_vertex_normals_ = obj_canon_mesh_data["vertex_normals"]  # Vx3

    # load rotation & translation
    _obj_angle_, _obj_trans_ = reader.get_objfit_params(frame_idx, OBJ_NAME_BEHAVE)
    _obj_rotmat_ = batch_rodrigues(torch.from_numpy(_obj_angle_).unsqueeze(0))  # 1x3 -> 1x3x3
    _obj_rotmat_ = _obj_rotmat_.squeeze().detach().cpu().numpy()  # 3x3
    _obj_trans_ = np.expand_dims(_obj_trans_, axis=0)  # 1x3

    # rotate & translate
    _obj_verts_canon_ = _obj_verts_.copy()
    _obj_verts_ = _obj_verts_ @ _obj_rotmat_.T + _obj_trans_
    _obj_vertex_normals_ = _obj_vertex_normals_ @ _obj_rotmat_.T

    # normalize normals
    _obj_vertex_normals_ = normalize_vectors_np(vecs=_obj_vertex_normals_, eps=args.eps)

    ## downsample using indices
    # get downsample indices
    human_downsample_indices = human_downsample_metadata["downsample_indices"]  # N_human
    object_downsample_indices = object_downsample_metadata["downsample_indices"]  # N_obj

    # downsample vertices
    human_verts = _human_verts_[human_downsample_indices]  # N_humanx3
    human_vertex_normals = _human_vertex_normals_[human_downsample_indices]  # N_humanx3
    obj_verts = _obj_verts_[object_downsample_indices]  # N_objx3
    obj_vertex_normals = _obj_vertex_normals_[object_downsample_indices]  # N_objx3

    ## check for missing normals
    for i in range(len(human_verts)):
        start = human_verts[i]
        assert human_vertex_normals[i].sum() != 0, "Should have been Already Processed in 'test_behave_downsample.py'"
    # object
    for i in range(len(obj_verts)):
        start = obj_verts[i]
        assert obj_vertex_normals[i].sum() != 0, "Should have been Already Processed in 'test_behave_downsample.py'"

    assert len(object_downsample_indices) == object_downsample_metadata["N"]

    if args.interactive:
        _obj_mesh_ = o3d.geometry.TriangleMesh()
        _obj_mesh_.vertices = o3d.utility.Vector3dVector(_obj_verts_)
        _obj_mesh_.triangles = o3d.utility.Vector3iVector(_obj_faces_)
        _obj_mesh_.compute_vertex_normals()

        o3d.visualization.draw_geometries([_human_mesh_, _obj_mesh_])

        human_pcd = o3d.geometry.PointCloud()
        obj_pcd = o3d.geometry.PointCloud()
        human_pcd.points = o3d.utility.Vector3dVector(_human_verts_)
        obj_pcd.points = o3d.utility.Vector3dVector(_obj_verts_)
        geometries = [human_pcd, obj_pcd]

        # o3d.visualization.draw_geometries(geometries)

        ## visualize the normals, too!
        from visualization.visualize_open3d import get_arrow

        # object
        for i in range(len(obj_verts)):
            start = obj_verts[i]
            assert obj_vertex_normals[i].sum() != 0, "Should have been Already Processed in 'test_behave_downsample.py'"
            end = obj_verts[i] + obj_vertex_normals[i] * 0.05
            geometries.append(get_arrow(end=end, origin=start))
        o3d.visualization.draw_geometries(geometries)

    return dict(
        _human_verts_=_human_verts_,
        _human_faces_=_human_faces_,
        _human_vertex_normals_=_human_vertex_normals_,
        _obj_verts_canon_=_obj_verts_canon_,
        _obj_verts_=_obj_verts_,
        _obj_faces_=_obj_faces_,
        _obj_vertex_normals_=_obj_vertex_normals_,
        _obj_rotmat_=_obj_rotmat_,
        _obj_trans_=_obj_trans_,
        human_downsample_indices=human_downsample_indices,
        object_downsample_indices=object_downsample_indices,
        human_verts=human_verts,
        human_vertex_normals=human_vertex_normals,
        obj_verts=obj_verts,
        obj_vertex_normals=obj_vertex_normals,
    )


def main(seq_folder, args, debug=False):
    image_size = 1200
    w, h = image_size, int(image_size * 0.75)

    # FrameDataReader is the core class for dataset reading
    reader = FrameDataReader(seq_folder)

    # handle transformations between different kinect color cameras
    # inside the constructor, the calibration info and kinect intrinsics are loaded
    kinect_transform = KinectTransform(seq_folder, kinect_count=reader.kinect_count)

    # defines the subfolder for loading fitting results
    smpl_name = args.smpl_name
    obj_name = args.obj_name

    pyt3d_wrapper = Pyt3DWrapper(image_size=(w, h))
    outdir = args.viz_dir
    seq_save_path = join(outdir, reader.seq_name)
    os.makedirs(outdir, exist_ok=True)
    seq_end = reader.cvt_end(args.end)
    # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    rend_video_path_before_processed = join(f"{seq_save_path}_smpl_{smpl_name}_obj_{obj_name}_s{args.start}_e{seq_end}_before_processed.mp4")
    if os.path.exists(rend_video_path_before_processed):
        os.remove(rend_video_path_before_processed)
    rend_video_path = join(f"{seq_save_path}_smpl_{smpl_name}_obj_{obj_name}_s{args.start}_e{seq_end}.mp4")

    if os.path.exists(rend_video_path) and args.skip_done:
        print(f"Skipping generating '{rend_video_path}' since already done!")
        return

    video_writer = None
    loop = tqdm(range(args.start, seq_end))
    loop.set_description(reader.seq_name)

    for i in loop:
        # load smpl and object fit meshes
        if args.use_smplx:
            smpl_fit = reader.get_smplfit_as_smplx(i, smpl_name)
        else:
            smpl_fit = reader.get_smplfit(i, smpl_name)

        obj_fit = reader.get_objfit(i, obj_name)
        if smpl_fit is None or obj_fit is None:
            print("no fitting result for frame: {}".format(reader.frame_time(i)))
            continue
        fit_meshes = [smpl_fit, obj_fit]

        # smpl vertices, obj vertices
        smpl_verts = smpl_fit.v
        smpl_faces = smpl_fit.f
        obj_verts = obj_fit.v
        obj_faces = obj_fit.f
        smpl_mesh = o3d.geometry.TriangleMesh()
        obj_mesh = o3d.geometry.TriangleMesh()
        smpl_mesh.vertices = o3d.utility.Vector3dVector(smpl_verts)
        smpl_mesh.triangles = o3d.utility.Vector3iVector(smpl_faces)
        obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
        obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)

        if debug:
            from IPython import embed

            embed()
            o3d.visualization.draw_geometries([smpl_mesh, obj_mesh])

        # get all color images in this frame
        kids = [1, 2]  # choose which kinect id to visualize
        imgs_all = reader.get_color_images(i, reader.kids)

        imgs_resize = [cv2.resize(x, (w, h)) for x in imgs_all]
        overlaps = [imgs_resize[1]]

        selected_imgs = [imgs_resize[x] for x in kids]  # here we render fitting in all 4 views
        for orig, kid in zip(selected_imgs, kids):
            # transform fitted mesh from world coordinate to local color coordinate, same for point cloud
            fit_meshes_local = kinect_transform.world2local_meshes(fit_meshes, kid)

            # render mesh
            rend = pyt3d_wrapper.render_meshes(fit_meshes_local, viz_contact=args.viz_contact)
            h, w = orig.shape[:2]
            overlap = cv2.resize((rend * 255).astype(np.uint8), (w, h))
            cv2.putText(overlap, f"kinect {kid}", (w // 3, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            overlaps.append(overlap)
        comb = np.concatenate(overlaps, 1)
        cv2.putText(comb, reader.frame_time(i), (w // 3, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        if video_writer is None:
            ch, cw = comb.shape[:2]
            video_writer = cv2.VideoWriter(rend_video_path_before_processed, 0x7634706D, 3, (cw, ch))
        video_writer.write(cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))

    video_writer.release()

    os.system(f'ffmpeg -i "{rend_video_path_before_processed}" -vcodec libx264 -f mp4 "{rend_video_path}" -y -loglevel "quiet"')
    os.remove(rend_video_path_before_processed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--seq_folders", nargs="+")
    parser.add_argument("-sn", "--smpl_name", help="smpl fitting save name, for final dataset, use fit02", default="fit02")
    parser.add_argument("-on", "--obj_name", help="object fitting save name, for final dataset, use fit01", default="fit01")
    parser.add_argument("-fs", "--start", type=int, default=0, help="start from which frame")
    parser.add_argument("-fe", "--end", type=int, default=None, help="ends at which frame")
    parser.add_argument("-v", "--viz_dir", default="./data_test/BEHAVE_viz", help="path to save you r visualization videos")
    parser.add_argument("--use_smplx", action="store_true", default=True)
    parser.add_argument("-vc", "--viz_contact", default=True, action="store_true", help="visualize contact sphere or not")
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    # seq_folder be like: './data_test/BEHAVE/sequences/Date03_Sub03_backpack_back'
    if args.seq_folders is None:
        args.seq_folders = sorted(list(glob(f"./data_test/BEHAVE/sequences/*")))

    for seq_folder in args.seq_folders:
        main(seq_folder=seq_folder, args=args)
