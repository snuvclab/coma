import argparse
import pickle
from copy import deepcopy

import numpy as np

from utils.misc import to_np_torch_recursive
from utils.coma_occupancy import ComA_Occupancy

from mayavi import mlab


def visualize(asset_downsample_pth, affordance_path):
    occupancy_info = np.load(affordance_path, allow_pickle=True).item()
    prob_field = occupancy_info["prob_field"]
    spatial_grid_metadata = occupancy_info["spatial_grid_metadata"]

    N_x, N_y, N_z = prob_field.shape

    # declare mayavi figure
    img_H = 1000
    img_W = 1000
    fig = mlab.figure(size=(img_W, img_H), bgcolor=(1, 1, 1))
    fig.scene

    ignore_percentage = 0.1
    prob_field = prob_field * (prob_field > ignore_percentage * prob_field.max())

    # add to 'mayavi.mlab' pipeline
    thres = 0.0
    mlab.pipeline.volume(mlab.pipeline.scalar_field(prob_field), vmin=thres, vmax=1.0)

    with open(asset_downsample_pth, "rb") as handle:
        object_downsample_metadata = deepcopy(pickle.load(handle))

    obj_verts = object_downsample_metadata["obj_vertices_original"]
    obj_faces = object_downsample_metadata["obj_faces_original"]

    selected_obj_idx = ComA_Occupancy.selected_obj_idxs[0]  # only 1 selected
    obj_verts_canon = obj_verts - obj_verts[selected_obj_idx][None]  # O x 3

    obj_verts_canon = to_np_torch_recursive(obj_verts_canon, use_torch=False, device="cpu")  # O x 3
    faces = to_np_torch_recursive(obj_faces, use_torch=False, device="cpu")
    # color
    color = np.array([0.8, 0.8, 0.8])
    color = to_np_torch_recursive(color, use_torch=False, device="cpu")
    if color.ndim == 1:
        color = np.stack([color] * obj_verts_canon.shape[0], axis=0)

    # add obj mesh to 'mayavi.mlab' pipeline
    voxel_size = spatial_grid_metadata["voxel_size"]
    obj_mesh = mlab.triangular_mesh(
        obj_verts_canon[:, 0] / voxel_size + N_x / 2,
        obj_verts_canon[:, 1] / voxel_size + N_y / 2,
        obj_verts_canon[:, 2] / voxel_size + N_z / 2,
        faces,
        representation="surface",
        scalars=np.arange(obj_verts_canon.shape[0]),
    )

    # add texture to 'posed human'
    alpha = 1.0
    obj_mesh.module_manager.scalar_lut_manager.lut.table = np.c_[(color * 255), alpha * 255 * np.ones(obj_verts.shape[0])].astype(np.uint8)

    # update to 'mayavi.mlab' engine
    obj_mesh.mlab_source.update()
    obj_mesh.parent.update()

    mlab.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--asset_downsample_pth", type=str, default=None)
    parser.add_argument("--affordance_path", type=str, default=None)

    args = parser.parse_args()

    assert args.asset_downsample_pth is not None, "assign downsampled mesh path to visualize human"
    assert args.affordance_path is not None, "assign affordance path to visualize human"

    visualize(asset_downsample_pth=args.asset_downsample_pth, affordance_path=args.affordance_path)
