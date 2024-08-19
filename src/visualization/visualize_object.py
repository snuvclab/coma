import blenderproc as bproc
import bpy

import argparse
import sys
import os

sys.path.append(os.getcwd())

from utils.blenderproc import initialize_scene, add_light, add_camera
from utils.visualization.colormap import MplColorHelper


def visualize(affordance_path):
    initialize_scene()
    add_light()
    camera = add_camera(resolution=(512, 512), name="CAMERA")

    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            space = area.spaces.active
            space.shading.type = "RENDERED"
            space.shading.color_type = "MATERIAL"

    bpy.ops.preferences.addon_enable(module="space_view3d_point_cloud_visualizer")
    bpy.ops.point_cloud_visualizer.load_ply_to_cache(filepath=affordance_path)
    bpy.ops.point_cloud_visualizer.draw()

    pcv = bpy.context.object.point_cloud_visualizer
    pcv.point_size = 10

    # pcv.mesh_type = "INSTANCER"
    # bpy.ops.point_cloud_visualizer.convert()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--affordance_path", type=str, default=None)

    args = parser.parse_args()

    assert args.affordance_path is not None, "assign affordance path to visualize human"

    visualize(affordance_path=args.affordance_path)
