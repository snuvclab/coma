import blenderproc as bproc
import bpy

import os
import sys

sys.path.append(os.getcwd())

import argparse

from mathutils import Matrix
import numpy as np

from utils.transformations import deg2rad
from utils.blenderproc import initialize_scene
from utils.reproducibility import seed_everything

from constants.metadata import DEFAULT_SEED

deg2rad = lambda deg: deg * np.pi / 180


def canonicalize(supercategory, category, obj_path):
    initialize_scene()

    bpy.ops.import_scene.obj(filepath=obj_path)
    asset_object = bpy.context.selected_objects[0]

    if supercategory == "BEHAVE" and category == "backpack":
        bpy.ops.object.origin_set(type="ORIGIN_CENTER_OF_MASS", center="MEDIAN")

        asset_object.location = [0.0, 0.0, 0.0]
        asset_object.rotation_euler = deg2rad(np.array([72.5, -6.0, 0.5]))

        os.makedirs("data/BEHAVE/objects/backpack/", exist_ok=True)
        bpy.ops.export_scene.obj(filepath="data/BEHAVE/objects/backpack/backpack.obj")

    if supercategory == "INTERCAP" and category == "suitcase":
        asset_object.location = [-0.09, 0.95, 0.015]

        bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="MEDIAN")

        asset_object.rotation_euler = deg2rad(np.array([0.0, 0.0, 90.0]))

        os.makedirs("data/INTERCAP/objects/suitcase/", exist_ok=True)
        bpy.ops.export_scene.obj(filepath="data/BEHAVE/objects/suitcase/mesh.obj")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategory", type=str)
    parser.add_argument("--category", type=str)
    parser.add_argument("--obj_path", type=str)
    parser.add_argument("--seed", default=DEFAULT_SEED)

    args = parser.parse_args()

    # seed for reproducible generation
    seed_everything(args.seed)

    ## render the assets in dataset
    canonicalize(
        supercategory=args.supercategory,
        category=args.category,
        obj_path=args.obj_path,
    )
