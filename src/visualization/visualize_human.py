import blenderproc as bproc
import bpy
import bmesh

import numpy as np

from mathutils import Matrix
import argparse
import pickle
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

    vertex_weight = np.load(affordance_path)

    mesh_path = "constants/mesh/smplx_star_downsampled_FULL.pickle"
    with open(mesh_path, "rb") as handle:
        mesh_data = pickle.load(handle)

    colormap = MplColorHelper(cmap_name="jet")
    color_layer_name = "color"

    collection = bpy.data.collections.new("SMPL Meshes")
    bpy.context.scene.collection.children.link(collection)

    if type(mesh_data) != str:
        verts, edges, faces = mesh_data["vertices"], [], mesh_data["faces"]

        mesh = bpy.data.meshes.new(name="SMPL")
        mesh.from_pydata(verts, edges, faces.astype(np.uint32))
        mesh.update()

        mesh_object = bpy.data.objects.new("SMPL", mesh)
        collection.objects.link(mesh_object)

        mesh = mesh_object.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        color_layer = bm.loops.layers.color.new(color_layer_name)
        for face in bm.faces:
            for loop in face.loops:
                weight = vertex_weight[loop.vert.index]
                loop[color_layer] = colormap.get_rgb(weight)

        # add material
        mat = bpy.data.materials.get("Material")
        if mat is None:
            mat = bpy.data.materials.new(name="Material")

        if mesh_object.data.materials:
            mesh_object.data.materials[0] = mat
        else:
            mesh_object.data.materials.append(mat)

        mat.use_nodes = True

        vc = mat.node_tree.nodes.new("ShaderNodeVertexColor")
        vc.layer_name = color_layer_name

        bsdf = mat.node_tree.nodes["Principled BSDF"]
        mat.node_tree.links.new(vc.outputs[0], bsdf.inputs[0])

        bm.to_mesh(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--affordance_path", type=str, default=None)

    args = parser.parse_args()

    assert args.affordance_path is not None, "assign affordance path to visualize human"

    visualize(affordance_path=args.affordance_path)
