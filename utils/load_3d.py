import open3d as o3d
import trimesh


def load_obj_as_o3d_preserving_face_order(file_pth, debug=False):
    # open3d uses ASSIMP for loading obj file, which changes the face order and vertex order
    # to mitigate, load with trimesh (process=False) first, and then convert to open3d triangle mesh

    trimesh_mesh = trimesh.load(file_pth, force="mesh", process=False)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    o3d_mesh.compute_vertex_normals()

    if debug:
        o3d.visualization.draw_geometries([o3d_mesh])

    return o3d_mesh
