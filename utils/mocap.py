import numpy as np


def extract_mesh_from_output(pred_output_list):
    pred_mesh_list = list()
    for pred_output in pred_output_list:
        if pred_output is not None:
            if "left_hand" in pred_output:  # hand mocap
                for hand_type in pred_output:
                    if pred_output[hand_type] is not None:
                        vertices = pred_output[hand_type]["pred_vertices_img"]
                        faces = pred_output[hand_type]["faces"].astype(np.int32)
                        pred_mesh_list.append(dict(vertices=vertices, faces=faces))
            else:  # body mocap (includes frank/whole/total mocap)
                vertices = pred_output["pred_vertices_img"]
                faces = pred_output["faces"].astype(np.int32)
                pred_mesh_list.append(dict(vertices=vertices, faces=faces))
    return pred_mesh_list
