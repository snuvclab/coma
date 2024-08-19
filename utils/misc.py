from typing import Union
from easydict import EasyDict

import numpy as np
import torch


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_np_torch_recursive(
    X: Union[dict, EasyDict, np.ndarray, torch.Tensor, list],
    use_torch=True,
    device="cuda",
    np_float_type=np.float32,
    np_int_type=np.int64,
    torch_float_type=torch.float32,
    torch_int_type=torch.int64,
):

    ## recursive approach
    # for dictionaries, run array-to-tensor recursively
    if type(X) == dict or type(X) == EasyDict:
        for key in X.keys():
            if type(X[key]) in [dict, EasyDict, np.ndarray, torch.Tensor, list]:
                X[key] = to_np_torch_recursive(X[key], use_torch, device)

    elif type(X) == list:
        for idx in range(len(X)):
            if type(X[idx]) in [dict, EasyDict, np.ndarray, torch.Tensor, list]:
                X[idx] = to_np_torch_recursive(X[idx], use_torch, device)

    # for np.ndarrays, send to torch.Tensor
    elif type(X) == np.ndarray:
        if use_torch:
            X = torch.tensor(X, device=device)
    # for torch.Tensor, set the device only
    elif type(X) == torch.Tensor:
        if use_torch:
            X = X.to(device)
        else:
            X = X.detach().cpu().numpy()

    ## dtype conversion
    if type(X) == torch.Tensor:
        if X.dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
            X = X.type(torch_float_type)
        elif X.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            X = X.type(torch_int_type)
        else:
            pass
    elif type(X) == np.ndarray:
        if X.dtype in [np.float32, np.float16, np.float64]:
            X = X.astype(np_float_type)
        elif X.dtype in [np.int64, np.int32, np.int16]:
            X = X.astype(np_int_type)
        else:
            pass

    return X


def get_3d_indexgrid_ijk(N_x, N_y, N_z, raveled=False):
    # Create indice grid
    indices = np.mgrid[0:N_x, 0:N_y, 0:N_z]
    if raveled:
        indices = np.stack([indices[0].ravel() + 1, indices[1].ravel() + 1, indices[2].ravel() + 1], axis=-1)
        # -> Shape: [(N_x+1) * (N_y+1) * (N_z+1), 3]
        # -> NOTE: Order would be
        #            0 0 0
        #            0 0 1
        #            0 0 2
        #            ...
        #            0 1 0
        #            0 1 1
        #            ...
        #            1 0 0
        #            1 0 1
        #            ...
    return indices


import cv2


def write_message_on_img(image_bgr, message: str):
    # save image with "no human" message
    image_bgr = cv2.putText(image_bgr, text=message, org=(image_bgr.shape[0] - 150, image_bgr.shape[0] - 20), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5, color=(0, 0, 255))
    return image_bgr
