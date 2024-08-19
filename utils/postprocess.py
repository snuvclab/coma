import numpy as np
import torch

from detectron2.structures.boxes import BoxMode
from detectron2.structures.masks import BitMasks

from constants.segmentation import COCO_CLASS_ID2NAME, LVIS_CLASS_ID2NAME


def intersection_over_union(seg1, seg2, use_torch=False):
    if use_torch:
        # make boolean
        seg1 = seg1.type(torch.bool)
        seg2 = seg2.type(torch.bool)

        # get intersection
        inter = torch.logical_and(seg1, seg2)
        area_inter = torch.sum(inter)

        # get union
        union = torch.logical_or(seg1, seg2)
        area_union = torch.sum(union)

    else:
        # make boolean
        seg1 = seg1.astype(np.bool)
        seg2 = seg2.astype(np.bool)

        # get intersection
        inter = np.logical_and(seg1, seg2)
        area_inter = np.sum(inter)

        # get union
        union = np.logical_or(seg1, seg2)
        area_union = np.sum(union)

    return area_inter / area_union


def intersection_over_maximum(seg1, seg2, use_torch=False):
    if use_torch:
        # make boolean
        seg1 = seg1.type(torch.bool)
        seg2 = seg2.type(torch.bool)

        # get intersection
        inter = torch.logical_and(seg1, seg2)
        area_inter = torch.sum(inter)

        # get maximum
        area_seg1 = torch.sum(seg1)
        area_seg2 = torch.sum(seg2)
        area_max = max(area_seg1, area_seg2)

    else:
        # make boolean
        seg1 = seg1.astype(np.bool)
        seg2 = seg2.astype(np.bool)

        # get intersection
        inter = np.logical_and(seg1, seg2)
        area_inter = np.sum(inter)

        # get union
        area_seg1 = np.sum(seg1)
        area_seg2 = np.sum(seg2)
        area_max = max(area_seg1, area_seg2)

    return area_inter / area_max


def intersection_over_chosen_seg(seg1, seg2, key="seg2", use_torch=False):
    if use_torch:
        # make boolean
        seg1 = seg1.type(torch.bool)
        seg2 = seg2.type(torch.bool)

        # get intersection
        inter = torch.logical_and(seg1, seg2)
        area_inter = torch.sum(inter)

        # get divisor
        if key == "seg1":
            area_divisor = torch.sum(seg1)
        elif key == "seg2":
            area_divisor = torch.sum(seg2)
        else:
            raise NotImplementedError

    else:
        # make boolean
        seg1 = seg1.astype(np.bool)
        seg2 = seg2.astype(np.bool)

        # get intersection
        inter = np.logical_and(seg1, seg2)
        area_inter = np.sum(inter)

        # get divisor
        if key == "seg1":
            area_divisor = np.sum(seg1)
        elif key == "seg2":
            area_divisor = np.sum(seg2)
        else:
            raise NotImplementedError

    return area_inter / area_divisor


def bbox_xy_to_wh(bbox):
    if not isinstance(bbox, (tuple, list)):
        original_shape = bbox.shape
        bbox = bbox.reshape((-1, 4))
    else:
        original_shape = None
    bbox = BoxMode.convert(box=bbox, from_mode=BoxMode.XYXY_ABS, to_mode=BoxMode.XYWH_ABS)
    if original_shape is not None:
        return bbox.reshape(original_shape)
    return bbox


def bbox_wh_to_xy(bbox):
    if not isinstance(bbox, (tuple, list)):
        original_shape = bbox.shape
        bbox = bbox.reshape((-1, 4))
    else:
        original_shape = None
    bbox = BoxMode.convert(box=bbox, from_mode=BoxMode.XYWH_ABS, to_mode=BoxMode.XYXY_ABS)
    if original_shape is not None:
        return bbox.reshape(original_shape)
    return bbox


def make_bbox_square(bbox, bbox_expansion=0.0):
    """

    Args:
        bbox (4 or B x 4): Bounding box in xywh format.
        bbox_expansion (float): Expansion factor to expand the bounding box extents from
            center.

    Returns:
        Squared bbox (same shape as bbox).
    """
    bbox = np.array(bbox)
    original_shape = bbox.shape
    bbox = bbox.reshape(-1, 4)
    center = np.stack((bbox[:, 0] + bbox[:, 2] / 2, bbox[:, 1] + bbox[:, 3] / 2), axis=1)
    b = np.expand_dims(np.maximum(bbox[:, 2], bbox[:, 3]), 1)
    b *= 1 + bbox_expansion
    square_bboxes = np.hstack((center - b / 2, b, b))
    return square_bboxes.reshape(original_shape)


def local_to_global_cam(bboxes, cams, L):
    """
    Converts a weak-perspective camera w.r.t. a bounding box to a weak-perspective
    camera w.r.t. to the entire image.

    Args:
        bboxes (N x 4): Bounding boxes in xyxy format.
        cams (N x 3): Weak perspective camera.
        L (int): Max of height and width of image.
    """
    square_bboxes = make_bbox_square(bbox_xy_to_wh(bboxes))
    global_cams = []
    for cam, bbox in zip(cams, square_bboxes):
        x, y, b, _ = bbox
        X = np.stack((x, y))
        # Bbox space [0, b]
        s_crop = b * cam[0] / 2
        t_crop = cam[1:] + 1 / cam[0]

        # Global image space [0, 1]
        s_og = s_crop / L
        t_og = t_crop + X / s_crop

        # Normalized global space [-1, 1]
        s = s_og * 2
        t = t_og - 0.5 / s_og
        global_cams.append(np.concatenate((np.array([s]), t)))
    return np.stack(global_cams)


def process_bbox_mask(bboxes_person, masks_person, confidence_person, final_keepidx):
    """
    Recall:
        bboxes_person: numpy.ndarray of shape (N, 4)
        masks_person: numpy.ndarray of shape (N, H, W)
        keepidx: list of humans to keep
    """
    return bboxes_person[final_keepidx], masks_person[final_keepidx], confidence_person[final_keepidx]


def process_remove_overlap(bbox_list, confidence_list, minoverlap=0.8, exconf=0.98):
    # list to keep remaining outputs
    keepidx = list(range(len(bbox_list)))

    # iterate through bboxes
    sorted_bbox_list = sorted(zip(list(range(len(bbox_list))), bbox_list, confidence_list), key=lambda tup: tup[-1])
    for original_idx, bbox, confidence in sorted_bbox_list:
        # if confidence is over threshold, keep it no matter what
        if confidence >= exconf:
            pass

        # else, check the bbox
        else:
            # area of confidence bbox
            confidence_area = bbox[-2] * bbox[-1]

            # compare with other bboxes
            for idx, comparison_bbox in enumerate(bbox_list):
                # if already removed, or is the same, continue
                if idx not in keepidx:
                    continue
                elif idx == original_idx:
                    continue
                # else, compare
                else:
                    # area of confidence bbox for comparison
                    comp_confidence_area = comparison_bbox[-2] * comparison_bbox[-1]

                    # area of intersection
                    start_xs = [bbox[0], comparison_bbox[0]]
                    end_xs = [bbox[0] + bbox[2], comparison_bbox[0] + comparison_bbox[2]]
                    start_ys = [bbox[1], comparison_bbox[1]]
                    end_ys = [bbox[1] + bbox[3], comparison_bbox[1] + comparison_bbox[3]]
                    intersection_w = max([min(end_xs) - max(start_xs), 0.0])
                    intersection_h = max([min(end_ys) - max(start_ys), 0.0])
                    inter = intersection_w * intersection_h

                    # if [intersection / smaller bounding box]
                    if inter / confidence_area >= minoverlap or inter / comp_confidence_area >= minoverlap:
                        keepidx.remove(original_idx)
                        break

    return keepidx


# for objects only
def process_segmentation(instances, minoverlap=0.8, exconf=0.98, verbose=False):  # Use minoverlap=0.8, exconf=0.98!!!
    # print progress
    if verbose:
        print("[Log] Post-processing started for segmentation instances...")

    # to not affect person data, we pre-extract the information
    is_person = instances.pred_classes == 0
    survived_instance_idxs = torch.tensor(list(range(len(instances.pred_classes))), device=is_person.device)[is_person]

    # for all classes
    for class_id in list(set(instances.pred_classes.cpu().numpy().tolist())):
        # skip if class_id denotes human
        if class_id == 0:
            continue

        # if class_id denotes object: lvis
        if hasattr(instances, "is_lvis"):
            # if contains "is_lvis", class must be a LVIS category
            is_class = instances.pred_classes == class_id
            assert instances.is_lvis[is_class].any(), "Class must be a LVIS category if instances contain 'is_lvis' attribute"
            # retrieve class name
            class_name = LVIS_CLASS_ID2NAME[class_id]

        # if class_id denotes object: coco
        else:
            class_name = COCO_CLASS_ID2NAME[class_id]

        # indices of object in instances
        is_class = instances.pred_classes == class_id
        index_in_instances = torch.tensor(list(range(len(instances.pred_classes))), device=is_class.device)[is_class]

        # bboxes, masks, confidences
        bboxes_class = instances[is_class].pred_boxes.tensor.cpu().numpy()  # xyxy
        instances[is_class].pred_masks
        confidence_class = instances[is_class].scores.cpu().numpy()  # ndarray of shape [B,]

        # process bboxes & confidences
        class_bbox_list = bbox_xy_to_wh(bboxes_class)
        confidence_list = confidence_class.tolist()

        # remove excessive overlaps
        keepidx = process_remove_overlap(class_bbox_list.tolist(), confidence_list, minoverlap=minoverlap, exconf=exconf)
        survived_instance_idxs = torch.cat([survived_instance_idxs, index_in_instances[keepidx]])

        # notify if removal occured for specific class
        if len(keepidx) < len(instances[is_class].pred_classes):
            if verbose:
                print(f'[Log] Some of instances of CLASS: "{class_name}" have been removed due to excessive occlusion.')

    # sort remaining instances
    survived_instance_idxs = torch.sort(survived_instance_idxs).values.detach().cpu().numpy()

    # print progress
    if len(survived_instance_idxs) < len(instances.pred_classes):
        if verbose:
            print("[Log] There has been removal of some instances. Postprocessing complete.")
    elif len(survived_instance_idxs) == len(instances.pred_classes):
        if verbose:
            print("[Log] No removed instance. Postprocessing complete.")
    else:
        assert False

    return instances[survived_instance_idxs]


def process_remove_none(mocap_output_lst, keepidx):
    # list to save remaining outputs
    mocap_output_lst2return = []
    final_keepidx = []

    # iterate through all human outputs
    for pred_output, idx in zip(mocap_output_lst, keepidx):
        """
        usually, human prediction is none when
            - bounding box is very small

        this occurs very rarely
        """
        # remove if none
        if pred_output is not None:
            mocap_output_lst2return.append(pred_output)
            final_keepidx.append(idx)

    return mocap_output_lst2return, final_keepidx


def process_mocap_predictions(mocap_predictions, bboxes, image_size, masks=None):
    """
    Rescales camera to follow HMR convention, and then computes the camera w.r.t. to
    image rather than local bounding box.

    Args:
        mocap_predictions (list).
        bboxes (N x 4): Bounding boxes in xyxy format.
        image_size (int): Max dimension of image.
        masks (N x H x W): Bit mask of people.

    Returns:
        dict {str: torch.cuda.FloatTensor}
            bbox: Bounding boxes in xyxy format (N x 3).
            cams: Weak perspective camera (N x 3).
            masks: Bitmasks used for computing ordinal depth loss, cropped to image
                space (N x L x L).
            local_cams: Weak perspective camera relative to the bounding boxes (N x 3).
    """
    verts = np.stack([p["pred_vertices_smpl"] for p in mocap_predictions])
    # All faces are the same, so just need one copy.
    faces = np.expand_dims(mocap_predictions[0]["faces"].astype(np.int32), 0)
    max_dim = np.max(bbox_xy_to_wh(bboxes)[:, 2:], axis=1)
    local_cams = []
    for b, pred in zip(max_dim, mocap_predictions):  # mocap predictions: same as pred_output_list
        local_cam = pred["pred_camera"].copy()
        scale_o2n = pred["bbox_scale_ratio"] * b / 224
        local_cam[0] /= scale_o2n
        local_cam[1:] /= local_cam[:1]
        local_cams.append(local_cam)
    local_cams = np.stack(local_cams)
    global_cams = local_to_global_cam(bboxes, local_cams, image_size)
    inds = np.argsort(bboxes[:, 0])  # Sort from left to right to make debugging easy.
    person_parameters = {
        "bboxes": bboxes[inds].astype(np.float32),
        "cams": global_cams[inds].astype(np.float32),
        "faces": faces,
        "local_cams": local_cams[inds].astype(np.float32),
        "verts": verts[inds].astype(np.float32),
        "indices": inds,  # NOTE: Added. --> indices for making the order from left bbox to right bbox
    }
    for k, v in person_parameters.items():
        person_parameters[k] = torch.from_numpy(v).cuda()
    if masks is not None:
        full_boxes = torch.tensor([[0, 0, image_size, image_size]] * len(bboxes))
        full_boxes = full_boxes.float().cuda()
        masks = BitMasks(masks).crop_and_resize(boxes=full_boxes, mask_size=image_size)
        person_parameters["masks"] = masks[inds].cuda()
    return person_parameters
