import os
import argparse
from tqdm import tqdm
import pickle
from copy import deepcopy
from easydict import EasyDict

import cv2
import torch

from utils.prepare_bodymocap import prepare_bodymocap
from utils.reproducibility import seed_everything
from utils.prepare_renders import prepare_inpainting_pths
from utils.postprocess import process_segmentation, process_remove_overlap, process_remove_none, process_bbox_mask, bbox_xy_to_wh
from utils.misc import write_message_on_img, to_np_torch_recursive
from utils.mocap import extract_mesh_from_output

from constants.metadata import DEFAULT_SEED


def extract_human(
    hparams,
    img_pth,
    seg_pth,
    bodymocap,
    visualize,
    verbose,
):
    # load image
    image_bgr = cv2.imread(img_pth)

    # load human segmentation
    with open(seg_pth, "rb") as handle:
        instances_orig = pickle.load(handle).to("cuda")

    # post-process all objects before post-processing humans
    if hparams.remocc:
        # remove excessive occlusions for same categories
        instances = process_segmentation(deepcopy(instances_orig), minoverlap=hparams.minoverlap, exconf=hparams.exconf, verbose=verbose)
        # print progress
        if verbose:
            print("[Log] Removed excessive occlusions from image segmentation results.")
    else:
        instances = deepcopy(instances_orig)

    # find human predictions (2D)
    is_person = instances.pred_classes == 0
    human_indices = torch.tensor(list(range(len(instances.pred_classes))), device=is_person.device)[is_person]
    bboxes_person = instances[is_person].pred_boxes.tensor.cpu().numpy()  # bbox type: xyxy
    masks_person = instances[is_person].pred_masks
    confidence_person = instances[is_person].scores.cpu().numpy()  # ndarray of shape [num_human,]

    # if no humans were detected during segmentation
    if bboxes_person.shape[0] == 0:
        if verbose:
            print(f"No human detected --> {img_pth}")
        return "NO HUMANS", write_message_on_img(image_bgr, message="NO HUMAN DETECTED")

    ## if humans were detected in seg, post-process bbox adaptable for human-prediction model
    body_bbox_list = bbox_xy_to_wh(bboxes_person)  # numpy ndarray, shape: [num_human, 4], xyxy -> xywh
    confidence_list = confidence_person.tolist()

    # calculate the overlapping bboxes and remove if too much is overlapping
    if hparams.remocc:
        keepidx = process_remove_overlap(bbox_list=body_bbox_list.tolist(), confidence_list=confidence_list, minoverlap=hparams.minoverlap, exconf=hparams.exconf)
    else:
        keepidx = list(range(len(body_bbox_list)))

    # if there are no human remaining, return image only
    if len(keepidx) == 0:
        if verbose:
            print("After removing overlapping, no human exists.")
        return "NO HUMANS", write_message_on_img(image_bgr, message="NO HUMAN DETECTED")

    ## regress body pose
    mocap_output_list, visualize_image_list = bodymocap.regress(image_bgr, body_bbox_list[keepidx], visualize=visualize)

    # if erroneous mocap-output exists, remove (cases include extremely small bboxes)
    if None in mocap_output_list:
        if verbose:
            print(f"Erroneous human prediction exists for {img_pth}\n--> Removing Erroneous")
    mocap_output_list, final_keepidx = process_remove_none(mocap_output_list, keepidx)

    # if there are no humans remaining, return image only
    if len(mocap_output_list) == 0:
        if verbose:
            print("After removing erroneous, no human exists.")
        return "NO HUMANS", write_message_on_img(image_bgr, message="NO HUMAN DETECTED")
    elif len(mocap_output_list) >= 2:
        if verbose:
            print("More than 2 humans after removing erroneous and post-processing!!")
        return "MORE THAN 2 HUMANS", write_message_on_img(image_bgr, message="MORE THAN 2 HUMANS DETECTED")

    # post-process remaining human bboxes & masks
    final_bboxes_person, final_masks_person, final_confidence_person = process_bbox_mask(bboxes_person, masks_person, confidence_person, final_keepidx)

    # extract SMPL (SMPL-X) mesh for rendering (vertices, faces in image space)
    pred_mesh_list = extract_mesh_from_output(mocap_output_list)
    human_params = dict(
        pred_mesh_list=pred_mesh_list,
        mocap_output_list=mocap_output_list,
        # --> used for 'human_kps' prediction
        kps_aux=dict(
            mask_person_list=final_masks_person,  # (N,H,W)
            body_bbox_list_xyxy=final_bboxes_person,  # xyxy format, (N,4)
            confidence_list=final_confidence_person,  # (N,)
        ),
    )

    return human_params, visualize_image_list


def run_3dhuman_prediction(
    supercategories,
    categories,
    prompts,
    inpaint_dir,
    human_seg_dir,
    save_dir,
    mode,
    exconf,
    minoverlap,
    remocc,
    parallel_num,
    parallel_idx,
    visualize,
    skip_done,
    verbose,
):
    ## prepare bodymocap model
    assert mode in ["hand4whole"], f"mode '{mode}' not implemented..."
    bodymocap = prepare_bodymocap(mode=mode)

    ## prepare inpainting paths
    inpaint_pths = prepare_inpainting_pths(inpaint_dir, supercategories, categories, prompts)

    ## iterate for all inpaintings
    images_to_extract_human = []
    for inpaint_pth in tqdm(inpaint_pths, "Predicting Human for Inpaintings..."):
        # metadata
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = inpaint_pth.split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        inpaint_id, ext = inpaint_id_ext.split(".")
        assert ext == "png", "Inpainting must have '.png' extension"
        """ FULL BODY ONLY """
        prompt.split(",")[0]
        if len(prompt.split(",")) == 1:
            pass
        else:
            if prompt.split(",")[-1].strip() != "full body":
                continue
        """ FULL BODY ONLY """
        # human-seg path
        seg_pth = f"{human_seg_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}/{inpaint_id}.pickle"

        # result-save directory
        result_save_dir = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}"

        # result-save paths
        result_img_save_pth = f"{result_save_dir}/{inpaint_id}.png"
        result_pred_save_pth = f"{result_save_dir}/{inpaint_id}.pickle"
        if os.path.exists(result_pred_save_pth) and skip_done:
            if verbose:
                print(f"Continueing '{result_img_save_pth}' Since Already Processed...")
            continue

        os.makedirs(result_save_dir, exist_ok=True)

        images_to_extract_human.append(
            dict(
                inpaint_pth=inpaint_pth,
                seg_pth=seg_pth,
                result_img_save_pth=result_img_save_pth,
                result_pred_save_pth=result_pred_save_pth,
            )
        )

    # parallel execution setting
    sub_length = (len(images_to_extract_human) // parallel_num) + 1
    start_idx = (parallel_idx) * sub_length
    end_idx = (parallel_idx + 1) * sub_length

    images_to_extract_human = sorted(images_to_extract_human, key=lambda x: x["result_pred_save_pth"])
    pbar = tqdm(images_to_extract_human[start_idx:end_idx])

    # Extract Humans
    for image_to_extract_human in pbar:
        inpaint_pth = image_to_extract_human["inpaint_pth"]
        seg_pth = image_to_extract_human["seg_pth"]
        result_img_save_pth = image_to_extract_human["result_img_save_pth"]
        result_pred_save_pth = image_to_extract_human["result_pred_save_pth"]

        # extract human (as SMPL-X) with post-processing / (optional) render SMPL human on image
        human_params, rendered_img = extract_human(
            hparams=EasyDict(dict(exconf=exconf, minoverlap=minoverlap, remocc=remocc)),
            img_pth=inpaint_pth,
            seg_pth=seg_pth,
            bodymocap=bodymocap,
            verbose=verbose,
            visualize=visualize,
        )

        ## save results
        # if no human was detected
        if human_params == "NO HUMANS":
            with open(result_pred_save_pth, "wb") as handle:
                pickle.dump(human_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if more than 2 humans were detected
        elif human_params == "MORE THAN 2 HUMANS":
            with open(result_pred_save_pth, "wb") as handle:
                pickle.dump(human_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # if 1 human was detected
        else:
            # save human-rendered image if it exists
            if visualize:
                cv2.imwrite(result_img_save_pth, rendered_img[0])

            verts, faces = human_params["pred_mesh_list"][0]["vertices"], human_params["pred_mesh_list"][0]["faces"]
            pelvis = human_params["mocap_output_list"][0]["pelvis_xyz"]  # shape: (3,)
            smplx_data = human_params["mocap_output_list"][0]["smplx_data"]
            convert_data = human_params["mocap_output_list"][0]["convert_data"]
            joints_proj = human_params["mocap_output_list"][0]["joints_proj"]
            kps_aux = human_params["kps_aux"]
            to_save = to_np_torch_recursive(
                dict(
                    verts=verts,
                    faces=faces,
                    pelvis=pelvis,
                    kps_aux=kps_aux,
                    smplx_data=smplx_data,
                    joints_proj=joints_proj,
                    convert_data=convert_data,
                ),
                use_torch=False,
                device="cpu",
            )
            # save SMPL results
            with open(result_pred_save_pth, "wb") as handle:
                pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--inpaint_dir", type=str, default="results/generation/inpaintings")
    parser.add_argument("--human_seg_dir", type=str, default="results/generation/human_segs")  # change later
    parser.add_argument("--save_dir", type=str, default="results/generation/human_preds")  # change later

    parser.add_argument("--mode", type=str, choices=["hand4whole"], default="hand4whole")
    parser.add_argument("--exconf", type=float, default=0.98)
    parser.add_argument("--minoverlap", type=float, default=0.8)
    parser.add_argument("--disable_remocc", action="store_true")
    parser.add_argument("--visualize", action="store_true", default=False)

    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]
    if args.prompts is not None:
        args.prompts = [prompt.lower() for prompt in args.prompts]

    # seed for reproducible generation
    seed_everything(args.seed)

    ## predict 3d human
    run_3dhuman_prediction(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        inpaint_dir=args.inpaint_dir,
        human_seg_dir=args.human_seg_dir,
        save_dir=args.save_dir,
        mode=args.mode,
        exconf=args.exconf,
        minoverlap=args.minoverlap,
        remocc=not args.disable_remocc,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
        visualize=args.visualize,
        skip_done=args.skip_done,
        verbose=args.verbose,
    )
