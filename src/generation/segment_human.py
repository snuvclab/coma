import os
import argparse
from tqdm import tqdm
from easydict import EasyDict
import pickle

import cv2
from PIL import Image

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.projects import point_rend
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils.reproducibility import seed_everything
from utils.prepare_renders import prepare_inpainting_pths

from constants.metadata import DEFAULT_SEED
from constants.segmentation import COCO_SEG_CONFIG_PTH, COCO_SEG_WEIGHTS_PTH


def human_segmentation_coco(
    supercategories,
    categories,
    prompts,
    inpaint_dir,
    save_dir,
    threshold,
    parallel_num,
    parallel_idx,
    save_full,
    save_vis_in_same_folder,
    save_image,
    skip_done,
    verbose,
):
    # setup coco metadata
    setup_logger()
    coco_metadata = MetadataCatalog.get("coco_2017_val")

    # get segmentation model (coco)
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)  # --> Add PointRend-specific config\
    cfg.merge_from_file(COCO_SEG_CONFIG_PTH)
    cfg.MODEL.WEIGHTS = COCO_SEG_WEIGHTS_PTH
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.DEVICE = "cuda"

    # get segmentation model
    SegmentationModel = DefaultPredictor(cfg)

    ## prepare inpainting paths
    inpaint_pths = prepare_inpainting_pths(inpaint_dir, supercategories, categories, prompts)

    ## iterate for inpaintings
    images_to_segment = []
    for inpaint_pth in tqdm(sorted(inpaint_pths), desc="Running Human-Segmentation..."):
        # metadata
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = inpaint_pth.split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        inpaint_id, ext = inpaint_id_ext.split(".")

        """ FULL BODY ONLY """
        prompt.split(",")[0]
        if len(prompt.split(",")) == 1:
            pass
        else:
            if prompt.split(",")[-1].strip() != "full body":
                continue
        """ FULL BODY ONLY """

        assert ext == "png", "Inpainting must have '.png' extension"

        # result-save directory
        result_save_dir = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{prompt}"

        # result-save paths
        result_img_save_pth = f"{result_save_dir}/{inpaint_id}.png"
        result_seg_save_pth = f"{result_save_dir}/{inpaint_id}.pickle"

        if save_vis_in_same_folder:
            vis_save_dir_human = f"{save_dir}_VIZ/{supercategory_str}/{category_str}/{asset_id}/HUMAN"
            vis_save_dir_nohuman = f"{save_dir}_VIZ/{supercategory_str}/{category_str}/{asset_id}/NO-HUMAN"
            os.makedirs(vis_save_dir_human, exist_ok=True)
            os.makedirs(vis_save_dir_nohuman, exist_ok=True)
            vis_img_save_pth_human = f"{vis_save_dir_human}/{view_id}:{asset_mask_id}:{prompt}:{inpaint_id}.png"
            vis_img_save_pth_nohuman = f"{vis_save_dir_nohuman}/{view_id}:{asset_mask_id}:{prompt}:{inpaint_id}.png"
        else:
            vis_img_save_pth_human = f"{result_save_dir}/vis:{inpaint_id}.png"
            vis_img_save_pth_nohuman = f"{result_save_dir}/vis:{inpaint_id}.png"

        # if os.path.exists(result_img_save_pth) and os.path.exists(result_seg_save_pth) and skip_done:
        if os.path.exists(result_seg_save_pth) and skip_done:
            if verbose:
                print(f"Continueing '{result_seg_save_pth}' Since Already Processed...")
            continue

        os.makedirs(result_save_dir, exist_ok=True)

        images_to_segment.append(
            dict(
                inpaint_pth=inpaint_pth,
                vis_img_save_pth_human=vis_img_save_pth_human,
                vis_img_save_pth_nohuman=vis_img_save_pth_nohuman,
                result_img_save_pth=result_img_save_pth,
                result_seg_save_pth=result_seg_save_pth,
            )
        )

    # parallel execution setting
    sub_length = (len(images_to_segment) // parallel_num) + 1
    start_idx = (parallel_idx) * sub_length
    end_idx = (parallel_idx + 1) * sub_length

    images_to_segment = sorted(images_to_segment, key=lambda x: x["result_seg_save_pth"])
    pbar = tqdm(images_to_segment[start_idx:end_idx])

    for image_to_segment in pbar:
        inpaint_pth = image_to_segment["inpaint_pth"]
        vis_img_save_pth_human = image_to_segment["vis_img_save_pth_human"]
        vis_img_save_pth_nohuman = image_to_segment["vis_img_save_pth_nohuman"]
        result_img_save_pth = image_to_segment["result_img_save_pth"]
        result_seg_save_pth = image_to_segment["result_seg_save_pth"]

        # load image
        im = cv2.imread(inpaint_pth)
        assert im is not None
        H, W, _ = im.shape

        # run segmentation
        outputs = SegmentationModel(im)
        instances = outputs["instances"]

        # visualize segmentation on image
        if save_image:
            v = Visualizer(im[:, :, ::-1], coco_metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
            rend_result = v.draw_instance_predictions(instances.to("cpu")).get_image()
            if 0 in instances.pred_classes:
                cv2.imwrite(vis_img_save_pth_human, rend_result[:, :, ::-1])
            else:
                cv2.imwrite(vis_img_save_pth_nohuman, rend_result[:, :, ::-1])

            # save human mask
            is_person = instances.pred_classes == 0
            masks_person = instances[is_person].pred_masks
            if len(masks_person) > 0:
                Image.fromarray(masks_person[0].detach().cpu().numpy()).convert("L").save(result_img_save_pth)

        # save segmentation information
        if save_full:
            with open(result_seg_save_pth, "wb") as handle:
                pickle.dump(instances.to("cpu"), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            instances_proc = EasyDict(
                dict(
                    num_instances=len(instances),
                    image_height=H,
                    image_width=W,
                    pred_boxes=instances.pred_boxes.tensor.detach().cpu().numpy(),
                    scores=instances.scores.detach().cpu().numpy(),
                    pred_classes=instances.pred_classes.detach().cpu().numpy(),
                    pred_masks=instances.pred_masks.detach().cpu().numpy(),
                )
            )
            with open(result_seg_save_pth, "wb") as handle:
                pickle.dump(instances_proc, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run_human_segmentation(mode, **kwargs):
    assert mode in ["coco", "lvis", "odise"], f"Segmentation Mode: {mode} --> Not implemented..."

    # coco categories
    if mode == "coco":
        human_segmentation_coco(**kwargs)
    # lvis categories
    if mode == "lvis":
        raise NotImplementedError
    # open-vocabulary categories
    if mode == "odise":
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--inpaint_dir", type=str, default="results/generation/inpaintings")
    parser.add_argument("--save_dir", type=str, default="results/generation/human_segs")

    parser.add_argument("--mode", type=str, choices=["coco", "lvis", "odise"], default="coco")
    parser.add_argument("--threshold", type=float, default=0.8, nargs="?", choices=[0.8, 0.95])

    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)

    parser.add_argument("--disable_save_full", action="store_true", help="If False, saves essential information only (without detectron2 dependency)")
    parser.add_argument("--save_vis_in_same_folder", action="store_true", default=False)
    parser.add_argument("--save_image", action="store_true", default=False)
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

    ## predict human segmentation
    run_human_segmentation(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        inpaint_dir=args.inpaint_dir,
        save_dir=args.save_dir,
        mode=args.mode,
        threshold=args.threshold,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
        save_full=not args.disable_save_full,
        save_vis_in_same_folder=args.save_vis_in_same_folder,
        save_image=args.save_image,
        skip_done=args.skip_done,
        verbose=args.verbose,
    )
