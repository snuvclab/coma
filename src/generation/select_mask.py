import os
import argparse
import pickle
from glob import glob
from tqdm import tqdm

import numpy as np
import cv2

from utils.reproducibility import seed_everything
from utils.prepare_renders import prepare_asset_render_pths

from constants.metadata import DEFAULT_SEED
from constants.generation.assets import CATEGORY2MASK_FILTER_CONFIG


def select_masks(supercategories, categories, asset_render_dir, asset_mask_dir, asset_seg_dir, save_dir, default_min_seg_overlap_ratio, default_max_seg_overlap_ratio, skip_done, verbose):

    ## prepare asset render paths
    asset_render_pths = prepare_asset_render_pths(asset_render_dir, supercategories, categories)

    ## run mask selection
    stats = dict()
    for asset_render_pth in tqdm(asset_render_pths):
        # metadata
        supercategory_str, category_str, asset_id, view_id_ext = asset_render_pth.split("/")[-4:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        view_id, ext = view_id_ext.split(".")
        assert ext == "png", "Rendering must have '.png' extension"

        # asset segmentation path
        asset_seg_pth = f"{asset_seg_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}.png"

        # result save directory & path
        result_save_dir = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}"
        os.makedirs(result_save_dir, exist_ok=True)

        # skip if already done
        result_save_pth = f"{result_save_dir}/{view_id}.pickle"
        if os.path.exists(result_save_pth) and skip_done:
            if verbose:
                print(f"Continueing '{result_save_pth}' Since Already Done!")
            continue

        # load asset segmentation & tight bbox
        asset_seg = cv2.imread(asset_seg_pth, cv2.IMREAD_GRAYSCALE)
        asset_seg_nonzero = np.array(asset_seg.nonzero())
        asset_bbox = np.zeros(asset_seg.shape, dtype=np.uint8)
        if len(asset_seg_nonzero[0]) == 0 or len(asset_seg_nonzero[1]) == 0:
            if verbose:
                print(f"Continueing '{asset_render_pth}' since Segmentation is Null")
            continue
        asset_bbox[asset_seg_nonzero[0].min() : asset_seg_nonzero[0].max() + 1, asset_seg_nonzero[1].min() : asset_seg_nonzero[1].max() + 1] = 1

        # area of segmentation
        asset_seg_area = (asset_seg > 0).astype(np.float32).sum()

        ## iterate for all possible masks
        valid_mask_ids = []
        asset_mask_pths = sorted(list(glob(f"{asset_mask_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/*.png")))
        for asset_mask_pth in asset_mask_pths:
            # metadata
            asset_mask_id_ext = asset_mask_pth.split("/")[-1]
            asset_mask_id, ext = asset_mask_id_ext.split(".")
            assert ext == "png", "Mask must have '.png' extension"

            # asset mask
            asset_mask = cv2.imread(asset_mask_pth, cv2.IMREAD_GRAYSCALE)

            # area of intersection between mask and segmentation
            asset_seg_mask_intersection_area = np.logical_and(asset_seg > 0, asset_mask > 0).astype(np.float32).sum()

            intersection_over_seg = asset_seg_mask_intersection_area / asset_seg_area
            min_seg_overlap_ratio = CATEGORY2MASK_FILTER_CONFIG[supercategory][category].get("minimum_seg_overlap_ratio", default_min_seg_overlap_ratio)
            max_seg_overlap_ratio = CATEGORY2MASK_FILTER_CONFIG[supercategory][category].get("maximum_seg_overlap_ratio", default_max_seg_overlap_ratio)

            is_over_min = intersection_over_seg >= min_seg_overlap_ratio
            is_below_max = intersection_over_seg <= max_seg_overlap_ratio
            # print(f"asset_mask_pth: {asset_mask_pth} | intersection_over_seg: {intersection_over_seg}")

            if is_over_min and is_below_max:
                valid_mask_ids.append(asset_mask_id)

        print(f"{asset_render_pth}: {valid_mask_ids}")
        ## save results
        if verbose:
            print(f"Number of selected masks for '{asset_render_dir}': {len(valid_mask_ids)}")
        to_save = {"supercategory": supercategory, "category": category, "asset_id": asset_id, "view_id": view_id, "valid_mask_ids": valid_mask_ids}
        with open(result_save_pth, "wb") as handle:
            pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # stats
        if supercategory not in stats.keys():
            stats[supercategory] = dict()
        if category not in stats[supercategory].keys():
            stats[supercategory][category] = 0

        stats[supercategory][category] += len(valid_mask_ids)

    # print stats
    for supercategory in stats.keys():
        print(f"[supercategory: {supercategory}]")
        for category in stats[supercategory].keys():
            print(f"\tcategory: {category} --> {stats[supercategory][category]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")

    # input dir
    parser.add_argument("--asset_render_dir", type=str, default="results/generation/asset_renders")
    parser.add_argument("--asset_seg_dir", type=str, default="results/generation/asset_segs")

    # save dir
    parser.add_argument("--asset_mask_dir", type=str, default="results/generation/asset_masks")

    parser.add_argument("--default_min_seg_overlap_ratio", type=float, default=0.8)
    parser.add_argument("--default_max_seg_overlap_ratio", type=float, default=0.9)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", default=DEFAULT_SEED)

    args = parser.parse_args()

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]

    # seed for reproducible generation
    seed_everything(args.seed)

    ## select the masks and save the metadata
    select_masks(
        supercategories=args.supercategories,
        categories=args.categories,
        asset_render_dir=args.asset_render_dir,
        asset_mask_dir=args.asset_mask_dir,
        asset_seg_dir=args.asset_seg_dir,
        save_dir=args.asset_mask_dir,
        default_min_seg_overlap_ratio=args.default_min_seg_overlap_ratio,
        default_max_seg_overlap_ratio=args.default_max_seg_overlap_ratio,
        skip_done=args.skip_done,
        verbose=args.verbose,
    )
