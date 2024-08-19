from glob import glob
from tqdm import tqdm
import argparse
import pickle
import os
import json

from constants.metadata import DEFAULT_SEED
from utils.reproducibility import seed_everything


def run_post_filtering(
    supercategories,
    categories,
    prompts,
    human_sample_dir,
    save_dir,
    IoU_threshold_min,
    intersection_volume_ratio_threshold_max,
    inlier_num_threshold_min,
    enable_aggregate_total_prompts,
    parallel_num,
    parallel_idx,
):

    if enable_aggregate_total_prompts:
        human_pths = sorted(list(glob(f"{human_sample_dir}/*/*/*/*/*/total*/*.pickle")))
    else:
        human_pths = sorted(list(glob(f"{human_sample_dir}/*/*/*/*/*/[!total]*/*.pickle")))

    filtered_human_pths = [
        human_pth
        for human_pth in human_pths
        if (not supercategories or human_pth.split("/")[-7].lower() in supercategories)
        and (not categories or human_pth.split("/")[-6].lower() in categories)
        and (not prompts or human_pth.split("/")[-2].lower() in prompts)
    ]

    REJECTED_FROM_IoU = 0
    REJECTED_FROM_INTERSECTION = 0
    REJECTED_FROM_INLIERS = 0
    NUM_MESH = 0

    to_save = dict()
    for human_pth in tqdm(filtered_human_pths):
        supercategory, category, asset_id, view_id, asset_mask_id, prompt, inpaint_id_with_ext = human_pth.split("/")[-7:]
        inpaint_id, ext = inpaint_id_with_ext.split(".")
        base_prompt = prompt.split(",")[0]
        if prompt.replace(base_prompt, "") not in [", full body", ""]:
            continue

        ### Make Empty Placeholders ###
        if enable_aggregate_total_prompts:
            if (supercategory, category, asset_id) not in to_save:
                to_save[(supercategory, category, asset_id)] = []

        else:
            if (supercategory, category, asset_id, base_prompt) not in to_save:
                to_save[(supercategory, category, asset_id, base_prompt)] = []

        with open(human_pth, "rb") as handle:
            human_mesh = pickle.load(handle)

        if type(human_mesh) == str:
            continue
        else:
            NUM_MESH += 1

            IoU = human_mesh["IoU"]
            interscetion_ratio = human_mesh["interscetion_ratio"]
            if "num_inliers" in human_mesh.keys():
                num_inliers = human_mesh["num_inliers"]
            else:
                num_inliers = None

            if IoU < IoU_threshold_min:
                REJECTED_FROM_IoU += 1
                continue

            if interscetion_ratio > intersection_volume_ratio_threshold_max:
                REJECTED_FROM_INTERSECTION += 1
                continue

            if num_inliers is not None:
                if num_inliers < inlier_num_threshold_min:
                    REJECTED_FROM_INLIERS += 1
                    continue

            if enable_aggregate_total_prompts:
                if to_save.get((supercategory, category, asset_id), None) is None:
                    to_save[(supercategory, category, asset_id)] = []

                to_save[(supercategory, category, asset_id)].append([view_id, asset_mask_id, prompt, inpaint_id])
            else:
                if to_save.get((supercategory, category, asset_id, base_prompt), None) is None:
                    to_save[(supercategory, category, asset_id, base_prompt)] = []
                to_save[(supercategory, category, asset_id, base_prompt)].append([view_id, asset_mask_id, prompt, inpaint_id])

    for data_config in to_save.keys():
        if enable_aggregate_total_prompts:
            supercategory, category, asset_id = data_config
            save_pth = f"{save_dir}/{supercategory}/{category}/{asset_id}/total.json"
        else:
            supercategory, category, asset_id, base_prompt = data_config
            save_pth = f"{save_dir}/{supercategory}/{category}/{asset_id}/{base_prompt}.json"

        save_directory = f"{save_dir}/{supercategory}/{category}/{asset_id}"

        print(save_directory)
        os.makedirs(save_directory, exist_ok=True)

        with open(save_pth, "w") as wf:
            json.dump(to_save[data_config], wf, indent=1)

    print("\n")
    print(f"================ POST-FILTERING RESULTS ================")
    print(f"1. REJECTED FROM IoU: {REJECTED_FROM_IoU}")
    print(f"3. REJECTED FROM INTERSECTION: {REJECTED_FROM_INTERSECTION}")
    print(f"4. REJECTED FROM INLINERS: {REJECTED_FROM_INLIERS}")
    print("\n")
    print(f"5. INITIAL MESHES: {NUM_MESH}")
    print(f"6. LEFTOVER MESHES: {NUM_MESH - (REJECTED_FROM_IoU + REJECTED_FROM_INTERSECTION + REJECTED_FROM_INLIERS)}")
    print(f"================ POST-FILTERING RESULTS ================")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--human_sample_dir", type=str, default="results/generation/human_sample")
    parser.add_argument("--save_dir", type=str, default="results/coma/human_postfilterings")

    parser.add_argument("--IoU_threshold_min", type=float, default=0.7)
    parser.add_argument("--intersection_volume_ratio_threshold_max", type=float, default=0.05)
    parser.add_argument("--inlier_num_threshold_min", type=int, default=1)

    parser.add_argument("--enable_aggregate_total_prompts", action="store_true")
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)
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

    ## run depth optimization
    run_post_filtering(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        human_sample_dir=args.human_sample_dir,
        save_dir=args.save_dir,
        IoU_threshold_min=args.IoU_threshold_min,
        intersection_volume_ratio_threshold_max=args.intersection_volume_ratio_threshold_max,
        inlier_num_threshold_min=args.inlier_num_threshold_min,
        enable_aggregate_total_prompts=args.enable_aggregate_total_prompts,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
    )
