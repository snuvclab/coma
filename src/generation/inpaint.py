import os
import argparse
from glob import glob
from tqdm import tqdm
import pickle
from easydict import EasyDict

import numpy as np
import torch
import cv2
from PIL import Image

from diffusers import DDIMScheduler

from utils.reproducibility import seed_everything
from utils.prepare_renders import prepare_asset_render_pths
from utils.adaptive_mask_inpainting import (
    AdaptiveMaskInpaintPipeline,
    PointRendPredictor,
    SAMHumanPredictor,
    SAMHumanPredictorWithAssetExclusion,
    SAMHumanPredictorWithDefaultBboxAssetExclusion,
    SAMHumanPredictorAccumulativeBboxAssetExclusion,
    MaskDilateScheduler,
    ProvokeScheduler,
)

from constants.metadata import DEFAULT_SEED
from constants.generation.inpaint_ldm import AVAILABLE_MODELS, HF_MODEL_KEYS
from constants.generation.prompts import ALLOWED_VIEWPOINT_AUGMENTATIONS, SCV2DIFFUSERCONFIG, SC2DIFFUSERCONFIG


def set_pipeline(
    ldm_model_key,
    adaptive_mask_model_type,
    default_ddim_steps,
    default_pointrend_threshold,
    enable_sam_multitask_output,
    enable_safety_checker,
    use_visualizer,
):
    # model metadata
    model_info = AVAILABLE_MODELS[HF_MODEL_KEYS[ldm_model_key]]
    use_diffusers_format = model_info["use_diffusers_format"]
    use_inpaint = model_info["use_inpaint"]
    key = model_info["key"]
    if not use_inpaint:
        raise NotImplementedError

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # System
    # DDIM Scheduling
    DEFAULT_BETA_START = 0.00085
    DEFAULT_BETA_END = 0.012
    DEFAULT_BETA_SCHEDULE = "scaled_linear"

    # DDIM scheduler
    scheduler = DDIMScheduler(beta_start=DEFAULT_BETA_START, beta_end=DEFAULT_BETA_END, beta_schedule=DEFAULT_BETA_SCHEDULE, clip_sample=False, set_alpha_to_one=False)
    scheduler.set_timesteps(num_inference_steps=default_ddim_steps)

    ## load models as pipelines
    if use_diffusers_format:
        pipeline = AdaptiveMaskInpaintPipeline.from_pretrained(key, scheduler=scheduler, torch_dtype=torch.float16, requires_safety_checker=False).to(device)
    else:
        pipeline = AdaptiveMaskInpaintPipeline.from_single_file(key).to(device)

    ## disable safety checker
    if not enable_safety_checker:
        pipeline.safety_checker = None

    # declare segmentation model used for adaptive inpainting for human insertion
    if adaptive_mask_model_type == "p" or adaptive_mask_model_type == "baseline":
        adaptive_mask_model = PointRendPredictor(pointrend_thres=default_pointrend_threshold, device="cuda", use_visualizer=use_visualizer)
    if adaptive_mask_model_type == "ps":
        adaptive_mask_model = SAMHumanPredictor(
            pointrend_thres=default_pointrend_threshold,
            device="cuda",
            use_visualizer=use_visualizer,
            is_sam_multitask_output=enable_sam_multitask_output,
        )
    if adaptive_mask_model_type == "ps_ae":
        adaptive_mask_model = SAMHumanPredictorWithAssetExclusion(
            pointrend_thres=default_pointrend_threshold,
            device="cuda",
            use_visualizer=use_visualizer,
            is_sam_multitask_output=enable_sam_multitask_output,
        )
    if adaptive_mask_model_type == "s_pdb_ae":
        adaptive_mask_model = SAMHumanPredictorWithDefaultBboxAssetExclusion(
            pointrend_thres=default_pointrend_threshold,
            device="cuda",
            use_visualizer=use_visualizer,
            is_sam_multitask_output=enable_sam_multitask_output,
        )
    if adaptive_mask_model_type == "s_db_ae":
        adaptive_mask_model = SAMHumanPredictorWithDefaultBboxAssetExclusion(
            pointrend_thres=default_pointrend_threshold,
            device="cuda",
            use_visualizer=use_visualizer,
            is_sam_multitask_output=enable_sam_multitask_output,
        )
    if adaptive_mask_model_type == "s_ab_ae":
        adaptive_mask_model = SAMHumanPredictorAccumulativeBboxAssetExclusion(
            pointrend_thres=default_pointrend_threshold,
            device="cuda",
            use_visualizer=use_visualizer,
            is_sam_multitask_output=enable_sam_multitask_output,
        )
    pipeline.register_adaptive_mask_model(adaptive_mask_model)

    step_num = int(default_ddim_steps * 0.1)
    final_step_num = default_ddim_steps - step_num * 7
    # adaptive mask settings
    adaptive_mask_settings = EasyDict(
        dict(
            dilate_scheduler=MaskDilateScheduler(
                max_dilate_num=20,
                num_inference_steps=default_ddim_steps,
                schedule=[20] * step_num + [10] * step_num + [5] * step_num + [4] * step_num + [3] * step_num + [2] * step_num + [1] * step_num + [0] * final_step_num
                if adaptive_mask_model_type == "p"
                else [10] * 50,
            ),
            dilate_kernel=np.ones((3, 3), dtype=np.uint8),
            provoke_scheduler=ProvokeScheduler(
                num_inference_steps=default_ddim_steps,
                schedule=list(range(2, 10 + 1, 2)) + list(range(12, 40 + 1, 2)) + [45] if adaptive_mask_model_type != "baseline" else [],
                is_zero_indexing=False,
            ),
        )
    )
    pipeline.register_adaptive_mask_settings(adaptive_mask_settings)

    if adaptive_mask_model_type == "baseline":
        assert len(adaptive_mask_settings.provoke_scheduler.schedule) == 0, "Baseline should have no adaptive mask inpainting enabled"

    return pipeline


def canny(image, threshold_low=0, threshold_high=200):
    edges = cv2.Canny(image, threshold_low, threshold_high)
    return edges


def inpaint_human(
    num_img_per_combination,
    supercategories,
    categories,
    asset_render_dir,
    asset_mask_dir,
    asset_seg_dir,
    prompts_dir,
    save_dir,
    ldm_model_key,
    adaptive_mask_model_type,
    default_cfg_scale,
    default_strength,
    default_ddim_steps,
    default_pointrend_threshold,
    default_enforce_full_mask_ratio,
    default_human_detection_thres,
    enable_sam_multitask_output,
    negative_prompt,
    enable_safety_checker,
    use_visualizer,
    skip_done,
    verbose,
    parallel_num,
    parallel_idx,
    debug=False,
):
    ## prepare asset render paths
    asset_render_pths = prepare_asset_render_pths(asset_render_dir, supercategories, categories)

    ## prepare pipeline for stable-diffusion with 'adaptive mask inpainting'
    pipeline = set_pipeline(
        ldm_model_key,
        adaptive_mask_model_type,
        default_ddim_steps,
        default_pointrend_threshold,
        enable_sam_multitask_output,
        enable_safety_checker,
        use_visualizer,
    )

    ## for all renders, prepare input & metadata for human inpainting
    inpaint_inputs = []
    for asset_render_pth in tqdm(asset_render_pths, desc="Preparing Inputs..."):
        # metadata
        supercategory_str, category_str, asset_id, view_id_ext = asset_render_pth.split("/")[-4:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")
        view_id, ext = view_id_ext.split(".")
        assert ext == "png", "Rendering must have '.png' extension"

        # asset segmentation path
        asset_seg_pth = f"{asset_seg_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}.png"

        # asset mask list
        asset_mask_metadata_pth = f"{asset_mask_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}.pickle"
        if os.path.exists(asset_mask_metadata_pth):
            with open(asset_mask_metadata_pth, "rb") as handle:
                asset_mask_metadata = pickle.load(handle)
            asset_mask_id_list = asset_mask_metadata["valid_mask_ids"]
        else:
            assert debug, "THIS SHOULD BE ONLY ALLOWED IN DEBUGGING MODE. RUN STEP2 PRIOR"
            asset_mask_id_list = [asset_mask_pth.split("/")[-1].split(".")[0] for asset_mask_pth in sorted(list(glob(f"{asset_mask_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/*")))]

        ## iterate for all masks
        for asset_mask_id in asset_mask_id_list:
            asset_mask_pth = f"{asset_mask_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}.png"

            prompts_path = f"{prompts_dir}/{supercategory}/{category}/{asset_id}/prompts.pickle"
            with open(prompts_path, "rb") as handle:
                prompts = pickle.load(handle)["prompts"]

            viewpoint_augs = SCV2DIFFUSERCONFIG[supercategory][category].get(view_id, SC2DIFFUSERCONFIG[supercategory][category]).get("view_text", ["original"])
            assert type(viewpoint_augs) == list

            for prompt in prompts:
                for viewpoint_aug in viewpoint_augs:
                    ## viewpoint_aug must be inside ALLOWED_VIEWPOINT_AUGMENTATIONS
                    assert viewpoint_aug in ALLOWED_VIEWPOINT_AUGMENTATIONS, f"viewpoint augmentation: '{viewpoint_aug}' not allowed"

                    ## prompt + viewpoint augmentation
                    if viewpoint_aug == "original":
                        input_prompt = prompt
                    elif viewpoint_aug != ", full body":
                        continue
                    else:
                        input_prompt = prompt + viewpoint_aug

                    ## result-saving directory / result-saving path
                    for inpaint_id in range(num_img_per_combination):
                        result_save_dir = f"{save_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{input_prompt}"
                        result_save_pth = f"{result_save_dir}/{inpaint_id:06}.png"
                        pbar_desc = f"Running '{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{input_prompt}/{inpaint_id:06}'"
                        visualization_save_dir = f"{result_save_dir}/{inpaint_id:06}" if use_visualizer else None

                        ## add to inpaint-input
                        inpaint_inputs.append(
                            {
                                "asset_render_pth": asset_render_pth,
                                "asset_mask_pth": asset_mask_pth,
                                "asset_seg_pth": asset_seg_pth,
                                "result_save_dir": result_save_dir,
                                "result_save_pth": result_save_pth,
                                "visualization_save_dir": visualization_save_dir,
                                "pbar_desc": pbar_desc,
                                "inpaint_id": inpaint_id,
                                "input_prompt": input_prompt,
                                "input_negprompt": negative_prompt,
                                "ddim_steps": SCV2DIFFUSERCONFIG[supercategory][category]
                                .get(view_id, SC2DIFFUSERCONFIG[supercategory][category])
                                .get("ddim_steps", SC2DIFFUSERCONFIG[supercategory][category].get("ddim_steps", default_ddim_steps)),
                                "cfg_scale": SCV2DIFFUSERCONFIG[supercategory][category]
                                .get(view_id, SC2DIFFUSERCONFIG[supercategory][category])
                                .get("cfg_scale", SC2DIFFUSERCONFIG[supercategory][category].get("cfg_scale", default_cfg_scale)),
                                "strength": SCV2DIFFUSERCONFIG[supercategory][category]
                                .get(view_id, SC2DIFFUSERCONFIG[supercategory][category])
                                .get("strength", SC2DIFFUSERCONFIG[supercategory][category].get("strength", default_strength)),
                                "enforce_full_mask_ratio": SCV2DIFFUSERCONFIG[supercategory][category]
                                .get(view_id, SC2DIFFUSERCONFIG[supercategory][category])
                                .get("enforce_full_mask_ratio", SC2DIFFUSERCONFIG[supercategory][category].get("enforce_full_mask_ratio", default_enforce_full_mask_ratio)),
                                "human_detection_thres": SCV2DIFFUSERCONFIG[supercategory][category]
                                .get(view_id, SC2DIFFUSERCONFIG[supercategory][category])
                                .get("human_detection_thres", SC2DIFFUSERCONFIG[supercategory][category].get("human_detection_thres", default_human_detection_thres)),
                            }
                        )

    # parallel execution setting
    sub_length = (len(inpaint_inputs) // parallel_num) + 1
    start_idx = (parallel_idx) * sub_length
    end_idx = (parallel_idx + 1) * sub_length

    ## for all input for human inpainting, run adaptive mask inpainting
    inpaint_inputs = sorted(inpaint_inputs, key=lambda x: x["result_save_pth"])
    pbar = tqdm(inpaint_inputs[start_idx:end_idx])
    for inpaint_input in pbar:
        # set pbar description
        pbar.set_description(desc=inpaint_input["pbar_desc"])

        # retrieve paths
        asset_render_pth = inpaint_input["asset_render_pth"]
        asset_mask_pth = inpaint_input["asset_mask_pth"]
        asset_seg_pth = inpaint_input["asset_seg_pth"]
        inpaint_id = inpaint_input["inpaint_id"]

        # retrieve save paths
        result_save_dir = inpaint_input["result_save_dir"]
        os.makedirs(result_save_dir, exist_ok=True)

        result_save_pth = inpaint_input["result_save_pth"]
        if os.path.exists(result_save_pth) and skip_done:
            if verbose:
                print(f"Continueing {result_save_pth} Since Already Done!")
            continue

        # input image & mask & asset-seg
        init_image = Image.open(asset_render_pth).convert("RGB")
        default_mask_image = Image.open(asset_mask_pth).convert("L")
        asset_seg = np.array(Image.open(asset_seg_pth).convert("L")) > 0  # boolean array

        # prepare canny image
        cv2.imread(asset_render_pth, cv2.IMREAD_GRAYSCALE)

        # seed
        generator = torch.Generator(device="cuda")
        generator.manual_seed(inpaint_id)

        # inpaint prompts
        input_prompt = inpaint_input["input_prompt"]
        input_negprompt = inpaint_input["input_negprompt"]

        # inpaint settings
        ddim_steps = inpaint_input["ddim_steps"]
        cfg_scale = inpaint_input["cfg_scale"]
        strength = inpaint_input["strength"]
        enforce_full_mask_ratio = inpaint_input["enforce_full_mask_ratio"]
        visualization_save_dir = inpaint_input["visualization_save_dir"]
        human_detection_thres = inpaint_input["human_detection_thres"]

        # inpaint
        if adaptive_mask_model_type == "ps_ae":  # pointrend+sam / asset-exclusion
            pipeline.adaptive_mask_model.set_presumed_asset_mask(asset_seg)
        if adaptive_mask_model_type == "s_db_ae":  # sam / default-bbox / asset-exclusion
            pipeline.adaptive_mask_model.set_presumed_asset_mask(asset_seg)
            pipeline.adaptive_mask_model.reset_initial_human_bbox()
            pipeline.adaptive_mask_model.set_initial_human_bbox(np.asarray(default_mask_image) > 0)
        if adaptive_mask_model_type == "s_pdb_ae":  # sam / default-bbox from pointrend / asset-exclusion
            pipeline.adaptive_mask_model.set_presumed_asset_mask(asset_seg)
            pipeline.adaptive_mask_model.reset_initial_human_bbox()
        if adaptive_mask_model_type == "s_ab_ae":  # sam /accumulative-bbox from pointrend / asset-exclusion
            pipeline.adaptive_mask_model.set_presumed_asset_mask(asset_seg)
            pipeline.adaptive_mask_model.reset_initial_human_bbox()

        inpaint_result = pipeline(
            prompt=input_prompt,
            negative_prompt=input_negprompt,
            image=init_image,
            default_mask_image=default_mask_image,
            guidance_scale=cfg_scale,
            strength=strength,
            use_adaptive_mask=True,
            generator=generator,
            num_inference_steps=ddim_steps,
            enforce_full_mask_ratio=enforce_full_mask_ratio,
            visualization_save_dir=visualization_save_dir,
            human_detection_thres=human_detection_thres,
        ).images[0]

        inpaint_result.save(result_save_pth)


""" HYPERPARAMS"""
NUM_IMG_PER_COMBINATION = 10
ASSET_RENDER_DIR = "results/generation/asset_renders"
ASSET_MASK_DIR = "results/generation/asset_masks"
ASSET_SEG_DIR = "results/generation/asset_segs"
PROMPTS_DIR = "results/generation/prompts"
SAVE_DIR = "results/generation/inpaintings"
LDM_MODEL_KEY = "realisticvision"
# ADAPTIVE_MASK_MODEL_TYPE="s_db_ae"
ADAPTIVE_MASK_MODEL_TYPE = "p"
DEFAULT_CFG_SCALE = 11.0
DEFAULT_STRENGTH = 0.98
DEFAULT_DDIM_STEPS = 50
DEFAULT_POINTREND_THRESHOLD = 0.2  # doesn't matter in 's_db_ae'
DEFAULT_ENFORCE_FULL_MASK_RATIO = 0.0
DEFAULT_HUMAN_DETECTION_THRES = 0.015
NEGATIVE_PROMPT = "worst quality, normal quality, low quality, bad anatomy, artifacts, blurry, cropped, watermark, greyscale, nsfw"
SKIP_DONE = True
""" HYPERPARAMS"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_img_per_combination", type=int, default=NUM_IMG_PER_COMBINATION)
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")

    parser.add_argument("--asset_render_dir", type=str, default=ASSET_RENDER_DIR)
    parser.add_argument("--asset_mask_dir", type=str, default=ASSET_MASK_DIR)
    parser.add_argument("--asset_seg_dir", type=str, default=ASSET_SEG_DIR)
    parser.add_argument("--prompts_dir", type=str, default=PROMPTS_DIR)
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)

    parser.add_argument("--ldm_model_key", type=str, default=LDM_MODEL_KEY, choices=HF_MODEL_KEYS.keys())
    ## "p": pointrend / "ps": pointrend+sam / "ps_ae": pointrend+sam+asset-exclusion
    parser.add_argument("--adaptive_mask_model_type", type=str, choices=["baseline", "p", "ps", "ps_ae", "s_pdb_ae", "s_db_ae", "s_ab_ae"], default=ADAPTIVE_MASK_MODEL_TYPE)  # "s_db_ae" is the best!
    parser.add_argument("--default_cfg_scale", type=float, default=DEFAULT_CFG_SCALE)
    parser.add_argument("--default_strength", type=float, default=DEFAULT_STRENGTH)
    parser.add_argument("--default_ddim_steps", type=int, default=DEFAULT_DDIM_STEPS)
    parser.add_argument("--default_pointrend_threshold", type=float, default=DEFAULT_POINTREND_THRESHOLD)

    parser.add_argument("--default_enforce_full_mask_ratio", type=float, default=DEFAULT_ENFORCE_FULL_MASK_RATIO)
    parser.add_argument("--default_human_detection_thres", type=float, default=DEFAULT_HUMAN_DETECTION_THRES)
    parser.add_argument("--enable_sam_multitask_output", action="store_true")

    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT)
    parser.add_argument("--enable_safety_checker", action="store_true")

    parser.add_argument("--use_visualizer", action="store_true")
    parser.add_argument("--skip_done", action="store_true", default=SKIP_DONE)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--parallel_num", type=int, default=1)
    parser.add_argument("--parallel_idx", type=int, default=0)

    args = parser.parse_args()

    ## prepare supercategories / categories
    if args.supercategories is not None:
        args.supercategories = [supercategory.lower() for supercategory in args.supercategories]
    if args.categories is not None:
        args.categories = [category.lower() for category in args.categories]

    ## Added for checking
    if args.adaptive_mask_model_type == "baseline":
        print("\n\n############################# RUNNING BASELINE MODE!! #############################\n\n")
        args.save_dir = f"{args.save_dir}_noadaptivemask"

    # seed for reproducible generation
    seed_everything(args.seed)

    ## inpaint with rendered masks
    inpaint_human(
        num_img_per_combination=args.num_img_per_combination,
        supercategories=args.supercategories,
        categories=args.categories,
        asset_render_dir=args.asset_render_dir,
        asset_mask_dir=args.asset_mask_dir,
        asset_seg_dir=args.asset_seg_dir,
        prompts_dir=args.prompts_dir,
        save_dir=args.save_dir,
        ldm_model_key=args.ldm_model_key,
        adaptive_mask_model_type=args.adaptive_mask_model_type,
        default_cfg_scale=args.default_cfg_scale,
        default_strength=args.default_strength,
        default_ddim_steps=args.default_ddim_steps,
        default_pointrend_threshold=args.default_pointrend_threshold,
        default_enforce_full_mask_ratio=args.default_enforce_full_mask_ratio,
        default_human_detection_thres=args.default_human_detection_thres,
        enable_sam_multitask_output=args.enable_sam_multitask_output,
        negative_prompt=args.negative_prompt,
        enable_safety_checker=args.enable_safety_checker,
        use_visualizer=args.use_visualizer,
        skip_done=args.skip_done,
        verbose=args.verbose,
        parallel_num=args.parallel_num,
        parallel_idx=args.parallel_idx,
    )
