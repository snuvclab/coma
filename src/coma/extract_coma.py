import os
from glob import glob
import argparse
import pickle
from copy import deepcopy
from tqdm import tqdm
import json

import random
import numpy as np
import torch
import open3d as o3d

from utils.reproducibility import seed_everything
from utils.load_3d import load_obj_as_o3d_preserving_face_order
from utils.coma import ComA, prepare_affordance_extraction_inputs, get_aggregated_contact
from utils.coma_occupancy import ComA_Occupancy
from utils.visualization.colormap import MplColorHelper

from constants.coma.qual import QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT
from constants.coma.quant import QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT
from constants.metadata import DEFAULT_SEED

mayavi_scaler = lambda x: x * 16.0 + 24.0

colormap = MplColorHelper("jet", 0.0, 1.0)


def apply_postfilter(
    enable_postfilter,
    human_postfilter_dir,
    _postfilter_remains,
    _supercategory,
    _category,
    _asset_id,
    _view_id,
    _asset_mask_id,
    _prompt,
    _mainprompt,
    _inpaint_id,
):
    _supercategory_str = _supercategory.replace("/", ":")
    _category_str = _category.replace("/", ":")

    # postfilter (after step8)
    if enable_postfilter:
        # save the filterings for not repeating the load!
        _human_postfilter_pth = f"{human_postfilter_dir}/{_supercategory_str}/{_category_str}/{_asset_id}/{_mainprompt}.json"
        assert os.path.exists(_human_postfilter_pth), _human_postfilter_pth

        # create filter remain list to save
        if (_supercategory, _category, _asset_id, _mainprompt) not in _postfilter_remains.keys():
            _postfilter_remains[(_supercategory, _category, _asset_id, _mainprompt)] = []

            with open(_human_postfilter_pth, "r") as rf:
                _human_postfilter_list = json.load(rf)
            _postfilter_remains[(_supercategory, _category, _asset_id, _mainprompt)] = [tuple(_human_postfilter) for _human_postfilter in _human_postfilter_list]

        _postfilter_checker = (_view_id, _asset_mask_id, _prompt, _inpaint_id)
        if _postfilter_checker not in _postfilter_remains[(_supercategory, _category, _asset_id, _mainprompt)]:
            return True, _postfilter_remains

    return False, _postfilter_remains


def run_affordance_extraction(
    supercategories,
    categories,
    prompts,
    camera_dir,
    human_params_dir,
    asset_downsample_dir,
    human_postfilter_dir,
    human_sample_dir,
    coma_save_dir,
    affordance_save_dir,
    smplx_canon_obj_pth,
    hyperparams,
    hyperparams_key,
    fovy,
    visualize,
    vis_example_num,
    tmp_cache_dir,
    selected_object_indices,
    scale_tolerance,
    interactive,
    vis_interactive,
    skip_done,
):
    ####### Hyperparameters ######
    human_res = hyperparams["human_res"]
    human_use_downsample_pcd_raw = hyperparams["human_use_downsample_pcd_raw"]  # if true, then uses pcd
    object_res = hyperparams["object_res"]
    object_use_downsample_pcd_raw = hyperparams["object_use_downsample_pcd_raw"]  # if true, then uses pcd
    principle_vec = hyperparams["principle_vec"]
    sub_principle_vec = hyperparams["sub_principle_vec"]
    rel_dist_method = hyperparams["rel_dist_method"]
    spatial_grid_size = hyperparams["spatial_grid_size"]
    spatial_grid_thres = hyperparams["spatial_grid_thres"]
    normal_gaussian_sigma = hyperparams["normal_gaussian_sigma"]
    normal_res = hyperparams["normal_res"]
    spatial_res = hyperparams["spatial_res"]
    eps = hyperparams["eps"]
    hyperparams["significant_contact_ratio"]
    enable_postfilter = hyperparams["enable_postfilter"]
    standardize_human_scale = hyperparams["standardize_human_scale"]
    scaler_range = hyperparams["scaler_range"]

    # visualization type
    vis_example_num = hyperparams["vis_example_num"] if "vis_example_num" in hyperparams else vis_example_num
    visualize_type = hyperparams["visualize_type"]
    quant_mode = hyperparams["quant_mode"]  # --> only does total

    ####### Load Human-Related Downsampling Metadata #######
    ## open smplx star pose template
    smplx_star = load_obj_as_o3d_preserving_face_order(smplx_canon_obj_pth)
    smplx_star.compute_vertex_normals()
    np.asarray(smplx_star.vertices)
    np.asarray(smplx_star.vertex_normals)
    np.asarray(smplx_star.triangles)

    ## human-downsample
    smplx_downsample_pth = f"./constants/mesh/smplx_star_downsampled_{human_res}.pickle"
    with open(smplx_downsample_pth, "rb") as handle:
        human_downsample_metadata = pickle.load(handle)

    ## human-resolution
    if human_res == "FULL":
        human_res = 10475
    else:
        human_res = int(human_res)

    ####### Load Object-Related Downsampling Metadata #######
    ## select object indices
    if selected_object_indices == "":
        selected_object_indices = None
    else:
        _selected_object_indices_ = []
        for selobjindex in selected_object_indices.split(" "):
            if "-" in selobjindex:
                start, end = selobjindex.split("-")
                _selected_object_indices_ += list(range(int(start), int(end) + 1))
            else:
                _selected_object_indices_ += [int(selobjindex)]
        selected_object_indices = sorted(list(set(_selected_object_indices_))) if len(_selected_object_indices_) > 0 else None

    ####### Find All Possible SCAMs (Supercategory / Category / AssetID / Mainprompt) #######
    _human_sample_pths = sorted(list(set(list(glob(f"{human_sample_dir}/*/*/*/*/*/*/*.pickle")))))
    # prepare all (supercategory/category/asset_id/mainprompt) pairs
    all_scams = []
    for _human_sample_pth in tqdm(_human_sample_pths, desc="Preparing SCAM Pairs"):
        # metadata
        _supercategory_str, _category_str, _asset_id, _, _, _prompt, _ = _human_sample_pth.split("/")[-7:]
        _supercategory = _supercategory_str.replace(":", "/")
        _category = _category_str.replace(":", "/")
        _mainprompt = "total" if "total:" in _prompt.split(",")[0] else _prompt.split(",")[0]

        # skip
        if supercategories is not None:
            if _supercategory.lower() not in supercategories:
                continue
        if categories is not None:
            if _category.lower() not in categories:
                continue
        if prompts is not None:
            if _mainprompt.lower() not in prompts:
                continue

        # add
        all_scams.append((_supercategory, _category, _asset_id, _mainprompt))

    # remove duplicates
    all_scams = sorted(list(set(all_scams)))

    ####### Prepare Inputs by Iterating for all Existing Files #######
    info_per_scams = dict()
    _postfilter_remains = dict()

    for _supercategory, _category, _asset_id, _mainprompt in all_scams:
        ## all related human_sample_pths
        _supercategory_str = _supercategory.replace("/", ":")
        _category_str = _category.replace("/", ":")
        _related_human_sample_pths = sorted(list(set(list(glob(f"{human_sample_dir}/{_supercategory_str}/{_category_str}/{_asset_id}/*/*/{_mainprompt}*/*.pickle")))))

        # result-save directory & result-save pths
        _result_save_dir = f"{coma_save_dir}/{_supercategory_str}/{_category_str}/{_asset_id}"
        _result_each_json_pth = f"{_result_save_dir}/{hyperparams_key}:{_mainprompt}.json"
        _result_each_save_pth = f"{_result_save_dir}/{hyperparams_key}:{_mainprompt}.pickle"
        _result_each_vis_name = f"{_result_save_dir}/{hyperparams_key}:{_mainprompt}"
        _tmp_cache_each = f"{tmp_cache_dir}/{_supercategory_str}/{_category_str}/{_asset_id}/{hyperparams_key}:{_mainprompt}"

        # input pths (asset)
        _asset_obj_pth = f"{asset_downsample_dir}/{_supercategory_str}/{_category_str}/{_asset_id}.obj"
        _asset_downsample_pth = f"{asset_downsample_dir}/{_supercategory_str}/{_category_str}/{_asset_id}_{object_res}.pickle"

        ## prepare input human pths
        for _human_sample_pth in tqdm(_related_human_sample_pths, desc=f"Preparing Inputs for SCAM: {_supercategory_str}/{_category_str}/{_asset_id}/{_mainprompt}"):
            # metadata
            _check1, _check2, _check3, _view_id, _asset_mask_id, _prompt, _inpaint_id_ext = _human_sample_pth.split("/")[-7:]
            _check4 = "total" if "total:" in _prompt.split(",")[0] else _prompt.split(",")[0]

            assert _supercategory_str == _check1
            assert _category_str == _check2
            assert _asset_id == _check3
            assert _mainprompt == _check4, (_mainprompt, _check4)
            _inpaint_id, _ext = _inpaint_id_ext.split(".")
            assert _ext == "pickle", "Human Finals must have '.pickle' extension"

            _camera_pth = f"{camera_dir}/{_supercategory_str}/{_category_str}/{_asset_id}/{_view_id}.pickle"

            # apply postfiltering
            _do_postfilter, _postfilter_remains = apply_postfilter(
                enable_postfilter=enable_postfilter,
                human_postfilter_dir=human_postfilter_dir,
                _postfilter_remains=_postfilter_remains,
                _supercategory=_supercategory,
                _category=_category,
                _asset_id=_asset_id,
                _view_id=_view_id,
                _asset_mask_id=_asset_mask_id,
                _prompt=_prompt,
                _mainprompt=_mainprompt,
                _inpaint_id=_inpaint_id,
            )
            if _do_postfilter:
                continue

            # save info per supercategory-category-asset_id pairs
            with open(_human_sample_pth, "rb") as handle:
                human_after_opt = pickle.load(handle)
                # this is base filtering, too. however, since they are detected during step7,
                # if you set enable_postfilter=True, this should have been removed.
                if human_after_opt in [
                    "NOT ALLOWED VIEWPOINT PROMPTS",
                    "ERRONEOUS SAMPLE DUE TO TOO SMALL HUMAN",
                    "TOO LITTLE INLIERS",
                    "LARGELY PENETRATED HUMAN",
                ]:
                    assert not enable_postfilter, _human_sample_pth
                    continue
                # other string errors?
                if type(human_after_opt) == str:
                    assert False, "What more errors could there be?"

            # create saveholders if non-existent
            if (_supercategory, _category, _asset_id, _mainprompt) not in info_per_scams.keys():
                info_per_scams[(_supercategory, _category, _asset_id, _mainprompt)] = dict(
                    input_human_pths=[],
                    asset_downsample_pth=_asset_downsample_pth,
                    asset_obj_pth=_asset_obj_pth,
                    camera_pth=_camera_pth,
                    result_save_dir=_result_save_dir,
                    result_json_pth=_result_each_json_pth,
                    result_save_pth=_result_each_save_pth,
                    result_vis_name=_result_each_vis_name,
                    tmp_cache=_tmp_cache_each,
                    pbar_desc=f"ComA Extraction for {_supercategory}/{_category}/{_asset_id}/{_mainprompt}",
                )
            assert _human_sample_pth not in info_per_scams[(_supercategory, _category, _asset_id, _mainprompt)], "should be the first time"
            info_per_scams[(_supercategory, _category, _asset_id, _mainprompt)]["input_human_pths"].append(_human_sample_pth)

    ####### Run Affordance Extractions (total) #######
    for supercategory, category, asset_id, mainprompt in sorted(list(info_per_scams.keys())):
        # quant mode
        if quant_mode:
            if mainprompt != "total":
                continue

        #### load info ####
        info = info_per_scams[(supercategory, category, asset_id, mainprompt)]
        asset_obj_pth = None  # info['asset_obj_pth']
        camera_pth = info["camera_pth"]
        asset_downsample_pth = info["asset_downsample_pth"]
        result_save_dir = info["result_save_dir"]
        result_json_pth = info["result_json_pth"]
        result_save_pth = info["result_save_pth"]
        info["result_vis_name"]
        info["tmp_cache"]
        input_human_pths = info["input_human_pths"]
        pbar_desc = info["pbar_desc"]

        #### Load Object with Downsampling ####
        # object downsample
        with open(asset_downsample_pth, "rb") as handle:
            object_downsample_metadata = deepcopy(pickle.load(handle))

        # object mesh
        object_downsample_metadata["obj_vertices_original"]
        object_downsample_metadata["obj_vertex_normals_original"]
        object_downsample_metadata["obj_faces_original"]

        # H & O & other settings
        H = human_downsample_metadata["N_raw"] if human_use_downsample_pcd_raw else human_downsample_metadata["N"]
        O = object_downsample_metadata["N_raw"] if object_use_downsample_pcd_raw else object_downsample_metadata["N"]

        # save info as json file
        if not os.path.exists(result_json_pth):
            #### Save Metadata ####
            save_info = deepcopy(info)
            save_info["H"] = H
            save_info["O"] = O
            save_info.update(hyperparams)

            os.makedirs(result_save_dir, exist_ok=True)
            with open(result_json_pth, "w") as wf:
                json.dump(save_info, wf, indent=1)

        else:
            with open(result_json_pth, "r") as rf:
                save_info = json.load(rf)

        #### Declare ComA ####
        if visualize_type == "occupancy":
            coma = ComA_Occupancy(
                scale_tolerance=scale_tolerance,
                human_res=H,
                obj_res=O,
                normal_res=normal_res,
                spatial_res=spatial_res,
                proximity_settings=dict(
                    spatial_grid_size=spatial_grid_size,
                    spatial_grid_thres=spatial_grid_thres,
                ),
                principle_vec=principle_vec,
                sub_principle_vec=sub_principle_vec,
                rel_dist_method=rel_dist_method,  # 'dist' or 'sdf'
                normal_gaussian_sigma=normal_gaussian_sigma,
                eps=eps,
                device="cuda",
            )
        else:
            coma = ComA(
                human_res=H,
                obj_res=O,
                normal_res=normal_res,
                spatial_res=spatial_res,
                proximity_settings=dict(
                    spatial_grid_size=spatial_grid_size,
                    spatial_grid_thres=spatial_grid_thres,
                ),
                principle_vec=principle_vec,
                sub_principle_vec=sub_principle_vec,
                rel_dist_method=rel_dist_method,  # 'dist' or 'sdf'
                normal_gaussian_sigma=normal_gaussian_sigma,
                eps=eps,
                device="cuda",
            )

        #### Aggregate Human Inputs ####
        if skip_done and os.path.exists(result_save_pth):
            coma.load(result_save_pth)
            random.shuffle(input_human_pths)
            main_samples = []
            if vis_example_num is None:
                vis_example_num = 0
            for input_human_pth in input_human_pths[:vis_example_num]:

                supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = input_human_pth.split("/")[-7:]
                mainprompt = prompt.replace("total:", "")
                inpaint_id, ext = inpaint_id_ext.split(".")
                assert ext == "pickle"
                human_params_pth = f"{human_params_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{mainprompt}/{inpaint_id}.pickle"

                main_samples += [
                    prepare_affordance_extraction_inputs(
                        human_mesh_pth=input_human_pth,
                        human_mesh_pth_type="pickle",
                        human_downsample_metadata=human_downsample_metadata,
                        object_downsample_metadata=object_downsample_metadata,
                        human_use_downsample_pcd_raw=human_use_downsample_pcd_raw,
                        object_use_downsample_pcd_raw=object_use_downsample_pcd_raw,
                        eps=eps,
                        standardize_human_scale=standardize_human_scale,
                        scaler_range=scaler_range,
                        camera_pth=camera_pth,
                        human_params_pth=human_params_pth,
                        object_mesh_for_check_pth=asset_obj_pth,
                        interactive=interactive,
                    )
                ]
            sample = dict()
        else:
            pbar = tqdm(input_human_pths, desc=pbar_desc)
            main_samples = []
            for input_human_pth in pbar:

                supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = input_human_pth.split("/")[-7:]
                mainprompt = prompt.replace("total:", "")
                inpaint_id, ext = inpaint_id_ext.split(".")
                assert ext == "pickle"
                human_params_pth = f"{human_params_dir}/{supercategory_str}/{category_str}/{asset_id}/{view_id}/{asset_mask_id}/{mainprompt}/{inpaint_id}.pickle"

                affordance_extraction_inputs = prepare_affordance_extraction_inputs(
                    human_mesh_pth=input_human_pth,
                    human_mesh_pth_type="pickle",
                    human_downsample_metadata=human_downsample_metadata,
                    object_downsample_metadata=object_downsample_metadata,
                    human_use_downsample_pcd_raw=human_use_downsample_pcd_raw,
                    object_use_downsample_pcd_raw=object_use_downsample_pcd_raw,
                    eps=eps,
                    standardize_human_scale=standardize_human_scale,
                    scaler_range=scaler_range,
                    camera_pth=camera_pth,
                    human_params_pth=human_params_pth,
                    object_mesh_for_check_pth=asset_obj_pth,
                    interactive=interactive,
                )
                if affordance_extraction_inputs is None:
                    continue

                sample = {
                    "human_verts": affordance_extraction_inputs["human_verts"],
                    "human_normals": affordance_extraction_inputs["human_vertex_normals"],
                    "obj_verts": affordance_extraction_inputs["obj_verts"],
                    "obj_normals": affordance_extraction_inputs["obj_vertex_normals"],
                }

                if vis_example_num is not None:
                    if len(main_samples) < vis_example_num:
                        main_samples.append(deepcopy(affordance_extraction_inputs))

                coma.register_sample_to_cache(**sample)

            # aggregate all samples to single distribution
            coma.aggregate_all_samples()
            coma.export(save_pth=result_save_pth)

        if visualize_type == "aggr-human-contact":
            aggregated_contact, significant_contact_vertex_indices = get_aggregated_contact(
                coma=coma,
                contact_map_type="human",
                significant_contact_ratio=hyperparams["significant_contact_ratio"],
            )

            os.makedirs(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}", exist_ok=True)
            np.save(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}/human_contact.npy", aggregated_contact / aggregated_contact.max())

        elif visualize_type == "aggr-object-contact":
            aggregated_contact, significant_contact_vertex_indices = get_aggregated_contact(
                coma=coma,
                contact_map_type="obj",
                significant_contact_ratio=hyperparams["significant_contact_ratio"],
            )

            obj_pcd_points = (object_downsample_metadata["downsampled_pcd_points_raw"],)
            obj_pcd_normals = (object_downsample_metadata["downsampled_pcd_normal_raw"],)

            score_on_geo = o3d.geometry.PointCloud()
            score_on_geo.points = o3d.utility.Vector3dVector(obj_pcd_points[0])
            score_on_geo.normals = o3d.utility.Vector3dVector(obj_pcd_normals[0])

            score = aggregated_contact / aggregated_contact.max()
            colors = colormap.get_rgb(score)[:, :3]
            score_on_geo.colors = o3d.utility.Vector3dVector(colors)

            os.makedirs(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}", exist_ok=True)
            o3d.io.write_point_cloud(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}/object_contact.ply", score_on_geo)

        elif visualize_type == "orientation":
            nonphysical_scores = coma.compute_nonphysical_response_sphere(n_bin=1e6, nonphysical_type="human", as_numpy=True,)[
                "human"
            ]  # H x O

            obj_i = 0
            nonphysical_score = nonphysical_scores[:, obj_i]  # H

            os.makedirs(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}", exist_ok=True)
            np.save(
                f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}/orientational_tendency.npy",
                (nonphysical_score - nonphysical_score.min()) / (nonphysical_score.max() - nonphysical_score.min()),
            )

        elif visualize_type == "occupancy":
            prob_field = coma.return_aggregated_spatial_grids(human_indices=None).cpu().numpy()  # N x N x N

            prob_field /= prob_field.max()
            prob_field = 0.7 * prob_field

            spatial_grid_metadata = coma.spatial_grid_metadata

            occupancy_info = dict(prob_field=prob_field, spatial_grid_metadata=spatial_grid_metadata)
            os.makedirs(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}", exist_ok=True)
            np.save(f"{affordance_save_dir}/{supercategory}/{category}/{asset_id}/{hyperparams_key}:{mainprompt}/occupancy.npy", occupancy_info)

        #### Delete Instances ####
        # detach and release from cuda, and remove
        for key in vars(coma):
            if type(vars(coma)[key]) == torch.Tensor:
                vars(coma)[key] = vars(coma)[key].detach().cpu()
        del coma

        # detach and release from cuda, and remove
        for key in sample.keys():
            if type(sample[key]) == torch.Tensor:
                sample[key] = sample[key].detach().cpu()
        del sample

        # detach and release from cuda, and remove
        for main_sample in main_samples:
            for key in main_sample.keys():
                if type(main_sample[key]) == torch.Tensor:
                    main_sample[key] = main_sample[key].detach().cpu()
            del main_sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")

    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--human_params_dir", type=str, default="results/generation/human_preds")
    parser.add_argument("--asset_downsample_dir", type=str, default="results/coma/asset_downsample")
    parser.add_argument("--human_postfilter_dir", type=str, default="results/coma/human_postfilterings")
    parser.add_argument("--human_sample_dir", type=str, default="results/generation/human_sample")
    parser.add_argument("--coma_save_dir", type=str, default="results/coma/extracted_coma")  # change later
    parser.add_argument("--affordance_save_dir", type=str, default="results/coma/affordance")  # change later

    parser.add_argument("--smplx_canon_obj_pth", type=str, default="./constants/mesh/smplx_star.obj")
    parser.add_argument("--hyperparams_key", choices=list(QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT.keys()) + list(QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT.keys()))

    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis_example_num", type=int)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--vis_interactive", action="store_true")  # interactive for vis only
    parser.add_argument("--fovy", type=float, default=27.5)  # for viz
    parser.add_argument("--tmp_cache_dir", type=str, default="results/coma_tmp_cache")
    parser.add_argument("--selected_object_indices", type=str, help="Type as '21 22' or '21-25'", default="")

    parser.add_argument("--scale_tolerance", type=float, default=3.0)  # choose above 1.0
    parser.add_argument("--skip_done", action="store_true")
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

    # hyperparameters
    assert args.hyperparams_key is not None, "You must Specify the 'args.hypeparams_key'"
    if "qual:" in args.hyperparams_key:
        hyperparams = QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[args.hyperparams_key]
    else:
        hyperparams = QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[args.hyperparams_key]

    # run affordance extraction
    run_affordance_extraction(
        supercategories=args.supercategories,
        categories=args.categories,
        prompts=args.prompts,
        camera_dir=args.camera_dir,
        human_params_dir=args.human_params_dir,
        asset_downsample_dir=args.asset_downsample_dir,
        human_postfilter_dir=args.human_postfilter_dir,
        human_sample_dir=args.human_sample_dir,
        coma_save_dir=args.coma_save_dir,
        affordance_save_dir=args.affordance_save_dir,
        smplx_canon_obj_pth=args.smplx_canon_obj_pth,
        hyperparams=hyperparams,
        hyperparams_key=args.hyperparams_key,
        fovy=args.fovy,
        visualize=args.visualize,
        vis_example_num=args.vis_example_num,
        tmp_cache_dir=args.tmp_cache_dir,
        selected_object_indices=args.selected_object_indices,
        scale_tolerance=args.scale_tolerance,
        interactive=args.interactive,
        vis_interactive=args.vis_interactive,
        skip_done=args.skip_done,
    )
