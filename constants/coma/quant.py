QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT = {
    "quant:full": dict(  # --> for visualizing contact on human
        human_res="750",  # 'FULL', '2000', '1000'
        human_use_downsample_pcd_raw=False,
        object_res="2048",  # '1500', '180', '1000'
        object_use_downsample_pcd_raw=True,
        principle_vec=[0, 0, 1],
        sub_principle_vec=[0, 1, 0],
        rel_dist_method="dist",  # 'dist', 'sdf'
        spatial_grid_size=0.04,  # note: the smpl-x body is usually within the radius 1 sphere
        spatial_grid_thres=0.1,
        normal_gaussian_sigma=0.2,
        normal_res=250,
        spatial_res=0,  # if 0 --> save only distance as discrete representation
        eps=1e-10,
        significant_contact_ratio=0.0,
        enable_prefilter=False,  ## DEFAULT FALSE!!!
        # enable_midfilter=True,
        enable_postfilter=True,
        standardize_human_scale=False,  # --> makes the object scale, not human
        scaler_range=(0.75, 1.25),
        visualize_type="none",
        vis_example_num=0,
        quant_mode=True,
        quant_keys=[
            "aggr_object_contact_metrics",
            "aggr_human_contact_metrics",
        ],
    ),
}

default_hyperparams = QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT["quant:full"]
for k in QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT.keys():
    for dhkey in default_hyperparams.keys():
        # fill in the default values if not existing
        if dhkey not in QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[k].keys():
            QUANT_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[k][dhkey] = default_hyperparams[dhkey]
