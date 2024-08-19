QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT = {
    "qual:001": dict(
        human_res="FULL",  # 'FULL', '2000', '1000'
        human_use_downsample_pcd_raw=False,
        object_res="180",  # '1500', '180'
        object_use_downsample_pcd_raw=True,
        principle_vec=[0, 0, 1],
        sub_principle_vec=[0, 1, 0],
        rel_dist_method="dist",  # 'dist', 'sdf'
        spatial_grid_size=0.06,  # note: the smpl-x body is usually within the radius 1 sphere
        spatial_grid_thres=0.24,
        normal_gaussian_sigma=0.2,
        normal_res=250,
        spatial_res=0,  # if 0 --> save only distance as discrete representation
        eps=1e-10,
        significant_contact_ratio=0.3,
        enable_postfilter=True,
        standardize_human_scale=False,  # --> makes the object scale, not human
        scaler_range=(0.75, 1.25),
        visualize_type="aggr-human-contact",
        vis_example_num=0,
        quant_mode=False,
        quant_keys=[],
    ),
    "qual:backpack_human_contact": dict(
        spatial_grid_size=0.07,
        spatial_grid_thres=0.03,
        normal_gaussian_sigma=0.25,
        significant_contact_ratio=0.1,
        standardize_human_scale=False,
        scaler_range=(0.75, 1.25),
        visualize_type="aggr-human-contact",
    ),
    "qual:backpack_object_contact": dict(
        spatial_grid_size=0.15,
        spatial_grid_thres=0.05,
        normal_gaussian_sigma=0.25,
        significant_contact_ratio=0.1,
        standardize_human_scale=False,
        scaler_range=(0.75, 1.25),
        human_res="1000",
        human_use_downsample_pcd_raw=False,
        object_res="1500",
        object_use_downsample_pcd_raw=True,
        visualize_type="aggr-object-contact",
    ),
    "qual:backpack_occupancy": dict(
        spatial_res=30,
        normal_res=0,
        standardize_human_scale=False,
        scaler_range=(0.75, 1.25),
        human_res="FULL",
        human_use_downsample_pcd_raw=False,
        object_res="1500",
        object_use_downsample_pcd_raw=False,
        visualize_type="occupancy",
    ),
    "qual:backpack_orientation": dict(
        spatial_grid_size=0.03,
        spatial_grid_thres=0.1,
        normal_gaussian_sigma=0.2,
        significant_contact_ratio=0.1,
        standardize_human_scale=False,
        scaler_range=(0.75, 1.25),
        visualize_type="orientation",
        vis_example_num=1,
    ),
}

default_hyperparams = QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT["qual:001"]
for k in QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT.keys():
    for dhkey in default_hyperparams.keys():
        # fill in the default values if not existing
        if dhkey not in QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[k].keys():
            QUAL_AFFORDANCE_EXTRACTION_HYPERPARAMS_DICT[k][dhkey] = default_hyperparams[dhkey]
