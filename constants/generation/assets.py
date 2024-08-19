DATASET_PTHS = {
    # 3D-Future
    "3D-FUTURE": "data/3D-FUTURE-model",
    # ShapeNet
    "SHAPENET": "data/ShapeNetCore.v2",
    # SketchFabl
    "SKETCHFAB": "data/SketchFab",
    # SAPIEN
    "SAPIEN": "data/SAPIEN",
    # BEHAVE
    "BEHAVE": "data/BEHAVE",
    # INTERCAP
    "INTERCAP": "data/INTERCAP",
}

# example categories
DATASET_TYPE2CATEGORIES = {
    "3D-FUTURE": [
        ("Chair", "Lounge Chair / Cafe Chair / Office Chair"),
    ],
    "SHAPENET": [
        ("motorcycle,bike", "motorcycle,bike"),
    ],
    "SKETCHFAB": [
        ("umbrella", "umbrella"),
        ("frypan", "frypan"),
    ],
    "SAPIEN": [
        ("cart", "cart"),
    ],
    "BEHAVE": [
        ("BEHAVE", "backpack"),
    ],
    "INTERCAP": [
        ("INTERCAP", "suitcase"),
    ],
}


CATEGORY2DATASET_TYPE = dict()
for dataset_type in DATASET_TYPE2CATEGORIES.keys():
    for supercat_cat in DATASET_TYPE2CATEGORIES[dataset_type]:
        CATEGORY2DATASET_TYPE[supercat_cat] = dataset_type


CATEGORY2ASSET = {
    # 3D-Future
    "Chair": {
        "Lounge Chair / Cafe Chair / Office Chair": [
            "0a5a346c-cc3b-4280-b358-ccd1c4d8a865",
        ],
    },
    # ShapeNet
    "motorcycle,bike": {
        "motorcycle,bike": [
            "9b9794dda0a6532215a11c390f7ca182",
        ]
    },
    # SketchFab
    "umbrella": {"umbrella": ["85fto9rtgcvsx2itzy9rd0gwh7758d64"]},
    "frypan": {"frypan": {"77kk57qyyj3tivpp51tpjw6xia2ds9d9"}},
    ##BEHAVE
    "BEHAVE": {
        "backpack": ["behave_asset"],
    },
    ## INTERCAP
    "INTERCAP": {
        "suitcase": ["intercap_asset"],
    },
}

CATEGORY2CAMERA_CONFIG = {
    # 3D-Future
    "Chair": {
        "Lounge Chair / Cafe Chair / Office Chair": dict(
            ortho_scale=1.75,
            z_scale=0.5,
            bbox_size=(0.6, 0.6, 1.1),
            elevation=15,
            ## asset-specific example ##
            asset_specific_config={
                "0a5a346c-cc3b-4280-b358-ccd1c4d8a865": dict(
                    ortho_scale=1.75,
                    z_scale=0.5,
                    bbox_size=(0.6, 0.6, 1.1),
                    elevation=15,
                )
            },
        )
    },
    # ShapeNet
    "motorcycle,bike": {"motorcycle,bike": dict(ortho_scale=1.0, z_scale=0.45, bbox_size=(0.25, 0.25, 0.5))},
    # SketchFab
    "umbrella": {"umbrella": dict(view_num=4, perturb_sample_num=10, ortho_scale=1.25, z_scale=1.0, bbox_size=(0.2, 0.2, 0.4), elevation=15)},
    "frypan": {
        "frypan": dict(stride_x=0.07, stride_y=0.06, view_num=4, perturb_sample_num=10, ortho_scale=1.3, z_scale=10.0, bbox_size=(0.15, 0.15, 0.3), elevation=15),
    },
    # SAPIEN
    "cart": {
        "cart": dict(stride_x=0.075, stride_y=0.075, ortho_scale=1.5, z_scale=0.6, bbox_size=(0.3, 0.3, 0.7), elevation=15),
    },
    # BEHAVE
    "BEHAVE": {
        "backpack": dict(stride_x=0.2, stride_y=0.2, view_num=4, perturb_sample_num=10, ortho_scale=2.0, z_scale=2.0, bbox_size=(0.45, 0.45, 0.95), elevation=15),
    },
    # INTERCAP
    "INTERCAP": {
        "suitcase": dict(stride_x=0.15, stride_y=0.15, view_num=4, ortho_scale=1.2, z_scale=1.0, bbox_size=(0.5, 0.5, 1.0), elevation=15),
    },
}

CATEGORY2MASK_FILTER_CONFIG = {
    # 3D-FUTURE
    "Chair": {"Lounge Chair / Cafe Chair / Office Chair": dict(minimum_seg_overlap_ratio=0.8, maximum_seg_overlap_ratio=0.9)},
    # SketchFab
    "umbrella": {"umbrella": dict(minimum_seg_overlap_ratio=0.7, maximum_seg_overlap_ratio=0.9)},
    "frypan": {
        "frypan": dict(minimum_seg_overlap_ratio=0.1, maximum_seg_overlap_ratio=0.5),
    },
    # SAPIEN
    "cart": {
        "cart": dict(minimum_seg_overlap_ratio=0.21, maximum_seg_overlap_ratio=0.6),
    },
    # BEHAVE
    "BEHAVE": {
        "backpack": dict(minimum_seg_overlap_ratio=0.55, maximum_seg_overlap_ratio=0.59),
    },
    # INTERCAP
    "INTERCAP": {
        "suitcase": dict(minimum_seg_overlap_ratio=0.3, maximum_seg_overlap_ratio=0.4),
    },
}


CATEGORY2PERTURB_CONFIG = {
    # 3D-Future
    "Chair": {"Lounge Chair / Cafe Chair / Office Chair": dict(need_perturb=False)},
    # SketchFab
    "umbrella": {
        "umbrella": dict(
            need_perturb=True,
            rotation_x=[[-20, 20]],
            rotation_y=[[-20, 20]],
            displacement_z=[[1.1, 1.2]],
        )
    },
    "frypan": {
        "frypan": dict(
            need_perturb=True,
            rotation_x=[[-10, 10]],
            rotation_y=[[-10, 10]],
            displacement_z=[[10.0, 11.0]],
        )
    },
    # SAPIEN
    "cart": {
        "cart": dict(
            need_perturb=False,
        )
    },
    # BEHAVE
    "BEHAVE": {
        "backpack": dict(
            need_perturb=True,
            rotation_x=[[-5, 5]],
            rotation_y=[[-5, 5]],
            displacement_z=[[1.73, 1.83]],
        ),
    },
    # INTERCAP
    "INTERCAP": {
        "suitcase": dict(
            need_perturb=True,
            rotation_x=[[-45, 45]],
        ),
    },
}
