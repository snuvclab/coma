HUMAN_DEFINED_PROMPTS = {
    # 3D-Future
    "Chair": {
        "Lounge Chair / Cafe Chair / Office Chair": {
            "0a5a346c-cc3b-4280-b358-ccd1c4d8a865": [
                "1 person sits on a chair",
                "1 person moves a chair",
                "1 person leans against a chair",
            ],
        }
    },
    # ShapeNet
    "motorcycle,bike": {
        "motorcycle,bike": {
            "9b9794dda0a6532215a11c390f7ca182": [
                "1 person rides the motorcycle",
                "1 person cleans the motorcycle",
                "1 person examines the motorcycle",
            ],
        }
    },
    ## SketchFab
    "umbrella": {
        "umbrella": {
            "85fto9rtgcvsx2itzy9rd0gwh7758d64": [
                "1 person opens the umbrella",
                "1 person holds the umbrella",
                "1 person closes the umbrella",
            ],
        }
    },
    "frypan": {
        "frypan": {
            "77kk57qyyj3tivpp51tpjw6xia2ds9d9": [
                "1 person cooks with the frypan",
                "1 person washes the frypan",
                "1 person heats the frypan",
            ],
        }
    },
    # BEHAVE
    "BEHAVE": {
        "backpack": {
            "behave_asset": [
                "1 person wears the backpack",
                "1 person zips the backpack",
                "1 person carries the backpack",
            ]
        },
    },
    # INTERCAP
    "INTERCAP": {
        "suitcase": {
            "intercap_asset": [
                "1 person pulls the suitcase",
                "1 person packs the suitcase",
                "1 person unzips the suitcase",
            ]
        },
    },
}

SC2DIFFUSERCONFIG = {
    # 3D-Future
    "Chair": {
        "Lounge Chair / Cafe Chair / Office Chair": {
            "strength": 1.0,
            "controlnet_conditioning_scale": 0.0,
        }
    },
    # ShapeNet
    "motorcycle,bike": {
        "motorcycle,bike": {
            "strength": 0.9,
            "controlnet_conditioning_scale": 0.0,
        },
    },
    # SketchFab
    "umbrella": {"umbrella": dict()},  # default setting
    "frypan": {"frypan": dict()},
    # BEHAVE
    "BEHAVE": {
        "backpack": {
            "strength": 0.98,
        },
    },
    # INTERCAP
    "INTERCAP": {
        "suitcase": {
            "strength": 0.98,
        },
    },
}

ALLOWED_VIEWPOINT_AUGMENTATIONS = [
    ", full body",
    "original",
]

SCV2DIFFUSERCONFIG = {
    # 3F-Future
    "Chair": {
        "Lounge Chair / Cafe Chair / Office Chair": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(8))
        }
    },
    # ShapeNet
    "motorcycle,bike": {
        "motorcycle,bike": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(8))
        },
    },
    # SketchFab
    "umbrella": {
        "umbrella": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(40))
        }
    },
    "frypan": {
        "frypan": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(40))
        }
    },
    # SAPIEN
    "cart": {
        "cart": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(8))
        }
    },
    # BEHAVE
    "BEHAVE": {
        "backpack": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(40))
        },
    },
    # INTERCAP
    "INTERCAP": {
        "suitcase": {
            f"view:{idx:05}": {
                "view_text": [", full body", "original"],
            }
            for idx in list(range(40))
        },
    },
}
