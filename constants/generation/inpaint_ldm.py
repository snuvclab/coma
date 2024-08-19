#### MODEL CHOICES ####
AVAILABLE_MODELS = {
    ## inpainting
    "stabilityai/stable-diffusion-2-inpainting": {
        "key": "stabilityai/stable-diffusion-2-inpainting",
        "use_diffusers_format": True,
        "use_inpaint": True,
    },
    "Lykon/dreamshaper-8-inpainting": {"key": "Lykon/dreamshaper-8-inpainting", "use_diffusers_format": True, "use_inpaint": True},
    "Lykon/absolute-realism-1.6525-inpainting": {"key": "Lykon/absolute-realism-1.6525-inpainting", "use_diffusers_format": True, "use_inpaint": True},
    "Uminosachi/realisticVisionV51_v51VAE-inpainting": {"key": "Uminosachi/realisticVisionV51_v51VAE-inpainting", "use_diffusers_format": True, "use_inpaint": True},
}

HF_MODEL_KEYS = {
    "sd2inpaint": "stabilityai/stable-diffusion-2-inpainting",
    "dreamshaper8": "Lykon/dreamshaper-8-inpainting",
    "absolutereal": "Lykon/absolute-realism-1.6525-inpainting",
    "realisticvision": "Uminosachi/realisticVisionV51_v51VAE-inpainting",
}
