from glob import glob

from constants.generation.assets import CATEGORY2ASSET


def prepare_asset_render_pths(asset_render_dir, supercategories, categories):
    # all rendered images of asset
    asset_render_pths_raw = sorted(list(glob(f"{asset_render_dir}/*/*/*/*.png")))

    # filter for asset ids in CATEGORY2ASSET 'constants.generation.assets'
    asset_render_pths = []
    for asset_render_pth in asset_render_pths_raw:
        supercategory_str, category_str, asset_id, view_id_ext = asset_render_pth.split("/")[-4:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")

        if supercategory not in CATEGORY2ASSET.keys():
            continue
        if category not in CATEGORY2ASSET[supercategory].keys():
            continue
        if asset_id not in CATEGORY2ASSET[supercategory][category]:
            continue

        asset_render_pths.append(asset_render_pth)

    # filter for given supercategories/categories
    if supercategories is not None:
        asset_render_pths = [asset_render_pth for asset_render_pth in asset_render_pths if asset_render_pth.split("/")[-4].lower() in supercategories]
    if categories is not None:
        asset_render_pths = [asset_render_pth for asset_render_pth in asset_render_pths if asset_render_pth.split("/")[-3].lower() in categories]

    return asset_render_pths


def prepare_inpainting_pths(inpaint_dir, supercategories, categories, prompts):
    # all inpaintings
    inpaint_pths_raw = sorted(list(glob(f"{inpaint_dir}/*/*/*/*/*/*/*.png")))

    # filter
    inpaint_pths = []
    for inpaint_pth in inpaint_pths_raw:
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = inpaint_pth.split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")

        if supercategory not in CATEGORY2ASSET.keys():
            continue
        if category not in CATEGORY2ASSET[supercategory].keys():
            continue
        if asset_id not in CATEGORY2ASSET[supercategory][category]:
            continue

        inpaint_pths.append(inpaint_pth)

    # filter for given supercategories/categories/prompts
    if supercategories is not None:
        inpaint_pths = [inpaint_pth for inpaint_pth in inpaint_pths if inpaint_pth.split("/")[-7].lower() in supercategories]
    if categories is not None:
        inpaint_pths = [inpaint_pth for inpaint_pth in inpaint_pths if inpaint_pth.split("/")[-6].lower() in categories]
    if prompts is not None:
        inpaint_pths = [inpaint_pth for inpaint_pth in inpaint_pths if inpaint_pth.split("/")[-2].lower() in prompts]

    return sorted(inpaint_pths)


def prepare_human_after_opt_pths(human_after_opt_dir, supercategories, categories, prompts):
    # all inpaintings
    human_after_opt_pths_raw = sorted(list(glob(f"{human_after_opt_dir}/*/*/*/*/*/*/*.png")))

    # filter
    human_after_opt_pths = []
    for human_after_opt_pth in human_after_opt_pths_raw:
        supercategory_str, category_str, asset_id, view_id, asset_mask_id, prompt, inpaint_id_ext = human_after_opt_pth.split("/")[-7:]
        supercategory = supercategory_str.replace(":", "/")
        category = category_str.replace(":", "/")

        if supercategory not in CATEGORY2ASSET.keys():
            continue
        if category not in CATEGORY2ASSET[supercategory].keys():
            continue
        if asset_id not in CATEGORY2ASSET[supercategory][category]:
            continue

        human_after_opt_pths.append(human_after_opt_pth)

    # filter for given supercategories/categories/prompts
    if supercategories is not None:
        human_after_opt_pths = [human_after_opt_pth for human_after_opt_pth in human_after_opt_pths if human_after_opt_pth.split("/")[-7].lower() in supercategories]
    if categories is not None:
        human_after_opt_pths = [human_after_opt_pth for human_after_opt_pth in human_after_opt_pths if human_after_opt_pth.split("/")[-6].lower() in categories]
    if prompts is not None:
        human_after_opt_pths = [human_after_opt_pth for human_after_opt_pth in human_after_opt_pths if human_after_opt_pth.split("/")[-2].lower() in prompts]

    return sorted(human_after_opt_pths)
