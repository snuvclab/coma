import argparse
from openai import OpenAI
import base64
from glob import glob
import pickle
import os

from constants.openai import ORGANIZATION, API_KEY, PROMPT_GENERATION_TEXT
from constants.metadata import DEFAULT_SEED
from constants.generation.prompts import HUMAN_DEFINED_PROMPTS

client = OpenAI(
    organization=ORGANIZATION,
    api_key=API_KEY,
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_prompt(
    supercategories,
    categories,
    selected_view,
    asset_render_dir,
    save_dir,
    use_vlm,
    skip_done,
):

    render_info_list = sorted(list(glob(f"{asset_render_dir}/*/*/*")))
    object_info_list = [render_info.split("/")[-3:] for render_info in render_info_list]

    for supercategory, category, asset_id in object_info_list:
        save_path = f"{save_dir}/{supercategory}/{category}/{asset_id}/prompts.pickle"

        if supercategories is not None and supercategory not in supercategories:
            continue
        if categories is not None and category not in categories:
            continue
        if skip_done and os.path.exists(save_path):
            continue

        if use_vlm:
            if not os.path.exists(f"{asset_render_dir}/{supercategory}/{category}/{asset_id}/view:{selected_view:05d}.png"):
                selected_view = 0
            query_image_path = f"{asset_render_dir}/{supercategory}/{category}/{asset_id}/view:{selected_view:05d}.png"

            base64_image = encode_image(query_image_path)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT_GENERATION_TEXT}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}]}],
                temperature=0.0,
                seed=DEFAULT_SEED,
            )

            prompts = [raw_prompt[raw_prompt.find("1 person") :].rstrip(".") for raw_prompt in response.choices[0].message.content.split("\n")]

        else:
            assert HUMAN_DEFINED_PROMPTS[supercategory][category][asset_id]  # if you don't use vision-language model, you need to specify prompts by yourself

            prompts = HUMAN_DEFINED_PROMPTS[supercategory][category][asset_id]

        prompt_info = dict(prompts=prompts, use_vlm=use_vlm)

        os.makedirs(f"{save_dir}/{supercategory}/{category}/{asset_id}", exist_ok=True)
        with open(save_path, "wb") as output_file:
            pickle.dump(prompt_info, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supercategories", type=str, nargs="+")
    parser.add_argument("--categories", type=str, nargs="+")

    parser.add_argument("--selected_view", type=int, default=0)

    parser.add_argument("--asset_render_dir", type=str, default="results/generation/asset_renders")
    parser.add_argument("--save_dir", type=str, default="results/generation/prompts")

    parser.add_argument("--use_vlm", action="store_true")
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    generate_prompt(
        supercategories=args.supercategories,
        categories=args.categories,
        selected_view=args.selected_view,
        asset_render_dir=args.asset_render_dir,
        save_dir=args.save_dir,
        use_vlm=args.use_vlm,
        skip_done=args.skip_done,
    )
