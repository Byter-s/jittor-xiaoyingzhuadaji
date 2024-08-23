import os
import json
import argparse
import jittor as jt
from pathlib import Path
from JDiffusion.pipelines import StableDiffusionPipeline

dataset_root = "./data/B"
sd_root = "./data/sdt"
style_folders = [f for f in Path(dataset_root).iterdir() if f.is_dir()]

prompts = {}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate oringinal images for test labels")
    parser.add_argument("--style", default=None, type=str, required=True, help="style folder name like 00")
    parser.add_argument("--step", default=25, type=int, help="diffusion steps")
    args = parser.parse_args()
    if args.style is None:
        raise ValueError("style folder name is required")
    return args

def gen_prompts():
    for num in range (28):
        folder_name = f"{num:02d}"
        test_prompt = json.load(open(dataset_root+f"/{folder_name}/prompt.json"))
        test_list = [_.lower() for _ in test_prompt.values()]
        prompts[folder_name] = test_list

    with open("./prompt/sd_test.json", "w") as file:
        json.dump(prompts, file, indent=4, ensure_ascii=False)

def gen_on_test_label(args):
    pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-2-1").to("cuda")
    prompts = json.load(open("./prompt/sd_test.json","r"))
    with jt.no_grad():
        for prompt in prompts[args.style]:
            filename = prompt.replace(" ", "_")
            prompt = f"an image of a realistic {prompt} placed at the very central place, similar to sketch or portrait composition. The whole {prompt} is prominently positioned in the center of the image to maximize focus. The background is uniformly white to highlight the {prompt} effectively."
            print(prompt)
            os.makedirs(sd_root+f"{args.step}/{args.style}", exist_ok=True)
            image = pipe(prompt, num_inference_steps=args.step, width=512, height=512).images[0]
            image.save(sd_root+f"{args.step}/{args.style}"+f"/{filename}.png")

if __name__ == "__main__":
    args = parse_args()
    gen_prompts()
    gen_on_test_label(args)