import json, os
import jtorch
import random
import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline

dataset_root = "../data/B"
ckpt_root = "style"

modifier_path = "prompt/modifier.json"
template_path = "prompt/template.json"
seeds_path = "seed.json"
device = "cuda"

def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    jtorch.cuda.manual_seed(seed)
    jtorch.cuda.manual_seed_all(seed)
    random.seed(seed)
    jtorch.set_global_seed(seed)
    return

def main(taskid):
    with open(template_path, "r") as file:
        templates = json.load(file)
    with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
        prompts = json.load(file)
    with open(modifier_path, "r") as file:
        modifier_all = json.load(file)
        modifier = modifier_all[taskid]
    with open(seeds_path, "r") as file:
        seeds = json.load(file)
    seed = seeds[taskid]

    output_folder = f"./output/{taskid}"
    with jt.no_grad():
        pipe = StableDiffusionPipeline.from_pretrained("../stable-diffusion-2-1").to(device)
        pipe.load_lora_weights(f"{ckpt_root}/style_{taskid}")

        for id, prompt in modifier.items():
            set_seed(seed[prompts[id]])
            full_prompt = templates[taskid].replace("(CLASSNAME)", prompt.lower())
            print(full_prompt)
            image = pipe(full_prompt, num_inference_steps=500, width=512, height=512).images[0]

            os.makedirs(output_folder, exist_ok=True)
            image.save(output_folder+f"/{prompts[id]}.png")


if __name__ == "__main__":
    for style in range(28):
        taskid = "{:0>2d}".format(style)
        main(taskid)