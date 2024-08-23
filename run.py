import json, os
import jtorch
import random
import jittor as jt
from JDiffusion.pipelines import StableDiffusionPipeline
import argparse

max_num = 28
dataset_root = "data/B"
modifier_path = "prompt/modifier.json"
ckpt_root = "style"

def set_seed():
    seed = random.randint(0, 2**32 - 1)
    
    jtorch.cuda.manual_seed(seed)
    jtorch.cuda.manual_seed_all(seed)
    random.seed(seed)
    jtorch.set_global_seed(seed)
    return seed

def parse_args():
    parser = argparse.ArgumentParser(description="Single run to generate one style")
    parser.add_argument("--mode", type=str, default="sd", help="prompt, lora, sd")
    parser.add_argument("--template_file", type=str, default=None, help="template for prompt")
    parser.add_argument("--step", type=int, default=25, help="number of steps")
    # parser.add_argument("--cuda", type=int, default=None, required=True, help="cuda number")
    parser.add_argument("--taskid", type=int, default=None, required=True, help="task id")
    args = parser.parse_args()
    if args.mode not in ["prompt", "lora", "sd"]:
        raise ValueError("Invalid mode")
    # if args.cuda is None:
    #     raise ValueError("Please specify cuda number")
    if args.taskid not in range(max_num):
        raise ValueError(f"taskid should be in [0, {max_num})")
    return args

def main(args):
    if args.mode in ["prompt", "sd"]: # 需要prompt模板
        if args.template_file is None:
            raise ValueError("Please specify template file")
        with open(("./prompt/"+args.template_file)) as file:
            templates = json.load(file)

    with jt.no_grad():
        taskid = "{:0>2d}".format(args.taskid)
        device = "cuda"
        pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-2-1").to(device)

        pipe.load_lora_weights(f"{ckpt_root}/style_{taskid}")

        # load json
        with open(f"{dataset_root}/{taskid}/prompt.json", "r") as file:
            prompts = json.load(file)
        with open(modifier_path, "r") as file:
            modifier_all = json.load(file)
            modifier = modifier_all[taskid]
        
        seeds = {}
        output_folder = f"./output_step{str(args.step)}_{args.mode[:2]}/{taskid}"
        for id, prompt in modifier.items():
            seed = set_seed()
            if args.mode == "prompt":
                full_prompt = templates[taskid].replace("(CLASSNAME)", "single "+prompt.lower())
                image = pipe(full_prompt, num_inference_steps=args.step, width=512, height=512).images[0]
            elif args.mode == "lora":
                # original
                image = pipe(prompt + f" in style_{taskid}", num_inference_steps=args.step, width=512, height=512).images[0]
            elif args.mode == "sd":
                full_prompt = templates[taskid].replace("(CLASSNAME)", prompt.lower())
                print(full_prompt)
                image = pipe(full_prompt, num_inference_steps=args.step, width=512, height=512).images[0]
            else:
                raise ValueError("Invalid mode")

            os.makedirs(output_folder, exist_ok=True)
            image.save(output_folder+f"/{prompts[id]}.png")
            seeds[prompts[id]] = seed
        json.dump(seeds, open(output_folder+"/seed.json", "w"))


if __name__ == "__main__":
    args = parse_args()
    main(args)