import diffusers
import torch
import numpy as np
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
stable_gamma_list = [49]
ckpt_steps = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# for stable_gamma in stable_gamma_list:
path=f"/home/u5649209/workspace/Diffusion_One/sd3-dog-singlecard-baseline-randomseed/checkpoint-"
output_dict = {}
for ckpt_step in ckpt_steps:
    ckpt_path = f"{path}{ckpt_step}"
    print(f"Loading checkpoint: {ckpt_path}")
    lora_state_dict1 = StableDiffusion3Pipeline.lora_state_dict(ckpt_path)


    fro_norms = {}
    lora_keys = list(lora_state_dict1.keys())
    for name in tqdm(lora_keys, desc="Computing Frobenius norms"):
        if "transformer.transformer_blocks" not in name:
            continue
        param = lora_state_dict1[name]
        if "lora_A" not in name:
            original_layer_name = '.'.join(name.split('.')[:-2])    
            lora_B_layer_name = original_layer_name + '.lora_B.weight'
            if lora_B_layer_name in lora_state_dict1:
                lora_B = lora_state_dict1[lora_B_layer_name].float().cpu().numpy()
                fro_norm = np.linalg.norm(param @ lora_B.T, ord='fro')
                fro_norms[original_layer_name] = fro_norm
    output_dict[ckpt_step] = fro_norms
np.save(f"fro_norms.npy", output_dict)