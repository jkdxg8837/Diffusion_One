import diffusers
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
path1="/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-base2/checkpoint-100"
path2="/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-multi-timestep_20bsz1/checkpoint-1"
raw_lora_weights_path = "/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-multi-timestep-anal_20bsz1/raw_lora_weights.pth"
re_init_weights_path = "/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-multi-timestep-anal_20bsz1/reinitialized_lora_weights.pth"
# Extract LoRA weights from both pipelines
from peft.utils import get_peft_model_state_dict
# Load raw lora weights
raw_lora_weights = torch.load(raw_lora_weights_path, map_location="cpu")
re_init_weights = torch.load(re_init_weights_path, map_location="cpu")
lora_state_dict1 = StableDiffusion3Pipeline.lora_state_dict(path1)
lora_state_dict2 = StableDiffusion3Pipeline.lora_state_dict(path2)

# Example: print keys of the LoRA weights
# print("Raw LoRA weight keys:", raw_lora_weights.keys())
# print("Re-initialized LoRA weight keys:", re_init_weights.keys())
# print("Pipeline 1 LoRA weight keys:", lora_state_dict1.keys())
# print("Pipeline 2 LoRA weight keys:", lora_state_dict2.keys())

# print("Pipeline 2 LoRA weight keys:", lora_weights2.keys())
import numpy as np 

def var_diff(params1, params2):
    variance_results_A = {}
    variance_results_B = {}
    for key in params1:
        key_for_param2 = key.split('.')
        # Join without the second last one
        key_for_param2 = '.'.join(key_for_param2[:-2])
        key_for_param2 = key_for_param2+".weight"
        key_for_param2 = "transformer."+key_for_param2
        if key_for_param2 not in params2:
            print(f"Key {key_for_param2} not in model2")
            continue

        tensor1 = params1[key].float().cpu().numpy()
        tensor2 = params2[key_for_param2].float().cpu().numpy()
        # Replace nan value into 0
        tensor1 = np.nan_to_num(tensor1, nan=0.0)
        tensor2 = np.nan_to_num(tensor2, nan=0.0)
        if tensor1.shape != tensor2.shape:
            print(f"Shape mismatch at {key}")
            continue

        diff = tensor1 - tensor2
        variance = np.var(diff)
        if "lora_A" in key:
            variance_results_A[key] = variance
        else:
            variance_results_B[key] = variance
    avg_variance_A = np.mean(list(variance_results_A.values()))
    avg_variance_B = np.mean(list(variance_results_B.values()))
    return avg_variance_A, avg_variance_B

def var_diff_clean(params1, params2):
    variance_results_A = {}
    variance_results_B = {}
    for key in params1:
        key_for_param2 = key
        if key_for_param2 not in params2:
            print(f"Key {key_for_param2} not in model2")
            continue

        tensor1 = params1[key].float().cpu().numpy()
        tensor2 = params2[key_for_param2].float().cpu().numpy()
        # Replace nan value into 0
        tensor1 = np.nan_to_num(tensor1, nan=0.0)
        tensor2 = np.nan_to_num(tensor2, nan=0.0)
        if tensor1.shape != tensor2.shape:
            print(f"Shape mismatch at {key}")
            continue

        diff = tensor1 - tensor2
        variance = np.var(diff)
        if "lora_A" in key:
            variance_results_A[key] = variance
        else:
            variance_results_B[key] = variance
    avg_variance_A = np.mean(list(variance_results_A.values()))
    avg_variance_B = np.mean(list(variance_results_B.values()))
    return avg_variance_A, avg_variance_B

avg_variance_A, avg_variance_B = var_diff(raw_lora_weights, lora_state_dict1)
print(f"Average variance for lora_A: {avg_variance_A}")
print(f"Average variance for lora_B: {avg_variance_B}")
# print(re_init_weights)
# print(lora_state_dict2)
avg_variance_A, avg_variance_B = var_diff(re_init_weights, lora_state_dict2)
print(f"Average variance for re_init lora_A: {avg_variance_A}")
print(f"Average variance for re_init lora_B: {avg_variance_B}")


# Calculating angles between lora_state_dict1 & re_init_weights
loraA_key = "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
loraB_key = "transformer.transformer_blocks.0.attn.to_q.lora_B.weight"
lora1_A = lora_state_dict1[loraA_key].float().cpu().numpy()
lora1_B = lora_state_dict1[loraB_key].float().cpu().numpy()
print(re_init_weights.keys())
lora2_A = re_init_weights["transformer_blocks.0.attn.to_q.lora_A.default.weight"].float().cpu().numpy()
lora2_B = re_init_weights["transformer_blocks.0.attn.to_q.lora_B.default.weight"].float().cpu().numpy()

def calculate_angle(v1, v2):
    v1 = v1.flatten()
    v2 = v2.flatten()
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip to avoid numerical errors
    return angle


#  (W+LoraA * LoraB) * W = W^2 + LoraA * LoraB * W
angle_A = calculate_angle(lora1_A, lora2_A)
angle_B = calculate_angle(lora1_B, lora2_B)
print(f"Angle between lora_A of pipeline 1 and re_init_weights: {angle_A} radians")
print(f"Angle between lora_B of pipeline 1 and re_init_weights: {angle_B} radians")
# Calculating angles between lora_state_dict2 & re_init_weights
lora1_A = lora_state_dict2[loraA_key].float().cpu().numpy()
lora1_B = lora_state_dict2[loraB_key].float().cpu().numpy()
lora2_A = re_init_weights["transformer_blocks.0.attn.to_q.lora_A.default.weight"].float().cpu().numpy()
lora2_B = re_init_weights["transformer_blocks.0.attn.to_q.lora_B.default.weight"].float().cpu().numpy()
# transformer_blocks.0.attn.to_q.base_layer.weight
angle_A = calculate_angle(lora1_A, lora2_A)
angle_B = calculate_angle(lora1_B, lora2_B)
print(f"Angle between lora_A of pipeline 2 and re_init_weights: {angle_A} radians")
print(f"Angle between lora_B of pipeline 2 and re_init_weights: {angle_B} radians")