# Load model
import time
import torch
import numpy as np
from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from torchvision.utils import make_grid
# visualization
import matplotlib.pyplot as plt

from matplotlib import cm

from peft import LoraConfig, get_peft_model
from sklearn.datasets import make_moons
import sys
sys.path.append("/home/u5649209/workspace/flow_matching")  # Adjust the path as necessary to import flow_matching_utils
from flow_matching_utils import MFMLP, evaluate_result, train_moon_gen, reinit_lora
from flow_matching_utils import MeanFlow
# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')
torch.manual_seed(42)
import warnings

warnings.filterwarnings("ignore")

import os

# training arguments
lr = 0.001
batch_size = 4096
iterations = 10000
print_every = 1000
hidden_dim = 512
gradient_base = 0
gradient_iter = 10000
pretrain_iter = 16000
is_pre_train = False
is_lora = False
is_eval = False 
is_reinit = False
is_baseline = True
gamma = 100
mode = "new"
loss_history = []
lora_init_mode_list = [\
    "lora-one", \
    # "lora-ga", \
    # "lora-sb"\
]
# Baseline for 20; MeanF for 5 or 1
meanF_step = 20
def weight_angle(fft_model, lora_model, update_value = False):
    layer_num = [2, 4, 6]
    state_dict = lora_model.state_dict()
    fft_model = fft_model.state_dict()
    for i in range(len(layer_num)):
        module = f'main.{layer_num[i]}.weight'
        if 'bias' in module:
            continue
        compared_weights = fft_model[module].to(device)
        if update_value:
            compared_weights = compared_weights - state_dict[f'base_model.model.main.{layer_num[i]}.base_layer.weight']
        loraB_weights = state_dict[f'base_model.model.main.{layer_num[i]}.lora_B.default.weight']
        loraA_weights = state_dict[f'base_model.model.main.{layer_num[i]}.lora_A.default.weight']

        lora_combined_weights = loraB_weights @ loraA_weights
        if not update_value:
            lora_combined_weights = 2 * lora_combined_weights + state_dict[f'base_model.model.main.{layer_num[i]}.base_layer.weight']
        
        # distance = torch.norm(after_optimization_weights - weights).item()
        # distance = wasserstein_distance(after_optimization_weights.cpu().numpy().flatten(), weights.cpu().numpy().flatten())
        dot_product = np.dot(compared_weights.flatten().cpu().numpy(), lora_combined_weights.flatten().cpu().numpy()) + 1e-5
        # print(dot_product)
        norm_after = np.linalg.norm(compared_weights.flatten().cpu().numpy())
        norm_weights = np.linalg.norm(lora_combined_weights.flatten().cpu().numpy())
        distance = np.arccos(np.clip(dot_product / (norm_after * norm_weights), -1.0, 1.0)) / np.pi * 180

        # print(f"Layer {layer_num[i]}, angle distance: {distance}")

    return distance
ckpt_number =[0, 1,2,3,4,5,6,7,8,9] + [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
fft_path = "/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/fft_16000_test"
# lora_path = "/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/lora-one_pretrainiter_16000_0_10000_gamma9_grad_reinit"
lora_path = "/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/lora_16000_r32"
angle_list = []
# Load fft model from ckpts
for ckpt_number_ in ckpt_number:
    vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    fft_ckpt = os.path.join(fft_path, f"{ckpt_number_}_up_shift.pth")
    vf.load_state_dict(torch.load(fft_ckpt))

    for j_ckpt_number_ in ckpt_number:
        lora_vf =MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
        lora_vf.load_state_dict(torch.load("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_16000.pth", map_location=device))
        from peft import PeftModel
        lora_vf = PeftModel.from_pretrained(lora_vf, os.path.join(lora_path, f"{j_ckpt_number_}_up_shift"))
        distance = weight_angle(vf, lora_vf, update_value = False)
        angle_list.append(distance)

# Draw a heatmap
angle_matrix = np.array(angle_list).reshape(len(ckpt_number), len(ckpt_number))
angle_value = angle_matrix.mean(1)
plt.figure(figsize=(8, 6))
plt.imshow(angle_matrix, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Angle Distance (degrees)')
plt.xticks(ticks=np.arange(len(ckpt_number)), labels=ckpt_number)
plt.yticks(ticks=np.arange(len(ckpt_number)), labels=ckpt_number)
plt.xlabel('Lora Checkpoint Number')
plt.ylabel('FFT Checkpoint Number')
plt.title('Angle Distance between FFT and Lora Models')
plt.savefig(f'/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/angle_heatmap_{lora_init_mode_list[0]}_gamma{gamma}_grad_reinit.png')
plt.show()

