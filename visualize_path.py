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
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
is_baseline = False
gamma = 100
mode = "up_shift"
loss_history = []
lora_init_mode_list = [\
    "lora-one", \
    "lora-ga", \
    # "lora-sb"\
]
# Baseline for 20; MeanF for 5 or 1
if is_baseline:
    meanF_step = 20
else:
    meanF_step = 5

from einops import rearrange
class special_MeanFlow(MeanFlow):
    def __init__(self, channels=1, image_size=32, num_classes=10, normalizer=..., flow_ratio=0.5, time_dist=..., cfg_ratio=0.1, cfg_scale=2, cfg_uncond='v', jvp_api='funtorch', baseline=False):
        super().__init__(channels, image_size, num_classes, normalizer, flow_ratio, time_dist, cfg_ratio, cfg_scale, cfg_uncond, jvp_api, baseline)

    
    def sample(self, model, z, sample_steps = 20, device = 'cuda'):
        z_dict = {}
        model.eval()
        # z = torch.randn(20000, 2, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)


            t_ = rearrange(t, "b -> b 1").detach().clone()
            r_ = rearrange(r, "b -> b 1").detach().clone()

            v = model(z, t, r, None)
            z = z - (t_-r_) * v
            z_dict[i] = z.detach().cpu()

        return z_dict
meanflow = special_MeanFlow(baseline = is_baseline)

ckpt_number_list = [1, 2000, 4000, 6000, 8000, 10000]
# Load model
def create_test_mean_flow_model(is_pretrain, is_baseline, is_lora, is_reinit, pretrain_iter,
        lora_init_mode, mode, gradient_base, gradient_iter, gamma, ckpt_number):

    vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    if not is_pretrain:
        if is_baseline:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_{pretrain_iter}.pth", map_location=device)
        else:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{pretrain_iter}.pth", map_location=device)
        vf.load_state_dict(state_dict)
    if is_lora == False:
        # Load FFT model
        if is_baseline:
            print("loading from fft")
            # state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/fft_16000/{ckpt_number}_new.pth", map_location=device)
            # Use stationary model ckpt
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/fft_16000/10000_{mode}.pth", map_location=device)
        else:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/fft_16000/10000_{mode}.pth", map_location=device)
        vf.load_state_dict(state_dict)
        # train
    elif is_reinit == False:
        # LoRA model
        print("loading from lora")
        if is_baseline:
            vf.load_state_dict(torch.load("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_16000.pth", map_location=device))
            path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/lora_16000/{ckpt_number}_{mode}"
        else:
            vf.load_state_dict(torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_16000.pth", map_location=device))
            path = f"/home/u5649209/workspace/flow_matching/meanf/lora_16000/{ckpt_number}_{mode}"
        from peft import PeftModel
        vf = PeftModel.from_pretrained(vf, path)
    elif is_reinit:
        print("loading from reinit")
        if is_baseline:
            vf.load_state_dict(torch.load("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_16000.pth", map_location=device))
            path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/lora-one_pretrainiter_16000_0_10000_gamma9/{ckpt_number}_{mode}"
        else:
            vf.load_state_dict(torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_16000.pth", map_location=device))
            path = f"/home/u5649209/workspace/flow_matching/meanf/lora-one_pretrainiter_16000_0_10000_gamma9/{ckpt_number}_{mode}"
        from peft import PeftModel
        vf = PeftModel.from_pretrained(vf, path)
    else:
        pass
    # only test reinit step1 data
    # else:
    #     vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    #     state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{pretrain_iter}.pth", map_location=device)
    #     vf.load_state_dict(state_dict)
    #     print(f"mode: {mode}, lora_init_mode: {lora_init_mode}, gradient_base: {gradient_base}, gradient_iter: {gradient_iter}, gamma: {gamma}\
    #         pretrain_iter: {pretrain_iter}")

    #     import pickle

    #     lora_config = LoraConfig(
    #         r=2,
    #         lora_alpha=4,
    #         target_modules=["main.0", "main.2", "main.4", "main.6"],
    #         init_lora_weights="gaussian",
    #     )
    #     vf = get_peft_model(vf, lora_config)

    #     with open(f'/home/u5649209/workspace/flow_matching/meanf/weights/fullP{pretrain_iter}_step{gradient_base}_data_new_iter_{gradient_iter}.pkl', 'rb') as f:
    #         named_grad = pickle.load(f)
    #     _ = reinit_lora(vf, gamma, named_grad, init_mode=lora_init_mode, lora_config=lora_config)
    #     for param in vf.parameters():
    #         param.data = param.data.contiguous()
    return vf
from flow_matching_utils import compute_emd_distance
source_x, _ = train_moon_gen(batch_size=20000, device=device, is_pretrain=False, mode=mode)

for ckpt_number in ckpt_number_list:
    is_lora = False
    vf = create_test_mean_flow_model(
        is_pre_train,
        is_baseline,
        is_lora,
        is_reinit,
        pretrain_iter,
        lora_init_mode_list[0],
        mode,
        gradient_base,
        gradient_iter,
        gamma,
        ckpt_number
    )

    # For fft model
    is_lora = True
    is_reinit = False
    vf_2 = create_test_mean_flow_model(
        is_pre_train,
        is_baseline,
        is_lora,
        is_reinit,
        pretrain_iter,
        lora_init_mode_list[1],
        mode,
        gradient_base,
        gradient_iter,
        gamma,
        ckpt_number
    )

    is_reinit = True
    vf_3 = create_test_mean_flow_model(
    is_pre_train,
    is_baseline,
    is_lora,
    is_reinit,
    pretrain_iter,
    "lora-one",
    mode,
    gradient_base,
    gradient_iter,
    gamma,
    ckpt_number
)

    # Using one extra graph for angle

    def weight_angle(fft_model, lora_model):
        layer_num = [0, 2, 4, 6]
        state_dict = lora_model.state_dict()
        fft_model = fft_model.state_dict()
        for i in range(len(layer_num)):
            module = f'main.{layer_num[i]}.weight'
            if 'bias' in module:
                continue
            compared_weights = fft_model[module].to(device)

            loraB_weights = state_dict[f'base_model.model.main.{layer_num[i]}.lora_B.default.weight']
            loraA_weights = state_dict[f'base_model.model.main.{layer_num[i]}.lora_A.default.weight']

            lora_combined_weights = loraB_weights @ loraA_weights

            
            # distance = torch.norm(after_optimization_weights - weights).item()
            # distance = wasserstein_distance(after_optimization_weights.cpu().numpy().flatten(), weights.cpu().numpy().flatten())
            dot_product = np.dot(compared_weights.flatten().cpu().numpy(), lora_combined_weights.flatten().cpu().numpy())
            norm_after = np.linalg.norm(compared_weights.flatten().cpu().numpy())
            norm_weights = np.linalg.norm(lora_combined_weights.flatten().cpu().numpy())
            distance = np.arccos(np.clip(dot_product / (norm_after * norm_weights), -1.0, 1.0)) / np.pi * 180

            print(f"Layer {layer_num[i]}, angle distance: {distance}")

    # Using a set of random noise points as input

    fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
    pred_x_dict = meanflow.sample(vf, fixed_random_noise, meanF_step)

    pred_x_dict_2 = meanflow.sample(vf_2, fixed_random_noise, meanF_step)

    pred_x_dict_3 = meanflow.sample(vf_3, fixed_random_noise, meanF_step)

    # EMD Distance
    # emd_1 = compute_emd_distance(pred_x_dict[meanF_step-1].detach().cpu().numpy(), source_x)
    # emd_2 = compute_emd_distance(pred_x_dict_2[meanF_step-1].detach().cpu().numpy(), source_x)
    # print(f"EMD for fft: {emd_1},   EMD for lora: {emd_2}")



    path_number = 10
    # Visualize the projection between two figs
    fixed_random_noise = fixed_random_noise.detach().cpu().numpy()
    fixed_random_noise = fixed_random_noise[:path_number, :]

    # Visualize two lines in one fig, and 20 subfigs for 20 selected fixed)random_noise

    # pick the step you want to visualize

    data1 = np.array([])
    data2 = np.array([])
    data3 = np.array([])
    # path_point_number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    path_point_number_list = range(meanF_step)
    for path_number_ in range(path_number):
        for path_point_number in path_point_number_list:
            data1 = np.append(data1, pred_x_dict[path_point_number][path_number_].detach().cpu().numpy())
            data2 = np.append(data2, pred_x_dict_2[path_point_number][path_number_].detach().cpu().numpy())

            data3 = np.append(data3, pred_x_dict_3[path_point_number][path_number_].detach().cpu().numpy())
    data1 = data1.reshape((path_number, len(path_point_number_list), 2))
    data2 = data2.reshape((path_number, len(path_point_number_list), 2))

    data3 = data3.reshape((path_number, len(path_point_number_list), 2))

        
    # Add start point
    fixed_random_noise = fixed_random_noise.reshape((path_number, 1, 2))
    data1 = np.concatenate((fixed_random_noise, data1), axis=1)
    data2 = np.concatenate((fixed_random_noise, data2), axis=1)

    data3 = np.concatenate((fixed_random_noise, data3), axis=1)
    def plot(data1, data2, data3 = None):
        # 1. 找到所有数据的全局范围

        all_data = np.concatenate([data1[:10], data2[:10], data3[:10]], axis=0)
        x_min, y_min = all_data[..., 0].min(), all_data[..., 1].min()
        x_max, y_max = all_data[..., 0].max(), all_data[..., 1].max()

        # 2. 计算最大跨度，并增加一点边距 (padding)
        x_range = x_max - x_min
        y_range = y_max - y_min
        max_range = max(x_range, y_range) * 1.01  # 增加1%的边距

        # 3. 计算新的中心点和统一的坐标限制
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        
        xlim = [x_center - max_range / 2, x_center + max_range / 2]
        ylim = [y_center - max_range / 2, y_center + max_range / 2]

        fig, axes = plt.subplots(1, 10, figsize=(30, 4))  # 1行10列 = 10 个子图

        axes = axes.flatten()

        for i in range(10):
            ax = axes[i]
            # 第一条曲线 (来自 data1)
            ax.plot(data1[i, :, 0], data1[i, :, 1], marker='o', label="curve1")
            # 第二条曲线 (来自 data2)
            ax.plot(data2[i, :, 0], data2[i, :, 1], marker='x', label="curve2")

            ax.plot(data3[i, :, 0], data3[i, :, 1], marker='^', label="curve3")
            ax.set_title(f"Sample {i+1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            ax.grid(True)
            ax.set_aspect("equal")  # 保持比例尺一致
            # ax.set_xlim(xlim)            # 设置统一的X轴范围
            # ax.set_ylim(ylim)            # 设置统一的Y轴范围

        plt.tight_layout()
        if is_baseline:
            plt.savefig(f'./10velocitypath_{ckpt_number}_stationary_baseline.png')
        else:   
            plt.savefig(f'./10velocitypath_{ckpt_number}_stationary.png')
        plt.show()

    data1 = data1.reshape((path_number, len(path_point_number_list) + 1, 2))
    data2 = data2.reshape((path_number, len(path_point_number_list) + 1, 2))

    data3 = data3.reshape((path_number, len(path_point_number_list) + 1, 2))
    plot(data1, data2, data3)




# Try to get some qualititative results
# How to measure the divergence bewtween a set of points?