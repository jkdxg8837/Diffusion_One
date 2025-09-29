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
    "lora-ga", \
    # "lora-sb"\
]
# Baseline for 20; MeanF for 5 or 1
meanF_step = 20

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

ckpt_number_list = [0,1, 2000, 4000, 6000, 8000, 10000]
# Load model
def create_test_mean_flow_model(is_pretrain, is_baseline, is_lora, is_reinit, pretrain_iter,
        lora_init_mode, mode, gradient_base, gradient_iter, gamma, ckpt_number, ckpt_path):

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
            if ckpt_path is not None:
                print(f"loading from 1234")
                state_dict = torch.load(f'{ckpt_path}/{ckpt_number}_new.pth', map_location=device)
            else:
                state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/fft_16000/{ckpt_number}_new.pth", map_location=device)
        else:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{pretrain_iter}.pth", map_location=device)
        vf.load_state_dict(state_dict)
        # train
    elif is_reinit == False:
        if is_baseline:
            print("loading from lora")
            vf.load_state_dict(torch.load("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_20000.pth", map_location=device))
            path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/lora_16000/{ckpt_number}_new"
        from peft import PeftModel
        vf = PeftModel.from_pretrained(vf, path)
    else:
        vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
        state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{pretrain_iter}.pth", map_location=device)
        vf.load_state_dict(state_dict)
        print(f"mode: {mode}, lora_init_mode: {lora_init_mode}, gradient_base: {gradient_base}, gradient_iter: {gradient_iter}, gamma: {gamma}\
            pretrain_iter: {pretrain_iter}")

        import pickle

        lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["main.0", "main.2", "main.4", "main.6"],
            init_lora_weights="gaussian",
        )
        vf = get_peft_model(vf, lora_config)

        with open(f'/home/u5649209/workspace/flow_matching/meanf/weights/fullP{pretrain_iter}_step{gradient_base}_data_new_iter_{gradient_iter}.pkl', 'rb') as f:
            named_grad = pickle.load(f)
        _ = reinit_lora(vf, gamma, named_grad, init_mode=lora_init_mode, lora_config=lora_config)
        for param in vf.parameters():
            param.data = param.data.contiguous()
    return vf
from flow_matching_utils import compute_emd_distance
source_x, _ = train_moon_gen(batch_size=20000, device=device, is_pretrain=False, mode="new")

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
        ckpt_number,
        None
    )
    ckpt_path = "/home/u5649209/workspace/flow_matching/meanf/new_baseline/fft_16000_1"
    # For fft model
    is_lora = True
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
        ckpt_number,
        ckpt_path = ckpt_path
    )


    # Using one extra graph for angle
    def weight_angle_2fft(fft_model, fft_model_1, update_value = True):
        layer_num = [0, 2, 4, 6]
        fft_model_1 = fft_model_1.state_dict()
        fft_model = fft_model.state_dict()
        for i in range(len(layer_num)):
            module = f'main.{layer_num[i]}.weight'
            if 'bias' in module:
                continue
            compared_weights = fft_model[module].to(device)

            fft_1_weights = fft_model_1[module].to(device)
            if update_value:

                pre_trained_weights = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_{pretrain_iter}.pth", map_location=device)
                compared_weights = compared_weights - pre_trained_weights[f'main.{layer_num[i]}.weight']
                fft_1_weights = fft_1_weights - pre_trained_weights[f'main.{layer_num[i]}.weight']
            # distance = torch.norm(after_optimization_weights - weights).item()
            # distance = wasserstein_distance(after_optimization_weights.cpu().numpy().flatten(), weights.cpu().numpy().flatten())
            dot_product = np.dot(compared_weights.flatten().cpu().numpy(), fft_1_weights.flatten().cpu().numpy())
            # print(dot_product)
            norm_after = np.linalg.norm(compared_weights.flatten().cpu().numpy())
            norm_weights = np.linalg.norm(fft_1_weights.flatten().cpu().numpy())
            distance = np.arccos(np.clip(dot_product / (norm_after * norm_weights), -1.0, 1.0)) / np.pi * 180

            print(f"Layer {layer_num[i]}, angle distance: {distance}")
    def weight_angle(fft_model, lora_model, update_value = False):
        layer_num = [0, 2, 4, 6]
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
            dot_product = np.dot(compared_weights.flatten().cpu().numpy(), lora_combined_weights.flatten().cpu().numpy())
            # print(dot_product)
            norm_after = np.linalg.norm(compared_weights.flatten().cpu().numpy())
            norm_weights = np.linalg.norm(lora_combined_weights.flatten().cpu().numpy())
            distance = np.arccos(np.clip(dot_product / (norm_after * norm_weights), -1.0, 1.0)) / np.pi * 180

            print(f"Layer {layer_num[i]}, angle distance: {distance}")

    # Using a set of random noise points as input

    fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
    pred_x_dict = meanflow.sample(vf, fixed_random_noise, meanF_step)

    pred_x_dict_2 = meanflow.sample(vf_2, fixed_random_noise, meanF_step)
    # emd_1 = compute_emd_distance(pred_x_dict[meanF_step-1].detach().cpu().numpy(), source_x)
    # emd_2 = compute_emd_distance(pred_x_dict_2[meanF_step-1].detach().cpu().numpy(), source_x)
    # print(f"EMD for fft: {emd_1},   EMD for lora: {emd_2}")
    # Visualize the projection between two figs
    fixed_random_noise = fixed_random_noise.detach().cpu().numpy()
    # for step_index in [0, 4, 9, 14, 19]:
    for step_index in [0]:
    # step_index = 19
        pred_x = pred_x_dict[step_index]

        pred_x_2 = pred_x_dict_2[step_index]
        weight_angle(vf, vf_2, True)
        
        # Raw vis
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].scatter(pred_x[:, 0], pred_x[:, 1], c='blue', alpha=0.6, label='pred_x')
        # axs[0].set_title('pred_x')
        # axs[0].set_xlabel('x')
        # axs[0].set_ylabel('y')
        # axs[0].axis('equal')
        # axs[0].legend()

        # axs[1].scatter(pred_x_2[:, 0], pred_x_2[:, 1], c='red', alpha=0.6, label='pred_x_2')
        # axs[1].set_title('pred_x_2')
        # axs[1].set_xlabel('x')
        # axs[1].set_ylabel('y')
        # axs[1].axis('equal')
        # axs[1].legend()

        # plt.tight_layout()
        # plt.savefig('compare_pred_x.png')
        # plt.show()

        # --- 2. 创建一个包含1行2列子图的Figure ---
        # fig是整个图片，axes是一个包含两个子图(ax1, ax2)的数组
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))

        # --- 3. 操作左边的子图 (axes[0]) ---
        ax1 = axes[0]
        # 绘制输入点 (蓝色)
        
        pred_x = pred_x.detach().cpu().numpy()
        pred_x_2 = pred_x_2.detach().cpu().numpy()
        ax1.scatter(fixed_random_noise[:, 0], fixed_random_noise[:, 1], c='royalblue', label='输入点', alpha=0.7, s=50, edgecolors='w')
        # 绘制第一个输出点 (红色)
        ax1.scatter(pred_x[:, 0], pred_x[:, 1], c='tomato', label='输出点 1', alpha=0.7, s=50, edgecolors='w')

        # 定义需要绘制箭头的点的下标
        indices_to_plot = [0] + list(range(9, 200, 10))

        # 为左图绘制箭头
        for i in indices_to_plot:
            start_point = fixed_random_noise[i]
            end_point = pred_x[i]
            ax1.annotate('', xy=end_point, xytext=start_point,
                        arrowprops=dict(arrowstyle="->", color='green', linewidth=1.5, shrinkA=5, shrinkB=5))

        # 设置左图的样式
        ax1.set_title('Projection between noise and target data', fontsize=16)
        ax1.set_xlabel('x axis', fontsize=12)
        ax1.set_ylabel('y axis', fontsize=12)
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_aspect('equal', adjustable='box')

        # --- 4. 操作右边的子图 (axes[1]) ---
        ax2 = axes[1]
        # 绘制输入点 (蓝色)
        ax2.scatter(fixed_random_noise[:, 0], fixed_random_noise[:, 1], c='royalblue', label='noise input', alpha=0.7, s=50, edgecolors='w')
        # 绘制第二个输出点 (紫色)
        ax2.scatter(pred_x_2[:, 0], pred_x_2[:, 1], c='purple', label='target', alpha=0.7, s=50, edgecolors='w')

        # Calculate Distance between 2 outputs
        distance = np.linalg.norm(pred_x - pred_x_2, axis=1)
        mean_distance = np.mean(distance)
        std_distance = np.std(distance)
        # print(f"Step {step_index}, ckpt {ckpt_number}")
        # print(f"Mean distance between two outputs: {mean_distance}, std: {std_distance}")
        # 为右图绘制箭头
        for i in indices_to_plot:
            start_point = fixed_random_noise[i]
            end_point = pred_x_2[i]
            ax2.annotate('', xy=end_point, xytext=start_point,
                        arrowprops=dict(arrowstyle="->", color='darkorange', linewidth=1.5, shrinkA=5, shrinkB=5))

        # 设置右图的样式
        ax2.set_title('Projection between noise and target data', fontsize=16)
        ax2.set_xlabel('x axis', fontsize=12)
        ax2.set_ylabel('y axis', fontsize=12)
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.set_aspect('equal', adjustable='box')

        # --- 5. 调整布局并保存整个图像 ---
        plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
        # plt.savefig(f'mean_velocity_baseline_vis/fig1_fft_fig2_lora_step{step_index}_ckptnumber_{ckpt_number}.png', dpi=300)






# Try to get some qualititative results
# How to measure the divergence bewtween a set of points?