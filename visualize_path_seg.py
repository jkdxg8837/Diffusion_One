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
from flow_matching_utils import segment_MeanFlow as MeanFlow
# To avoide meshgrid warning
import warnings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
mode = "up_down_shift"
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
# lora-one
# ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg/lora_one_gamma9_seg_{seg_ratio}_reverse/ckpt"
# lora
# ckpt_path =f"/home/u5649209/workspace/flow_matching/meanf/seg/lora_seg_{seg_ratio}/ckpt"
# meanflow = MeanFlow(baseline = is_baseline)


# ckpt_number_list = [1, 10, 200, 400, 1000, 5000, 10000]
ckpt_number_list = [1]
from flow_matching_utils import compute_emd_distance
source_x, _ = train_moon_gen(batch_size=20000, device=device, is_pretrain=False, mode=mode)
fft_data_target_filter_dict = {}
lora_data_target_filter_dict = {}
lora_one_data_target_filter_dict = {}
fft_filter_idx = None
lora_filter_idx = None
lora_one_filter_idx = None
for ckpt_number in ckpt_number_list:
    # Load fft
    fft_model_mf = MeanFlow(baseline = True, segment_point=0.0, is_lora = False, is_reinit = False, gamma = 9, reverse=True)
    ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/fft_seg_0.0_up_down_shift_100/ckpt"
    path = f"{ckpt_path}/{ckpt_number}"
    # path = f"{ckpt_path}/10000"
    print(path)
    fft_model_mf.load_model(path, is_lora = False)
    # Load Lora
    lora_model_mf = MeanFlow(baseline = True, segment_point=0.0, is_lora = False, is_reinit = False, gamma = 9, reverse=False)
    ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_seg_0.0_up_down_shift_100/ckpt"
    path = f"{ckpt_path}/{ckpt_number}"
    print(path)
    lora_model_mf.load_model(path, is_lora = True)
    # Load Lora+Lora-one
    lora_one_mf = MeanFlow(baseline = True, segment_point=0.0, is_lora = False, is_reinit = False, gamma = 9, reverse=True)
    ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_one_gamma4_seg_0.0_reverse_up_down_shift_100/ckpt"
    path = f"{ckpt_path}/{ckpt_number}"
    print(path)
    lora_one_mf.load_model(path, is_lora = True)
    # Load Seg-lora
    lora_ga_mf = MeanFlow(baseline = True, segment_point=0.0, is_lora = False, is_reinit = False, gamma = 9, reverse=True)
    ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_ga_gamma64_seg_0.0_reverse_up_down_shift_100/ckpt"
    path = f"{ckpt_path}/{ckpt_number}"
    print(path)
    lora_ga_mf.load_model(path, is_lora = True)
    # Load Seg-lora+Lora-one
    seg_loraone_mf = MeanFlow(baseline = True, segment_point=0.1, is_lora = False, is_reinit = False, gamma = 9, reverse=True)
    ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/lora-one/lora_one_gamma9_seg_0.1_reverse/ckpt"
    path = f"{ckpt_path}/{ckpt_number}"
    print(path)
    seg_loraone_mf.load_model(path, is_lora = True)
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
            lora_combined_weights = 2.0 * lora_combined_weights + state_dict[f'base_model.model.main.{layer_num[i]}.base_layer.weight']
            
            # distance = torch.norm(after_optimization_weights - weights).item()
            # distance = wasserstein_distance(after_optimization_weights.cpu().numpy().flatten(), weights.cpu().numpy().flatten())
            dot_product = np.dot(compared_weights.flatten().cpu().numpy(), lora_combined_weights.flatten().cpu().numpy())
            norm_after = np.linalg.norm(compared_weights.flatten().cpu().numpy())
            norm_weights = np.linalg.norm(lora_combined_weights.flatten().cpu().numpy())
            distance = np.arccos(np.clip(dot_product / (norm_after * norm_weights), -1.0, 1.0)) / np.pi * 180

            print(f"Layer {layer_num[i]}, angle distance: {distance}")
    def single_weight_angle(weight1, weight2):
        dot_product = np.dot(weight1.flatten().cpu().numpy(), weight2.flatten().cpu().numpy())
        norm_after = np.linalg.norm(weight1.flatten().cpu().numpy())
        norm_weights = np.linalg.norm(weight2.flatten().cpu().numpy())
        distance = np.arccos(np.clip(dot_product / (norm_after * norm_weights), -1.0, 1.0)) / np.pi * 180

        return distance
    # Using a set of random noise points as input

    fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)



    pred_x_dict = fft_model_mf.sample(meanF_step, "cuda", fixed_random_noise, )

    pred_x_dict_2 = lora_model_mf.sample(meanF_step, "cuda", fixed_random_noise, )
    pred_x_dict_3 = lora_one_mf.sample(meanF_step, "cuda", fixed_random_noise, )
    # pred_x_dict_4 = seg_loraone_mf.sample(meanF_step, "cuda", fixed_random_noise, )
    pred_x_dict_4 = lora_ga_mf.sample(meanF_step, "cuda", fixed_random_noise, )

    # EMD Distance
    # emd_1 = compute_emd_distance(pred_x_dict[meanF_step-1].detach().cpu().numpy(), source_x)
    # emd_2 = compute_emd_distance(pred_x_dict_2[meanF_step-1].detach().cpu().numpy(), source_x)
    # print(f"EMD for fft: {emd_1},   EMD for lora: {emd_2}")



    path_number = 10000
    # Visualize the projection between two figs
    fixed_random_noise = fixed_random_noise.detach().cpu().numpy()
    # Remove some idx in fixed_random_noise
    # removed_idx = [2, 5, 6, 7, 8, 9, 10, 15]
    # fixed_random_noise = np.delete(fixed_random_noise, removed_idx, axis=0)
    fixed_random_noise = fixed_random_noise[:path_number, :]

    # Visualize two lines in one fig, and 16 subfigs for 16 selected fixed)random_noise

    # pick the step you want to visualize

    data1 = np.array([])
    data2 = np.array([])
    data3 = np.array([])
    data4 = np.array([])
    data5 = np.array([])
    # path_point_number_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    path_point_number_list = range(meanF_step)
    # for path_number_ in range(path_number + 8):

    # for path_number_ in range(path_number):
    #     for path_point_number in path_point_number_list:
    #         data1 = np.append(data1, pred_x_dict[path_point_number][path_number_].detach().cpu().numpy())
    #         data2 = np.append(data2, pred_x_dict_2[path_point_number][path_number_].detach().cpu().numpy())
    #         data3 = np.append(data3, pred_x_dict_3[path_point_number][path_number_].detach().cpu().numpy())
    #         data4 = np.append(data4, pred_x_dict_4[path_point_number][path_number_].detach().cpu().numpy())
    #         data5 = np.append(data5, pred_x_dict_5[path_point_number][path_number_].detach().cpu().numpy())
    data_list = [[], [], [], []]  # 分别对应 data1~data4

    for path_number_ in range(path_number):
        for path_point_number in path_point_number_list:
            for i, pred_dict in enumerate([pred_x_dict, pred_x_dict_2, pred_x_dict_3, pred_x_dict_4]):
                data_list[i].append(pred_dict[path_point_number][path_number_].detach().cpu().numpy())

    # 循环结束后一次性合并
    data1, data2, data3, data4 = [np.concatenate(d) for d in data_list]

    data1 = data1.reshape((path_number, len(path_point_number_list), 2))
    data2 = data2.reshape((path_number, len(path_point_number_list), 2))
    data3 = data3.reshape((path_number, len(path_point_number_list), 2))
    data4 = data4.reshape((path_number, len(path_point_number_list), 2))

        
    # Add start point
    fixed_random_noise = fixed_random_noise.reshape((path_number, 1, 2))
    data1 = np.concatenate((fixed_random_noise, data1), axis=1)
    data2 = np.concatenate((fixed_random_noise, data2), axis=1)
    data3 = np.concatenate((fixed_random_noise, data3), axis=1)
    data4 = np.concatenate((fixed_random_noise, data4), axis=1)
    # np.save(f"fft_baseline_path.npy", data1)
    # print("save done")
    # break
    def normalize_x_y(data1, data2, data3, data4):
        for i in range(data1.shape[0]):
            plot_1 = data1[i]
            plot_2 = data2[i]
            plot_3 = data3[i]
            plot_4 = data4[i]

            all_data = np.concatenate([plot_1, plot_2, plot_3, plot_4], axis=0)
            x_min, y_min = all_data[..., 0].min(), all_data[..., 1].min()
            x_max, y_max = all_data[..., 0].max(), all_data[..., 1].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_amplify = 1.0 / x_range * 3.0
            y_amplify = 1.0 / y_range * 3.0
            data1[i,:,0] = data1[i,:,0] * x_amplify
            data1[i,:,1] = data1[i,:,1] * y_amplify
            data2[i,:,0] = data2[i,:,0] * x_amplify
            data2[i,:,1] = data2[i,:,1] * y_amplify
            data3[i,:,0] = data3[i,:,0] * x_amplify
            data3[i,:,1] = data3[i,:,1] * y_amplify
            data4[i,:,0] = data4[i,:,0] * x_amplify
            data4[i,:,1] = data4[i,:,1] * y_amplify
            # Shift to 0 to 3
            # data1[i,:,0] = data1[i,:,0] - data1[i,0,0]
            # data1[i,:,1] = data1[i,:,1] - data1[i,0,1]
            # data2[i,:,0] = data2[i,:,0] - data2[i,0,0]
            # data2[i,:,1] = data2[i,:,1] - data2[i,0,1]
            # data3[i,:,0] = data3[i,:,0] - data3[i,0,0]
            # data3[i,:,1] = data3[i,:,1] - data3[i,0,1]
            # data4[i,:,0] = data4[i,:,0] - data4[i,0,0]
            # data4[i,:,1] = data4[i,:,1] - data4[i,0,1]

        return data1, data2, data3, data4
    


    def relative_angle(reference_data, g1_data):
        # Calculate relative angles between 2 sets of points
        angles_list = []
        for i in range(reference_data.shape[0]):
            angles = []
            ref_points = reference_data[i]
            g1_points = g1_data[i]
            total_angle = 0
            for j in range(1, ref_points.shape[0]):
                vec1 = ref_points[j] - ref_points[j-1]
                vec2 = g1_points[j] - g1_points[j-1]
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                cos_angle = dot_product / (norm_vec1 * norm_vec2 + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure within valid range for arccos
                angle = np.arccos(cos_angle)
                total_angle += angle
                angles.append(total_angle)
            angles_list.append(angles)
        return angles_list
    def relative_angle_customize(reference_data, g1_data, end_point):
        # Calculate relative angles between 2 sets of points
        angles_list = []
        for i in range(reference_data.shape[0]):
            angles = []
            ref_points = reference_data[i]
            g1_points = g1_data[i]
            total_angle = 0
            for j in range(1, end_point):
                vec1 = ref_points[j] - ref_points[j-1]
                vec2 = g1_points[j] - g1_points[j-1]
                dot_product = np.dot(vec1, vec2)
                norm_vec1 = np.linalg.norm(vec1)
                norm_vec2 = np.linalg.norm(vec2)
                cos_angle = dot_product / (norm_vec1 * norm_vec2 + 1e-8)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure within valid range for arccos
                angle = np.arccos(cos_angle)
                total_angle += angle
                angles.append(total_angle)

                # Using 2-norm distance as distance metric
                # dist = np.linalg.norm(ref_points[j] - g1_points[j])
                # total_angle += dist
                # angles.append(total_angle)
            angles_list.append(angles)
        return angles_list

    def plot(data1, data2, data3 = None, data4 = None):
        # 1. 找到所有数据的全局范围
        all_data = np.concatenate([data1[:path_number], data2[:path_number], data3[:path_number], data4[:path_number]], axis=0)

        # fig, axes = plt.subplots(1, path_number, figsize=(30 / 10 * path_number, 4 / 10 * path_number))  # 1行10列 = 10 个子图
        fig, axes = plt.subplots(4, 4, figsize=(10 / 10 * path_number, 10 / 10 * path_number))  # 1行10列 = 10 个子图
        axes = axes.flatten()

        # Plot inference paths
        for i in range(path_number):
            ax = axes[i]
            data1, data2, data3, data4 = normalize_x_y(data1, data2, data3, data4)
            ax.plot(data1[i, :, 0], data1[i, :, 1], marker='o', label="FFT")
            ax.plot(data2[i, :, 0], data2[i, :, 1], marker='x', label="Lora")
            ax.plot(data3[i, :, 0], data3[i, :, 1], marker='^', label="Failed Lora-One")
            ax.plot(data4[i, :, 0], data4[i, :, 1], marker='s', label="Seg Lora-One")
            
            # ax.plot(data1[i, :, 0], data1[i, :, 1], marker='o')
            # ax.plot(data2[i, :, 0], data2[i, :, 1], marker='x')
            # ax.plot(data3[i, :, 0], data3[i, :, 1], marker='^')
            # ax.plot(data4[i, :, 0], data4[i, :, 1], marker='s')
            ax.set_title(f"Sample {i+1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            ax.grid(True)
            ax.set_aspect("equal")

            # ax.set_xlim(xlim)            # 设置统一的X轴范围
            # ax.set_ylim(ylim)            # 设置统一的Y轴范围

        plt.tight_layout()
        if is_baseline:
            plt.savefig(f'./fig1/{path_number}velocitypath_{ckpt_number}_base.png')
        else:   
            plt.savefig(f'./fig1/{path_number}velocitypath_{ckpt_number}.png')
        plt.show()
    def plot_diff(data1, data2, data3 = None, data4 = None):
        # 1. 找到所有数据的全局范围
        all_data = np.concatenate([data1[:path_number], data2[:path_number], data3[:path_number], data4[:path_number]], axis=0)
        

        # fig, axes = plt.subplots(1, path_number, figsize=(30 / 10 * path_number, 4 / 10 * path_number))  # 1行10列 = 10 个子图
        fig, axes = plt.subplots(4, 4, figsize=(10 / 10 * path_number, 10 / 10 * path_number))  # 1行10列 = 10 个子图
        axes = axes.flatten()
        category_label = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
        # Plot inference paths
        # 16 x 20+1 x 2
        # 先不normalize未免影响relative angle的效果
        # data1, data2, data3, data4 = normalize_x_y(data1, data2, data3, data4)
        data_m1 = relative_angle_customize(data1, data2, 2)
        # set mean to data_m1
        mean_data_m1 = np.mean([data_m1[i][-1] for i in range(len(data_m1))])
        
        data_m2 = relative_angle_customize(data1, data3, 2)
        mean_data_m2 = np.mean([data_m2[i][-1] for i in range(len(data_m2))])
        data_m3 = relative_angle_customize(data1, data4, 2)
        mean_data_m3 = np.mean([data_m3[i][-1] for i in range(len(data_m3))])
        print(f"Mean relative angle up to step 2: Lora: {mean_data_m1}, Lora-One: {mean_data_m2}, Lora-ga: {mean_data_m3}")

        # print(data_m2[0])
        # print(data_m3[0])
        # for i in range(path_number):
        #     ax = axes[i]
            
            
        #     ax.bar(category_label, data_m1[i], color='b', label="Lora")
        #     ax.bar(category_label, data_m2[i], color='g', label="Failed Lora-One")
        #     # ax.bar(category_label, data_m3[i], color='r', label="Seg Lora-One")

        #     # ax.plot(data1[i, :, 0], data1[i, :, 1], marker='o')
        #     # ax.plot(data2[i, :, 0], data2[i, :, 1], marker='x')
        #     # ax.plot(data3[i, :, 0], data3[i, :, 1], marker='^')
        #     # ax.plot(data4[i, :, 0], data4[i, :, 1], marker='s')
        #     ax.set_title(f"Sample {i+1}")
        #     ax.set_xlabel("x")
        #     ax.set_ylabel("y")
        #     ax.legend()
        #     ax.grid(True)
        #     ax.set_aspect("equal")

        #     # ax.set_xlim(xlim)            # 设置统一的X轴范围
        #     # ax.set_ylim(ylim)            # 设置统一的Y轴范围

        # plt.tight_layout()
        # if is_baseline:
        #     plt.savefig(f'./fig1/{path_number}diff_plot{ckpt_number}_baseline.png')
        # else:   
        #     plt.savefig(f'./fig1/{path_number}diff_plot{ckpt_number}_dynamic.png')
        # plt.show()
        return data_m1, data_m2, data_m3
    def plot_grid(data1, data2, data3, data_m2, data_m3, data_m4):
        # 2 for lora, 3 for failed lora-one, 4 for seg-lora-one
        x_edges = np.arange(-1, 2.0, 0.05)
        y_edges = np.arange(-0.5, 1.5, 0.05)
        empty_data_m = np.zeros(data_m2.shape[0])
        # handle data_m
        # Set value to 1 if data_m2[i] < data_m3[i]
        for i in range(len(data_m2)):
            if data_m3[i] > data_m4[i]:
                empty_data_m[i] = 1
            else:
                empty_data_m[i] = 0


        heatmap_sum, _, _ = np.histogram2d(data1[:, 0], data1[:, 1],
                                   bins=[x_edges, y_edges],
                                   weights=empty_data_m)

        # 同时统计每个格子中点的数量，以便算平均
        heatmap_count, _, _ = np.histogram2d(data1[:, 0], data1[:, 1],
                                            bins=[x_edges, y_edges])

        # 避免除零错误
        heatmap_avg = np.divide(heatmap_sum, heatmap_count,
                                out=np.zeros_like(heatmap_sum),
                                where=heatmap_count > 0)

        # ---------------------------
        # Step 3: 绘制热图
        # ---------------------------
        plt.figure(figsize=(8, 6))
        # extent 指定坐标范围
        plt.imshow(
            heatmap_avg.T,
            origin='lower',
            extent=[-1, 2.0, -0.5, 1.5],
            cmap='inferno',   # 颜色越深值越大
            aspect='auto'
        )
        plt.colorbar(label='Average Value')
        plt.title('2D Heatmap of data_m1 over [-1, 2.0]×[-0.5, 1.5]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        plt.savefig(f'./fig1-1/{path_number}grid_plot{ckpt_number}_m3_newone_is_better_higher_half.png')
    def point_to_polyline_distance(point, polyline):
        """
        计算一个点到由若干点组成的折线的最短距离。

        参数:
            point: (2,) ndarray
            polyline: (N, 2) ndarray

        返回:
            min_dist: float, 最短距离
            closest_point: (2,) ndarray, 折线上距离该点最近的点坐标
        """
        p = np.array(point)
        min_dist = float('inf')
        closest_point = None

        for i in range(len(polyline) - 1):
            a = polyline[i]
            b = polyline[i + 1]
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = np.clip(t, 0, 1)  # 限制在线段内
            projection = a + t * ab  # 投影点
            dist = np.linalg.norm(p - projection)
            if dist < min_dist:
                min_dist = dist
                closest_point = projection

        return min_dist, closest_point
    from tqdm import tqdm
    def plot_target_filter(data):
        target_point = data[:, -1, :]
        data_points, _ = train_moon_gen(4000, device=device, is_pretrain=False, mode="up_shift")
        distance_list = []
        for idx in tqdm(range(2000)):
            distance, _ = point_to_polyline_distance(target_point[idx], data_points)
            distance_list.append(distance)
        distance_list = np.array(distance_list)
        
        sorted_idx = np.argsort(distance_list)
        filtered_idx = sorted_idx[:100]
        # Shuffle sorted_idx
        np.random.seed(42)
        np.random.shuffle(filtered_idx)
        # filtered_idx.sort()
        # filtered_data = data[filtered_idx]
        return filtered_idx
    
    data1 = data1.reshape((path_number, len(path_point_number_list) + 1, 2))
    data2 = data2.reshape((path_number, len(path_point_number_list) + 1, 2))

    data3 = data3.reshape((path_number, len(path_point_number_list) + 1, 2))
    data4 = data4.reshape((path_number, len(path_point_number_list) + 1, 2))

    data_m2, data_m3, data_m4 = plot_diff(data1,data2, data3, data4)
    # plot_grid(data1[:, 0, :], data2[:, 0, :], data3[:, 0, :], np.array(data_m2)[:,15], np.array(data_m3)[:,15], np.array(data_m4)[:,15])
    # visualize data1
    # if ckpt_number == 10000:
    #     fft_filter_idx = plot_target_filter(data1)
    #     # lora_filter_idx = plot_target_filter(data2)
    #     # lora_one_filter_idx = plot_target_filter(data3)
    #     fft_data_target_filter_dict[ckpt_number] = data1[fft_filter_idx]
    #     lora_data_target_filter_dict[ckpt_number] = data2[fft_filter_idx]
    #     lora_one_data_target_filter_dict[ckpt_number] = data3[fft_filter_idx]
    # else:
    #     fft_data_target_filter_dict[ckpt_number] = data1[fft_filter_idx]
    #     lora_data_target_filter_dict[ckpt_number] = data2[fft_filter_idx]
    #     lora_one_data_target_filter_dict[ckpt_number] = data3[fft_filter_idx]

    # plot(data1, data2, data3, data4)

def plot_target_visualize(data_dict, name):
    """
    data_dict:
        key: ckpt 名称
        value: shape = [20, N, 2]
    功能:
        画出 16 个子图，每个子图对应同一个 path 的不同 ckpt 曲线
    """
    os.makedirs('./fig_target', exist_ok=True)
    path_number = 16  # 前 16 条路径

    ckpt_names = list(data_dict.keys())
    n_ckpt = len(ckpt_names)

    # 创建 4x4 网格的子图
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes = axes.flatten()

    for i in range(path_number):
        ax = axes[i]
        for ckpt, data in data_dict.items():
            ax.plot(data[i, :, 0], data[i, :, 1], label=ckpt)
        ax.set_title(f'Path {i}', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # 仅在左上角几个子图显示图例，避免太密集
        if i == 0:
            ax.legend(fontsize=8)

    # 调整布局
    plt.suptitle(f'Target Visualization - {name}', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'./fig_target/target_visualize_{name}_grid_up_down_shift.png', dpi=300)
    plt.show()
# plot_target_visualize(fft_data_target_filter_dict, "FFT")
# plot_target_visualize(lora_data_target_filter_dict, "Lora")
# plot_target_visualize(lora_one_data_target_filter_dict, "Lora-One")


# Try to get some qualititative results
# How to measure the divergence bewtween a set of points?