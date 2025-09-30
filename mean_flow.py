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
is_lora = True
is_eval = False 
is_reinit = False
is_baseline = True
gamma = 9
mode = "up_shift"
loss_history = []
lora_init_mode_list = [\
    "lora-one", \
    # "lora-ga", \
    # "lora-sb"\
]
# Baseline for 20; MeanF for 5 or 1
if is_baseline:
    meanF_step = 20
else:
    meanF_step = 5

meanflow = MeanFlow(baseline = is_baseline)
def save_points(points_dict, path):
    # Final step visualization
    path_number = 10
    path_point_number_list = range(meanF_step)
    points = points_dict[meanF_step-1]
    points_np = points.detach().cpu().numpy()
    # 画散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(points_np[:, 0], points_np[:, 1], s=1, alpha=0.5)  # s 控制点大小
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter plot of 20,000 points")
    plt.show()
    plt.savefig(path)
    # Now for the path visualization
    plt.clf()
    fig, axes = plt.subplots(1, 10, figsize=(30, 4))  # 1行10列 = 10 个子图
    data1 = np.array([])
    axes = axes.flatten()
    for path_number_ in range(path_number):
        for path_point_number in path_point_number_list:
            data1 = np.append(data1, points_dict[path_point_number][path_number_].detach().cpu().numpy())
    data1 = data1.reshape((path_number, len(path_point_number_list), 2))
    for i in range(10):
        ax = axes[i]
        # 第一条曲线 (来自 data1)
        ax.plot(data1[i][:, 0], data1[i][:, 1], marker='o', label="curve1")
        # 第二条曲线 (来自 data2)
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal")  # 保持比例尺一致
        # ax.set_xlim(xlim)            # 设置统一的X轴范围
        # ax.set_ylim(ylim)            # 设置统一的Y轴范围

    plt.tight_layout()
    process_path = path[:-4] + "_path.png"
    plt.savefig(process_path)
    plt.show()

def save_model(step_name = None, image_object = None, loss_history = None, visualize_path = False):
    if not is_pre_train:
        if is_lora:
            if is_reinit:
                if is_baseline:
                    save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/{lora_init_mode}_pretrainiter_{pretrain_iter}_{gradient_base}_{gradient_iter}_gamma{gamma}_grad_reinit_r32/{step_name}_{mode}"
                else:
                    save_path = f"/home/u5649209/workspace/flow_matching/meanf/{lora_init_mode}_pretrainiter_{pretrain_iter}_{gradient_base}_{gradient_iter}_gamma{gamma}/{step_name}_{mode}"
                os.makedirs(save_path, exist_ok=True)
                vf.save_pretrained(save_path)
                if image_object is not None:
                    if is_baseline:
                        img_save_path =f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/{lora_init_mode}_pretrainiter_{pretrain_iter}_{gradient_base}_{gradient_iter}_gamma{gamma}_grad_reinit_r32/images-{meanF_step}"
                    else:
                        img_save_path =f"/home/u5649209/workspace/flow_matching/meanf/{lora_init_mode}_pretrainiter_{pretrain_iter}_{gradient_base}_{gradient_iter}_gamma{gamma}/images-{meanF_step}"           
            else:
                if is_baseline:
                    save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/lora_{pretrain_iter}_r32/{step_name}_{mode}"
                else:
                    save_path = f"/home/u5649209/workspace/flow_matching/meanf/final_width_compare/lora_{pretrain_iter}_r32/{step_name}_{mode}"
                os.makedirs(save_path, exist_ok=True)
                vf.save_pretrained(save_path)
                if image_object is not None:
                    if is_baseline:
                        img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/lora_{pretrain_iter}_r32/images-{meanF_step}"
                    else:
                        img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/lora_{pretrain_iter}_r32/images-{meanF_step}"
        else:
            if is_baseline:
                save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/fft_{pretrain_iter}_test/{step_name}_{mode}.pth"
            else:
                save_path = f"/home/u5649209/workspace/flow_matching/meanf/final_width_compare/fft_{pretrain_iter}/{step_name}_{mode}.pth"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(vf.state_dict(), save_path)
            if image_object is not None:
                if is_baseline:
                    img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/fft_{pretrain_iter}_test/images-{meanF_step}"
                else:
                    img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/fft_{pretrain_iter}/images-{meanF_step}"    
    else:
        if is_baseline:
            save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/weights/raw_model_{step_name}.pth"
        else:
            save_path = f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{step_name}.pth"
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(vf.state_dict(), save_path)
        if image_object is not None:
            if is_baseline:
                img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/final_width_compare/weights/images-{meanF_step}"
            else:
                img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/weights/images-{meanF_step}"
    if image_object is not None:
        os.makedirs(img_save_path, exist_ok=True)
        save_points(image_object, f"{img_save_path}/step_{step_name}_{mode}.png")
    if step_name != 0 and loss_history is not None:
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history[:])
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        plt.show()
        plt.savefig(f"{img_save_path}/loss_curve.png")
def train_process_mean_flow():
    start_time = time.time()
    fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
    for i in range(iterations):
        optim.zero_grad() 
        
        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
        x_1, y = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=is_pre_train, mode = mode) # sample data
        # print(y)
        x_1 = torch.tensor(x_1).float().to(device)


        # Mean flow insert
        loss, mse_val = meanflow.loss(vf, x_1)

        loss_history.append(loss.item())
        if i == 0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                .format(i+1, elapsed*1000/print_every, loss.item())) 
            start_time = time.time()
            save_model(step_name=0)
        # optimizer step
        loss.backward() # backward
        if i==0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                .format(i, elapsed*1000/print_every, loss.item())) 
            z = meanflow.sample(vf, meanF_step, random_noise = fixed_random_noise)
            start_time = time.time()
            save_model(i+1, z, loss_history)
        optim.step() # update
        # log loss
        if ((i+1) % print_every == 0) or (i in range(10)):
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                .format(i, elapsed*1000/print_every, loss.item())) 
            z = meanflow.sample(vf, meanF_step, random_noise = fixed_random_noise)
            start_time = time.time()
            save_model(i+1, z, loss_history)
        
if not is_eval:
    vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    if not is_pre_train:
        if is_baseline:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_{pretrain_iter}.pth", map_location=device)
        else:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{pretrain_iter}.pth", map_location=device)
        vf.load_state_dict(state_dict)
    if is_lora == False:
        optim = torch.optim.Adam(vf.parameters(), lr=lr)
        # train
        start_time = time.time()
        path = AffineProbPath(scheduler=CondOTScheduler())
        train_process_mean_flow()
    elif is_reinit == False:
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["main.2", "main.4", "main.6"],  # target Linear layers in MLP
            )
        vf = get_peft_model(vf, lora_config)
        optim = torch.optim.Adam(vf.parameters(), lr=lr)
        optim.param_groups[0]['params'] = [p for n, p in vf.named_parameters() if 'lora_' in n]
        # train
        start_time = time.time()
        path = AffineProbPath(scheduler=CondOTScheduler())
        train_process_mean_flow()
    else:
        for lora_init_mode in lora_init_mode_list:
            vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device) 
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/weights/raw_model_{pretrain_iter}.pth", map_location=device)
            if is_baseline:
                state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_{pretrain_iter}.pth", map_location=device)
            vf.load_state_dict(state_dict)
            print(f"mode: {mode}, lora_init_mode: {lora_init_mode}, gradient_base: {gradient_base}, gradient_iter: {gradient_iter}, gamma: {gamma}\
                pretrain_iter: {pretrain_iter}")
            
            # reinit the model
            import pickle

            lora_config = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=["main.2", "main.4", "main.6"],  # target Linear layers in MLP
                init_lora_weights="gaussian",
            )
            vf = get_peft_model(vf, lora_config)
            
            # with open(f'/home/u5649209/workspace/flow_matching/raw_model_gradients/models_grads_step19999_new.pkl', 'rb') as f:
            # with open(f'/home/u5649209/workspace/flow_matching/meanf/weights/fullP{pretrain_iter}_step{gradient_base}_data_new_iter_{gradient_iter}.pkl', 'rb') as f:  # IGNORE
            # with open('/home/u5649209/workspace/flow_matching/meanf/weights/full_sample_up_shift.pkl', 'rb') as f:
            # Using baseline + meanflow
            # with open('/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/old_loraone_up_shift.pkl', 'rb') as f:
            # Using path loss
            with open("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/pretrained_16000_up_shift_path_grad.pkl", 'rb') as f:
                named_grad = pickle.load(f)
            _ = reinit_lora(vf, gamma, named_grad, init_mode = lora_init_mode, lora_config = lora_config)
            for param in vf.parameters():
                param.data = param.data.contiguous()
    
            optim = torch.optim.Adam(vf.parameters(), lr=lr)
            optim.param_groups[0]['params'] = [p for n, p in vf.named_parameters() if 'lora_' in n]
            # train
            start_time = time.time()
            path = AffineProbPath(scheduler=CondOTScheduler())
            train_process_mean_flow()
else:
    vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    if is_pre_train:
        step_list = [0, 1 , 2, 3, 4, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]
    else:
        step_list = [0, 1 , 2, 3, 4, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    for step in step_list:
        if is_pre_train:
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_{step}.pth", map_location=device)
            vf.load_state_dict(state_dict)
            z = meanflow.sample(vf, meanF_step)
            start_time = time.time()
            save_model(step, z)
        else:
            if not is_lora:
                state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/fft_16000/{step}_new.pth", map_location=device)
                vf.load_state_dict(state_dict)
                z = meanflow.sample(vf, meanF_step)
                start_time = time.time()
                save_model(step, z)
            elif not is_reinit:
                state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/lora_{pretrain_iter}/{step}_new", map_location=device)
                vf = get_peft_model(vf, LoraConfig(
                    r=2,
                    lora_alpha=4,
                    target_modules=["main.0", "main.2", "main.4", "main.6"],  # target Linear layers in MLP
                    ))
                vf.load_state_dict(state_dict)
                z = meanflow.sample(vf, meanF_step)
                start_time = time.time()
                save_model(step, z)