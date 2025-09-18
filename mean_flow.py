import time
import torch
import numpy as np
from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
from torchvision.utils import make_grid, save_image
# visualization
import matplotlib.pyplot as plt

from matplotlib import cm

from peft import LoraConfig, get_peft_model
from sklearn.datasets import make_moons
import sys
sys.path.append("/home/u5649209/workspace/flow_matching")  # Adjust the path as necessary to import flow_matching_utils
from flow_matching_utils import MFMLP, evaluate_result, train_moon_gen, reinit_lora

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
iterations = 20000
print_every = 1000
hidden_dim = 512
gradient_base = 0
gradient_iter = 15000
pretrain_iter = 16000
is_pre_train = True
is_lora = False
is_eval = False 
is_reinit = False
gamma = 100
mode = "new"
loss_history = []
lora_init_mode_list = [\
    "lora-ga", \
    "lora-one", \
    "lora-sb"\
]


from flow_matching_utils import MeanFlow
meanflow = MeanFlow()
def train_process_mean_flow():
    start_time = time.time()
    for i in range(iterations):
        optim.zero_grad() 
        def save_points(points, path):
            points_np = points.detach().cpu().numpy()
            # 画散点图
            plt.figure(figsize=(6, 6))
            plt.scatter(points_np[:, 0], points_np[:, 1], s=1, alpha=0.5)  # s 控制点大小
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Scatter plot of 20,000 points")
            plt.show()
            plt.savefig(path)

        def save_model(step_name = None, image_object = None):
            if step_name is None:
                step_name = i + 1
            if not is_pre_train:
                if is_lora:
                    if is_reinit:
                        save_path = f"/home/u5649209/workspace/flow_matching/ckpts/meanf/{lora_init_mode}_pretrainiter_{pretrain_iter}_{gradient_base}_{gradient_iter}_gamma{gamma}/{step_name}_{mode}"
                        os.makedirs(save_path, exist_ok=True)
                        vf.save_pretrained(save_path)
                        if image_object is not None:
                            os.makedirs(f"{save_path}/images", exist_ok=True)
                            img_save_path = f"{save_path}/images/step_{step_name}.png"
                            save_points(image_object, img_save_path)
                    else:
                        save_path = f"/home/u5649209/workspace/flow_matching/ckpts/meanf/lora_{pretrain_iter}/{step_name}_{mode}"
                        os.makedirs(save_path, exist_ok=True)
                        vf.save_pretrained(save_path)
                        if image_object is not None:
                            os.makedirs(f"{save_path}/images", exist_ok=True)
                            img_save_path = f"{save_path}/images/step_{step_name}.png"
                            save_points(image_object, img_save_path)
                else:
                    save_path = f"/home/u5649209/workspace/flow_matching/ckpts/meanf/fft_{pretrain_iter}/{step_name}_{mode}.pth"
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(vf.state_dict(), save_path)
                    if image_object is not None:
                        os.makedirs(f"{save_path}/images", exist_ok=True)
                        img_save_path = f"{save_path}/images/step_{step_name}.png"
                        save_points(image_object, img_save_path)
            else:
                save_path = f"/home/u5649209/workspace/flow_matching/ckpts/meanf/weights_wo_embeds/raw_model_{step_name}.pth"
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(vf.state_dict(), save_path)
                if image_object is not None:
                    save_path = f"/home/u5649209/workspace/flow_matching/ckpts/meanf/weights_wo_embeds"
                    os.makedirs(f"{save_path}/images-10", exist_ok=True)
                    img_save_path = f"{save_path}/images-10/step_{step_name}.png"
                    save_points(image_object, img_save_path)
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
        optim.step() # update

        
        
        # log loss
        if ((i+1) % print_every == 0) or (i in [0, 1, 2, 3]):
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                .format(i, elapsed*1000/print_every, loss.item())) 
            z = meanflow.sample(vf, 10)
            start_time = time.time()
            save_model(None, z)
if not is_eval:
    vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    if not is_pre_train:
        state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/ckpts/weights/raw_model_{pretrain_iter-1}.pth", map_location=device)
        vf.load_state_dict(state_dict)
    if is_lora == False:
        optim = torch.optim.Adam(vf.parameters(), lr=lr)
        # train
        start_time = time.time()
        path = AffineProbPath(scheduler=CondOTScheduler())
        train_process_mean_flow()
    elif is_reinit == False:
        lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["main.0", "main.2", "main.4", "main.6"],  # target Linear layers in MLP
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
            state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/ckpts/weights/raw_model_{pretrain_iter-1}.pth", map_location=device)
            vf.load_state_dict(state_dict)
            print(f"mode: {mode}, lora_init_mode: {lora_init_mode}, gradient_base: {gradient_base}, gradient_iter: {gradient_iter}, gamma: {gamma}\
                pretrain_iter: {pretrain_iter}")
            
            # reinit the model
            import pickle

            lora_config = LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["main.0", "main.2", "main.4", "main.6"],  # target Linear layers in MLP
                init_lora_weights="gaussian",
            )
            vf = get_peft_model(vf, lora_config)
            
            # with open(f'/home/u5649209/workspace/flow_matching/ckpts/raw_model_gradients/models_grads_step19999_new.pkl', 'rb') as f:
            with open(f'/home/u5649209/workspace/flow_matching/ckpts/raw_model_gradients/fullP{pretrain_iter}_step{gradient_base}_data_new_iter_{gradient_iter}.pkl', 'rb') as f:  # IGNORE
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
