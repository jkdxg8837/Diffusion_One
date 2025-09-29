import time
import torch
import numpy as np
from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm

from peft import LoraConfig, get_peft_model
from sklearn.datasets import make_moons
import sys
sys.path.append("/home/u5649209/workspace/flow_matching")  # Adjust the path as necessary to import flow_matching_utils
from flow_matching_utils import MFMLP, evaluate_result, train_moon_gen, reinit_lora, MeanFlow

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
iterations = 100
# iterations = 10000
print_every = 50
hidden_dim = 512
gradient_base = 0
gradient_iter = 15000
pretrain_iter = 16000
is_pre_train = False
is_lora = True
is_eval = False 
is_reinit = True
gamma = 100
mode = "up_shift"
loss_history = []
lora_init_mode_list = [\
    "lora-one", \
    "lora-ga", \
    # "lora-sb"
]
is_baseline = True
# velocity field model init
layer_gradients = {}

def save_gradient(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                if n not in record_dict:
                    record_dict[n] = [p.grad.cpu()]
                else:
                    record_dict[n].append(p.grad.cpu())
                p.grad = None
        return grad

    return record_gradient_hook

vf = MFMLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
hooks = []

for name, param in vf.named_parameters():
    if param.requires_grad == True:
        hook = param.register_hook(save_gradient(vf, layer_gradients))
        hooks.append(hook) 
# Load path from which checkpoint
# state_dict_path = f'/home/u5649209/workspace/flow_matching/ckpts/full/{gradient_step}_new.pth'
dir_path = f'/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights'
state_dict_path = f'{dir_path}/raw_model_{pretrain_iter}.pth'
state_dict = torch.load(state_dict_path, map_location=device)
vf.load_state_dict(state_dict)

from einops import rearrange
start_time = time.time()
class special_MeanFlow(MeanFlow):
    
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
            z_dict[i] = z

        return z_dict
meanflow = special_MeanFlow(baseline = is_baseline)
u_list = np.array([])
u_tgt_list = np.array([])
iterations = 1
for i in range(iterations):
    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
    x_1, y = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=is_pre_train, mode = mode) # sample data
    # # print(y)
    x_1 = torch.tensor(x_1).float().to(device)


    # # Mean flow insert
    # # False + is_baseline indicates the raw method of lora-one re-init
    # True indicates using full sample mean flow as re-init
    loss, mse_val, u, u_tgt, end_point, start_point= meanflow.loss(vf, x_1, None, False, True)
    u_list = np.concatenate((u_list, u.detach().cpu().numpy()), axis=0) if u_list.size else u.detach().cpu().numpy()
    u_tgt_list = np.concatenate((u_tgt_list, u_tgt.detach().cpu().numpy()), axis=0) if u_tgt_list.size else u_tgt.detach().cpu().numpy()


    # New gradient method, path loss
    gth_path = np.load('/home/u5649209/workspace/flow_matching/fft_baseline_path.npy')
    gth_path = gth_path[:, 1:, :]   # shape -> (20000, 20, 2)
    gth_path = gth_path.swapaxes(0, 1)
    gth_path = torch.tensor(gth_path).float().to(device)
    fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
    pred_x_dict = meanflow.sample(vf, fixed_random_noise, 20)
    # Convert dict to tensor
    pred_path = []
    for key in pred_x_dict.keys():
        pred_path.append(pred_x_dict[key].unsqueeze(0))
    pred_path = torch.cat(pred_path, dim=0).to(device)  # shape -> (20, 20000, 2)
    loss = nn.MSELoss()(pred_path, gth_path)
    # loss = 0

    loss_history.append(loss.item())
    if i == 0:
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
            .format(i+1, elapsed*1000/print_every, loss.item())) 
        start_time = time.time()

    # optimizer step
    loss.backward() # backward

    save_gradient(vf, layer_gradients)(None)
    
    
    # log loss
    if ((i+1) % print_every == 0) or (i in [0, 1, 2, 3]):
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
            .format(i, elapsed*1000/print_every, loss.item())) 
        start_time = time.time()
    for n, p in vf.named_parameters():
        if p.grad is not None:
            p.grad = None
# Visualization
if is_baseline:
    # Using instant velocity for visualization
    pass
else:
    # Using mean flow velocity for visualization
    pass
from tqdm import tqdm
for key in layer_gradients.keys():
    layer_gradients[key] = torch.stack(layer_gradients[key], dim=0).mean(dim=0)
import pickle

# If using pretrained gradients, use this save
with open(f"{dir_path}/pretrained_{pretrain_iter}_{mode}_path_grad.pkl", "wb") as f:
    pickle.dump(layer_gradients, f)