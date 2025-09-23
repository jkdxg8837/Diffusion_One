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
iterations = 10000
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
# is_baseline = False
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


start_time = time.time()
meanflow = MeanFlow(baseline = False)
for i in range(iterations):
    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
    x_1, y = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=is_pre_train, mode = mode) # sample data
    # print(y)
    x_1 = torch.tensor(x_1).float().to(device)


    # Mean flow insert
    loss, mse_val = meanflow.loss(vf, x_1, None, True)

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

from tqdm import tqdm
for key in layer_gradients.keys():
    layer_gradients[key] = torch.stack(layer_gradients[key], dim=0).mean(dim=0)
import pickle

# If using pretrained gradients, use this save
with open(f"{dir_path}/pretrained_{pretrain_iter}_{mode}.pkl", "wb") as f:
    pickle.dump(layer_gradients, f)