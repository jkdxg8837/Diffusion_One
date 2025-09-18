gradient_base = 0
gradient_iter = 15000
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
from flow_matching_utils import MLP, evaluate_result

# To avoide meshgrid warning
import warnings
device = 'cuda:0'
mode = "new"
hidden_dim = 512    
warnings.filterwarnings("ignore", category=UserWarning, module='torch')
import pickle
from peft import LoraConfig, get_peft_model, PeftModel
# step_list = [2]
step_list = [1,2,3,4, 50, 100,150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
lora_ga_nll_list = []
lora_one_nll_list = []
lora_sb_nll_list = []
from flow_matching_utils import evaluate_result, MLP

for step in step_list:
    path_list = [
        # f"/home/u5649209/workspace/flow_matching/ckpts/lora-sb_{gradient_base}_{gradient_iter}/{step}_new",
        f"/home/u5649209/workspace/flow_matching/ckpts/lora-ga_{gradient_base}_{gradient_iter}/{step}_new",
        # f"/home/u5649209/workspace/flow_matching/ckpts/lora-one_{gradient_base}_{gradient_iter}/{step}_new",
    ]
    for path in path_list:

        vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
        vf.load_state_dict(torch.load("/home/u5649209/workspace/flow_matching/ckpts/weights/raw_model_19999.pth", map_location=device))
        vf = PeftModel.from_pretrained(vf, path)
            
        _, exact_nll = evaluate_result(vf, data_mode=mode, visualize=False)
        
        if "lora-one" in path:
            lora_one_nll_list.append(exact_nll.cpu().item())
        elif "lora-ga" in path:
            lora_ga_nll_list.append(exact_nll.cpu().item())
        elif "lora-sb" in path:
            lora_sb_nll_list.append(exact_nll.cpu().item())
        else:
            raise NotImplementedError
np.savez('nll_moons_lora_ga.npz', step=step_list, lora_ga_nll=lora_ga_nll_list, lora_one_nll=lora_one_nll_list, lora_sb_nll=lora_sb_nll_list)
