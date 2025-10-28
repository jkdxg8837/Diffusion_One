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
from flow_matching_utils import segment_MeanFlow
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
is_reinit = True
is_baseline = True
batch_size_scale_list = [2,3,4,5, 6, 7, 8,9,10]
gamma = 64
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
if is_reinit:
    # for batch_size_scale in batch_size_scale_list:
    for segment_point in [0.0]:
        meanflow = segment_MeanFlow(baseline = is_baseline, segment_point=segment_point, is_lora = is_lora, is_reinit = is_reinit, gamma = gamma, reverse=True, data_mode=mode, batch_size_scale = 1)
        meanflow.train()
        # Add fft and lora training
        # meanflow = segment_MeanFlow(baseline = is_baseline, segment_point=segment_point, is_lora = False, is_reinit = False, gamma = gamma, reverse=True, data_mode=mode, batch_size_scale = 1)
        # meanflow.train()
        # meanflow = segment_MeanFlow(baseline = is_baseline, segment_point=segment_point, is_lora = True, is_reinit = False, gamma = gamma, reverse=True, data_mode=mode, batch_size_scale = 1)
        # meanflow.train()
else:
    for segment_point in [0.0, 0.1]:
        meanflow = segment_MeanFlow(baseline = is_baseline, segment_point=segment_point, is_lora = is_lora, is_reinit = is_reinit, gamma = gamma, reverse=True, data_mode=mode, batch_size_scale = 1)
        meanflow.train()