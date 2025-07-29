import copy
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
import argparse
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
import sys
from train_cal_S_mean import(
    DreamBoothDataset, 
    PromptDataset,
    _encode_prompt_with_t5, 
    _encode_prompt_with_clip, 
    encode_prompt, 
    tokenize_prompt,
    parse_args,
    collate_fn,
    load_text_encoders,
    import_model_class_from_model_name_or_path
)

import os
import torch

def svd_weights(weights):
    # Only operate on 2D tensors (matrices)
    if weights.ndim == 2:
        U, S, Vh = torch.linalg.svd(weights, full_matrices=False)
        # return {'U': U.cpu(), 'S': S.cpu(), 'V': Vh.cpu()}
        return {'U': None, 'S': S.cpu(), 'V': None}
    return None

def process_pt_file(filepath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(filepath, map_location=device)
    if isinstance(data, dict) and 'state_dict' in data:
        state_dict = data['state_dict']
    elif isinstance(data, dict):
        state_dict = data
    else:
        return {}

    svd_dict = {}
    for name, weights in state_dict.items():
        svd_result = svd_weights(weights.float())
        if svd_result is not None:
            svd_dict[name] = svd_result
    return svd_dict

result_path = "./svd_results.txt"
# Prev singular value cal
output_dict = {}


args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)
# args.output_dir
named_grads_path = args.output_dir+"/named_grads.pt"
print(f'Processing {named_grads_path}')
svd_dict = process_pt_file(named_grads_path)
if svd_dict:  # Only save if there are SVD results
    svd_fname = named_grads_path.replace('.pt', '_s_only.pt')
    torch.save(svd_dict, svd_fname)
