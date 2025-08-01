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

from train_dreambooth_lora_one_sd3 import DreamBoothDataset
from train_dreambooth_lora_one_sd3 import(
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

stable_gamma_list = [49]
ckpt_steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
path=f"/home/u5649209/workspace/Diffusion_One/ckpts/sd3-dog-singlecard-reinit80-randomseed-woprecondition-POSmedium-scaleLR_lr_scale4.5/checkpoint-"
# Prev singular value cal
prev_path = path+"0"
output_dict = {}
prev_lora_state_dict = StableDiffusion3Pipeline.lora_state_dict(prev_path)

args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)

accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    kwargs_handlers=[kwargs],
)

weight_dtype = torch.float32
if accelerator.mixed_precision == "fp16":
    weight_dtype = torch.float16
elif accelerator.mixed_precision == "bf16":
    weight_dtype = torch.bfloat16
pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            revision="main",
            variant="fp16",
            torch_dtype=weight_dtype,
        )
        # load attention processors

pipeline.load_lora_weights(prev_path, lora_scale = 5.7)

from lora_one_utils_nfn import estimate_nfn, get_group_metrics
import yaml
# Construct Temp Set
# Dataset and DataLoaders creation:
train_dataset = DreamBoothDataset(
    instance_data_root=args.instance_data_dir,
    instance_prompt=args.instance_prompt,
    class_prompt=args.class_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_num=args.num_class_images,
    size=args.resolution,
    repeats=args.repeats,
    center_crop=args.center_crop,
    args=args,
)
temp_dataset = DreamBoothDataset(
    instance_data_root=args.instance_data_dir+"4reinit",
    instance_prompt=args.instance_prompt,
    class_prompt=args.class_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_num=args.num_class_images,
    size=args.resolution,
    repeats=args.repeats,
    center_crop=args.center_crop,
    args=args,
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    num_workers=args.dataloader_num_workers,
)
# import correct text encoder classes
text_encoder_cls_one = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision
)
text_encoder_cls_two = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
)
text_encoder_cls_three = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
)
# Load the tokenizers
tokenizer_one = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=args.revision,
)
tokenizer_two = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_2",
    revision=args.revision,
)
tokenizer_three = T5TokenizerFast.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer_3",
    revision=args.revision,
)

noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="scheduler"
)
noise_scheduler_copy = copy.deepcopy(noise_scheduler)
text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
    text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three, args=args
)


text_encoder_one.to(accelerator.device, dtype=weight_dtype)
text_encoder_two.to(accelerator.device, dtype=weight_dtype)
text_encoder_three.to(accelerator.device, dtype=weight_dtype)
temp_dataloader = torch.utils.data.DataLoader(
    temp_dataset,
    batch_size=5,
    shuffle=True,
    collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
    num_workers=args.dataloader_num_workers,
    drop_last=True
)
# Calculate named_grad
named_grads = None
transformer = pipeline.transformer.to("cuda")
vae = pipeline.vae.to("cuda")
vae.to(accelerator.device, dtype=torch.float32)
transformer.to(accelerator.device, dtype=weight_dtype)
for name, param in transformer.named_parameters():
    if "lora" in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
# named_grads = estimate_nfn([transformer, vae], temp_dataloader, args, noise_scheduler_copy, accelerator\
#             , [text_encoder_one, text_encoder_two, text_encoder_three]\
#             , [tokenizer_one, tokenizer_two, tokenizer_three], 1)
with open("nfn_metrics.json", "r") as f:
    nfn_metrics = yaml.safe_load(f)

get_group_metrics(nfn_metrics)