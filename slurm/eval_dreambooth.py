import os
import subprocess

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "dog"
os.environ["OUTPUT_DIR"] = "/dcs/pg24/u5649209/data/workspace/diffusers/slurm/trained-sd3-lora-dog-r32-singlecard-wnumel-reinit80-randomseed-loraga_stable_gamma144"

# 构造命令``
cmd = [
    "accelerate", "launch", "/dcs/pg24/u5649209/data/workspace/diffusers/eval_dreambooth_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--instance_prompt", "a photo of sks dog",
    "--rank","32",
    "--resolution", "64",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--max_train_steps", "2",
    "--learning_rate", "5e-4",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--lora_scale", "1.0",
    # "--push_to_hub"
]

# 运行命令
subprocess.run(cmd)
