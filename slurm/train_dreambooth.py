import os
import subprocess

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "dog"
os.environ["OUTPUT_DIR"] = "trained-sd3-lora-multi-timestep"
time_step = 0.2
re_init_schedule = "multi"
re_init_bsz = 1
re_init_samples = 20
# 构造命令
cmd = [
    "accelerate", "launch", "/dcs/pg24/u5649209/data/workspace/diffusers/train_dreambooth_lora_one_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR")+"_"   + str(re_init_samples) + "bsz" + str(re_init_bsz),       # 使用 os.environ.get 提供默认值
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "4e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "100",
    "--validation_prompt", "A photo of sks dog in a bucket",
    "--validation_epochs", "25",
    "--seed", "0",
    "--time_step", str(time_step),
    "--re_init_schedule", re_init_schedule,
    "--re_init_bsz", str(re_init_bsz),
    "--re_init_samples", str(re_init_samples)
    # "--push_to_hub"
]

# 运行命令
subprocess.run(cmd)