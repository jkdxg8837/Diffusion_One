import os
import subprocess

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "dog"
os.environ["OUTPUT_DIR"] = "/dcs/pg24/u5649209/data/workspace/diffusers/slurm/past_expr/sd3-dog-singlecard-baseline-randomseed"

time_step = 0.2
re_init_schedule = "multi"
re_init_bsz = 1
re_init_samples = 32
noise_samples = 1
stable_gamma = 1
# stable_gamma_list = [289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024,]
stable_gamma_list = [4, 4.5 ,5]
# 构造命令

cmd = [
    "accelerate", "launch", 
    "../train_dreambooth_lora_one_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--lr_scale", str(stable_gamma),
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--rank", "32",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "1e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "300",
    "--validation_prompt", "A photo of sks dog in front of a building",
    "--validation_epochs", "25",
    # "--seed", "0",
    "--time_step", str(time_step),
    "--re_init_schedule", re_init_schedule,
    # "--re_init_bsz", str(re_init_bsz),
    "--re_init_samples", str(re_init_samples),
    "--repeats","8",
    "--reinit_depth", "medium",
    # "--baseline",
    # "--fixed_noise",
    # "--noise_samples", str(noise_samples),
    # "--reinit_only"
            # "--push_to_hub"
]

# 构造命令``
eval_cmd = [
    "accelerate", "launch", "../eval_dreambooth_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "64",
    "--rank", "32",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "1",
    "--max_train_steps", "2",
    "--learning_rate", "5e-4",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--lora_scale", "5.7",
    # "--push_to_hub"
]

named_grads_cmd = [
    "accelerate", "launch", 
    "../cal_S_mean_second_stage.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--lr_scale", str(stable_gamma),
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "512",
    "--rank", "32",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "1e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "300",
    "--validation_prompt", "A photo of sks dog in front of a building",
    "--validation_epochs", "25",
    # "--seed", "0",
    "--time_step", str(time_step),
    "--re_init_schedule", re_init_schedule,
    # "--re_init_bsz", str(re_init_bsz),
    "--re_init_samples", str(re_init_samples),
    "--repeats","8",
    "--reinit_depth", "medium",
    # "--baseline",
    # "--fixed_noise",
    # "--noise_samples", str(noise_samples),
    # "--reinit_only"
            # "--push_to_hub"
]

# Define the checkpoints you want to evaluate
checkpoints = [0, 50, 100, 150, 200, 250, 300]  # Add or remove steps as needed
# Baseline
for step in checkpoints:
    checkpoint_path = f"{cmd[8]}/checkpoint-{step}"
    named_grads_cmd[named_grads_cmd.index("--output_dir") + 1] = checkpoint_path
    subprocess.run(named_grads_cmd)

# for stable_gamma in stable_gamma_list:
#     cmd[8] = os.environ.get("OUTPUT_DIR") + "_" + "lr_scale" + str(stable_gamma)
#     cmd[cmd.index("--lr_scale") + 1] = str(stable_gamma)
#     # subprocess.run(cmd)

#     for step in checkpoints:
#         checkpoint_path = f"{cmd[8]}/checkpoint-{step}"
#         named_grads_cmd[named_grads_cmd.index("--output_dir") + 1] = checkpoint_path
#         subprocess.run(named_grads_cmd)