import os
import subprocess

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "dog"
os.environ["OUTPUT_DIR"] = "trained-sd3-lora-dog-r32-singlecard-real-fixedseed-loraga"

time_step = 0.2
re_init_schedule = "multi"
re_init_bsz = 1
re_init_samples = 32
noise_samples = 1
stable_gamma = 1
# stable_gamma_list = [289, 324, 361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024,]
stable_gamma_list = [361, 400, 441, 484, 529, 576, 625, 676, 729, 784, 841, 900, 961, 1024]
# 构造命令

cmd = [
    "accelerate", "launch", 
    # "--num_machines", "1",
    # "--num_processes", str(os.environ.get("NUMPROCESS", 1)),  # 使用 os.environ.get 提供默认值
    # "--main_process_port", "44444",
    # "--dynamo_backend", "no",
    # "--mixed_precision", "bf16",
    # "--num_cpu_threads_per_process", "8",
    "/dcs/pg24/u5649209/data/workspace/diffusers/train_dreambooth_lora_one_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--stable_gamma", str(stable_gamma),
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
    "--max_train_steps", "50",
    "--validation_prompt", "A photo of sks dog in front of a building",
    "--validation_epochs", "25",
    "--seed", "0",
    "--time_step", str(time_step),
    "--re_init_schedule", re_init_schedule,
    # "--re_init_bsz", str(re_init_bsz),
    "--re_init_samples", str(re_init_samples),
    "--repeats","8",
    "--direction", "LoRA-GA",
    # "--baseline",
    # "--fixed_noise",
    # "--noise_samples", str(noise_samples),
    # "--reinit_only"
            # "--push_to_hub"
]

# 构造命令``
eval_cmd = [
    "accelerate", "launch", "/dcs/pg24/u5649209/data/workspace/diffusers/eval_dreambooth_sd3.py",
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

# Baseline commmands
# subprocess.run(cmd)
# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-1"
# subprocess.run(eval_cmd)

# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-10"
# subprocess.run(eval_cmd)

# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-20"
# subprocess.run(eval_cmd)

# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-30"
# subprocess.run(eval_cmd)

# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-40"
# subprocess.run(eval_cmd)

# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-50"
# subprocess.run(eval_cmd)

# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8]
# subprocess.run(eval_cmd)

# cmd[8] = os.environ.get("OUTPUT_DIR") + "_" + "stable_gamma" + str(stable_gamma)
# cmd[cmd.index("--stable_gamma") + 1] = str(stable_gamma)
# subprocess.run(cmd)


for stable_gamma in stable_gamma_list:
    cmd[8] = os.environ.get("OUTPUT_DIR") + "_" + "stable_gamma" + str(stable_gamma)
    cmd[cmd.index("--stable_gamma") + 1] = str(stable_gamma)
    subprocess.run(cmd)
    eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-0"
    subprocess.run(eval_cmd)
    eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/checkpoint-30"
    subprocess.run(eval_cmd)

    eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8]
    subprocess.run(eval_cmd)
# reinit_weights
# eval_cmd[eval_cmd.index("--output_dir") + 1] = cmd[8] + "/reinit_weights"
# subprocess.run(eval_cmd)
