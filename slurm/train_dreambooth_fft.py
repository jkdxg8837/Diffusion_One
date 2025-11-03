import os
import subprocess

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "dog"
# os.environ["OUTPUT_DIR"] = "sd3-dog-singlecard-reinit80-randomseed-woprecondition-POS-crossAtt-scaleLR"
os.environ["OUTPUT_DIR"] = "dog-fft"
# 构造命令

cmd = [
    "accelerate", "launch", 
    # "--num_machines", "1",
    # "--num_processes", str(os.environ.get("NUMPROCESS", 1)),  # 使用 os.environ.get 提供默认值
    # "--main_process_port", "44444",
    # "--dynamo_backend", "no",
    # "--mixed_precision", "bf16",
    # "--num_cpu_threads_per_process", "8",
    "../train_dreambooth_lora_one_sd3_fft.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR"),       # 使用 os.environ.get 提供默认值
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks dog",
    "--resolution", "1024",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "1e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "4",
    "--validation_prompt", "A photo of sks dog in front of a building",
    "--validation_epochs", "25",
    "--seed", "0",

    # "--re_init_bsz", str(re_init_bsz),

    # "--baseline",
    # "--fixed_noise",
    # "--noise_samples", str(noise_samples),
    # "--reinit_only"
            # "--push_to_hub"
]

# 构造命令
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


subprocess.run(cmd)
