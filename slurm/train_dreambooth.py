import os
import subprocess

# 设置环境变量
os.environ["MODEL_NAME"] = "stabilityai/stable-diffusion-3-medium-diffusers"
os.environ["INSTANCE_DIR"] = "car"
os.environ["OUTPUT_DIR"] = "trained-sd3-lora-car"
# os.environ["LORA_LAYERS"] ="time_text_embed.timestep_embedder.linear_1,time_text_embed.timestep_embedder.linear_2,time_text_embed.text_embedder.linear_1,time_text_embed.text_embedder.linear_2,pos_embed.proj,context_embedder,norm1.linear,norm2.linear,attn.to_k,attn.to_q,attn.to_v,attn.to_out.0,attn.add_k_proj,attn.add_q_proj,attn.add_v_proj,attn.to_add_out,ff.net.0.proj,ff.net.2,norm_out.linear,proj_out"
time_step = 0.2
re_init_schedule = "multi"
re_init_bsz = 1
re_init_samples = 20
noise_samples = 100
# 构造命令
cmd = [
    "accelerate", "launch", "/dcs/pg24/u5649209/data/workspace/diffusers/train_dreambooth_lora_one_sd3.py",
    "--pretrained_model_name_or_path", os.environ.get("MODEL_NAME"), # 使用 os.environ.get 提供默认值以防环境变量未设置
    "--instance_data_dir", os.environ.get("INSTANCE_DIR"),   # 使用 os.environ.get 提供默认值
    "--output_dir", os.environ.get("OUTPUT_DIR")+"_"   + str(re_init_samples) + "bsz" + str(re_init_bsz) + "noise_samples" + str(noise_samples),       # 使用 os.environ.get 提供默认值
    "--mixed_precision", "fp16",
    "--instance_prompt", "a photo of sks car",
    "--resolution", "512",
    "--train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--learning_rate", "4e-4",
    "--report_to", "wandb",
    "--lr_scheduler", "constant",
    "--lr_warmup_steps", "0",
    "--max_train_steps", "100",
    "--validation_prompt", "A photo of sks car in front of a building",
    "--validation_epochs", "25",
    "--seed", "0",
    "--time_step", str(time_step),
    "--re_init_schedule", re_init_schedule,
    "--re_init_bsz", str(re_init_bsz),
    "--re_init_samples", str(re_init_samples),
    # "--baseline"
    "--fixed_noise",
    "--noise_samples", str(noise_samples)
        
            # "--push_to_hub"
]

# 运行命令
subprocess.run(cmd)