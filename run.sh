MODEL_NAME="stabilityai/stable-diffusion-3-medium-diffusers" 
INSTANCE_DIR="dog" 
OUTPUT_DIR="trained-sd3-lora-one" 
export NUMPROCESS=1 

acc_args=(
--num_machines 1 \
--num_processes $NUMPROCESS \
--main_process_port 234222 \
--dynamo_backend no \
--mixed_precision bf16 \
--num_cpu_threads_per_process 8 \
#--multi_gpu \
)

train_args=(
  --pretrained_model_name_or_path "$MODEL_NAME" \
  --instance_data_dir "$INSTANCE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --mixed_precision fp16 \
  --instance_prompt "a photo of [sks] cat" \
  --resolution 512 \
  --train_batch_size 3 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --report_to tensorboard \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 100 \
  --validation_prompt "A photo of [sks] cat" \
  --validation_epochs 25 \
  # --seed 0 \
  --time_step 0.2 \
  --rank 32 \
  --re_init_samples 24 \
  --center_crop \
  --repeats 11 \
)

CUDA_VISIBLE_DEVICES=7 accelerate launch "${acc_args[@]}" train_dreambooth_lora_one_sd3.py "${train_args[@]}"

