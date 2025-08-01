#!/bin/bash

# List of checkpoint folders
checkpoints=(0 50 100 150 200 250 300)

# Base directory containing the checkpoints
base_dir="/dcs/pg24/u5649209/data/workspace/diffusers/slurm/past_expr/sd3-dog-singlecard-reinit80-randomseed-woprecondition-POSmedium-scaleLR_lr_scale4.5"

# Remote destination
remote_user="u5649209"
remote_host="urus.dcs.warwick.ac.uk"
remote_base="/home/u5649209/workspace/Diffusion_One/named_grads/scale_lr/4.5"

for ckpt in "${checkpoints[@]}"; do
    folder="checkpoint-$ckpt"
    src_path="$base_dir/$folder/named_grads_s_only.pt"
    if [[ -f "$src_path" ]]; then
        # Create the corresponding folder on the remote side
        ssh urus "mkdir -p '$remote_base/$folder'"
        # Copy the file, preserving the folder structure
        scp "$src_path" "urus:$remote_base/$folder/"
    else
        echo "Warning: $src_path does not exist."
    fi
done