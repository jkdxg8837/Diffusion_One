stable_gamma_list = [49, 81, 121, 169, 225]
import matplotlib.pyplot as plt

import numpy as np
path = "/home/u5649209/workspace/Diffusion_One/Fro_value"
fro_norms_list = []
filename = "fro_norms.npy"
gamma_wise_lower_stage_means = {}
gamma_wise_medium_stage_means = {}
gamma_wise_higher_stage_means = {}
for gamma in stable_gamma_list:
    # Assumes each file is named like "49_fro_norms.py", "81_fro_norms.py", etc.
    filename = f"{gamma}_fro_norms.npy"
    fro_norm = np.load(f"{path}/{filename}", allow_pickle=True).item()

    step_wise_lower_stage_means = np.array([])
    step_wise_medium_stage_means = np.array([])
    step_wise_higher_stage_means = np.array([])

    for step_num in fro_norm.keys():
        lower_stage_means = np.array([])
        medium_stage_means = np.array([])
        higher_stage_means = np.array([])
        layer_fro_norms = fro_norm[step_num]
        for weight_name in layer_fro_norms.keys():
            if "transformer_blocks" not in weight_name:
                continue
            # Extract the block number from the weight name
            block_num = int(weight_name.split('.')[2])
            if block_num < 8:
                lower_stage_means = np.append(lower_stage_means, layer_fro_norms[weight_name])
            elif block_num < 16:
                medium_stage_means = np.append(medium_stage_means, layer_fro_norms[weight_name])
            else:
                higher_stage_means = np.append(higher_stage_means, layer_fro_norms[weight_name])

        step_wise_lower_stage_means = np.append(step_wise_lower_stage_means, np.mean(lower_stage_means) if lower_stage_means.size > 0 else 0)
        step_wise_medium_stage_means = np.append(step_wise_medium_stage_means, np.mean(medium_stage_means) if medium_stage_means.size > 0 else 0)
        step_wise_higher_stage_means = np.append(step_wise_higher_stage_means, np.mean(higher_stage_means) if higher_stage_means.size > 0 else 0)
    gamma_wise_lower_stage_means[gamma] = step_wise_lower_stage_means
    gamma_wise_medium_stage_means[gamma] = step_wise_medium_stage_means
    gamma_wise_higher_stage_means[gamma] = step_wise_higher_stage_means

    plt.figure(figsize=(12, 8))

    ckpt_steps = list(fro_norm.keys())
    plt.plot(ckpt_steps, step_wise_lower_stage_means, label=f'Lower Stage (Gamma {gamma})', color='blue', marker='o')
    plt.plot(ckpt_steps, step_wise_medium_stage_means, label=f'Medium Stage (Gamma {gamma})', color='orange', marker='o')
    plt.plot(ckpt_steps, step_wise_higher_stage_means, label=f'Higher Stage (Gamma {gamma})', color='green', marker='o')

    plt.xlabel('Step')
    plt.ylabel('Mean Frobenius Norm')
    plt.title(f'Stage-wise Mean Frobenius Norms Across Steps for {gamma}  Gamma')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{path}/{gamma}.png")
    plt.close()
pass
            
