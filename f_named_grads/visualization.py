import torch
import numpy as np
import matplotlib.pyplot as plt
steps = [2, 4, 6, 8, 10, 12]
for step in steps:
    base_path = "/dcs/pg24/u5649209/data/workspace/diffusers/f_named_grads/"
    state_dict = torch.load(f"{base_path}named_grads_{step}_svd.pt")

    attn_groups = {
        "0-7": [],
        "8-15": [],
        "16-23": [],
    }

    for key, value in state_dict.items():
        for i in range(24):
            prefix = f"transformer_blocks.{i}.attn"
            if key.startswith(prefix):
                mean_val = value["S"][:5].mean().item()
                if 0 <= i <= 7:
                    attn_groups["0-7"].append(mean_val)
                elif 8 <= i <= 15:
                    attn_groups["8-15"].append(mean_val)
                elif 16 <= i <= 23:
                    attn_groups["16-23"].append(mean_val)

    # 更高区分度的配色
    colors = {
        "0-7": "#1f77b4",    # 蓝色
        "8-15": "#ff7f0e",   # 橙色
        "16-23": "#2ca02c",  # 绿色
    }

    plt.figure(figsize=(10, 5))

    for group, values in attn_groups.items():
        plt.hist(values, bins=100, alpha=0.6, label=f"Layer {group}", color=colors[group])

    plt.title("Histogram of transformer_blocks.X.attn values (X=0~23)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{base_path}{step}_layer_histogram.png")
    plt.show()
