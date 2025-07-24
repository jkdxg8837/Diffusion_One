import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


state_dict = torch.load("/home/u5649209/workspace/Diffusion_One/named_grads/100.pt")

svd_results = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
keys = list(state_dict.keys())
for key in tqdm(keys, desc="Processing state_dict"):
    for i in range(24):
        prefix = f"transformer_blocks.{i}.attn"
        if key.startswith(prefix):
            value = state_dict[key].float()
            tensor = value.to(device)
            try:
                U, S, V = torch.linalg.svd(tensor)
                svd_results[key] = {
                    "U": U.cpu(),
                    "S": S.cpu(),
                    "V": V.cpu()
                }
            except Exception as e:
                print(f"SVD failed for {key}: {e}")

torch.save(svd_results, "/home/u5649209/workspace/Diffusion_One/100_svd.pt")


import torch
import numpy as np
import matplotlib.pyplot as plt


# 加载 state_dict
state_dict = torch.load("/home/u5649209/workspace/Diffusion_One/named_grads/20_svd.pt")

# 每个block的平均值（长度为24）
block_avg_values = []

# 对 transformer_blocks.0.attn ~ transformer_blocks.23.attn 分别处理
for i in range(24):
    prefix = f"transformer_blocks.{i}.attn"
    values = []
    for key, value in state_dict.items():
        if key.startswith(prefix):
            val = value["S"][:5].mean().item()
            values.append(val)
    if values:
        block_avg = np.mean(values)
        block_avg_values.append(block_avg)
    else:
        block_avg_values.append(np.nan)  # 没找到时占位

# 按三个段落分组
attn_groups = {
    "0-7": block_avg_values[0:8],
    "8-15": block_avg_values[8:16],
    "16-23": block_avg_values[16:24],
}

# 高区分度配色
colors = {
    "0-7": "#1f77b4",    # 蓝色
    "8-15": "#ff7f0e",   # 橙色
    "16-23": "#2ca02c",  # 绿色
}

# 绘图
plt.figure(figsize=(10, 5))
for group, values in attn_groups.items():
    plt.hist(values, bins=10, alpha=0.7, label=f"Layer {group}", color=colors[group])

plt.title("Histogram of avg transformer_blocks.X.attn values per block (X=0~23)")
plt.xlabel("Mean S[:5] Value (per block)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/u5649209/workspace/Diffusion_One/attn_histogram_grouped.png")
plt.show()




# 用后缀分组，例如 'q_proj', 'k_proj', 'out_proj'
suffix_groups = defaultdict(list)

for key, value in state_dict.items():
    if key.startswith("transformer_blocks.") and ".attn." in key:
        try:
            # 解析 key 的后缀：transformer_blocks.{i}.attn.{suffix}
            suffix = key.split(".attn.")[-1]
            val = value["S"][:5].mean().item()
            suffix_groups[suffix].append(val)
        except Exception as e:
            print(f"Warning: failed to process key {key}: {e}")

# 定义高区分度颜色
import itertools
color_cycle = itertools.cycle([
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf"
])

# 画直方图
plt.figure(figsize=(10, 5))
for suffix, values in suffix_groups.items():
    plt.hist(values, bins=20, alpha=0.6, label=suffix, color=next(color_cycle))

plt.title("Histogram grouped by attn submodules (e.g., q_proj, k_proj, etc.)")
plt.xlabel("Mean S[:5] Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/home/u5649209/workspace/Diffusion_One/attn_histogram_by_suffix_20.png")
plt.show()
