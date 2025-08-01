import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 假设你的数据是这个格式
with open("group_metrics.json", "r") as f:
    data = pd.read_json(f)

# 构造 DataFrame：行是模块类型，列是 layer index
df = pd.DataFrame(data).T  # 转置，使 module type 为行
df.columns = list(range(df.shape[1]))  # 列命名为 0~15

plt.figure(figsize=(12, 4))
ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="cividis", cbar=True)

# 设置标签
plt.xlabel("Layer Index")
plt.ylabel("Module Type")
plt.title("NFN-Map for Stabel diffusion 3.1", fontsize=12)

# 模仿图中的标题样式
plt.text(0, -1.2, "NFN-Map", fontsize=12, fontweight="bold", color="white",
         bbox=dict(facecolor='purple', boxstyle='round,pad=0.3'))

plt.tight_layout()
plt.show()
plt.savefig("nfn_map.png", dpi=300, bbox_inches='tight')
