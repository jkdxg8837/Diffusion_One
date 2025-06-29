import torch

# 创建一个形状为 (1, 16, 16, 64) 的样本张量
example = torch.empty(1, 16, 64, 64)

# 生成 100 个与 example 相同形状的随机张量
noise = torch.stack([torch.randn_like(example) for _ in range(2000)], dim=0)  # 形状为 (100, 1, 16, 16, 64)

# 保存到文件
torch.save(noise.squeeze(1), "noise.pt")
