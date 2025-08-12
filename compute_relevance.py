import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
lr = 0.001
batch_size = 4096
iterations = 20001
print_every = 2000 
hidden_dim = 512
# Model class
# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim+time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )
    

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)
if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')
torch.manual_seed(42)
with open('raw_models_grads.pkl', 'rb') as f:
    raw_models_grads = pickle.load(f)

checkpoint_steps = [1999,7999,13999,19999]
for ckpt in checkpoint_steps:
    weight_path = f'/home/u5649209/workspace/flow_matching/ckpts/raw_models/raw_model_{ckpt}.pth'
    vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    original_weights = vf.state_dict()
    loaded_weights = torch.load(weight_path, map_location=device)
    distance_module_dict = {}
    learning_rate_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
    for step in raw_models_grads.keys():
        step_gradients = raw_models_grads[step]
        for module in step_gradients:
            if 'bias' in module:
                continue
            weights = loaded_weights[module].to(device)
            gradients = step_gradients[module][0].to(device)
            for lr in learning_rate_list:
                # Reset model weights
                ori_module_weights= original_weights[module].to(device)
                after_optimization_weights = ori_module_weights + lr * gradients[0]
                # distance = torch.norm(after_optimization_weights - weights).item()
                # distance = wasserstein_distance(after_optimization_weights.cpu().numpy().flatten(), weights.cpu().numpy().flatten())
                dot_product = np.dot(after_optimization_weights.flatten().cpu().numpy(), weights.flatten().cpu().numpy())
                norm_after = np.linalg.norm(after_optimization_weights.flatten().cpu().numpy())
                norm_weights = np.linalg.norm(weights.flatten().cpu().numpy())
                distance = np.arccos(dot_product / (norm_after * norm_weights))
                # Print the step, module, learning rate, and distance
                print(f"Step: {step}, Module: {module}, LR: {lr}, Distance: {distance}")
                distance_module_dict[(step, module, lr)] = distance

# Plot
fig, axs = plt.subplots(1, len([m for m in step_gradients if 'bias' not in m]), figsize=(5 * len([m for m in step_gradients if 'bias' not in m]), 4))
if len([m for m in step_gradients if 'bias' not in m]) == 1:
    axs = [axs]
for idx, module in enumerate([m for m in step_gradients if 'bias' not in m]):
    for step in checkpoint_steps:
        distances = [distance_module_dict[(step, module, lr)] for lr in learning_rate_list]
        axs[idx].plot(learning_rate_list, distances, marker='o', label=f"Step {step}")
    axs[idx].set_xscale("log")
    axs[idx].set_xlabel("Learning Rate")
    axs[idx].set_ylabel("Distance")
    axs[idx].set_title(f"Module: {module}")
    axs[idx].legend()
plt.tight_layout()
plt.savefig("distance_vs_lr.png")
plt.show()