import torch
import time
import torch
import numpy as np
from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm

from peft import LoraConfig, get_peft_model
from sklearn.datasets import make_moons
from torch.distributions import Independent, Normal

# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
import torch.nn as nn

from tqdm import tqdm
from peft.tuners.lora import LoraLayer

# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor: 
        return torch.sigmoid(x) * x

# Model class
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



@torch.no_grad()
def reinit_lora_modules(name, module, gamma, named_grad, init_mode):
    # r"""
    # Reinitialize the lora model with the given configuration.
    # """
    if init_mode =="lora-one":
        lora_r = 2


        # print("*************************")
        
        # grad_name = name + '.weight'
        grad_name = ".".join(name.split(".")[-2:]) + '.weight'
        # print(grad_name)
        grads = named_grad[grad_name]

        grads = -grads.cuda().float()
        m, n = grads.shape

        grads = grads * (m**0.5)
        U, S, V = torch.linalg.svd(grads)
        B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
        A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
        B = B / gamma**0.5
        A = A / gamma**0.5

        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())
        return
    elif init_mode == "lora-ga":
        lora_r = 2
        grad_name = ".".join(name.split(".")[-2:]) + '.weight'
        grads = named_grad[grad_name]

        m, n = grads.shape
        U, S, V = torch.linalg.svd(grads.float())
        B = U[:, lora_r : 2 * lora_r]
        A = V[:lora_r, :]
        m, n = grads.shape # m: feature_out, n: feature_in
        # the scale of output is only related to the feature_out
        
        B = B * m**0.25 / gamma**0.5
        A = A * m**0.25 / gamma**0.5

        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())
        return
def reinit_lora(model, gamma, named_grad, init_mode="lora-one"):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    inited_modules = []
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        
        if isinstance(module, LoraLayer):
            reinit_lora_modules(name, module, gamma, named_grad, init_mode=init_mode)

        pass
    return model
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)
def train_moon_gen(batch_size: int = 200, device: str = "cpu", is_pretrain: bool = False, mode = "raw"):
    
    if is_pretrain:
        full_x, full_y = make_moons(n_samples=batch_size, noise=0, random_state=42)
        return full_x, full_y
    else:
        if mode != "raw":
            n_sample_out = batch_size // 2
            n_sample_in = batch_size - n_sample_out
            out_variable = np.linspace(-1, 1, n_sample_out)
            mask = out_variable <= 0
            out_variable_leq_05 = out_variable[mask]
            out_variable_gt_05 = out_variable[~mask]

            in_variable = np.linspace(0, 2, n_sample_in)
            mask = in_variable <= 1
            in_variable_leq_05 = in_variable[mask]
            in_variable_gt_05 = in_variable[~mask]

            # For gt 0.5, we use the half circle chart
            outer_gt_05_x = np.cos(out_variable_gt_05 * np.pi / 2)
            outer_gt_05_y = np.sin(out_variable_gt_05 * np.pi / 2)

            inner_leq_05_x = in_variable_leq_05
            inner_leq_05_y = -inner_leq_05_x + 0.5


            # For leq 0.5, we use the line chart
            outer_leq_05_x = out_variable_leq_05
            outer_leq_05_y = outer_leq_05_x + 1

            inner_gt_05_x = 1 - np.cos(in_variable_gt_05 * np.pi / 2)
            inner_gt_05_y = 0.5 - np.sin(in_variable_gt_05 * np.pi / 2)


            # outer_circ_x = np.cos(np.linspace(0, np.pi, n_sample_out))
            # outer_circ_y = np.sin(np.linspace(0, np.pi, n_sample_out))

            # inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_sample_in))
            # inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_sample_in)) - 0.5

            X = np.vstack(
                [np.append(np.append(outer_leq_05_x, outer_gt_05_x), np.append(inner_leq_05_x, inner_gt_05_x)),\
                np.append(np.append(outer_leq_05_y, outer_gt_05_y), np.append(inner_leq_05_y, inner_gt_05_y))]
            ).T
            y = np.hstack(
                [np.zeros(n_sample_out, dtype=np.intp), np.ones(n_sample_in, dtype=np.intp)]
            )
            return X, y
        else:
            full_x, full_y = make_moons(n_samples=3 * batch_size, noise=0, random_state=42)
            full_x = full_x[full_y == 1]
            full_y = full_y[full_y == 1]
            full_x = full_x[:batch_size]
            full_y = full_y[:batch_size]
            return full_x, full_y
def evaluate_result(vf, data_mode="new"):
    vf.eval()
    wrapped_vf = WrappedModel(vf)
    # step size for ode solver
    step_size = 0.05

    norm = cm.colors.Normalize(vmax=50, vmin=0)

    batch_size = 50000  # batch size
    eps_time = 1e-2
    T = torch.linspace(0,1,10)  # sample times
    T = T.to(device=device)

    x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
    solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
    # sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  # sample from the model


    # sample with likelihood

    T = torch.tensor([1., 0.])  # sample times
    T = T.to(device=device)

    grid_size = 200
    # x_1 = torch.meshgrid(torch.linspace(-2, 3, grid_size), torch.linspace(-1, 2, grid_size))
    # Using target distribution as sampled points
    x_1, _ = train_moon_gen(batch_size=grid_size**2, device=device, is_pretrain=False, mode=data_mode)
    x_1 = torch.tensor(x_1, dtype=torch.float32).to(device)
    # x_1 = torch.stack([x_1[0].flatten(), x_1[1].flatten()], dim=1).to(device)
    print(x_1.shape)
    # source distribution is an isotropic gaussian
    gaussian_log_density = Independent(Normal(torch.zeros(2, device=device), torch.ones(2, device=device)), 1).log_prob

    # compute log likelihood with unbiased hutchinson estimator, average over num_acc
    num_acc = 10
    log_p_acc = 0

    for i in range(num_acc):
        _, log_p = solver.compute_likelihood(x_1=x_1, method='midpoint', step_size=step_size, exact_divergence=False, log_p0=gaussian_log_density)
        log_p_acc += log_p
    
    log_p_acc /= num_acc
    log_p_acc_raw = torch.exp(-log_p_acc.clone())
    # compute with exact divergence
    _, exact_log_p = solver.compute_likelihood(x_1=x_1, method='midpoint', step_size=step_size, exact_divergence=True, log_p0=gaussian_log_density)
    exact_log_p_raw = torch.exp(-exact_log_p.clone())

    fig, axs = plt.subplots(1, 2,figsize=(10,10))
    import seaborn as sns

    sns.kdeplot(
    x=x_1[:, 0].cpu().numpy(),
    y=x_1[:, 1].cpu().numpy(),
    weights=torch.exp(exact_log_p).reshape(-1).cpu().numpy(),
        fill=True,           # 填充等高线内部
        cmap='viridis',      # 使用 'viridis' 色谱
        ax=axs[0]            # !!! 关键参数：指定在哪个子图上绘制 !!!
    )
    # 导入一个必要的模块
    from matplotlib.ticker import MultipleLocator

    # ... (您的数据准备和 subplots 创建代码) ...

    # --- 针对 axs[0] 的设置 ---
    axs[0].set_title('Weighted Kernel Density Estimation (KDE)')
    axs[0].set_xlabel('X Coordinate')
    axs[0].set_ylabel('Y Coordinate')

    # 1. 设置坐标轴比例为1:1，确保网格是正方形
    # adjustable='box' 会调整绘图框的大小来强制实现这个比例
    axs[0].set_aspect('equal', adjustable='box')

    # 2. 设置主刻度的间隔为 1
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(1))

    # 3. 显示网格 (您的代码)
    # grid() 函数会根据上面设置的刻度来绘制网格线
    axs[0].grid(True, linestyle='--', alpha=0.6)


    # Visualization 
    x_1 = torch.meshgrid(torch.linspace(-2, 3, grid_size), torch.linspace(-1, 2, grid_size))
    x_1 = torch.stack([x_1[0].flatten(), x_1[1].flatten()], dim=1).to(device)
    num_acc = 10
    log_p_acc = 0

    for i in range(num_acc):
        _, log_p = solver.compute_likelihood(x_1=x_1, method='midpoint', step_size=step_size, exact_divergence=False, log_p0=gaussian_log_density)
        log_p_acc += log_p
    
    log_p_acc /= num_acc
    _, exact_log_p = solver.compute_likelihood(x_1=x_1, method='midpoint', step_size=step_size, exact_divergence=True, log_p0=gaussian_log_density)


    likelihood = torch.exp(log_p_acc).cpu().reshape(grid_size, grid_size).t().detach().numpy()

    exact_likelihood = torch.exp(exact_log_p).cpu().reshape(grid_size, grid_size).t().detach().numpy()

    

    cmin = 0.0
    cmax = 1/32 # 1/32 is the gt likelihood value

    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    

    axs[1].imshow(exact_likelihood, extent=(-3, 3, -3, 3), origin='lower', cmap='viridis', norm=norm)
    axs[1].set_title('Exact Model Likelihood')

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=axs, orientation='horizontal', label='density')
    plt.show()
    return log_p_acc_raw.mean(), exact_log_p_raw.mean()


if __name__ == "__main__":
    # Example usage: train a model and evaluate
    batch_size = 200
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    X, y = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=True)
    X = torch.tensor(X, dtype=torch.float32, device=device)
    t = torch.zeros(X.shape[0], 1, device=device)

    vf = MLP(input_dim=2, time_dim=1, hidden_dim=128).to(device)
    evaluate_result(vf)