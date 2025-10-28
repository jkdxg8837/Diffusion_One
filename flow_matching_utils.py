import torch
import time
import torch
import numpy as np
from torch import nn, Tensor
from peft import PeftModel
# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import pickle
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
import math
class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

# Model class
class MFMLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
            )
        # self.t_embedder = TimestepEmbedder(time_dim)
        # self.r_embedder = TimestepEmbedder(time_dim)
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)

    def forward(self, x: Tensor, t: Tensor, r: Tensor, y: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        r = r.reshape(-1, self.time_dim).float() 
        # t = self.t_embedder(t)
        # r = self.r_embedder(r)

        # The choice of concat t & r
        # t = torch.cat([t, r], dim=1)
        # print(t.shape, r.shape)
        t = t + r
        h = torch.cat([x, t], dim=1)
        output = self.main(h)
        
        return output.reshape(*sz)
# Model class
import os
class seg_MFMLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 512, segment_point=0.5, is_lora = True, is_reinit = True, reverse = False, data_mode = "up_down_shift", gamma =4, 
                 batch_size_scale = 1):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.segment_point = segment_point


        self.smaller_main = MFMLP(input_dim=input_dim, time_dim=time_dim, hidden_dim=hidden_dim)
        self.larger_main = MFMLP(input_dim=input_dim, time_dim=time_dim, hidden_dim=hidden_dim)
        self.gamma = gamma
        self.is_lora = is_lora
        self.is_reinit = is_reinit
        self.reverse = reverse
        self.batch_size_scale = batch_size_scale
        self.data_mode = data_mode
        self.init_weights(is_lora = is_lora, is_reinit = is_reinit)
    def init_weights(self, is_lora = False, is_reinit = False):
        state_dict = torch.load(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/raw_model_16000.pth", map_location=device)
        self.smaller_main.load_state_dict(state_dict)
        self.larger_main.load_state_dict(state_dict)
        if not is_lora:
            pass
        elif not is_reinit:
            lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["0", "2", "4", "6"],  # 对应 Sequential 里 Linear 层
            )
            self.smaller_main = get_peft_model(self.smaller_main, lora_config)
            self.larger_main = get_peft_model(self.larger_main, lora_config)
        elif is_reinit:
            # separate by self.segment_point
            lora_config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["0", "2", "4", "6"],  # 对应 Sequential 里 Linear 层
            )
            self.smaller_main = get_peft_model(self.smaller_main, lora_config)
            self.larger_main = get_peft_model(self.larger_main, lora_config)
            
            # with open("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/full_sample_up_shift.pkl", 'rb') as f:
            # 必须马上改回来
            with open("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/old_loraone_up_shift.pkl", 'rb') as f:
                named_grad = pickle.load(f)
            # 必须马上改回来 x2
            # with open("/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/pretrained_16000_half_up_shift_grad_5000.pkl", 'rb') as f:
            # with open(f"/home/u5649209/workspace/flow_matching/meanf/new_baseline/weights/pretrained_16000_up_down_shift_grad_{self.batch_size_scale}.pkl", 'rb') as f:
            with open("/home/u5649209/workspace/flow_matching/temp_gradient_100.pkl", 'rb') as f:
                named_grad = pickle.load(f)
            if not self.reverse:
                _ = reinit_lora(self.smaller_main, self.gamma, named_grad, init_mode = "lora-ga", lora_config = lora_config)
                for param in self.smaller_main.parameters():
                    param.data = param.data.contiguous()
                print("Smaller main reinit done")
            else:
                _ = reinit_lora(self.larger_main, self.gamma, named_grad, init_mode = "lora-ga", lora_config = lora_config)
                for param in self.larger_main.parameters():
                    param.data = param.data.contiguous()
                print("Larger main reinit done")
        else:
            raise NotImplementedError("is_lora and is_reinit cannot be both False or both True")
    def forward(self, x: Tensor, t: Tensor, r: Tensor, y: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()
        r = r.reshape(-1, self.time_dim).float()

        # # 简单加和
        # t = t + r
        # h = torch.cat([x, t], dim=1)  # [B, input_dim+time_dim]

        # 生成mask
        mask = (t[:, 0] < self.segment_point)  # [B]
        t_smaller = t[mask]
        t_larger = t[~mask]

        x_smaller = x[mask]
        x_larger = x[~mask]

        r_smaller = r[mask]
        r_larger = r[~mask]
        # 分别forward
        out = torch.zeros(t.size(0), self.input_dim, device=t.device, dtype=t.dtype)
        if t_smaller.numel() > 0:
            out[mask] = self.smaller_main(x_smaller, t_smaller, r_smaller, None)
        if t_larger.numel() > 0:
            out[~mask] = self.larger_main(x_larger, t_larger, r_larger, None)

        return out.reshape(*sz)
    def save_points(self, points_dict, path):
        # Final step visualization
        path_number = 10
        path_point_number_list = range(20)
        points = points_dict[20-1]
        points_np = points.detach().cpu().numpy()
        # 画散点图
        plt.figure(figsize=(6, 6))
        plt.scatter(points_np[:, 0], points_np[:, 1], s=1, alpha=0.5)  # s 控制点大小
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Scatter plot of 20,000 points")
        plt.show()
        plt.savefig(path)
        # Now for the path visualization
        plt.clf()
        fig, axes = plt.subplots(1, 10, figsize=(30, 4))  # 1行10列 = 10 个子图
        data1 = np.array([])
        axes = axes.flatten()
        for path_number_ in range(path_number):
            for path_point_number in path_point_number_list:
                data1 = np.append(data1, points_dict[path_point_number][path_number_].detach().cpu().numpy())
        data1 = data1.reshape((path_number, len(path_point_number_list), 2))
        for i in range(10):
            ax = axes[i]
            # 第一条曲线 (来自 data1)
            ax.plot(data1[i][:, 0], data1[i][:, 1], marker='o', label="curve1")
            # 第二条曲线 (来自 data2)
            ax.set_title(f"Sample {i+1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            ax.grid(True)
            ax.set_aspect("equal")  # 保持比例尺一致
            # ax.set_xlim(xlim)            # 设置统一的X轴范围
            # ax.set_ylim(ylim)            # 设置统一的Y轴范围
        plt.tight_layout()
        process_path = path[:-4] + "_path.png"
        plt.savefig(process_path)
        plt.show()
    def save(self, step_name, image_object = None, loss_history = None):
        if self.is_lora:
            if self.is_reinit:
                save_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_ga_gamma{self.gamma}_seg_{self.segment_point}_reverse_{self.data_mode}_100/ckpt/{step_name}"
                os.makedirs(save_path, exist_ok=True)
                # 保存 LoRA adapter
                self.smaller_main.save_pretrained(save_path + "/smaller_lora")
                self.larger_main.save_pretrained(save_path + "/larger_lora")
                if image_object is not None:
                    img_save_path =f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_ga_gamma{self.gamma}_seg_{self.segment_point}_reverse_{self.data_mode}_100/images"
            else:
                save_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_seg_{self.segment_point}_{self.data_mode}_100/ckpt/{step_name}"
                os.makedirs(save_path, exist_ok=True)
                self.smaller_main.save_pretrained(save_path + "/smaller_lora")
                self.larger_main.save_pretrained(save_path + "/larger_lora")
                if image_object is not None:
                    img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/lora_seg_{self.segment_point}_{self.data_mode}_100/images"
        else:
            save_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/fft_seg_{self.segment_point}_{self.data_mode}_100/{step_name}.pth"
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "smaller_main": self.smaller_main.state_dict(),
                "larger_main": self.larger_main.state_dict(),
                "gamma": self.gamma,
            }, save_path)
            if image_object is not None:
                img_save_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/100_test/fft_seg_{self.segment_point}_{self.data_mode}_100/images"
        if image_object is not None:
            os.makedirs(img_save_path, exist_ok=True)
            self.save_points(image_object, f"{img_save_path}/step_{step_name}.png")
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
def reinit_lora(model, gamma, named_grad, init_mode="lora-one", lora_config = None):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    from lorasb_utils.initialization_utils import find_and_initialize_grad
    if init_mode == "lora-sb":
        lora_rank = 2
        import yaml
        with open("/home/u5649209/workspace/flow_matching/config/reconstruct_config.yaml", 'r') as stream:
            reconstr_config = yaml.load(stream, Loader=yaml.FullLoader)
            
        adapter_name = "default"  # assuming a single LoRA adapter per module to be transformed to LoRA-SB
        peft_config_dict = {adapter_name: lora_config}

        # specifying LoRA rank for the SVD initialization
        reconstr_config['svd']['rank'] = lora_rank
            
        named_grads_new = {f'base_model.model.{k}': v for k, v in named_grad.items()}

        # convert to LoRA-SB model
        find_and_initialize_grad(
            model=model,
            peft_config=peft_config_dict,
            adapter_name=adapter_name,
            reconstr_type='svd',
            reconstruct_config=reconstr_config,
            writer=None,
            named_grads=named_grads_new,
        )

        # perform training as usual
        for param in model.parameters():
            param.data = param.data.cuda()
            if param.grad is not None:
                param.grad = param.grad.cuda()
        # You can merge LoRA-SB into the base model using `merge_and_unload` in PEFT
        # model = model.merge_and_unload() 
        # pass
    else:
        inited_modules = []
        for name, module in tqdm(
            model.named_modules(),
            desc="Reinitializing Lora",
            total=len(list(model.named_modules())),
        ):
            
            if isinstance(module, LoraLayer):
                reinit_lora_modules(name, module, gamma, named_grad, init_mode=init_mode)
    return model
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)
    
def train_moon_gen(batch_size: int = 200, device: str = "cpu", is_pretrain: bool = False, mode = "raw"):
    
    if is_pretrain:
        full_x, full_y = make_moons(n_samples=batch_size, noise=0, random_state=42)
        return full_x, full_y
    else:
        if "half" in mode:
            n_sample_out = batch_size
            out_variable = np.linspace(-1, 1, n_sample_out)
            mask = out_variable <= 0
            out_variable_leq_05 = out_variable[mask]
            out_variable_gt_05 = out_variable[~mask]
            outer_gt_05_x = np.cos(out_variable_gt_05 * np.pi / 2)
            outer_gt_05_y = np.sin(out_variable_gt_05 * np.pi / 2)

            outer_leq_05_x = out_variable_leq_05
            outer_leq_05_y = outer_leq_05_x + 1
            X=np.vstack([np.append(outer_leq_05_x, outer_gt_05_x), np.append(outer_leq_05_y, outer_gt_05_y)]).T
            y = np.hstack([np.zeros(n_sample_out, dtype=np.intp)])
            if mode == "half_up_shift":
                X[:, 1] += 0.25
            return X, y
        elif mode != "raw":
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
            if mode == "up_shift":
                X[:, 1] += 0.25
            elif mode == "up_down_shift":
                X[:n_sample_out, 1] += 0.25
                X[n_sample_out:, 1] -= 0.5
            return X, y
        else:
            full_x, full_y = make_moons(n_samples=3 * batch_size, noise=0, random_state=42)
            full_x = full_x[full_y == 1]
            full_y = full_y[full_y == 1]
            full_x = full_x[:batch_size]
            full_y = full_y[:batch_size]
            return full_x, full_y
def evaluate_result(vf, data_mode="new", visualize = True, emd_value = True, segment = True):
    # if using nll for evaluation
    if segment:
        fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
        z = vf.sample(20, random_noise = fixed_random_noise)
        source_x, _ = train_moon_gen(batch_size=4096, device=device, is_pretrain=False, mode=data_mode)
        # source_x = torch.tensor(source_x, dtype=torch.float32).to(device)
        emd_distance = compute_emd_distance(source_x, z[19][:4096].detach().cpu().numpy()) * 1000
        # print(f"EMD distance is {emd_distance}")
        return "hello", emd_distance
    vf.eval()
    wrapped_vf = WrappedModel(vf)
    # step size for ode solver
    step_size = 0.05

    norm = cm.colors.Normalize(vmax=50, vmin=0)

    T = torch.linspace(0,1,10)  # sample times
    T = T.to(device=device)

    solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
    
    if emd_value:
        # step size for ode solver
        step_size = 0.05

        norm = cm.colors.Normalize(vmax=50, vmin=0)

        batch_size = 3000  # batch size
        eps_time = 1e-2
        T = torch.linspace(0,1,10)  # sample times
        T = T.to(device=device)

        x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
        solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
        sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  # sample from the model
        sol = sol.cpu().numpy()
        T = T.cpu()
        target_x = sol[9, :, :]

        source_x, _ = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=False, mode=data_mode)
        # source_x = torch.tensor(source_x, dtype=torch.float32).to(device)
        emd_distance = compute_emd_distance(source_x, target_x)
        print(f"EMD distance is {emd_distance}")
        return "hello", emd_distance
    



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
    if visualize:
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
        print(exact_log_p_raw.mean().item())
    return log_p_acc_raw.mean(), exact_log_p_raw.mean().cpu().item()
from scipy.stats import wasserstein_distance
def select_source_distribution():
    target_x, target_y = train_moon_gen(batch_size=1000, device=device, is_pretrain=False, mode="new")
    rand_float = np.random.rand()
    print(f"Random float between 0 and 1: {rand_float}")
    source_x = np.random.randn(target_x.shape[0], target_x.shape[1]) * rand_float
    import ot
    n_source_points = source_x.shape[0]
    n_target_points = target_x.shape[0]
    a = np.ones(n_source_points) / n_source_points
    b = np.ones(n_target_points) / n_target_points

    # 3. Calculate the cost matrix
    # This matrix contains the cost of moving from any source point to any target point.
    # We'll use the standard Euclidean distance as the cost.
    cost_matrix = ot.dist(source_x, target_x)

    # 4. Compute the Earth Mover's Distance
    emd_value = ot.emd2(a, b, cost_matrix)
    print(f"EMD between standard normal and target distribution: {emd_value} from noise scale {rand_float}")
def compute_emd_distance(source_x, target_x):
    import ot
    n_source_points = source_x.shape[0]
    n_target_points = target_x.shape[0]
    a = np.ones(n_source_points) / n_source_points
    b = np.ones(n_target_points) / n_target_points

    # 3. Calculate the cost matrix
    # This matrix contains the cost of moving from any source point to any target point.
    # We'll use the standard Euclidean distance as the cost.
    cost_matrix = ot.dist(source_x, target_x)

    # 4. Compute the Earth Mover's Distance
    emd_value = ot.emd2(a, b, cost_matrix)
    return emd_value






from einops import rearrange 
from functools import partial
class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        
class MeanFlow:
    def __init__(self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='funtorch',
        baseline = False,
        compare_data = None):
        super().__init__()
        # self.normer = Normalizer.from_list(normalizer)
        self.time_dist = time_dist
        self.flow_ratio = flow_ratio
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        if baseline:
            self.flow_ratio = 1.0
        self.compare_data = None
        self.is_baseline = baseline
        # compare_data is for visualization, serves as a baseline path for flow matching model
        # Normally, baseline is the fft result of baseline model
        if not compare_data:
            self.compare_data = compare_data
    def stopgrad(self, x):
        return x.detach()
    def adaptive_l2_loss(self, error, gamma=0.5, c=1e-3):
        """
        Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
        Args:
            error: Tensor of shape (B, C, W, H)
            gamma: Power used in original ||Δ||^{2γ} loss
            c: Small constant for stability
        Returns:
            Scalar loss
        """
        delta_sq = torch.mean(error ** 2, dim=(1), keepdim=False)
        p = 1.0 - gamma
        w = 1.0 / (delta_sq + c).pow(p)
        loss = delta_sq  # ||Δ||^2
        return (self.stopgrad(w) * loss).mean()
    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        # flow_ratio controls how many pairs are orginal flow matching t & r
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, model, x, c=None, gradient_generation=False, return_u = False):
        batch_size = x.shape[0]
        device = x.device
        if gradient_generation and self.flow_ratio < 1.0:
            self.flow_ratio = 0.0
        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1").detach().clone()
        r_ = rearrange(r, "b -> b 1").detach().clone()
        # if gradient_generation:
        #     t_ = torch.ones_like(t_) * 0.999
        #     r_ = torch.zeros_like(r_)
        # print("For 100 fixed point test~!!!!!!!!!!!")
        e = torch.randn_like(x)
        # e = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
        # e = e[:x.shape[0], :]


        # x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    # offical JAX repo uses original v for unconditional items
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
        else:
            v_hat = v

        # forward pass
        u = model(z, t, r, y=c)
        model_partial = partial(model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - self.stopgrad(u_tgt)
        loss = self.adaptive_l2_loss(error)
        # loss = torch.pow(error, 2).mean()
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (self.stopgrad(error) ** 2).mean()
        if return_u:
            return loss, mse_val, u, u_tgt, t_, r_

        return loss, mse_val
    @torch.no_grad()
    def sample(self, model, sample_steps = 5, device = 'cuda', random_noise = None):
        z_dict = {}
        model.eval()
        if random_noise is not None:
            z = random_noise.to(device)
        else:
            z = torch.randn(20000, 2, device=device)
        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)


            t_ = rearrange(t, "b -> b 1").detach().clone()
            r_ = rearrange(r, "b -> b 1").detach().clone()

            v = model(z, t, r, None)
            z = z - (t_-r_) * v
            z_dict[i] = z.detach().cpu()

        # z = self.normer.unnorm(z)
        return z_dict

class segment_MeanFlow:
    def __init__(self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        # mean flow settings
        flow_ratio=0.50,
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # set scale as none to disable CFG distill
        cfg_scale=2.0,
        # experimental
        cfg_uncond='v',
        jvp_api='funtorch',
        baseline = False,
        compare_data = None,
        segment_point = 0.6,
        is_lora = False, is_reinit = False, gamma = 9, reverse =False, data_mode = "half_up_shift",
        batch_size_scale = 1):
        super().__init__()
        # self.normer = Normalizer.from_list(normalizer)
        self.time_dist = time_dist
        self.flow_ratio = flow_ratio
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        if baseline:
            self.flow_ratio = 1.0
        self.compare_data = None
        self.is_baseline = baseline
        # compare_data is for visualization, serves as a baseline path for flow matching model
        # Normally, baseline is the fft result of baseline model
        if not compare_data:
            self.compare_data = compare_data
        self.batch_size_scale = batch_size_scale
        self.data_mode = data_mode
        lr = 0.001 * batch_size_scale
        model = seg_MFMLP(segment_point=segment_point, is_lora = is_lora, is_reinit = is_reinit, reverse = reverse, data_mode = data_mode, gamma=gamma, batch_size_scale = batch_size_scale)
        if is_lora:
            self.optim = torch.optim.Adam(model.parameters(), lr=lr)
            self.optim.param_groups[0]['params'] = [p for n, p in model.named_parameters() if 'lora_' in n]
        else:
            self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.model = model.to(device)

    def stopgrad(self, x):
        return x.detach()
    def load_model(self, path, is_lora = True):
        if is_lora:
            self.model.smaller_main = PeftModel.from_pretrained(self.model.smaller_main, f"{path}/smaller_lora")
            self.model.larger_main = PeftModel.from_pretrained(self.model.larger_main, f"{path}/larger_lora")
        else:
            path = path+ ".pth"
            checkpoint = torch.load(path)

            self.model.smaller_main.load_state_dict(checkpoint["smaller_main"])
            self.model.larger_main.load_state_dict(checkpoint["larger_main"])
    def train(self):
        loss_history = []
        print_every = 2000
        meanF_step = 20
        fixed_random_noise = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
        for i in range(10):
            start_time = time.time()
            self.optim.zero_grad() 
            
            # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
            x_1, y = train_moon_gen(batch_size=4096 * self.batch_size_scale, device=device, is_pretrain=False, mode = self.data_mode) # sample data
            # print(y)
            x_1 = torch.tensor(x_1).float().to(device)
            x_fixed_100 = torch.load("/home/u5649209/workspace/flow_matching/fixed_100_target_point.pt").to(device)
            x_1 = x_fixed_100

            # Mean flow insert
            loss, mse_val = self.loss(x_1)

            loss_history.append(loss.item())
            if i == 0:
                elapsed = time.time() - start_time
                print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                    .format(i+1, elapsed*1000/print_every, loss.item())) 
                start_time = time.time()
                self.model.save(step_name=0)
            # optimizer step
            loss.backward() # backward
            if i==0:
                elapsed = time.time() - start_time
                print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                    .format(i, elapsed*1000/print_every, loss.item())) 
                z = self.sample(meanF_step, random_noise = fixed_random_noise)
                start_time = time.time()
                self.model.save(i+1, z, loss_history)
            self.optim.step() # update
            # log loss
            if ((i+1) % print_every == 0) or (i in range(100) and (i+1) % 1 == 0):
                elapsed = time.time() - start_time
                print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                    .format(i, elapsed*1000/print_every, loss.item())) 
                z = self.sample(meanF_step, random_noise = fixed_random_noise)
                start_time = time.time()
                self.model.save(i+1, z, loss_history)
    def adaptive_l2_loss(self, error, gamma=0.5, c=1e-3):
        """
        Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
        Args:
            error: Tensor of shape (B, C, W, H)
            gamma: Power used in original ||Δ||^{2γ} loss
            c: Small constant for stability
        Returns:
            Scalar loss
        """
        delta_sq = torch.mean(error ** 2, dim=(1), keepdim=False)
        p = 1.0 - gamma
        w = 1.0 / (delta_sq + c).pow(p)
        loss = delta_sq  # ||Δ||^2
        return (self.stopgrad(w) * loss).mean()
    # fix: r should be always not larger than t
    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

        # Assign t = max, r = min, for each pair
        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        # flow_ratio controls how many pairs are orginal flow matching t & r
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r

    def loss(self, x, c=None, gradient_generation=False, return_u = False):
        batch_size = x.shape[0]
        device = x.device
        if gradient_generation and self.flow_ratio < 1.0:
            self.flow_ratio = 0.0
        t, r = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1").detach().clone()
        r_ = rearrange(r, "b -> b 1").detach().clone()
        # if gradient_generation:
        #     t_ = torch.ones_like(t_) * 0.999
        #     r_ = torch.zeros_like(r_)
        # e = torch.randn_like(x)
        print("For 100 fixed point test~!!!!!!!!!!!")
        e = torch.load("/home/u5649209/workspace/flow_matching/random_noise.pt").to(device)
        e = e[:x.shape[0], :]
        # x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        if c is not None:
            assert self.cfg_ratio is not None
            uncond = torch.ones_like(c) * self.num_classes
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            if self.w is not None:
                with torch.no_grad():
                    u_t = self.model(z, t, t, uncond)
                v_hat = self.w * v + (1 - self.w) * u_t
                if self.cfg_uncond == 'v':
                    # offical JAX repo uses original v for unconditional items
                    cfg_mask = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                    v_hat = torch.where(cfg_mask, v, v_hat)
        else:
            v_hat = v

        # forward pass
        u = self.model(z, t, r, y=c)
        model_partial = partial(self.model, y=c)
        jvp_args = (
            lambda z, t, r: model_partial(z, t, r),
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
        )

        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)

        u_tgt = v_hat - (t_ - r_) * dudt

        error = u - self.stopgrad(u_tgt)
        loss = self.adaptive_l2_loss(error)
        # loss = torch.pow(error, 2).mean()
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (self.stopgrad(error) ** 2).mean()
        if return_u:
            return loss, mse_val, u, u_tgt, t_, r_

        return loss, mse_val
    @torch.no_grad()
    def sample(self, sample_steps = 5, device = 'cuda', random_noise = None):
        z_dict = {}
        self.model.eval()
        if random_noise is not None:
            z = random_noise.to(device)
        else:
            z = torch.randn(20000, 2, device=device)
        if self.is_baseline:
            t_vals = torch.linspace(1.0, 0.0, sample_steps+1, device=device)
        else:
            t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            if self.is_baseline:
                r = torch.full((z.size(0),), t_vals[i + 1], device=device)
            else:
                r = torch.full((z.size(0),), t_vals[i + 1], device=device)


            t_ = rearrange(t, "b -> b 1").detach().clone()
            r_ = rearrange(r, "b -> b 1").detach().clone()

            v = self.model(z, t, r, None)
            z = z - (t_-r_) * v
            z_dict[i] = z.detach().cpu()

        # z = self.normer.unnorm(z)
        return z_dict


if __name__ == "__main__":
    import pickle
    import sys
    hidden_dim = 512
    gradient_base = 3
    gradient_iter = 4000
    is_lora = False
    

    # for seg_ratio in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
    for seg_ratio in [0.0]:
        # lora-one
        # ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/lora_one_gamma9_seg_{seg_ratio}_reverse_up_down_shift/ckpt"
        # lora
        # ckpt_path =f"/home/u5649209/workspace/flow_matching/meanf/seg_base/lora_seg_{seg_ratio}_up_down_shift/ckpt"
        # fft
        ckpt_path = f"/home/u5649209/workspace/flow_matching/meanf/seg_base/fft_seg_{seg_ratio}_up_down_shift/ckpt"
        emd_distance_list = []
        best_step = -1
        best_emd = 1e10
        for i in [1, 10, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            meanflow = segment_MeanFlow(baseline = True, segment_point=seg_ratio, is_lora = False, is_reinit = False, gamma = 9, reverse=True)
            path = f"{ckpt_path}/{i}"
            # print(path)
            meanflow.load_model(path, is_lora = is_lora)
            
            _, emd_distance = evaluate_result(meanflow, data_mode="up_down_shift", visualize=False, emd_value=True, segment=True)
            if emd_distance < best_emd:
                best_emd = emd_distance
                best_step = i
            emd_distance_list.append(emd_distance)
        print(f"Best step is {best_step} with EMD {best_emd}")
        # Write emd_distance_list into txt file under ckpt_path
        # if last dir is ckpt, use parent dir
        if ckpt_path.split("/")[-1] == "ckpt":
            ckpt_path = "/".join(ckpt_path.split("/")[:-1])
        with open(f"{ckpt_path}/emd_distance.txt", "w") as f:
            for item in emd_distance_list:
                f.write("%s\n" % item)
        print(emd_distance_list)
        # Plot the emd_distance_list with number on the points
        import matplotlib.pyplot as plt
        plt.plot(emd_distance_list, marker='o')
        plt.title("EMD Distance over Checkpoints")
        plt.xlabel("Checkpoint")
        plt.ylabel("EMD Distance")
        plt.xticks(ticks=range(len(emd_distance_list)), labels=range(1, len(emd_distance_list) + 1))
        plt.grid()
        plt.savefig(f"{ckpt_path}/emd_distance.png")
        plt.show()

   