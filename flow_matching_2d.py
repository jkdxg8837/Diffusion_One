import time
import torch

from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm
from sklearn.datasets import make_moons

# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

torch.manual_seed(42)
def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size, ), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45
    
    return data.float()
import numpy as np
def train_moon_gen(batch_size: int = 200, device: str = "cpu", is_pretrain: bool = False, mode = "raw"):
    
    if is_pretrain:
        full_x, full_y = make_moons(n_samples=batch_size, noise=0, random_state=42)
        return full_x, full_y
    else:
        if mode is not "raw":
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

from flow_matching_utils import MLP
from peft import LoraConfig, get_peft_model
import pickle

# training arguments
lr = 0.001
load_steps_list = [19999]
for load_steps in load_steps_list:
    batch_size = 4096
    iterations = 20001
    rest_iterations = 2000
    print_every = 2000 
    hidden_dim = 512

    # velocity field model init
    layer_gradients = {}

    def save_gradient(model, record_dict):
        def record_gradient_hook(grad):
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if n not in record_dict:
                        record_dict[n] = [p.grad.cpu()]
                    else:
                        record_dict[n].append(p.grad.cpu())
                    p.grad = None
            return grad

        return record_gradient_hook

    vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)
    hooks = []
    for name, param in vf.named_parameters():
        if param.requires_grad == True:
            hook = param.register_hook(save_gradient(vf, layer_gradients))
            hooks.append(hook) 
    state_dict_path = f'/home/u5649209/workspace/flow_matching/ckpts/weights/full_model_2_new.pth'
    state_dict = torch.load(state_dict_path, map_location=device)
    vf.load_state_dict(state_dict)

    # instantiate an affine path object
    path = AffineProbPath(scheduler=CondOTScheduler())

    # init optimizer
    optim = torch.optim.Adam(vf.parameters(), lr=lr) 

    # lora_config = LoraConfig(
    #     r=8,
    #     lora_alpha=16,
    #     target_modules=["main.0", "main.2", "main.4", "main.6"],  # target Linear layers in MLP
    #     init_lora_weights="gaussian",
    # )
    # vf = get_peft_model(vf, lora_config)


    # train
    start_time = time.time()
    named_grads = {}
    for i in range(rest_iterations):
        optim.zero_grad() 

        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
        mode = "new"
        x_1, y = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=False, mode = mode) # sample data
        
        # print(y)
        x_1 = torch.tensor(x_1).float().to(device)
        
        x_0 = torch.randn_like(x_1).to(device)

        # sample time (user's responsibility)
        t = torch.rand(x_1.shape[0]).to(device) 

        # sample probability path
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

        # flow matching l2 loss
        loss = torch.pow(vf(path_sample.x_t,path_sample.t) - path_sample.dx_t, 2).mean() 

        # optimizer step
        loss.backward() # backward
        save_gradient(vf, layer_gradients)(None)
        
        # optim.step() # update

        
        # log loss
        if (i+1) % print_every == 0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} ' 
                .format(i+1, elapsed*1000/print_every, loss.item())) 
            start_time = time.time()
        for n, p in vf.named_parameters():
            if p.grad is not None:
                p.grad = None
        optim.zero_grad(set_to_none=True)

    from tqdm import tqdm
    for key in layer_gradients.keys():
        layer_gradients[key] = torch.stack(layer_gradients[key], dim=0).mean(dim=0)
    with open(f"/home/u5649209/workspace/flow_matching/ckpts/raw_model_gradients/fullP_step1_{mode}.pkl", "wb") as f:
        pickle.dump(layer_gradients, f)

class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

wrapped_vf = WrappedModel(vf)





# # step size for ode solver
# step_size = 0.05

# norm = cm.colors.Normalize(vmax=50, vmin=0)

# batch_size = 50000  # batch size
# eps_time = 1e-2
# T = torch.linspace(0,1,10)  # sample times
# T = T.to(device=device)

# x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
# solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
# sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  # sample from the model

# sol = sol.cpu().numpy()
# T = T.cpu()

# fig, axs = plt.subplots(1, 10,figsize=(20,20))

# for i in range(10):
#     H= axs[i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,5), (-5,5)))
    
#     cmin = 0.0
#     cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()
    
#     norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
#     _ = axs[i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,5), (-5,5)), norm=norm)
    
#     axs[i].set_aspect('equal')
#     axs[i].axis('off')
#     axs[i].set_title('t= %.2f' % (T[i]))
    
# plt.tight_layout()
# plt.show()

