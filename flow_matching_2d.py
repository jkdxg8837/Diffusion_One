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
def train_moon_gen(batch_size: int = 200, device: str = "cpu", is_pretrain: bool = False):
    full_x, full_y = make_moons(n_samples=batch_size * 2, noise=0, random_state=42)
    if is_pretrain:
        return torch.tensor(full_x[:batch_size], dtype=torch.float32, device=device)
    else:
        return torch.tensor(full_x[full_y == 1][:batch_size], dtype=torch.float32, device=device)

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

from peft import LoraConfig, get_peft_model
import pickle

# training arguments
lr = 0.001
load_steps_list = [1999, 3999, 5999, 7999, 9999, 11999, 13999, 15999, 17999, 19999]
for load_steps in load_steps_list:
    batch_size = 4096
    iterations = 20001
    rest_iterations = iterations - load_steps
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
    state_dict_path = f'/home/u5649209/workspace/flow_matching/ckpts/pretrain_weights/raw_model_{load_steps}.pth'
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
        x_1 = train_moon_gen(batch_size=batch_size, device=device, is_pretrain=False) # sample data
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
    with open(f"/home/u5649209/workspace/flow_matching/ckpts/raw_model_gradients/models_grads_step{load_steps}.pkl", "wb") as f:
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

