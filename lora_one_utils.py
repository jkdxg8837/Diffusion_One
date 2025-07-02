import torch
from tqdm import tqdm
import math
import itertools
from peft.tuners.lora.layer import Linear as LoraLinear
import logging
log = logging.getLogger(__name__)
from typing import Tuple, List, Dict
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
def get_record_gradient_hook(model, record_dict):
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


def print_gpu_memory_usage(device_id=0):
    allocated = torch.cuda.memory_allocated(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    ratio = allocated / total
    print(f"显存占用：{ratio:.2%} （{allocated / (1024 ** 2):.2f} MB / {total / (1024 ** 2):.2f} MB）")

def estimate_gradient(
    models, dataloader, args, noise_scheduler_copy, accelerator, text_encoders, tokenizers, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    r"""
    Estimate the gradient of the model on the given dataset
    """
    transformer, vae = models[0], models[1]
    # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    log.info("Estimating gradient")
    transformer.train()
    
    named_grads = {}
    hooks = []
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    for name, param in transformer.named_parameters():
        if param.requires_grad == True:
            # print("adding hook to grad params")
            hook = param.register_hook(get_record_gradient_hook(transformer, named_grads))
            hooks.append(hook)
    num = 0
    weight_dtype = torch.float16
    # deal with time step
    # time_step = args.time_step
    if args.re_init_schedule == "multi":
        epochs = args.re_init_samples // len(dataloader)
        print("********************************")
        print("len of dataloader is : ", len(dataloader))
        print(f"Reinitializing LoRA modules every {epochs} epochs")
        print("********************************")
    else:
        epochs = 1
    from tqdm import tqdm
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc="Estimating gradient"):
            num += 1
            # print(batch)
            # batch = {k: v.to(transformer.device) for k, v in batch.items()}
            # Calculate diffusion model loss
            prompts = batch["prompts"]
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            pixel_values = pixel_values.to(vae.device)
            model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
            model_input = model_input.to(dtype=weight_dtype)
            
            # Sample noise that we'll add to the latents
            if not args.fixed_noise:
                noise = torch.randn_like(model_input)
            else:
                noise_tensor = torch.load("/dcs/pg24/u5649209/data/workspace/diffusers/noise.pt")
                sample_number = args.noise_samples
                # Randomly sample 'sample_number' indices from the noise tensor's first dimension
                total_samples = noise_tensor.shape[0]
                if sample_number > total_samples:
                    raise ValueError(f"Requested {sample_number} samples, but noise tensor only has {total_samples} samples.")
                indices = torch.randperm(total_samples)[:sample_number]
                # Select the noise samples based on the random indices
                noise_bank = noise_tensor[indices].to(model_input.device, dtype=model_input.dtype)
                noise = noise_bank[torch.randperm(noise_bank.shape[0])[:model_input.shape[0]]]
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )

            if args.re_init_schedule == "multi":
                pass
            elif args.time_step:
                # assert args.time_step.dtype == torch.float32, "time_step should be float32"
                u = torch.tensor([args.time_step])
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

            print("u is set to ", u)
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
            def compute_text_embeddings(prompt, text_encoders, tokenizers):
                with torch.no_grad():
                    from train_dreambooth_lora_one_sd3 import encode_prompt
                    prompt_embeds, pooled_prompt_embeds = encode_prompt(
                        text_encoders, tokenizers, prompt, args.max_sequence_length
                    )
                    prompt_embeds = prompt_embeds.to(accelerator.device)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                return prompt_embeds, pooled_prompt_embeds
            instance_prompt_hidden_states, instance_pooled_prompt_embeds = compute_text_embeddings(
                args.instance_prompt, text_encoders, tokenizers
            )
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds

            # Predict the noise residual
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            model_pred = model_pred * (-sigmas) + noisy_model_input

            from diffusers.training_utils import compute_loss_weighting_for_sd3
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
            target = model_input
            loss = torch.mean(
                        (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                        1,
                    )
            loss = loss.mean()
            print(loss.item())
            # loss.backward()
            accelerator.backward(loss)
            # print_gpu_memory_usage(0)
            if accelerator.sync_gradients:
                transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
                params_to_clip = (transformer_lora_parameters)
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            get_record_gradient_hook(transformer, named_grads)(None)  # get gradient of last layer
            # make sure the gradient is cleared
            for n, p in transformer.named_parameters():
                if p.grad is not None:
                    p.grad = None
        torch.cuda.empty_cache()
    for n, g in named_grads.items():
        g_list = named_grads[n]
        for i in range(len(g_list)):
            g_list[i] /= num
        named_grads[n] = g_list
    # for key in tqdm(named_grads.keys(),
    #                 desc="Merging gradient in list",):
    #     try:
    #         tensors = named_grads[key]
    #         named_grads[key] = torch.stack(tensors, dim=0).mean(dim=0)
    #     except Exception as e:
    #         log.error(f"Error processing key {key}: {e}")
    #         # print(e)
    #         continue
    # Using batch size
    if args.re_init_bsz:
        batch_size = args.re_init_bsz
        from tqdm import tqdm
        for key in tqdm(named_grads.keys(),
                        desc="Merging gradient in list",):
            try:
                data_list = named_grads[key]
                prev_tensor = None
                for i in range(0, len(data_list), batch_size):
                    # 获取当前批次的数据
                    current_batch = data_list[i : i + batch_size]
                    current_batch_tensor = torch.cat(current_batch, dim=0)
                    # print(data_list[0].shape)
                    # print(batch_size)
                    # print(len(data_list))
                    # print(current_batch_tensor.shape)
                    if prev_tensor is not None:
                        prev_tensor += current_batch_tensor
                        # prev_tensor /= 2
                    else:
                        prev_tensor = current_batch_tensor
                    # 计算平均值并添加到新列表
                named_grads[key] = prev_tensor
            except Exception as e:
                log.error(f"Error processing key {key}: {e}")
                # print(e)
                continue
        
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    del transformer
    return named_grads


@torch.no_grad()
def reinit_lora_modules(name, module, init_config, additional_info):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    lora_r = min(module.lora_A.default.weight.shape)
    a_dim = max(module.lora_A.default.weight.shape)
    b_dim = max(module.lora_B.default.weight.shape)
    print(f"Reinitializing {name} with LoRA rank {lora_r}, lora_A dim: {a_dim}, lora_B dim: {b_dim}")
    init_mode = init_config['mode']
    if init_mode == "simple":
        match init_config.lora_A:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=init_config.lora_A_std
                )
            case "kaiming":
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                torch.nn.init.kaiming_uniform_(module.lora_A.default.weight, a=math.sqrt(5))
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_A.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_A.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_A.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_A.default.weight, mean=0.0, std=1.0 / (a_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_A.default.weight)
            case _:
                raise ValueError(f"Unknown lora_A initialization: {init_config.lora_A}")
        match init_config.lora_B:
            case "gaussian":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=init_config.lora_B_std
                )
            case "kaiming":
                torch.nn.init.kaiming_normal_(module.lora_B.default.weight)
            case "fan_out_kaiming":
                torch.nn.init.kaiming_normal_(
                    module.lora_B.default.weight, mode="fan_out"
                )
            case "xavier":
                torch.nn.init.xavier_normal_(module.lora_B.default.weight)
            case "zeros":
                torch.nn.init.zeros_(module.lora_B.default.weight)
            case "unit":
                torch.nn.init.normal_(
                    module.lora_B.default.weight, mean=0.0, std=1.0 / (b_dim**0.5)
                )
            case "orthogonal":
                torch.nn.init.orthogonal_(module.lora_B.default.weight)
            case _:
                raise ValueError(f"Unknown lora_B initialization: {init_config.lora_B}")
        if init_config.get("scale", "") == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight.data *= (m**0.25) / gamma**0.5
            module.lora_A.default.weight.data *= (n**0.25) / gamma**0.5
    elif init_mode == "svd":
        U, S, V = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
        V = V.T
        m, n = module.weight.shape
        if init_config.scale == "default":
            S = S / module.scaling["default"]
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])).T.contiguous()
            )
        elif init_config.scale == "stable":
            gamma = init_config.stable_gamma
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * (m**0.25) / gamma**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :] * (n**0.25) / gamma**0.5).contiguous()
            )
        elif init_config.scale == "unit":
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r]).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :]).contiguous()
            )
        elif init_config.scale == "normalized":
            S_sum = S[:lora_r].sum()
            module.lora_B.default.weight = torch.nn.Parameter(
                (U[:, :lora_r] * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).contiguous()
            )
            module.lora_A.default.weight = torch.nn.Parameter(
                (V[:lora_r, :].T * torch.sqrt(S[:lora_r])/torch.sqrt(S_sum)*lora_r**0.5).T.contiguous()
            )
    elif init_mode == "gradient":
        named_grad = additional_info["named_grads"]

        grad_name = name + ".base_layer.weight"
        # grad_name = ".".join(name.split(".")[2:]) + ".weight"
        grads = named_grad[grad_name]
        # grads = named_grad[name]
        if init_config['direction'] == "LoRA-One":
            # Using full svd for LoRA-One
            # U, S, V = torch.svd_lowrank(-grads.cuda().float(), q=512, niter=16)
            U, S, V = torch.linalg.svd(-grads.cuda().float())
        else:
            U, S, V = torch.svd_lowrank(grads.cuda().float(), q=512, niter=16)
        V = V.T
        if init_config['direction'] == "LoRA-One":
            B = U[:, :lora_r] @ torch.diag(torch.sqrt(S[:lora_r])) / torch.sqrt(S[0])
            A = torch.diag(torch.sqrt(S[:lora_r])) @ V[:lora_r, :] / torch.sqrt(S[0])
        elif init_config['direction'] == "LoRA-GA":
            B = U[:, lora_r : 2 * lora_r]
            A = V[:lora_r, :]
        scaling_factor = module.scaling["default"]
        if init_config["scale"] == "gd":
            A = A / scaling_factor
            B = B / scaling_factor
        elif init_config["scale"] == "unit":
            # Because A,B is orthogonal, do not need to scale
            pass
        elif init_config["scale"] == "stable":
          if init_config["direction"] == "LoRA-One":
            gamma = init_config["stable_gamma"]
            B = B / gamma**0.5
            A = A / gamma**0.5
          else:
            m, n = grads.shape # m: feature_out, n: feature_in
            # the scale of output is only related to the feature_out
            gamma = init_config.stable_gamma
            B = B * m**0.25 / gamma**0.5
            A = A * m**0.25 / gamma**0.5
        elif init_config["scale"] == "weightS":
            _, S, _ = torch.svd_lowrank(module.weight.float(), q=4 * lora_r, niter=4)
            S = S / module.scaling["default"]
            avg_s = torch.sqrt(S[:lora_r]).mean().to(A.device)
            B = B * avg_s
            A = A * avg_s

        # construct new magnitude vectors if use DoRA
        # if peft_conf.get("dora", False):
        #    # temp matrix
        #    V = module.weight.float() + (peft_conf.lora_alpha/math.sqrt(lora_r)) * B @ A
        #    mag_vec = torch.norm(V, p=2, dim=1)
        # else:
        #    pass        

        module.lora_B.default.weight = torch.nn.Parameter(B.contiguous().cuda())
        module.lora_A.default.weight = torch.nn.Parameter(A.contiguous().cuda())
        # if peft_conf.get("dora", False):
        #    module.lora_magnitude_vector.default.weight = torch.nn.Parameter(mag_vec.contiguous().cuda())

    with torch.no_grad():
        # if peft_conf.get("dora", False): #DoRA uses fp16
        #         module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
        #             torch.float16
        #         )
        #         module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
        #             torch.float16
        #         )
        #         module.lora_magnitude_vector.default.weight.data = module.lora_magnitude_vector.default.weight.data.to(
        #             torch.float16
        #         )
        # else:
        # consider dtype not in init_config
        if "dtype" not in init_config:
            pass
        elif init_config["dtype"] == "bf16":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.bfloat16
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.bfloat16
            )
        elif init_config["dtype"] == "fp32":
            module.lora_A.default.weight.data = module.lora_A.default.weight.data.to(
                torch.float32
            )
            module.lora_B.default.weight.data = module.lora_B.default.weight.data.to(
                torch.float32
            )

        # If lora_A@lora_B is not zero, then we need to subtract lora_A@lora_B from the original weight matrix
        if init_config["direction"] == "LoRA-One":
          pass
        else:
          offset = (module.lora_B.default.weight @ module.lora_A.default.weight).to(
              module.weight.data.device
          )
          scaling_factor = module.scaling["default"]
          offset *= scaling_factor
          if "norm_clip" in init_config and init_config.norm_clip:
              # for numerical stability, offset's largest value must be less then weight's largest value
              ratio = torch.max(torch.abs(module.weight.data)) / torch.max(
                  torch.abs(offset)
              )
              if ratio < 1:
                  offset *= ratio
                  module.lora_A.default.weight.data *= ratio**0.5
                  module.lora_B.default.weight.data *= ratio**0.5
                  log.warning(f"Clipping offset by {ratio}")
          try:
              module.weight.data -= offset
          except:
              breakpoint()

def reinit_lora(model, init_config, additional_info):
    r"""
    Reinitialize the lora model with the given configuration.
    """
    for name, module in tqdm(
        model.named_modules(),
        desc="Reinitializing Lora",
        total=len(list(model.named_modules())),
    ):
        from peft.tuners.lora import LoraLayer
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # print(name)
            reinit_lora_modules(name, module, init_config, additional_info)

    return model
