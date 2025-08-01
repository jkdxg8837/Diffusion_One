import torch
from tqdm import tqdm
import math
import itertools
from peft.tuners.lora.layer import Linear as LoraLinear
import logging
import numpy as np
from scipy.stats import norm
from peft.tuners.lora import LoraLayer
import os
import json
log = logging.getLogger(__name__)
from typing import Tuple, List, Dict
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from collections import defaultdict
def calculate_nfn_scores(model, batch, random_baseline=True):
    """
    Calculate NFN scores for all weight matrices.
    Args:
        model: Model to calculate NFN scores for.
        batch: Batch of problems.
        random_baseline: Whether to calculate the random baseline (this is True by default since it's needed for the NFN score).
    Returns:
        Dictionary of NFN scores for all weight matrices.
    """
    # Move batch to GPU if needed
    if next(model.parameters()).device != batch['input_ids'].device:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        
    # Initialize metrics dictionary
    metrics = defaultdict(dict)
    
    # Define hook function to calculate NFN scores for each weight matrix
    def hook_fn(name):
        """
        Hook function to calculate NFN scores for each weight matrix.
        Args:
            name: Name of the weight matrix.
        Returns:
            Hook function to calculate NFN scores for each weight matrix.
        """
        # Define inner hook function to calculate NFN scores for each weight matrix
        def hook(module, input, output):
            """
            Inner hook function to calculate NFN scores for each weight matrix.
            Args:
                module: Module to calculate NFN scores for.
                input: Input to the module.
                output: Output from the module (won't be used here).
            """
            if hasattr(module, 'weight') and module.weight is not None:
                # Get input and weight matrices
                z = input[0] if isinstance(input, tuple) else input
                W = module.weight
                z = z.float()
                W = W.float()
                
                # Reshape input if it's a 3D tensor
                if len(z.shape) > 2:
                    batch_size, seq_len, hidden_dim = z.shape
                    z = z.reshape(-1, hidden_dim)
                
                # Calculate NFN scores
                try:
                    # We calculate the Frobenius norm of W to normalize W for stability, but it is not necessary.
                    W_norm = (W**2).mean().sqrt()
                    z_norm = torch.norm(z, dim=1, keepdim=True)
                    W_normalized = W / (W_norm + 1e-8)
                    z_normalized = z / (z_norm + 1e-8)
                    Wz = torch.mm(z_normalized, W_normalized.t())
                    metrics[name]['actual'] = torch.norm(Wz, dim=1).mean().item()/np.sqrt(z.shape[1])
                    if random_baseline:
                        z_random = torch.randn_like(z_normalized)
                        z_random_norm = torch.norm(z_random, dim=1, keepdim=True)
                        z_random_normalized = z_random / (z_random_norm + 1e-8)
                        Wz_random = torch.mm(z_random_normalized, W_normalized.t())
                        metrics[name]['random'] = torch.norm(Wz_random, dim=1).mean().item()/np.sqrt(z.shape[1])
                    metrics[name]['nfn'] = metrics[name]['actual']/metrics[name]['random']
                except RuntimeError as e:
                    print(f"Error in layer {name}:")
                    print(f"Input shape: {z.shape}")
                    print(f"Weight shape: {W.shape}")
                    raise e
        return hook
    hooks = []
    for name, module in model.named_modules():
        embedding_filter = isinstance(module, torch.nn.Embedding)
        ln_filter = isinstance(module, torch.nn.LayerNorm) or 'norm' in name.lower()
        if hasattr(module, 'weight') and (module.weight is not None) and (not embedding_filter) and (not ln_filter):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        _ = model(**batch)
    for hook in hooks:
        hook.remove()

    return metrics

def get_group_metrics(metrics, groups=['add_q_proj', 'add_k_proj', 'add_v_proj', 'to_q', 'to_k', 'to_v'], individual=False):
    """
    Calculate group metrics.
    Args:
        metrics: Dictionary of NFN scores for all weight matrices.
        groups: List of groups to calculate metrics for.
    Returns:
        Dictionary of group metrics.
    """
    group_metrics = defaultdict(list)
    for group in groups:
        group_metrics[group] = []
    
    for name, values in metrics.items():
        for group in groups:
            if group in name and "lora_B" not in name and "base_layer" in name:
                group_metrics[group].append(values.get('nfn', 0.0))
    # results = {}
    # for group, data in group_metrics.items():
    #     count = data['count']
    #     if count > 0:
    #         if not individual:
    #             results[group] = {
    #             'actual': data['actual_sum'] / count,
    #                 'random': data['random_sum'] / count if 'random_sum' in data else 0.0,
    #                 'nfn': data['actual_sum'] / data['random_sum'] if 'random_sum' in data else 0.0
    #             }
    #         else:
    #             results[group] = {
    #                 'actual': data['actual_sum'] / count,
    #                 'random': data['random_sum'] / count if 'random_sum' in data else 0.0,
    #                 'nfn': data['nfn_sum'] / count
    #             }
    #     else:
    #         results[group] = {'actual': 0.0, 'random': 0.0, 'nfn': 0.0}
    # Optionally save group metrics to a file if needed
    save_path = "./group_metrics.json"
    with open(save_path, "w") as f:
        json.dump(group_metrics, f, indent=2)
    return group_metrics

def get_record_gradient_hook(model, record_dict):
    def record_gradient_hook(grad):
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                # print(p.grad.shape)
                if n not in record_dict:
                    record_dict[n] = [p.grad.cpu()]
                    
                else:
                    record_dict[n].append(p.grad.cpu())
                p.grad = None
                # Estimate gradient scale and variance
                # grads = torch.stack(record_dict[n], dim=0)
                # record_dict[n + "_scale"] = grads.abs().mean().item()
                # record_dict[n + "_var"] = grads.var().item()
        return grad

    return record_gradient_hook
metrics = defaultdict(dict)
def get_record_nfn_hook(name):
    """
    Hook function to calculate NFN scores for each weight matrix.
    Args:
        name: Name of the weight matrix.
    Returns:
        Hook function to calculate NFN scores for each weight matrix.
    """
    # Define inner hook function to calculate NFN scores for each weight matrix
    def hook(module, input, output):
        """
        Inner hook function to calculate NFN scores for each weight matrix.
        Args:
            module: Module to calculate NFN scores for.
            input: Input to the module.
            output: Output from the module (won't be used here).
        """
        if hasattr(module, 'weight') and module.weight is not None and "pos_embed" not in name:
            # Get input and weight matrices
            z = input[0] if isinstance(input, tuple) else input
            W = module.weight
            z = z.float()
            W = W.float()
            
            # Reshape input if it's a 3D tensor
            if len(z.shape) > 2:
                batch_size, seq_len, hidden_dim = z.shape
                z = z.reshape(-1, hidden_dim)
            
            # Calculate NFN scores
            try:
                # We calculate the Frobenius norm of W to normalize W for stability, but it is not necessary.
                W_norm = (W**2).mean().sqrt()
                z_norm = torch.norm(z, dim=1, keepdim=True)
                W_normalized = W / (W_norm + 1e-8)
                z_normalized = z / (z_norm + 1e-8)
                Wz = torch.mm(z_normalized, W_normalized.t())
                metrics[name]['actual'] = torch.norm(Wz, dim=1).mean().item()/np.sqrt(z.shape[1])
                # Random baseline
                z_random = torch.randn_like(z_normalized)
                z_random_norm = torch.norm(z_random, dim=1, keepdim=True)
                z_random_normalized = z_random / (z_random_norm + 1e-8)
                Wz_random = torch.mm(z_random_normalized, W_normalized.t())
                metrics[name]['random'] = torch.norm(Wz_random, dim=1).mean().item()/np.sqrt(z.shape[1])
                metrics[name]['nfn'] = metrics[name]['actual']/metrics[name]['random']
            except RuntimeError as e:
                print(f"Error in layer {name}:")
                print(f"Input shape: {z.shape}")
                print(f"Weight shape: {W.shape}")
                raise e
    return hook

def sample_with_matched_distribution(n=32, mean=0.0, std=1.0):
    shift=2.0
    # Get evenly spaced quantiles (excluding 0 and 1)
    quantiles = np.linspace(1 / (n + 1), n / (n + 1), n)
    # Compute quantile-matched normal values
    samples = norm.ppf(quantiles, loc=mean, scale=std)
    # Convert to torch tensor
    samples = torch.tensor(samples, dtype=torch.float32)
    samples = torch.nn.functional.sigmoid(samples)
    # samples = (shift*samples)/(1+(shift-1) * samples)
    # Center to mean=0 and std=1
    # samples = (samples - samples.mean()) / samples.std()
    # samples = 0.1 + 0.85 * samples 
    samples = (shift*samples)/(1+(shift-1) * samples)
    return samples


def print_gpu_memory_usage(device_id=0):
    allocated = torch.cuda.memory_allocated(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    ratio = allocated / total
    print(f"显存占用：{ratio:.2%} （{allocated / (1024 ** 2):.2f} MB / {total / (1024 ** 2):.2f} MB）")

def estimate_nfn(
    models, dataloader, args, noise_scheduler_copy, accelerator, text_encoders, tokenizers, batch_size: int = 4
) -> Dict[str, List[torch.Tensor]]:
    # named_grads = torch.load("/home/u5649209/workspace/Diffusion_One/named_grads/wo_sigmas/120.pt")
    # return named_grads
    
    # Using flowmatching noise scheduler inside sigma corresponding to the timesteps
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
    # for name, param in transformer.named_parameters():
    #     param.requires_grad = not param.requires_grad
    #     # print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
    log.info("Estimating gradient")
    transformer.train()
    for name, param in transformer.named_parameters():
        if "transformer_blocks.0" in name and "weight" in name:
            param.requires_grad = True

    # Debug: Check if transformer parameters require gradients
    grad_params = [p for p in transformer.parameters() if p.requires_grad]
    log.info(f"Number of parameters requiring gradients: {len(grad_params)}")
    if len(grad_params) == 0:
        log.warning("No parameters require gradients! This will cause the backward pass to fail.")
    
    nfn_scores = {}
    hooks = []
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    for name, module in transformer.named_modules():
        embedding_filter = isinstance(module, torch.nn.Embedding)
        ln_filter = isinstance(module, torch.nn.LayerNorm) or 'norm' in name.lower()
        if hasattr(module, 'weight') and (module.weight is not None) and (not embedding_filter) and (not ln_filter):
            hooks.append(module.register_forward_hook(get_record_nfn_hook(name)))
    num = 0
    weight_dtype = torch.float16
    # deal with time step
    # time_step = args.time_step
    # if args.re_init_schedule == "multi":
    #     epochs = args.re_init_samples // len(dataloader)
    #     print("********************************")
    #     print("len of dataloader is : ", len(dataloader))
    #     print(f"Reinitializing LoRA modules every {epochs} epochs")
    #     print("********************************")
    # else:
    #     epochs = args.re_init_samples // len(dataloader)

    epochs = 1

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    from tqdm import tqdm
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Estimating gradient"):
            print(batch)
            num += 1
            
            # batch = {k: v.to(transformer.device) for k, v in batch.items()}
            # Calculate diffusion model loss
            prompts = batch["prompts"]
            pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
            pixel_values = pixel_values.to(vae.device)
            print(pixel_values.shape)
            with torch.no_grad():
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            print(noise[0][0][0][:])

            u = sample_with_matched_distribution(n=bsz, mean=0, std=1.0)
    
            print("u is set to ", u)
            
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
            print("timesteps is ", timesteps)
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
            sigmas = sigmas.detach()
            # print(timesteps)
            print(sigmas)
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
            print("precondition outputs is ", args.precondition_outputs)
            args.precondition_outputs = 0
            if args.precondition_outputs:
                # model_pred = model_pred * (-sigmas) + noisy_model_input
                model_pred = model_pred * (-sigmas.detach()) + noisy_model_input

            # model_pred.loss.backward()
            # from diffusers.training_utils import compute_loss_weighting_for_sd3
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
            weighting = weighting.detach()
            print(weighting)
            # flow matching loss
            if args.precondition_outputs:
                target = model_input.detach()  
            else:
                target = noise - model_input.detach()  
            loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
            loss = loss.mean()
            print(loss.item())
            # loss.backward()
            print_gpu_memory_usage(0)
            # get_record_nfn_hook(transformer, nfn_scores)(None)  # get gradient of last layer
            # make sure the gradient is cleared
            for n, p in transformer.named_parameters():
                if p.grad is not None:
                    p.grad = None
        torch.cuda.empty_cache()

    from tqdm import tqdm
    
    for key in tqdm(nfn_scores.keys(), desc="Computing gradient averages"):
        try:
            # Stack all tensors in the list along dim=0 and compute mean
            # named_grads[key] size = epochs* [input_dim, output_dim]
            tensors = nfn_scores[key]
            nfn_scores[key] = torch.stack(tensors, dim=0).mean(dim=0)
        except Exception as e:
            log.error(f"Error processing key {key}: {e}")

    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    # Save metrics to a file
    save_path = getattr(args, "nfn_metrics_save_path", "./nfn_metrics.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in metrics.items()}, f, indent=2)
    return nfn_scores