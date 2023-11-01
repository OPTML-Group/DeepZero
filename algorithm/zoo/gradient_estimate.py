import torch
from copy import deepcopy

@torch.no_grad()
def f(params_dict, network, x, y, loss_func):
    state_dict_backup = network.state_dict()
    network.load_state_dict(params_dict, strict=False)
    loss = loss_func(network(x), y).detach().item()
    network.load_state_dict(state_dict_backup)
    return loss

@torch.no_grad()
def rge(func, params_dict, sample_size, step_size, base=None):
    if base == None:
        base = func(params_dict)
    grads_dict = {}
    for _ in range(sample_size):
        perturbs_dict, perturbed_params_dict = {}, {}
        for key, param in params_dict.items():
            perturb = torch.randn_like(param)
            perturb /= (torch.norm(perturb) + 1e-8)
            perturb *= step_size
            perturbs_dict[key] = perturb
            perturbed_params_dict[key] = perturb + param
        directional_derivative = (func(perturbed_params_dict) - base) / step_size
        if len(grads_dict.keys()) == len(params_dict.keys()):
            for key, perturb in perturbs_dict.items():
                grads_dict[key] += perturb * directional_derivative / sample_size
        else:
            for key, perturb in perturbs_dict.items():
                grads_dict[key] = perturb * directional_derivative / sample_size
    return grads_dict

@torch.no_grad()
def cge(func, params_dict, mask_dict, step_size, base=None):
    if base == None:
        base = func(params_dict)
    grads_dict = {}
    for key, param in params_dict.items():
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        for idx in mask_flat.nonzero().flatten():
            perturbed_params_dict = deepcopy(params_dict)
            p_flat = perturbed_params_dict[key].flatten()
            p_flat[idx] += step_size
            directional_derivative_flat[idx] = (func(perturbed_params_dict) - base) / step_size
        grads_dict[key] = directional_derivative.to(param.device)
    return grads_dict