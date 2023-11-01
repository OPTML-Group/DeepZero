from random import shuffle
import torch
from torch.distributed import rpc

from .tools import weighted_allocate

def network_synchronize(remote_networks, network, gpus, process_per_gpu):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    mask_dict = {
        name: p for name, p in network.named_buffers() if 'mask' in name
    }
    params_to_be_updated = []
    for name_id, (key, param) in enumerate(params_dict.items()):
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        param_flat = param.flatten()
        indices = mask_flat.nonzero().flatten().tolist()
        params_to_be_updated += [[name_id, idx, param_flat[idx].item()] for idx in indices]
    params_to_be_updated_rref = rpc.RRef(torch.Tensor(params_to_be_updated))
    for gpu in gpus:
        for i in range(process_per_gpu):
            remote_networks[f"{gpu}-{i}"].rpc_async().synchronize(params_to_be_updated_rref)

def cge_weight_allocate_to_process(remote_networks, network, gpus, process_per_gpu, param_name_to_module_id, time_consumption_per_layer):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    mask_dict = {
        name: p for name, p in network.named_buffers() if 'mask' in name
    }
    whole_size = len(gpus) * process_per_gpu
    params_to_be_perturbed = []
    time_consumption = []
    for name_id, (key, param) in enumerate(params_dict.items()):
        if 'orig' in key:
            mask_key = key.replace('orig', 'mask')
            mask_flat = mask_dict[mask_key].flatten()
        else:
            mask_flat = torch.ones_like(param).flatten()
        indices = mask_flat.nonzero().flatten().tolist()
        params_to_be_perturbed += [(name_id, idx) for idx in indices]
        time_consumption += [time_consumption_per_layer[param_name_to_module_id(key)] for _ in indices]
    params_to_be_perturbed = weighted_allocate(params_to_be_perturbed, time_consumption, whole_size)
    params_set_signal = []
    param_names = list(params_dict.keys())
    for j, gpu in enumerate(gpus):
        for i in range(process_per_gpu):
            idx = (j * process_per_gpu) + i
            params_to_be_perturbed[idx] = rpc.RRef(torch.Tensor(params_to_be_perturbed[idx]))
            params_set_signal.append(remote_networks[f"{gpu}-{i}"].rpc_async().set_params_to_be_perturbed(params_to_be_perturbed[idx], param_names))
    for pss in params_set_signal:
        pss.wait()

def cge_calculation(remote_networks, network, gpus, process_per_gpu, x, y, cge_step_size):
    params_dict = {
        name: p for name, p in network.named_parameters() if p.requires_grad
    }
    device = next(network.parameters()).device
    x_rref, y_rref = rpc.RRef(x), rpc.RRef(y)
    grads_signal = []
    for gpu in gpus:
        for i in range(process_per_gpu):
            grads_signal.append(remote_networks[f"{gpu}-{i}"].rpc_async(timeout=0).calculate_grads(x_rref, y_rref, cge_step_size))
    grads = []
    for g in grads_signal:
        grads.append(g.wait())
    grads = torch.cat(grads, dim=0).to(device)
    for name_id, (_, param) in enumerate(params_dict.items()):
        param.grad = torch.zeros_like(param)
        grads_indices_and_values = grads[grads[:, 0]==name_id, 1:]
        param_grad_flat = param.grad.flatten()
        param_grad_flat[grads_indices_and_values[:, 0].long()] = grads_indices_and_values[:, 1]

