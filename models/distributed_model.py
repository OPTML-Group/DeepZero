import torch
import sys
from functools import reduce

sys.path.append(".")
from algorithm.prune import extract_mask, custom_prune

class DistributedCGEModel(object):
    def __init__(self, device, network_init_func, loss_function, param_name_to_module_id=None, init_file=None, pruned=True, feature_reuse=True) -> None:
        super().__init__()
        self.device = device
        self.pruned = pruned
        self.feature_reuse = feature_reuse
        self.network = network_init_func().to(device)
        self.param_name_to_module_id = param_name_to_module_id
        if init_file is not None:
            init_state_dict = torch.load(init_file, map_location=self.device)
            mask = extract_mask(init_state_dict)
            custom_prune(self.network, mask)
            self.network.load_state_dict(init_state_dict)
        self.loss_function = loss_function
        self.network.requires_grad_(False)

    @torch.no_grad()
    def calculate_grads(self, x_rref, y_rref, cge_step_size):
        assert hasattr(self, 'instruction')
        x, y = x_rref.to_here().to(self.device), y_rref.to_here().to(self.device)
        fxs = [x]
        with torch.no_grad():
            fxs += self.network(x, return_interval = self.feature_reuse) if self.feature_reuse else [self.network(x, return_interval = self.feature_reuse)]
            base = self.loss_function(fxs[-1], y)
        grads = torch.zeros(self.instruction.size(0), device=self.device)
        for i, (name_id, idx) in enumerate(self.instruction):
            self.perturb_a_param(self.param_names[name_id], idx, cge_step_size)
            starting_id = self.param_name_to_module_id(self.param_names[name_id]) if self.feature_reuse else 0
            with torch.no_grad():
                fx = self.network(fxs[starting_id], starting_id=starting_id)
            self.perturb_a_param(self.param_names[name_id], idx, cge_step_size, reset=True)
            grads[i] = self.loss_function(fx, y)
        grads = (grads - base) / cge_step_size
        grads = grads.cpu()
        return torch.cat([self.instruction, grads.unsqueeze(1)], dim=1)

    def synchronize(self, params_to_be_updated_rref):
        params_to_be_updated = params_to_be_updated_rref.to_here().to(self.device)
        for name_id, key in enumerate(self.param_names):
            params_indices_and_values = params_to_be_updated[params_to_be_updated[:, 0]==name_id, 1:]
            names, attr = key.split('.')[:-1], key.split('.')[-1]
            module = self.get_module_by_name(self.network, names)
            param = getattr(module, attr).flatten()
            param[params_indices_and_values[:, 0].long()] = params_indices_and_values[:, 1]

    def set_params_to_be_perturbed(self, instruction_rref, names):
        self.instruction = instruction_rref.to_here().long()
        self.set_param_names(names)

    def set_param_names(self, names):
        if not self.pruned:
            names = [name.replace('_orig', '') for name in names]
        self.param_names = names

    def perturb_a_param(self, key, idx, cge_step_size, reset=False):
        names, attr = key.split('.')[:-1], key.split('.')[-1]
        module = self.get_module_by_name(self.network, names)
        param = getattr(module, attr).flatten()
        if reset:
            param[idx] -= cge_step_size
        else:
            param[idx] += cge_step_size

    @staticmethod
    def get_module_by_name(module, names):
        return reduce(getattr, names, module)