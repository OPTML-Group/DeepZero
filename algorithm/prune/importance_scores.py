import torch
from torch.autograd import grad
from functools import partial

from .tools import fetch_data, extract_conv2d_and_linear_weights
from ..zoo import rge, f

def random_importance_score(
    model
    ):
    score_dict = {}
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            score_dict[(m, 'weight')] = torch.randn_like(m.weight)
    return score_dict

def grasp_importance_score(
    model,
    dataloader,
    samples_per_class,
    class_num,
    loss_func = torch.nn.CrossEntropyLoss()
    ):

    temperature = 200
    score_dict = {}
    model.zero_grad()
    device = next(model.parameters()).device
    x, y = fetch_data(dataloader, class_num, samples_per_class)
    x, y = x.to(device), y.to(device)
    loss = loss_func(model(x) / temperature, y)
    gs = grad(loss, model.parameters(), create_graph=True)
    model.zero_grad()
    t = sum([(g*g.data).sum() for g in gs])
    t.backward()

    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(m, "weight_orig"):
                score_dict[(m, 'weight')] = (m.weight_orig.grad.clone().detach() * m.weight.clone().detach()).abs()
            else:
                score_dict[(m, 'weight')] = (m.weight.grad.clone().detach() * m.weight.clone().detach()).abs()
    model.zero_grad()
    for g in gs:
        del g.grad
    return score_dict

def zoo_grasp_importance_score(
    model,
    dataloader,
    samples_per_class,
    class_num,
    zoo_rs_size,
    zoo_step_size,
    loss_func = torch.nn.CrossEntropyLoss()
    ):

    score_dict = {}
    device = next(model.parameters()).device
    x, y = fetch_data(dataloader, class_num, samples_per_class)
    x, y = x.to(device), y.to(device)

    params = extract_conv2d_and_linear_weights(model)
    
    f_theta = partial(f, network=model, x=x, y=y, loss_func=loss_func)

    g0 = rge(f_theta, params, zoo_rs_size, zoo_step_size)
    modified_params = {}
    for key, param in params.items():
        modified_params[key] = param.data + g0[key].data * zoo_step_size
    g1 = rge(f_theta, modified_params, zoo_rs_size, zoo_step_size)
    Hg = {}
    for key, param in params.items():
        Hg[key] = (g1[key].data - g0[key].data) / zoo_step_size

    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            if hasattr(m, "weight_orig"):
                score_dict[(m, 'weight')] = -m.weight_orig.clone().detach() * Hg[f'{name}.weight_orig']
            else:
                score_dict[(m, 'weight')] = -m.weight.clone().detach() * Hg[f'{name}.weight']

    return score_dict