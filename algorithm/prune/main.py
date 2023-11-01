import torch
from torch.nn.utils import prune
from copy import deepcopy
from .importance_scores import zoo_grasp_importance_score, grasp_importance_score, random_importance_score

__all__ = ['global_prune', 'check_sparsity', 'check_grad_sparsity', 'custom_prune', 'extract_mask', 'remove_prune', 'layer_sparsity']

def global_prune(model, ratio, method, class_num=None, dataloader=None, sample_per_classes=25, zoo_sample_size=None, zoo_step_size=None, layer_wise_sparsity=None):
    if method == 'grasp':
        score_dict = grasp_importance_score(model, dataloader, sample_per_classes, class_num)
        prune.global_unstructured(
            parameters=score_dict.keys(),
            pruning_method=prune.L1Unstructured,
            amount=ratio,
            importance_scores=score_dict,
        )
    elif method == 'zo_grasp':
        score_dict = zoo_grasp_importance_score(model, dataloader, sample_per_classes, class_num, zoo_sample_size, zoo_step_size)
        prune.global_unstructured(
            parameters=score_dict.keys(),
            pruning_method=prune.L1Unstructured,
            amount=ratio,
            importance_scores=score_dict,
        )
    elif method == 'random':
        score_dict = random_importance_score(model)
        prune.global_unstructured(
            parameters=score_dict.keys(),
            pruning_method=prune.L1Unstructured,
            amount=ratio,
            importance_scores=score_dict,
        )
    elif method == 'layer_wise_random':
        if layer_wise_sparsity is None:
            raise ValueError(f"Sparsity ckpt is None!!!")
        layer_wise_sparsity_dict = layer_wise_sparsity
        for name, module in model.named_modules():
            if name in layer_wise_sparsity_dict.keys():
                prune.random_unstructured(module, 'weight', layer_wise_sparsity_dict[name])   
    else:
        raise NotImplementedError(f'Pruning Method {method} not Implemented')


def check_sparsity(model, if_print=False):
    sum_list = 0
    zero_sum = 0

    for m in model.modules():
        if prune.is_pruned(m) and hasattr(m, 'weight'):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    if zero_sum:
        remain_weight_ratio = 100*(1-zero_sum/sum_list)
        if if_print:
            print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        if if_print:
            print('no weight for calculating sparsity')
        remain_weight_ratio = 100

    return remain_weight_ratio

def check_grad_sparsity(model, if_print=False):
    sum_list = 0
    zero_sum = 0

    for m in model.modules():
        if prune.is_pruned(m) and hasattr(m, 'weight'):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight.grad == 0))  

    if zero_sum:
        remain_weight_ratio = 100*(1-zero_sum/sum_list)
        if if_print:
            print('* remain weight ratio = ', 100*(1-zero_sum/sum_list),'%')
    else:
        if if_print:
            print('no weight for calculating sparsity')
        remain_weight_ratio = 100

    return remain_weight_ratio

def layer_sparsity(model, if_print=False):
    sum_list_all = 0
    zero_sum_all = 0
    sparsity_ckpt = {}

    for name, m in model.named_modules():

        if prune.is_pruned(m) and hasattr(m, 'weight'):
            sum_list_all = sum_list_all + float(m.weight.nelement())
            zero_sum_all = zero_sum_all + float(torch.sum(m.weight == 0))

            sum_list = float(m.weight.nelement())
            zero_sum = float(torch.sum(m.weight == 0))

            layer_sparsity_rate = zero_sum/sum_list
            sparsity_ckpt[name] = layer_sparsity_rate

    if zero_sum_all:
        remain_weight_ratio = 100*(1-zero_sum_all/sum_list_all)
        if if_print:
            print('============= Layer-wise Sparsity ===========')
            print('* global remain weight ratio = ', 100*(1-zero_sum_all/sum_list_all),'%')
            print(sparsity_ckpt)
    else:
        if if_print:
            print('no weight for calculating sparsity')

    return sparsity_ckpt

def custom_prune(model, mask_dict):
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            mask_name = name + '.weight_mask'
            if mask_name in mask_dict.keys():
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'])
            else:
                print('can not find [{}] in mask_dict, skipping'.format(mask_name))

def extract_mask(model_dict):
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = deepcopy(model_dict[key])
    return new_dict

def remove_prune(model):
    for m in model.modules():
        if prune.is_pruned(m) and hasattr(m, 'weight'):
            prune.remove(m, 'weight')
