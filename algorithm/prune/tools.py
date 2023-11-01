import torch
from torch.nn.utils import prune

def fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break
    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    del dataloader_iter
    return X, y


def extract_conv2d_and_linear_weights(model):
    
    if prune.is_pruned(model):
        return {
            f'{name}.weight_orig': m.weight_orig for name, m in model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        }
    else:
        return {
            f'{name}.weight': m.weight for name, m in model.named_modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        }