from torch import nn

def split_model(model):
    modules = []
    for m in model.children():
        if isinstance(m, (nn.Sequential,)):
            modules += split_model(m)
        else:
            modules.append(m)
    return modules

def time_consumption_per_layer(network = 'resnet20'):
    if network == 'resnet20':
        return [546, 508, 482, 416, 365, 318, 257, 210, 160, 102, 57, 11] # Calculated by 10 time test average
    else:
        raise NotImplementedError