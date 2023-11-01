from torch.nn import functional as F

def warmup_lr(optimizer, current_epoch, current_step, steps_per_epoch, warmup_epoch, base_lr):
    overall_steps = warmup_epoch * steps_per_epoch
    current_steps = current_epoch * steps_per_epoch + current_step
    lr = base_lr * current_steps/overall_steps
    for p in optimizer.param_groups:
        p['lr']=lr

def mean_squared_loss(x, y):
    y = F.one_hot(y) - 0.1
    return ( ( x - y )**2 ).mean()