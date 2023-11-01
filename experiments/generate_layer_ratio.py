import os
import sys
import argparse
import torch
sys.path.append(".")
from algorithm.prune import global_prune, layer_sparsity
from data import prepare_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--network', type=str, choices=['resnet20'])
    p.add_argument('--method', type=str, choices=['zo_grasp', 'grasp'])
    p.add_argument('--sparsity', type=float, default=0.)
    args = p.parse_args()

    loaders, class_num = prepare_dataset("cifar10")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.network == "resnet20":
        from models.resnet_s import resnet20, param_name_to_module_id_rn20
        param_name_to_module_id = param_name_to_module_id_rn20
        network_init_func = resnet20
        network_kwargs = {
            'num_classes': class_num
        }
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network = network_init_func(**network_kwargs).to(device)
    loss_function = torch.nn.CrossEntropyLoss()

    global_prune(network, args.sparsity, args.method, 10, loaders['train'], zoo_sample_size=192, zoo_step_size=5e-3)

    os.makedirs(f"Layer_Sparsity/{args.network}", exist_ok=True)
    torch.save(layer_sparsity(network), f"Layer_Sparsity/{args.network}/{args.method}_{args.sparsity}.pth")