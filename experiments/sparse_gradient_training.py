import os
from tqdm import tqdm
import argparse
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from  torch.distributed import rpc
import torch.multiprocessing as mp

import sys
sys.path.append(".")
from cfg import results_path
from tools import *
from algorithm.prune import global_prune, check_sparsity, extract_mask, custom_prune, remove_prune
from algorithm.zoo import cge_weight_allocate_to_process, cge_calculation, network_synchronize
from data import prepare_dataset
from models.tools import time_consumption_per_layer
from models.distributed_model import DistributedCGEModel

def main(args):
    # Misc
    device = f"cuda:{args.gpus[-1]}"
    set_seed(args.seed)
    exp = os.path.basename(__file__.split('.')[0])
    save_path = os.path.join(results_path, exp, gen_folder_name(args, ignore=['log', 'gpus', 'process_per_gpu', 'master_addr', 'master_port', 'momentum', 'weight_decay', 'sparsity_folder', 'sparsity_ckpt']))

    # Data
    loaders, class_num = prepare_dataset(args.dataset, args.batch_size)

    # Network
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

    # Load Lay-wise sparsity ckpt
    sparsity_ckpt = torch.load(os.path.join(args.sparsity_folder, args.network, args.sparsity_ckpt + '.pth'), map_location=device) if args.sparsity_ckpt is not None else None

    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=args.nesterov)
    global_length = (args.epoch - args.warmup_epochs) * len(loaders['train'])
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*global_length), int(0.75*global_length)], gamma=0.1)
    else:
        raise NotImplementedError(f'scheduler {args.scheduler} not implemented')

    # Makedir or Resume
    if args.log:
        try:
            os.makedirs(save_path, exist_ok=False)
            best_acc = 0.
            epoch = 0
        except FileExistsError:
            state_dict = torch.load(os.path.join(save_path, "ckpt.pth"), map_location=device)
            for key, val in state_dict["state_dicts"].items():
                eval(f"{key}.load_state_dict(val)")
            current_mask = state_dict["current_mask"]
            best_acc = state_dict["best_acc"]
            epoch = state_dict["epoch"]
        logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    else:
        epoch = 0

    # Init subprocess networks
    remote_networks = {}
    os.makedirs('.cache', exist_ok=True)
    cache_file_path = f'.cache/pruned_model_{args.master_port}.pth'
    torch.save(network.state_dict(), cache_file_path)
    for gpu in args.gpus:
        for i in range(args.process_per_gpu):
            remote_networks[f"{gpu}-{i}"] = rpc.remote(f"{gpu}-{i}", DistributedCGEModel,
                                                       args=(f"cuda:{gpu}", partial(network_init_func, **network_kwargs),
                                                             F.cross_entropy, param_name_to_module_id, cache_file_path, False))

    # Subprocess resume
    if epoch > 0:
        state_dict_to_restore = network.state_dict()
        custom_prune(network, current_mask)
        cge_weight_allocate_to_process(remote_networks, network, args.gpus, args.process_per_gpu, param_name_to_module_id, time_consumption_per_layer(args.network))
        remove_prune(network)
        network.load_state_dict(state_dict_to_restore)

    while epoch < args.epoch:
        epoch += 1
        if (epoch-1) % args.mask_shuffle_interval == 0:
            # ReGenerate Mask
            state_dict_to_restore = network.state_dict()
            if 0. < args.sparsity < 1.:
                global_prune(network, args.sparsity, args.score, class_num, loaders['train'], zoo_sample_size=192, zoo_step_size=5e-3, layer_wise_sparsity=sparsity_ckpt)
            elif args.sparsity == 0:
                pass
            else:
                raise ValueError('sparsity not valid')
            assert abs(args.sparsity - (1 - check_sparsity(network, if_print=False) / 100)) < 0.01, check_sparsity(network, if_print=False)
            current_mask = extract_mask(network.state_dict())
            cge_weight_allocate_to_process(remote_networks, network, args.gpus, args.process_per_gpu, param_name_to_module_id, time_consumption_per_layer(args.network))
            remove_prune(network)
            network.load_state_dict(state_dict_to_restore)
        # Train
        network.train()
        acc = AverageMeter()
        loss = AverageMeter()
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                desc=f"Epo {epoch} Training", ncols=160)
        for i, (x, y) in enumerate(pbar):
            if epoch <= args.warmup_epochs:
                warmup_lr(optimizer, epoch-1, i+1, len(loaders['train']), args.warmup_epochs, args.lr)
            x_cuda, y_cuda = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                fx = network(x_cuda, return_interval = False)
                loss_batch = F.cross_entropy(fx, y_cuda).cpu()
            lr = optimizer.param_groups[0]['lr']
            cge_calculation(remote_networks, network, args.gpus, args.process_per_gpu, x, y, lr if args.zoo_step_size == -1 else args.zoo_step_size)
            optimizer.step()
            network_synchronize(remote_networks, network, args.gpus, args.process_per_gpu)
            acc.update(torch.argmax(fx, 1).eq(y_cuda).float().mean().item(), y.size(0))
            loss.update(loss_batch.item(), y.size(0))
            if epoch > args.warmup_epochs:
                scheduler.step()
            pbar.set_postfix_str(f"Lr {lr:.2e} Acc {100*acc.avg:.2f}%")
        if args.log:
            logger.add_scalar("train/acc", acc.avg, epoch)
            logger.add_scalar("train/loss", loss.avg, epoch)

        # Test
        network.eval()
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=120)
        acc = AverageMeter()
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            acc.update(torch.argmax(fx, 1).eq(y).float().mean(), y.size(0))
            pbar.set_postfix_str(f"Acc {100*acc.avg:.2f}%")
        if args.log:
            logger.add_scalar("test/acc", acc.avg, epoch)

        # Save CKPT
        if args.log:
            state_dict = {
                "state_dicts":{
                    "network": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                "current_mask": current_mask,
                "epoch": epoch,
                "best_acc": best_acc,
            }
            if acc.avg > best_acc:
                best_acc = acc.avg
                state_dict['best_acc'] = best_acc
                torch.save(state_dict, os.path.join(save_path, 'best.pth'))
            torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))
        

def init_process(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    if rank == 0:
        rpc.init_rpc(
                f"master", rank=rank, world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.process_per_gpu*world_size+1, rpc_timeout=0.)
            )
        main(args)
    else:
        gpu = args.gpus[(rank-1)//args.process_per_gpu]
        i = (rank-1) % args.process_per_gpu
        rpc.init_rpc(
                f"{gpu}-{i}", rank=rank, world_size=world_size,
                rpc_backend_options=rpc.TensorPipeRpcBackendOptions(num_worker_threads=args.process_per_gpu*world_size+1, rpc_timeout=0.)        
            )
    rpc.shutdown()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p = process_cli(p)
    args = p.parse_args()

    args.gpus = args.gpus.split(',')
    world_size = 1 + len(args.gpus) * args.process_per_gpu
    mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)
