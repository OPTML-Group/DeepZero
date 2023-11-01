import torch

def generate_data_index_selected_class(retain_classes, train_ys_file = 'results/train_ys.pth', val_ys_file = 'results/val_ys.pth'):
    train_ys = torch.load(train_ys_file)
    val_ys = torch.load(val_ys_file)

    train_subset_indices = []
    val_subset_indices = []
    for c in retain_classes.flatten():
        train_subset_indices.append(torch.where(train_ys == c)[0])
        val_subset_indices.append(torch.where(val_ys == c)[0])
    train_subset_indices = torch.cat(train_subset_indices).unsqueeze(1)
    val_subset_indices = torch.cat(val_subset_indices).unsqueeze(1)

    return train_subset_indices, val_subset_indices

def split_data_and_move_to_device(data, device):
    def process(x):
        return x[0].to(device) if isinstance(x, list) else x.to(device)
    if isinstance(data, dict):
        results = [process(x) for x in data.values()]
    elif isinstance(data, list):
        results = [process(x) for x in data]
    return results