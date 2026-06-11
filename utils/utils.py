import yaml
import os
import shutil
import argparse
import math
import numpy as np
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
import random
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt


def eval_this_epoch(epoch, eval_freq: dict):
    if epoch == 0:
        return True

    ef = 1
    for v in eval_freq:
        if epoch < int(v):
            ef = eval_freq[v]
            break

    if (epoch + 1) % ef == 0:
        return True

    return False


def dict_to_str(data):
    msg = f''
    for name in data:
        msg += f'{name}:{data[name]}, '
    msg = msg[:-2]
    return msg


def load_yaml(args, yml):
    with open(yml, 'r', encoding='utf-8') as fyml:
        dic = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dic:
            setattr(args, k, dic[k])


def build_record_folder(args):

    if not os.path.isdir("./records/"):
        os.mkdir("./records/")
    
    args.save_dir = "./records/" + args.project_name + "/" + args.exp_name + "/"
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.save_dir + "backup/", exist_ok=True)
    os.makedirs(args.save_dir + "conf_mat/", exist_ok=True)
    os.makedirs(args.save_dir + "cdf/", exist_ok=True)

    shutil.copy(args.c, args.save_dir+"config.yaml")


def get_args():

    parser = argparse.ArgumentParser("Soccer Player Action Detection Task")
    parser.add_argument("--c", default="", type=str, help="config file path")
    args = parser.parse_args()

    load_yaml(args, args.c)
    build_record_folder(args)

    return args


def get_schedule(schedule_info, max_epochs, train_batchs, warmup_batchs=0):
    """
    train_batchs: The number of batchs within one epoch
    """

    assert schedule_info['name'] in ['cosine_decay', 'cosine_increase', \
                                     'step_decay', 'step_increase', \
                                     'linear_decay', 'linear_increase']

    if 'increase' in schedule_info['name']:
        warmup_batchs = 0

    if 'cosine' in schedule_info['name']:
        lr_schedule = cosine_decay(max_epochs, schedule_info['max_value'], schedule_info['min_value'], 
            train_batchs, warmup_batchs)
    elif 'step' in schedule_info['name']:
        lr_schedule = step_decay(max_epochs, schedule_info['max_value'], schedule_info['min_value'], 
            schedule_info['steps'], schedule_info['step_decay_scale'], train_batchs, warmup_batchs)
    elif 'linear' in schedule_info['name']:
        lr_schedule = linear_decay(max_epochs, schedule_info['max_value'], schedule_info['min_value'], 
            schedule_info['linear_type'], train_batchs, warmup_batchs)

    if 'increase' in schedule_info['name']:
        lr_schedule = lr_schedule.max() - lr_schedule + lr_schedule.min()

    return lr_schedule


def cosine_decay(max_epochs, max_lr, min_lr, batchs: int, warmup_batchs: int, decay_type: int = 1):
      
    assert decay_type in [1, 2]

    total_batchs = max_epochs * batchs
    iters = np.arange(total_batchs - warmup_batchs)

    if decay_type == 1:
        schedule = np.array([min_lr + 0.5 * (max_lr - min_lr) * (1 + \
                             math.cos(math.pi * t / total_batchs)) for t in iters])
    elif decay_type == 2:
        schedule = max_lr * np.array([math.cos(7 * math.pi * t / (16 * total_batchs)) for t in iters])
    
    if warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(min_lr, max_lr, warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule


def step_decay(max_epochs, max_lr, min_lr, decay_at_epochs, step_decay_scale, batchs: int, warmup_batchs: int):
    
    total_batchs = max_epochs * batchs
    iters = np.arange(total_batchs - warmup_batchs)

    i = 0
    lr = max_lr
    lr_list = []
    for t in iters:
        if i < len(decay_at_epochs) and t == decay_at_epochs[i] * batchs - warmup_batchs:
            lr *= step_decay_scale
            i += 1
        lr_list.append(lr)
    
    schedule = np.array(lr_list)
    if warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(min_lr, max_lr, warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule


def linear_decay(max_epochs, max_lr, min_lr, liner_type, batchs: int, warmup_batchs: int):
    
    assert liner_type in ['epoch', 'batch']
    assert not (liner_type == 'linear' and max_epochs == 1)

    total_batchs = max_epochs * batchs
    iters = np.arange(total_batchs - warmup_batchs)

    assert not (liner_type == 'batch' and total_batchs == 1)

    lr_list = []
    present_epoch = None
    for t in iters:
        if liner_type == 'epoch':
            present_batchs = t + warmup_batchs
            present_epoch = present_batchs // batchs
            _lr = max_lr - (max_lr - min_lr) * (present_epoch / (max_epochs - 1))
            lr_list.append(_lr)
        elif liner_type == 'batch':
            _lr = max_lr - (max_lr - min_lr) * t / (total_batchs - warmup_batchs - 1)
            lr_list.append(_lr)

    schedule = np.array(lr_list)
    if warmup_batchs > 0:
        warmup_lr_schedule = np.linspace(min_lr, max_lr, warmup_batchs)
        schedule = np.concatenate((warmup_lr_schedule, schedule))

    return schedule


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]


def adjust_lr(iteration, schedule, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = schedule[iteration]


def get_value(iteration, schedule):
    t = schedule[iteration]
    return t


def visualized(data, epoch, args, device):
# def visualized(data, epoch, args, pred, label, img_index):
    device = args.device
    data = data.to(device)

    # Ensure the epoch subfolder exists within combined_data
    epoch_dir = os.path.join(args.save_dir, "vis_data", f"ep{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    batch_size, c, t, h, w = data.shape
    for i in range(batch_size):
        fig, axs = plt.subplots(1, t, figsize=(20, 4))

        for j in range(t):
            img = data[i, :, j, :, :].permute(1, 2, 0).detach().cpu().numpy()  # Convert to numpy array
            img = (img * 255.0).astype(np.float32)

            axs[j].imshow(img)
            axs[j].axis('off')
        # Save the figure
        plt.savefig(os.path.join(epoch_dir, f'{i}.jpg'))
        # # file_name = f'{i}_pred_{pred}_label_{label}.jpg'
        # file_name = f'{img_index}_pred_{pred}_label_{label}.jpg'
        # plt.savefig(os.path.join(epoch_dir, file_name))
        plt.close()

# def visualized(data, epoch, args, pred, label, img_index):
   
#     base_vis_dir = os.path.join(args.save_dir, "vis_data", f"ep{epoch}")
#     folder_name = f"label_{label}_pred_{pred}"
#     folder_path = os.path.join(base_vis_dir, folder_name)

#     os.makedirs(folder_path, exist_ok=True)

#     batch_size, c, t, h, w = data.shape
#     for i in range(batch_size):
#         fig, axs = plt.subplots(1, t, figsize=(20, 4))

#         for j in range(t):
#             img = data[i, :, j, :, :].permute(1, 2, 0).detach().cpu().numpy()  # Convert to numpy array
#             img = (img * 255.0).astype(np.float32)

#             axs[j].imshow(img)
#             axs[j].axis('off')

#         file_name = f'{img_index}.jpg'
#         plt.savefig(os.path.join(folder_path, file_name))
#         plt.close()



def set_seed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False     












