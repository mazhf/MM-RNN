import os
from config import cfg

# gpus 需要放在torch之前
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

import torch
from torch import nn
from model import Model
from loss import Weighted_mse_mae
from train_and_test import train_and_test
from net_params import params
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from util.load_data import load_data
import argparse
from collections import OrderedDict


# fix init
def fix_random(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)  # 固定random.random()生成的随机数
    np.random.seed(seed)  # 固定np.random()生成的随机数
    torch.manual_seed(seed)  # 固定CPU生成的随机数
    torch.cuda.manual_seed(seed)  # 固定GPU生成的随机数-单卡
    torch.cuda.manual_seed_all(seed)  # 固定GPU生成的随机数-多卡
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


fix_random(2021)

# params
gpu_nums = cfg.gpu_nums
batch_size = cfg.batch
train_epoch = cfg.train_epoch
valid_epoch = cfg.valid_epoch
save_checkpoint_epoch = cfg.valid_epoch
LR = cfg.LR

# 设置并行——以下设置顺序不可颠倒！run: python -m torch.distributed.launch --nproc_per_node=4 --master_port 39985 main.py
# torch.distributed.launch arguments 该参数只能这么用，其他参数只能放在命令行。。。
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='node rank for distributed training')
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)
print('local_rank: ', args.local_rank)

# parallel group
torch.distributed.init_process_group(backend="nccl")

# model
model = Model(params[0], params[1], params[2]).cuda()

# load checkpoint
if cfg.resume:
    multi_GPU_dict = torch.load(cfg.resume_pth, map_location='cuda:{}'.format(args.local_rank))
    single_GPU_dict = OrderedDict()
    for k, v in multi_GPU_dict.items():
        single_GPU_dict[k[7:]] = v  # 去掉module.
    model.load_state_dict(single_GPU_dict, strict=True)
    start_epoch = int(cfg.resume_pth.split('.')[0].split('/')[-1].split('_')[1])
else:
    start_epoch = 0

# 加入module.
model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[args.local_rank],
                                            output_device=args.local_rank)

# dataloader DataLoader的shuffle和DistributedSampler的shuffle为True只能使用一个，valid和test可以shuffle，但是为了取test demo就不了
threads = cfg.dataloader_thread
train_data, valid_data, test_data = load_data()
train_sampler = DistributedSampler(train_data, shuffle=True)
valid_sampler = DistributedSampler(valid_data, shuffle=False)
train_loader = DataLoader(train_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=train_sampler)
test_loader = DataLoader(test_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_loader = DataLoader(valid_data, num_workers=threads, batch_size=batch_size, shuffle=False, pin_memory=True,
                          sampler=valid_sampler)
loader = [train_loader, test_loader, valid_loader]

# optimizer
if cfg.optimizer == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
elif cfg.optimizer == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
else:
    optimizer = None
# loss
criterion = Weighted_mse_mae().cuda()

# train valid test
train_and_test(model, optimizer, criterion, start_epoch, train_epoch, valid_epoch, save_checkpoint_epoch, loader,
               train_sampler)
