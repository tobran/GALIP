import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image,make_grid
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data.distributed import DistributedSampler
import multiprocessing as mp

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p,get_rank,merge_args_yaml,get_time_stamp,save_args
from lib.utils import load_netG,load_npz,save_models
from lib.perpare import prepare_dataloaders
from lib.perpare import prepare_models
from lib.modules import test as test


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='../cfg/coco.yml',
                        help='optional config file')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers(default: {0})'.format(mp.cpu_count() - 1))
    parser.add_argument('--stamp', type=str, default='normal',
                        help='the stamp of model')
    parser.add_argument('--pretrained_model_path', type=str, default='model',
                        help='the model for training')
    parser.add_argument('--log_dir', type=str, default='new',
                        help='file path to log directory')
    parser.add_argument('--model', type=str, default='GALIP',
                        help='the model for training')
    parser.add_argument('--state_epoch', type=int, default=100,
                        help='state epoch')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--train', type=str, default='True',
                        help='if train model')
    parser.add_argument('--mixed_precision', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--multi_gpus', type=str, default='False',
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
                        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def main(args): 
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(args.model),str(args.stamp),str(args.CONFIG_NAME),str(args.imsize),time_stamp])
    log_dir = osp.join(ROOT_PATH, 'logs/{0}'.format(osp.join(str(args.CONFIG_NAME), 'train', stamp)))
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        mkdir_p(osp.join(ROOT_PATH, 'logs'))
    # prepare TensorBoard
    if (args.multi_gpus==True) and (get_rank() != 0):
        writer = None
    else:
        writer = SummaryWriter(log_dir)
    # Build and load the generator
    # prepare dataloader, models, data
    train_dl, valid_dl ,train_ds, valid_ds, sampler = prepare_dataloaders(args)
    CLIP4trn, CLIP4evl, image_encoder, text_encoder, netG, netD, netC = prepare_models(args)
    state_path = args.pretrained_model_path
    multi_gpus = args.multi_gpus
    m1, s1 = load_npz(args.npz_path)
    netG = load_netG(netG, state_path, multi_gpus, args.train)

    save_models(netG, netD, netC, 0, args.multi_gpus, './tmp')

    netG.eval()
    FID, TI_score = test(valid_dl, text_encoder, netG, CLIP4evl, args.device, m1, s1, -1, -1, \
                    args.sample_times, args.z_dim, args.batch_size)
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('FID: %.2f, CLIP_Score: %.2f' % (FID, TI_score*100))


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
        #args.manualSeed = random.randint(1, 10000)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)


