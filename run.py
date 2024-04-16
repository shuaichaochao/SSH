import torch
import argparse

from loguru import logger
from data.data_loader import load_data
import os
import numpy as np
import random
from torchcam.methods import CAM
from torchcam.methods import GradCAM
from torchvision.io.image import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import normalize, resize, to_pil_image, center_crop
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask
from models.baseline import Baseline

import baseline_train as baseline
# import cv2
# from skimage import io
import torchvision.utils as vutils



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def run():
    seed_everything(68)
    args = load_config()

    # if not os.path.exists('logs/' + args.info):
    #     os.makedirs('logs/' + args.info)
    
    logger.add('logs/{time}_' + args.info + '_' + args.dataset  +'.log', rotation='50 MB', level='DEBUG')
    logger.info(args)


    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.batch_size,
        args.num_workers,
    )

    if args.arch == 'baseline':
        net_arch = baseline


    for code_length in args.code_length:
        mAP = net_arch.train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args)
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, mAP))



def load_config():
    """
    Load configuration.
    Args
        None
    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='CSH_PyTorch')
    parser.add_argument('--dataset', default="fgvc-aircraft-2013b", help='Dataset name(CUB_200_2011,fgvc-aircraft-2013b,nabirds,stanforddog,food-101,vegfru).')
    # parser.add_argument('--root',default="./dataset/fine-grained/CUB_200_2011/",help='Path of dataset')
    parser.add_argument('--root', default="./dataset/fine-grained/fgvc-aircraft-2013b/", help='Path of dataset')
    # parser.add_argument('--root', default="./dataset/fine-grained/food-101/",help='Path of dataset')
    # parser.add_argument('--root', default="./dataset/fine-grained/food-101/",help='Path of dataset')
    # parser.add_argument('--root', default="./dataset/fine-grained/vegfru/", help='Path of dataset')
    # parser.add_argument('--root', default="./dataset/fine-grained/nabirds/",help='Path of dataset')
    # parser.add_argument('--true_hash', default="dataset/CUB_200_2011/",help='Path of true_hash path')
    parser.add_argument('--true_hash', default="dataset/fgvc-aircraft-2013b/", help='Path of true_hash path')
    # parser.add_argument('--true_hash', default="dataset/food-101/", help='Path of true_hash path')
    # parser.add_argument('--true_hash', default="dataset/nabirds/", help='Path of true_hash path')
    # parser.add_argument('--true_hash', default="dataset/vegfru/", help='Path of true_hash path')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.(default: 16)')
    # parser.add_argument('--lr', default=2.5e-5, type=float, help='Learning rate.(default: 1e-4), food-101 is 7.5e-5')
    parser.add_argument('--lr', default=2e-5, type=float, help='Learning rate.(default: 1e-4)')
    parser.add_argument('--wd', default=1e-4, type=float, help='Weight Decay.(default: 1e-4)')
    parser.add_argument('--optim', default='RMSprop', type=str, help='Optimizer(SGD or Adam or RMSprop)')
    parser.add_argument('--code_length', default='12,24,32,48', type=str, help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max_iter', default=660, type=int, help='Number of iterations.(default: 40)')
    # parser.add_argument('--max_iter', default=40, type=int,help='Number of iterations.(default: 40)')
    parser.add_argument('--max_epoch', default=60, type=int, help='Number of epochs.(default: 30)')
    parser.add_argument('--num_samples', default=2000, type=int, help='Number of sampling data points.(default: 2000)')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of loading data threads.(default: 4)')
    parser.add_argument('--topk', default=-1, type=int, help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=2, type=int, help='Using gpu.(default: False)')
    parser.add_argument('--gamma', default=0.05, type=float, help='Hyper-parameter.(default: 200)')
    # parser.add_argument('--info', default='CSH_centerloss_add_MLPattn', help='Train info')
    # parser.add_argument('--info', default='MDSHC', help='Train info')
    parser.add_argument('--info', default='CSH_centerloss_crossattn_add', help='Train info')
    # parser.add_argument('--info', default='Nest_centerloss-B', help='Train info')
    # parser.add_argument('--info', default='ResNet50_centerloss-B', help='Train info')
    parser.add_argument('--arch', default='baseline', help='Net arch (semicon or baseline)')
    parser.add_argument('--save_ckpt', default='checkpoint/', help='result_save')
    parser.add_argument('--lr_step', default='40', type=str, help='lr decrease step.(default: 40)')
    parser.add_argument('--align_step', default=50, type=int, help='Step of start aligning.(default: 50)')
    parser.add_argument('--pretrain', action='store_true', help='Using image net pretrain')
    parser.add_argument('--momen', default=0.9, type=float, help='Hyper-parameter.(default: 0.9)')
    parser.add_argument('--nesterov', action='store_true', help='Using SGD nesterov')
    parser.add_argument('--num_classes', default=100, type=int, help='Number of data classes.(default: 200)')

    parser.add_argument('--test_step', default=30, type=int, help='Step of test.(default: 2000)')
    # parser.add_argument('--pretrained_dir', type=str, default="checkpoint/nest_small.pth", help="nest_small.pth")
    parser.add_argument('--pretrained_dir', type=str, default="checkpoint/jx_nest_small-422eaded.pth", help="nest_small.pth")
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--in_chans', type=int, default=3)
    # parser.add_argument('--out_chans', type=int, default=96)

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    args.lr_step = list(map(int, args.lr_step.split(',')))

    return args

if __name__ == '__main__':
    run()
