import torch
import torch.optim as optim
import time
import random
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from loguru import logger
from models.Loss import Loss
from models.MDSHC_Loss import OurLoss
from data.data_loader import sample_dataloader
import models.baseline as baseline
import models.SEMICON as SEMICON
# from apex import amp
from ptflops import get_model_complexity_info
import numpy as np
import os

def train(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):

    model = baseline.baseline(args, code_length=code_length, num_classes=args.num_classes, pretrained=True)
    args.hash_bit = code_length

    # num_classes, att_size, feat_size = args.num_classes, 1, 2048
    # model = SEMICON.semicon(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
    #                         device=args.device, pretrained=True)

    model.to(args.device)




    # (Statistical model calculation and number of parameters)
    # flops, num_params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
    # # logger.info("{}".format(config))
    # logger.info("Total Parameter: \t%s" % num_params)
    # logger.info("Total Flops: \t%s" % flops)

    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momen)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)


    scaler = GradScaler()

    file_name = str(args.hash_bit) + '_'+ args.dataset + '_' + str(args.num_classes) + '_class.pkl'

    # Hash_center = torch.load(args.true_hash+file_name).to(args.device)
    Hash_center =torch.from_numpy(np.load(args.true_hash+file_name)).float().to(args.device)

    criterion = Loss(code_length, args.gamma)

    num_retrieval = len(retrieval_dataloader.dataset)

    B = torch.randn(num_retrieval, code_length).to(args.device)

    retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(args.device)
    start = time.time()
    best_mAP = 0

    for batch, (data, targets, index) in enumerate(retrieval_dataloader):
        data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)

        hash_label = (targets == 1).nonzero(as_tuple=False)[:, 1]
        hash_center = Hash_center[hash_label]
        B[index, :] = hash_center


    for it in range(args.max_iter):
        iter_start = time.time()
        # Sample training data for cnn learning
        train_dataloader, sample_index = sample_dataloader(retrieval_dataloader, args.num_samples, args.batch_size,
                                                           args.root, args.dataset, args.num_workers)
        total_loss = []


        # Training CNN model

        for batch, (data, targets, index) in enumerate(train_dataloader):
            data, targets, index = data.to(args.device), targets.to(args.device), index.to(args.device)

            optimizer.zero_grad()

            hash_label = (targets == 1).nonzero(as_tuple=False)[:, 1]
            hash_center = Hash_center[hash_label]

            with autocast():
                orginal_hash, x_orginal, x_room, sample_map= model(data)
                cnn_loss = criterion(orginal_hash, hash_center)   

            scaler.scale(cnn_loss).backward()
            
            scaler.step(optimizer)

            scaler.update()

            total_loss.append(cnn_loss.data.cpu().numpy())


        logger.info('[epoch:{}/{}][total_loss:{:.4f}][Training Time:{:.2f}]'.format(it+1,
                                                    args.max_iter, np.mean(total_loss), time.time()-iter_start))





        if (it + 1) % args.test_step == 0 :
            query_code = generate_code(model, query_dataloader, code_length, args.device)

            mAP = mean_average_precision(
                query_code.to(args.device),
                B,
                query_dataloader.dataset.get_onehot_targets().to(args.device),
                retrieval_targets,
                args.device,
                args.topk,
            )
            logger.info(
                '[iter:{}/{}][code_length:{}][mAP:{:.4f}]'.format(it + 1, args.max_iter, code_length, mAP))
            if mAP > best_mAP:
                best_mAP = mAP
                # ret_path = os.path.join(args.save_ckpt, )
                # if not os.path.exists(ret_path):
                #     os.makedirs(ret_path)
                # torch.save(query_code.cpu(), os.path.join(ret_path, 'query_code.t'))
                # torch.save(B.cpu(), os.path.join(ret_path, 'database_code.t'))
                # torch.save(query_dataloader.dataset.get_onehot_targets, os.path.join(ret_path, 'query_targets.t'))
                # torch.save(retrieval_targets.cpu(), os.path.join(ret_path, 'database_targets.t'))
                torch.save(model.state_dict(), os.path.join(args.save_ckpt, args.info+'_'+args.dataset+'_'+str(code_length)+'_model.pkl'))
            logger.info('[iter:{}/{}][code_length:{}][mAP:{:.4f}][best_mAP:{:.4f}]'.format(it+1, args.max_iter, code_length, mAP, best_mAP))
    logger.info('[Training time:{:.2f}]'.format(time.time()-start))

    return best_mAP

def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code
    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.
    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        # code = torch.zeros([N, code_length]).half().to(device)
        code = torch.zeros([N, code_length]).to(device)
        for batch, (data, targets, index) in enumerate(dataloader):
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            hash_code,_,_,_= model(data)
            # hash_code,_= model(data)
            code[index, :] = hash_code.sign()

            # code[index, :] = hash_code.sign()
    model.train()
    return code


def mean_average_precision(query_code,
                           database_code,
                           query_labels,
                           database_labels,
                           device,
                           topk=None,
                           ):
    """
    Calculate mean average precision(map).
    Args:
        query_code (torch.Tensor): Query data hash code.
        database_code (torch.Tensor): Database data hash code.
        query_labels (torch.Tensor): Query data targets, one-hot
        database_labels (torch.Tensor): Database data targets, one-host
        device (torch.device): Using CPU or GPU.
        topk (int): Calculate top k data map.
    Returns:
        meanAP (float): Mean Average Precision.
    """
    num_query = query_labels.shape[0]
    mean_AP = 0.0

    for i in tqdm(range(num_query),ncols=60):
        # Retrieve images from database
        retrieval = (query_labels[i, :] @ database_labels.t() > 0).float()

        # Calculate hamming distance
        hamming_dist = 0.5 * (database_code.shape[1] - query_code[i, :] @ database_code.t())

        # Arrange position according to hamming distance
        retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

        # Retrieval count
        retrieval_cnt = retrieval.sum().int().item()

        # Can not retrieve images
        if retrieval_cnt == 0:
            continue

        # Generate score for every position
        score = torch.linspace(1, retrieval_cnt, retrieval_cnt).to(device)

        # Acquire index
        index = (torch.nonzero(retrieval == 1).squeeze() + 1.0).float()

        mean_AP += (score / index).mean()

    mean_AP = mean_AP / num_query
    return mean_AP
