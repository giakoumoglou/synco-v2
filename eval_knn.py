#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from src.models import build_model
from src.data import build_loader
from src.utils import get_config, create_logger
from src.utils import load_pretrained, reduce_tensor
from src.utils import extract_features, knn_classifier


def parse_option():
    parser = argparse.ArgumentParser('KNN evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    
    # KNN settings
    parser.add_argument('--nb-knn', default=[10, 20, 100, 200], nargs='+', type=int, help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.07, type=float, help='Temperature used in the voting coefficient')
    parser.add_argument('--dump-features', default=None, type=str, help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load-features', default=None, type=str, help='If the features have already been computed, where to find them.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    # ================ overwrite for KNN evaluation ================
    config.defrost()
    # base
    config.KNN_EVAL = type('', (), {})()  # create namespace
    config.KNN_EVAL.PRETRAINED = os.path.join(config.OUTPUT, 'checkpoint.pth')
    config.OUTPUT = os.path.join(config.OUTPUT, f'knn_{config.DATA.DATASET}')
    # model - use the base encoder without classification head
    config.MODEL.TYPE = 'linear' # to load model correctly
    # aug - use simple augmentation for KNN
    config.AUG.SSL_AUG = False
    config.AUG.SSL_LINEAR_AUG = True
    config.AUG.MIXUP = 0.0
    config.AUG.CUTMIX = 0.0
    config.AUG.CUTMIX_MINMAX = None
    config.freeze()

    return args, config


def main(config, args):    
    print('Saving under', config.OUTPUT)
    
    # ================ data ================
    dataset_train, dataset_val, data_loader_train, data_loader_val, _ = build_loader(config)
    logger.info("Number of training samples: {}".format(len(data_loader_train.dataset)))
    logger.info("Number of validation samples: {}".format(len(data_loader_val.dataset)))
    logger.info("Number of classes: {}".format(config.MODEL.NUM_CLASSES))

    # ================ model ================
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    
    # ================ freeze all parameters ================
    for name, p in model.named_parameters():
        p.requires_grad = False
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module
    
    # ================ load pre-trained model ================
    load_pretrained(model_without_ddp, config.KNN_EVAL.PRETRAINED, logger)

    # ================ print parameters & flops ================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} - should be 0% for KNN)")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    # ================ extract features or load precomputed features ================
    if args.load_features:
        logger.info(f"Loading features from {args.load_features}")
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
        
        # Move to GPU
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()
    else:
        # Extract features
        logger.info("Start feature extraction")
        start_time = time.time()
        
        train_features, train_labels = extract_features(config, data_loader_train, model, logger)
        test_features, test_labels = extract_features(config, data_loader_val, model, logger)
        
        extraction_time = time.time() - start_time
        extraction_time_str = str(datetime.timedelta(seconds=int(extraction_time)))
        logger.info(f'Feature extraction time {extraction_time_str}')
        
        # Save features if requested
        if args.dump_features and dist.get_rank() == 0:
            os.makedirs(args.dump_features, exist_ok=True)
            torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
            torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
            torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
            torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
            logger.info(f"Features saved to {args.dump_features}")

    # ================ KNN classification ================
    logger.info("Features are ready! Start the k-NN classification.")
    
    for k in args.nb_knn:
        top1, top5 = knn_classifier(
            train_features, train_labels, 
            test_features, test_labels, 
            k, args.temperature, 
            config.MODEL.NUM_CLASSES, 
            logger
        )
        logger.info(f"{k}-NN classifier result: Top1: {top1:.2f}%, Top5: {top5:.2f}%")
    
    logger.info("KNN evaluation completed!")


if __name__ == '__main__':
    args, config = parse_option()

    # ================ distributed setting ================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # ================ init dist ================
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # ================ set the seed ================
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ================ create logger ================
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    # ================ save config ================
    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # ================ print config ================
    logger.info(config.dump())

    # ================ run ================
    main(config, args)