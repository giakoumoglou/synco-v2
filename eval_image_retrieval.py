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
from torchvision import transforms as pth_transforms

from src.models import build_model
from src.utils import get_config, create_logger
from src.utils import load_pretrained

from src.utils import OxfordParisDataset, retrieval_extract_features, evaluate_retrieval


def parse_option():
    parser = argparse.ArgumentParser('Image retrieval evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    
    # Retrieval settings
    parser.add_argument('--dataset', default='roxford5k', type=str, choices=['roxford5k', 'rparis6k'], help='Dataset for retrieval')
    parser.add_argument('--multiscale', action='store_true', help='Use multiscale feature extraction')
    parser.add_argument('--imsize', default=224, type=int, help='Image size')
    parser.add_argument('--dump-features', default=None, type=str, help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load-features', default=None, type=str, help='If the features have already been computed, where to find them.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    # ================ overwrite for retrieval evaluation ================
    config.defrost()
    # base
    config.RETRIEVAL_EVAL = type('', (), {})()  # Create namespace
    config.RETRIEVAL_EVAL.PRETRAINED = os.path.join(config.OUTPUT, 'checkpoint.pth')
    config.OUTPUT = os.path.join(config.OUTPUT, f'retrieval_{args.dataset}')
    # model - use the base encoder without classification head
    config.MODEL.TYPE = 'linear'
    # aug - use simple augmentation for retrieval
    config.AUG.SSL_AUG = False
    config.AUG.SSL_LINEAR_AUG = False
    config.AUG.MIXUP = 0.0
    config.AUG.CUTMIX = 0.0
    config.AUG.CUTMIX_MINMAX = None
    config.freeze()

    return args, config


def build_retrieval_loader(args, config):
    """Build data loaders for retrieval evaluation"""
    # Define transforms
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Create datasets
    dataset_train = OxfordParisDataset(
        args.data_path, 
        args.dataset, 
        split="train", 
        transform=transform, 
        imsize=args.imsize
    )
    dataset_query = OxfordParisDataset(
        args.data_path, 
        args.dataset, 
        split="query", 
        transform=transform, 
        imsize=args.imsize
    )
    
    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        sampler=None  # No distributed sampler for retrieval
    )
    data_loader_query = torch.utils.data.DataLoader(
        dataset_query,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        sampler=None  # No distributed sampler for retrieval
    )
    
    return dataset_train, dataset_query, data_loader_train, data_loader_query


def main(config, args):    
    print('Saving under', config.OUTPUT)
    
    # ================ data ================
    dataset_train, dataset_query, data_loader_train, data_loader_query = build_retrieval_loader(args, config)
    logger.info(f"Train (database): {len(dataset_train)} imgs / Query: {len(dataset_query)} imgs")

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
    load_pretrained(model_without_ddp, config.RETRIEVAL_EVAL.PRETRAINED, logger)

    # ================ print parameters & flops ================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} - should be 0% for retrieval)")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    # ================ extract features or load precomputed features ================
    if args.load_features:
        logger.info(f"Loading features from {args.load_features}")
        train_features = torch.load(os.path.join(args.load_features, "train_features.pth"))
        query_features = torch.load(os.path.join(args.load_features, "query_features.pth"))
        
        # Move to GPU
        train_features = train_features.cuda()
        query_features = query_features.cuda()
    else:
        # Extract features
        logger.info("Start feature extraction")
        start_time = time.time()
        
        logger.info("\nExtracting features from train (database) set...")
        train_features = retrieval_extract_features(config, data_loader_train, model, logger, multiscale=args.multiscale)
        
        logger.info("\nExtracting features from query set...")
        query_features = retrieval_extract_features(config, data_loader_query, model, logger, multiscale=args.multiscale)
        
        extraction_time = time.time() - start_time
        extraction_time_str = str(datetime.timedelta(seconds=int(extraction_time)))
        logger.info(f'Feature extraction time {extraction_time_str}')
        
        # Save features if requested
        if args.dump_features and dist.get_rank() == 0:
            os.makedirs(args.dump_features, exist_ok=True)
            torch.save(train_features.cpu(), os.path.join(args.dump_features, "train_features.pth"))
            torch.save(query_features.cpu(), os.path.join(args.dump_features, "query_features.pth"))
            logger.info(f"Features saved to {args.dump_features}")

    # ================ Image Retrieval Evaluation ================
    logger.info("\nFeatures are ready! Start image retrieval evaluation.")
    
    # Get ground truth
    gnd = dataset_train.cfg['gnd']
    
    # Evaluate
    ks = [1, 5, 10]
    results = evaluate_retrieval(train_features, query_features, gnd, ks=ks, logger=logger)
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"{'='*60}")
    logger.info(f"mAP Medium (easy + hard): {results['mAP_M']:.2f}%")
    logger.info(f"mAP Hard:                 {results['mAP_H']:.2f}%")
    if results['mP@k_M'] is not None:
        logger.info(f"mP@k{ks} Medium:           {np.around(results['mP@k_M'], decimals=2)}")
        logger.info(f"mP@k{ks} Hard:             {np.around(results['mP@k_H'], decimals=2)}")
    logger.info(f"{'='*60}\n")
    
    logger.info("Image retrieval evaluation completed!")


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