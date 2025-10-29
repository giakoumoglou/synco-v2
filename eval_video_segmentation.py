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
from src.utils import get_config, create_logger
from src.utils import load_pretrained

from video_seg_utils import eval_video_tracking, read_frame_list, read_seg, load_color_palette


def parse_option():
    parser = argparse.ArgumentParser('Video object segmentation evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--data-path', type=str, help='path to DAVIS dataset')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    
    # Video segmentation settings
    parser.add_argument("--n-last-frames", type=int, default=7, help="Number of preceding frames to use")
    parser.add_argument("--size-mask-neighborhood", default=12, type=int, help="Spatial neighborhood size for restricting source nodes")
    parser.add_argument("--topk", type=int, default=5, help="Accumulate label from top k neighbors")
    parser.add_argument("--patch-size-override", type=int, default=None, help="Override patch size (default: use from config)")

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    # ================ overwrite for video segmentation evaluation ================
    config.defrost()
    # base
    config.VIDEO_SEG_EVAL = type('', (), {})()  # Create namespace
    config.VIDEO_SEG_EVAL.PRETRAINED = os.path.join(config.OUTPUT, 'checkpoint.pth')
    config.OUTPUT = os.path.join(config.OUTPUT, 'video_segmentation_davis2017')
    # model - use the base encoder without classification head
    config.MODEL.TYPE = 'linear'
    # aug - no augmentation for video segmentation
    config.AUG.SSL_AUG = False
    config.AUG.SSL_LINEAR_AUG = False
    config.AUG.MIXUP = 0.0
    config.AUG.CUTMIX = 0.0
    config.AUG.CUTMIX_MINMAX = None
    config.freeze()

    return args, config


def main(config, args):    
    print('Saving under', config.OUTPUT)
    
    # ================ model ================
    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    
    # ================ freeze all parameters ================
    for name, p in model.named_parameters():
        p.requires_grad = False
    
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False
    )
    model_without_ddp = model.module
    
    # ================ load pre-trained model ================
    load_pretrained(model_without_ddp, config.VIDEO_SEG_EVAL.PRETRAINED, logger)

    # ================ print parameters & flops ================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} - should be 0% for video seg)")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    # ================ set patch size ================
    if args.patch_size_override is not None:
        patch_size = args.patch_size_override
        logger.info(f"Using override patch size: {patch_size}")
    else:
        patch_size = config.MODEL.PATCH_SIZE if hasattr(config.MODEL, 'PATCH_SIZE') else 16
        logger.info(f"Using default patch size: {patch_size}")

    # ================ load color palette ================
    logger.info("Loading color palette...")
    color_palette = load_color_palette()

    # ================ load video list ================
    video_list_path = os.path.join(args.data_path, "ImageSets/2017/val.txt")
    if not os.path.exists(video_list_path):
        logger.error(f"Video list not found: {video_list_path}")
        logger.error(f"Please check that --data-path points to DAVIS dataset root")
        return
    
    video_list = open(video_list_path).readlines()
    logger.info(f"Found {len(video_list)} videos to process")

    # ================ video object segmentation evaluation ================
    logger.info("\nStart video object segmentation evaluation")
    start_time = time.time()
    
    for i, video_name in enumerate(video_list):
        video_name = video_name.strip()
        logger.info(f'\n[{i+1}/{len(video_list)}] Processing video: {video_name}')
        
        # Read frame list
        video_dir = os.path.join(args.data_path, "JPEGImages/480p", video_name)
        if not os.path.exists(video_dir):
            logger.warning(f"Video directory not found: {video_dir}, skipping...")
            continue
        
        frame_list = read_frame_list(video_dir)
        logger.info(f"  Found {len(frame_list)} frames")
        
        # Read first frame segmentation
        seg_path = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
        if not os.path.exists(seg_path):
            logger.warning(f"Segmentation not found: {seg_path}, skipping...")
            continue
        
        first_seg, seg_ori = read_seg(seg_path, patch_size)
        
        # Evaluate video tracking
        eval_video_tracking(
            config=config,
            model=model,
            frame_list=frame_list,
            video_name=video_name,
            first_seg=first_seg,
            seg_ori=seg_ori,
            color_palette=color_palette,
            n_last_frames=args.n_last_frames,
            size_mask_neighborhood=args.size_mask_neighborhood,
            topk=args.topk,
            patch_size=patch_size,
            logger=logger
        )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f'\nTotal evaluation time: {total_time_str}')
    logger.info(f'Average time per video: {total_time/len(video_list):.2f}s')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Video object segmentation evaluation completed!")
    logger.info(f"Results saved to: {config.OUTPUT}")
    logger.info(f"To evaluate mIoU, run the DAVIS evaluation toolkit on the output folder")
    logger.info(f"{'='*60}\n")


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