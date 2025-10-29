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
from timm.utils import accuracy, AverageMeter

from src.models import build_model
from src.data import build_loader
from src.optim import build_scheduler, build_optimizer
from src.utils import create_logger, get_config
from src.utils import get_grad_norm, get_component_parameters, load_checkpoint, save_checkpoint, auto_resume_helper, reduce_tensor

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def parse_option():
    parser = argparse.ArgumentParser('Self-supervised pretraining script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):

    # ================ data ================
    dataset_train, _, data_loader_train, _, _ = build_loader(config)
    logger.info("Number of training samples: {}".format(len(dataset_train)))
    
    config.defrost()
    config.DATA.TRAINING_IMAGES = len(dataset_train)
    config.freeze()

    # ================ model ================
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    # ================ optimizer ================
    optimizer = build_optimizer(config, model)
    if config.AMP_OPT_LEVEL != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    # ================ print ================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"Number of GFLOPs: {flops / 1e9}")

    # ================ lr scheduler ================
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # ================ auto resume ================
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"Auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'Auto resuming from {resume_file}')
        else:
            logger.info(f'No checkpoint found in {config.OUTPUT}, ignoring auto resume')

    # ================ resume from a checkpoint ================
    if config.MODEL.RESUME:
        _ = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)

    # ================ start training ================
    logger.info("Start self-supervised pre-training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, data_loader_train, optimizer, epoch, lr_scheduler)
        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            save_checkpoint(config, epoch, model_without_ddp, 0.0, optimizer, lr_scheduler, logger)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, data_loader, optimizer, epoch, lr_scheduler):
    """
    One epoch training for self-supervised learning with component gradient tracking.

    args:
        config (config): config file
        model (nn.Module): model
        data_loader (torch.utils.data.DataLoader): data loader
        optimizer (torch.optim): optimizer
        epoch (int): current epoch
        lr_scheduler (LRScheduler): learning rate scheduler
    returns:
        None
    """
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    
    # accuracy meters
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    # overall model gradient average meters
    model_l2_meter = AverageMeter()
    model_inf_meter = AverageMeter()
    
    # component gradient average meters
    encoder_l2_meter = AverageMeter()
    projector_l2_meter = AverageMeter()
    predictor_l2_meter = AverageMeter()
    encoder_inf_meter = AverageMeter()
    projector_inf_meter = AverageMeter()
    predictor_inf_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples_1, samples_2, targets) in enumerate(data_loader):
        samples_1 = samples_1.cuda(non_blocking=True)
        samples_2 = samples_2.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        model_outputs = model(samples_1, samples_2)
        
        if isinstance(model_outputs, (tuple, list)) and len(model_outputs) == 3:
            loss, output, target = model_outputs
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            acc1_meter.update(acc1.item(), output.size(0))
            acc5_meter.update(acc5.item(), output.size(0))
        else:
            loss = model_outputs
            acc1_meter.update(float('nan'), targets.size(0))
            acc5_meter.update(float('nan'), targets.size(0))

        optimizer.zero_grad()
        if config.AMP_OPT_LEVEL != "O0":
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(amp.master_params(optimizer))
        else:
            loss.backward()
            
            # calculate overall model gradients
            all_params = get_component_parameters(model, 'overall')
            model_l2_norm = get_grad_norm(all_params, norm_type=2)
            model_inf_norm = get_grad_norm(all_params, norm_type=float('inf'))
            
            # calculate component gradients using your existing get_grad_norm function
            encoder_params = get_component_parameters(model, 'encoder')
            projector_params = get_component_parameters(model, 'projector')
            predictor_params = get_component_parameters(model, 'predictor')
            
            # L_2 norms
            encoder_l2_norm = get_grad_norm(encoder_params, norm_type=2)
            projector_l2_norm = get_grad_norm(projector_params, norm_type=2)
            predictor_l2_norm = get_grad_norm(predictor_params, norm_type=2)
            
            # L_inf norms
            encoder_inf_norm = get_grad_norm(encoder_params, norm_type=float('inf'))
            projector_inf_norm = get_grad_norm(projector_params, norm_type=float('inf'))
            predictor_inf_norm = get_grad_norm(predictor_params, norm_type=float('inf'))
            
            # update average meters
            model_l2_meter.update(model_l2_norm, targets.size(0))
            model_inf_meter.update(model_inf_norm, targets.size(0))
            encoder_l2_meter.update(encoder_l2_norm, targets.size(0))
            projector_l2_meter.update(projector_l2_norm, targets.size(0))
            predictor_l2_meter.update(predictor_l2_norm, targets.size(0))
            encoder_inf_meter.update(encoder_inf_norm, targets.size(0))
            projector_inf_meter.update(projector_inf_norm, targets.size(0))
            predictor_inf_meter.update(predictor_inf_norm, targets.size(0))
            
            if config.TRAIN.CLIP_GRAD:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
        
        optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB\t'
                f'[Unsupervised Acc] '
                f'acc1 {acc1_meter.val:.2f} ({acc1_meter.avg:.2f})\t'
                f'acc5 {acc5_meter.val:.2f} ({acc5_meter.avg:.2f})\t'
                f'[Grad Norms] '
                f'model_l2 {model_l2_meter.val:.4f} ({model_l2_meter.avg:.4f})\t'
                f'model_inf {model_inf_meter.val:.6f} ({model_inf_meter.avg:.6f})\t'
                f'enc_l2 {encoder_l2_meter.val:.4f} ({encoder_l2_meter.avg:.4f})\t'
                f'proj_l2 {projector_l2_meter.val:.4f} ({projector_l2_meter.avg:.4f})\t'
                f'pred_l2 {predictor_l2_meter.val:.4f} ({predictor_l2_meter.avg:.4f})\t'
                f'enc_inf {encoder_inf_meter.val:.6f} ({encoder_inf_meter.avg:.6f})\t'
                f'proj_inf {projector_inf_meter.val:.6f} ({projector_inf_meter.avg:.6f})\t'
                f'pred_inf {predictor_inf_meter.val:.6f} ({predictor_inf_meter.avg:.6f})'
                )
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


if __name__ == '__main__':
    _, config = parse_option()

    # ================ amp ================
    print(f"AMP_OPT_LEVEL: {config.AMP_OPT_LEVEL}")
    if config.AMP_OPT_LEVEL != "O0":
        assert amp is not None, "amp not installed!"

    # ================ distributed setting ================
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # ================ initialize dist ================
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    # ================ set the seed ================
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ================ linear scale ================
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0

    # ================ gradient accumulation ================
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

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
    main(config)
