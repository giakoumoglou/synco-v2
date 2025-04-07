# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.BATCH_SIZE = 64
_C.DATA.DATA_PATH = ''
_C.DATA.DATASET = 'imagenet'
_C.DATA.IMG_SIZE = 224
_C.DATA.INTERPOLATION = 'bicubic'
_C.DATA.ZIP_MODE = False
_C.DATA.CACHE_MODE = 'part'
_C.DATA.PIN_MEMORY = True
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.RESUME = ''
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.DROP_RATE = 0.0
_C.MODEL.DROP_PATH_RATE = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.1

# -----------------------------------------------------------------------------
# Self-supervised model settings
# -----------------------------------------------------------------------------
_C.MODEL.TYPE = 'moby'
_C.MODEL.ENCODER = 'swin_tiny'
_C.MODEL.NAME = 'moby__swin_tiny'
_C.MODEL.ONLINE_DROP_PATH_RATE = 0.1 # trick 1: asymmetric drop path
_C.MODEL.TARGET_DROP_PATH_RATE = 0.0
_C.MODEL.CONTRAST_MOMENTUM = 0.99
_C.MODEL.CONTRAST_TEMPERATURE = 0.2
_C.MODEL.CONTRAST_NUM_NEGATIVE = 4096
_C.MODEL.PROJ_NUM_LAYERS = 2
_C.MODEL.PRED_NUM_LAYERS = 2
_C.MODEL.STOP_GRAD_CONV1 = False # trick 2: stop grad conv1
_C.MODEL.REPLACE_LN_WITH_BN = False # trick 3: replace ln with bn
_C.MODEL.N_HARD = 256
_C.MODEL.N1 = 128
_C.MODEL.N2 = 128
_C.MODEL.N3 = 128
_C.MODEL.N4 = 128
_C.MODEL.N5 = 128
_C.MODEL.N6 = 128
_C.MODEL.WARMUP_EPOCHS = 30
_C.MODEL.COOLDOWN_EPOCHS = 100

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.ACCUMULATION_STEPS = 0
_C.TRAIN.USE_CHECKPOINT = False

# -----------------------------------------------------------------------------
# LR scheduler
# -----------------------------------------------------------------------------
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Linear eval settings
# -----------------------------------------------------------------------------
_C.LINEAR_EVAL = CN()
_C.LINEAR_EVAL.PRETRAINED = ''

# -----------------------------------------------------------------------------
# Augmentation settings (from default swin training)
# -----------------------------------------------------------------------------
_C.AUG = CN()
_C.AUG.COLOR_JITTER = 0.4
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
_C.AUG.REPROB = 0.25
_C.AUG.REMODE = 'pixel'
_C.AUG.RECOUNT = 1
_C.AUG.MIXUP = 0.8
_C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX_MINMAX = None
_C.AUG.MIXUP_PROB = 1.0
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Self-supervised augmentation settings
# -----------------------------------------------------------------------------
_C.AUG.SSL_AUG = False
_C.AUG.SSL_AUG_TYPE = 'byol'
_C.AUG.SSL_AUG_CROP = 0.08
_C.AUG.SSL_LINEAR_AUG = False

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.AMP_OPT_LEVEL = 'O0'
_C.OUTPUT = 'output'
_C.TAG = 'default'
_C.SAVE_FREQ = 10
_C.PRINT_FREQ = 10
_C.SEED = 0
_C.EVAL_MODE = False
_C.THROUGHPUT_MODE = False
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
