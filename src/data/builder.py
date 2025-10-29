#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from termcolor import colored

import torch
import torch.distributed as dist
from torchvision import datasets
from timm.data import Mixup

from .dataset import CustomImageFolder
from .transform import build_transform
from .samplers import SubsetRandomSampler


def build_loader(config):
    """
    Build data loader for training and validation.

    Args:
        config: Configuration object with data settings
        
    Returns:
        dataset_train: Training dataset
        dataset_val: Validation dataset
        data_loader_train: Training data loader
        data_loader_val: Validation data loader
        mixup_fn: Mixup/CutMix function or None
    """
    config.defrost()

    # ================ build datasets ================
    # Use dummy dataset for evaluation mode
    dataset_train = datasets.FakeData(
        size=1281167,
        image_size=(3, 224, 224),
        num_classes=1000,
        transform=build_transform(is_train=True, config=config)
    )
    
    if not config.EVAL_MODE:
        dataset_train, _ = build_dataset(is_train=True, config=config)
    
    dataset_val, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config)
    config.freeze()

    # ================ build samplers ================
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True
    )
    
    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    # ================ build data loaders ================
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # ================ setup mixup / cutmix ================
    mixup_fn = None
    mixup_active = (
        config.AUG.MIXUP > 0 or
        config.AUG.CUTMIX > 0. or
        config.AUG.CUTMIX_MINMAX is not None
    )
    
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    """
    Build dataset for training or validation.
    
    Currently supports ImageNet only.
    For zip mode, only ImageNet ILSVRC2012 is supported.

    Args:
        is_train: True for training, False for validation
        config: Configuration object
        
    Returns:
        dataset: PyTorch Dataset
        nb_classes: Number of classes in the dataset
        
    Raises:
        NotImplementedError: If dataset is not supported
    """
    transform = build_transform(is_train, config)
    
    print(colored(
        f"==============> Building {'training' if is_train else 'validation'} "
        f"dataset {config.DATA.DATASET} ....................",
        "red"
    ))

    # ================ imagenet ================
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = CustomImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError(
            f"-----> Unknown dataset: {config.DATA.DATASET}. "
            f"Only 'imagenet' is currently supported."
        )
    
    return dataset, nb_classes