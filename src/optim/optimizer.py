#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from termcolor import colored
import torch
from torch import optim as optim

from .lars import LARS


def build_optimizer(config, model):
    """
    Build optimizer with selective weight decay.
    
    Automatically sets weight decay to 0 for normalization layers and biases.
    Supports SGD, AdamW, and LARS optimizers.
    
    Args:
        config: Configuration object with optimizer settings
        model: PyTorch model to optimize
        
    Returns:
        optimizer: Initialized optimizer
        
    Raises:
        NotImplementedError: If optimizer type is not supported
    """
    # ================ setup parameter groups ================
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    
    # ================ SGD optimizer ================
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            parameters,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            nesterov=True,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        
    # ================ AdamW optimizer ================
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(
            parameters,
            eps=config.TRAIN.OPTIMIZER.EPS,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            lr=config.TRAIN.BASE_LR,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
        )
        
    # ================ LARS optimizer ================
    elif opt_lower == 'lars':
        trust_coefficient = getattr(config.TRAIN, 'TRUST_COEFFICIENT', 0.001)
        optimizer = LARS(
            parameters,
            lr=config.TRAIN.BASE_LR,
            momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
            weight_decay=config.TRAIN.WEIGHT_DECAY,
            trust_coefficient=trust_coefficient,
        )
        
    else:
        raise NotImplementedError(
            f"-----> Unknown optimizer: {opt_lower}. "
            f"Only 'sgd', 'adamw', and 'lars' are currently supported."
        )
    
    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """
    Separate parameters into groups with and without weight decay.
    
    Parameters without weight decay include:
    - All 1D parameters (normalization layers)
    - Bias terms
    - Parameters in skip_list
    - Parameters matching skip_keywords
    
    Args:
        model: PyTorch model
        skip_list: Set of parameter names to exclude from weight decay
        skip_keywords: Tuple of keywords; params containing these skip weight decay
        
    Returns:
        List of parameter groups with different weight decay settings
    """
    has_decay = []
    no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
            
        if (len(param.shape) == 1 or 
            name.endswith(".bias") or 
            (name in skip_list) or
            check_keywords_in_name(name, skip_keywords)):
            no_decay.append(param)
        else:
            has_decay.append(param)
    
    return [
        {'params': has_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]


def check_keywords_in_name(name, keywords=()):
    """
    Check if any keyword appears in the parameter name.
    
    Args:
        name: Parameter name string
        keywords: Tuple of keyword strings to search for
        
    Returns:
        True if any keyword is found in name, False otherwise
    """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin