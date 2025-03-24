# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Nikolaos Giakoumoglou
# --------------------------------------------------------

import torch
from torch import optim as optim
from termcolor import colored


def build_optimizer(config, model, logger):
    """
    Build optimizer, set weight decay of normalization to 0 by default
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None
    if opt_lower == 'sgd':
        optimizer = optim.SGD(parameters, momentum=config.TRAIN.OPTIMIZER.MOMENTUM, nesterov=True,
                              lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
        print(f"Using SGD optimizer with momentum={config.TRAIN.OPTIMIZER.MOMENTUM}, "
                    f"lr={config.TRAIN.BASE_LR}, weight_decay={config.TRAIN.WEIGHT_DECAY}")
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=config.TRAIN.BASE_LR, weight_decay=config.TRAIN.WEIGHT_DECAY)
        print(f"Using AdamW optimizer with eps={config.TRAIN.OPTIMIZER.EPS}, "
                    f"betas={config.TRAIN.OPTIMIZER.BETAS}, lr={config.TRAIN.BASE_LR}, "
                    f"weight_decay={config.TRAIN.WEIGHT_DECAY}")
    elif opt_lower == 'lars':
        trust_coefficient = config.TRAIN.TRUST_COEFFICIENT if hasattr(config.TRAIN, 'TRUST_COEFFICIENT') else 0.001
        optimizer = LARS(parameters, lr=config.TRAIN.BASE_LR, 
                         momentum=config.TRAIN.OPTIMIZER.MOMENTUM,
                         weight_decay=config.TRAIN.WEIGHT_DECAY,
                         trust_coefficient=trust_coefficient)
        print(f"Using LARS optimizer with momentum={config.TRAIN.OPTIMIZER.MOMENTUM}, "
                    f"lr={config.TRAIN.BASE_LR}, weight_decay={config.TRAIN.WEIGHT_DECAY}, "
                    f"trust_coefficient={trust_coefficient}")

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    """
    Set weight decay of normalization to 0, and set weight decay of bias to 0
    """
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    """
    Check if the name contains keywords
    """
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


class LARS(optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])