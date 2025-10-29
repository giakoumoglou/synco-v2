#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import optim as optim


class LARS(optim.Optimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer.
    
    LARS applies layer-wise adaptive learning rates with trust coefficient.
    No rate scaling or weight decay for parameters <= 1D (normalization/bias).
    
    Reference:
        "Large Batch Training of Convolutional Networks" (You et al., 2017)
        https://arxiv.org/abs/1708.03888
    """
    
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        """
        Initialize LARS optimizer.
        
        Args:
            params: Iterable of parameters to optimize or parameter groups
            lr: Learning rate (default: 0)
            weight_decay: Weight decay coefficient (default: 0)
            momentum: Momentum factor (default: 0.9)
            trust_coefficient: Trust coefficient for layer-wise adaptation (default: 0.001)
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if trust_coefficient < 0.0:
            raise ValueError(f"Invalid trust_coefficient value: {trust_coefficient}")
            
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Applies LARS update rule with layer-wise adaptive learning rates.
        Weight decay and rate scaling are only applied to multi-dimensional parameters.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss (optional)
            
        Returns:
            loss: Loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                    
                dp = p.grad

                if p.ndim > 1:  # if not normalization gamma/beta or bias
                    # add weight decay
                    dp = dp.add(p, alpha=g['weight_decay'])
                    
                    # compute local learning rate
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    
                    # trust coefficient formula: local_lr = trust_coeff * ||w|| / ||g||
                    q = torch.where(
                        param_norm > 0.,
                        torch.where(
                            update_norm > 0,
                            (g['trust_coefficient'] * param_norm / update_norm),
                            one
                        ),
                        one
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                
                p.add_(mu, alpha=-g['lr'])
        
        return loss