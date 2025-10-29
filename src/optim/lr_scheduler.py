#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from termcolor import colored
import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def build_scheduler(config, optimizer, n_iter_per_epoch):
    """
    Build learning rate scheduler.
    
    Supports cosine annealing, linear decay, and step decay schedules.
    All schedulers include optional warmup period.
    
    Args:
        config: Configuration object with scheduler settings
        optimizer: PyTorch optimizer instance
        n_iter_per_epoch: Number of iterations per epoch
        
    Returns:
        lr_scheduler: Initialized learning rate scheduler
        
    Raises:
        NotImplementedError: If scheduler type is not supported
    """
    num_steps = int(config.TRAIN.EPOCHS * n_iter_per_epoch)
    warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch) if hasattr(
        config.TRAIN.LR_SCHEDULER, 'DECAY_EPOCHS'
    ) else 0
    
    lr_scheduler = None
    
    # ================ cosine annealing ================
    if config.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=config.TRAIN.MIN_LR,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
        
    # ================ linear decay ================
    elif config.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
        
    # ================ step decay ================
    elif config.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=config.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
        
    else:
        raise NotImplementedError(
            f"-----> Unknown scheduler: {config.TRAIN.LR_SCHEDULER.NAME}. "
            f"Only 'cosine', 'linear', and 'step' are currently supported."
        )
    
    return lr_scheduler


class LinearLRScheduler(Scheduler):
    """
    Linear learning rate scheduler with warmup.
    
    Linearly decays learning rate from base_lr to base_lr * lr_min_rate
    over the course of training, with optional warmup period.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        """
        Initialize Linear LR Scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            t_initial: Total number of training steps/epochs
            lr_min_rate: Minimum learning rate as fraction of base_lr
            warmup_t: Number of warmup steps/epochs
            warmup_lr_init: Initial warmup learning rate
            t_in_epochs: If True, t_initial and warmup_t are in epochs, else steps
            noise_range_t: Optional noise schedule range
            noise_pct: Noise percentage for stochastic schedule
            noise_std: Standard deviation of noise
            noise_seed: Random seed for noise generation
            initialize: Whether to initialize scheduler state
        """
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )
        
        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        
        if self.warmup_t:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_values
            ]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        """
        Calculate learning rate for given time step.
        
        Args:
            t: Current time step (epoch or iteration)
            
        Returns:
            lrs: List of learning rates for each parameter group
        """
        if t < self.warmup_t:
            # warmup phase
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            # linear decay phase
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        
        return lrs

    def get_epoch_values(self, epoch: int):
        """
        Get learning rate values for epoch-based scheduling.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Learning rates if t_in_epochs=True, else None
        """
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        """
        Get learning rate values for iteration-based scheduling.
        
        Args:
            num_updates: Current iteration number
            
        Returns:
            Learning rates if t_in_epochs=False, else None
        """
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None