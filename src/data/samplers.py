#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


class SubsetRandomSampler(torch.utils.data.Sampler):
    """
    Samples elements randomly from a given list of indices.
    
    Used for distributed validation where we want each GPU to
    evaluate a non-overlapping subset of the validation set.
    
    Args:
        indices: Sequence of indices to sample from
        
    Example:
        >>> # Distribute validation across 4 GPUs
        >>> world_size = 4
        >>> rank = 0  # This GPU's rank
        >>> indices = np.arange(rank, len(dataset), world_size)
        >>> sampler = SubsetRandomSampler(indices)
        >>> loader = DataLoader(dataset, sampler=sampler)
    """
    
    def __init__(self, indices):
        """
        Initialize sampler with indices.
        
        Args:
            indices: Sequence of indices (e.g., numpy array, list)
        """
        self.epoch = 0
        self.indices = indices
    
    def __iter__(self):
        """
        Iterator that returns shuffled indices.
        
        Yields:
            Shuffled indices
        """
        # Randomly permute the indices
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
    
    def __len__(self):
        """Return number of samples."""
        return len(self.indices)
    
    def set_epoch(self, epoch):
        """
        Set epoch for reproducibility.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch