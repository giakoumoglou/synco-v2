# --------------------------------------------------------
# BYOL: A New Approach to Self-Supervised Learning
# Copyright (c) 2025 Imperial College London
# Licensed under The MIT License
# Written by Nikolaos Giakoumoglou
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffdist import functional


class BYOL(nn.Module):
    def __init__(self,
                 cfg,
                 encoder,
                 encoder_k,
                 contrast_momentum=0.99,
                 proj_num_layers=2,
                 pred_num_layers=2,
                 **kwargs):
        super().__init__()
        
        self.cfg = cfg
        
        self.encoder = encoder
        self.encoder_k = encoder_k
        
        self.contrast_momentum = contrast_momentum
        
        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers

        self.projector = MLP(in_dim=self.encoder.num_features, num_layers=proj_num_layers)
        self.projector_k = MLP(in_dim=self.encoder.num_features, num_layers=proj_num_layers)
        self.predictor = MLP(num_layers=pred_num_layers)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.K = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.EPOCHS
        self.k = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.START_EPOCH

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - self.contrast_momentum) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def forward(self, im_1, im_2):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        Output:
            loss: negative cosine similarity
        """
        # compute query features: NxC
        feat_1 = self.encoder(im_1)
        proj_1 = self.projector(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.encoder(im_2)
        proj_2 = self.projector(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        # compute key features: NxC
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(im_1) 
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(im_2)
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        # calculate symmetric loss
        loss = 2 - 2 * (pred_1 * proj_2_ng).sum(dim=1).mean() \
            + 2 - 2 * (pred_2 * proj_1_ng).sum(dim=1).mean()
        loss = loss * 0.5

        return loss
    
    
class MLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(MLP, self).__init__()
        
        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        # output layer
        self.linear_out = nn.Linear(in_dim if num_layers == 1 else inner_dim, out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return x


def dist_collect(x):
    """ 
    Collect all tensor from all GPUs

    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()