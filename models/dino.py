# Copyright (C) 2026.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class DINO(nn.Module):
    def __init__(self,
                 cfg,
                 encoder,
                 encoder_k,
                 contrast_momentum=0.99,
                 out_dim=65536,
                 hidden_dim=2048,
                 bottleneck_dim=256,
                 proj_num_layers=3,
                 use_bn=False,
                 norm_last_layer=True,
                 student_temp=0.1,
                 teacher_temp=0.04,
                 warmup_teacher_temp=0.04,
                 warmup_teacher_temp_epochs=0,
                 center_momentum=0.9,
                 freeze_last_layer_epochs=1,
                 **kwargs):
        super().__init__()

        self.cfg = cfg

        self.encoder = encoder
        self.encoder_k = encoder_k

        self.contrast_momentum = contrast_momentum

        self.proj_num_layers = proj_num_layers

        self.projector = DINOHead(in_dim=self.encoder.num_features,
                                  out_dim=out_dim,
                                  hidden_dim=hidden_dim,
                                  bottleneck_dim=bottleneck_dim,
                                  num_layers=proj_num_layers,
                                  use_bn=use_bn,
                                  norm_last_layer=norm_last_layer,
                                  )
        self.projector_k = DINOHead(in_dim=self.encoder.num_features,
                                    out_dim=out_dim,
                                    hidden_dim=hidden_dim,
                                    bottleneck_dim=bottleneck_dim,
                                    num_layers=proj_num_layers,
                                    use_bn=use_bn,
                                    norm_last_layer=True,
                                    )

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if "resnet" in cfg.MODEL.ENCODER or ("swin" in cfg.MODEL.ENCODER and "bn" in cfg.MODEL.ENCODER):
            self.encoder = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.encoder_k = nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)

        self.K = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.EPOCHS
        self.k = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.START_EPOCH

        # dino: https://arxiv.org/abs/2104.14294
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.freeze_last_layer_epochs = freeze_last_layer_epochs

        # linear warmup of the teacher temperature, constant afterwards
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs

        # create the center
        self.register_buffer("center", torch.zeros(1, out_dim))

        # freeze the last layer of the student head during the first epochs
        if self.freeze_last_layer_epochs > 0:
            self.projector.last_layer.weight_v.register_hook(self._cancel_last_layer_grad)

    def _current_epoch(self):
        """
        Current epoch inferred from the number of momentum updates so far
        """
        steps_per_epoch = self.cfg.DATA.TRAINING_IMAGES // self.cfg.DATA.BATCH_SIZE
        return self.cfg.TRAIN.START_EPOCH + (self.k // steps_per_epoch)

    def _cancel_last_layer_grad(self, grad):
        """
        Zero out the gradients of the student head last layer during the first epochs
        """
        if self._current_epoch() < self.freeze_last_layer_epochs:
            return torch.zeros_like(grad)
        return grad

    def _get_teacher_temp(self):
        """
        Linear warmup of the teacher temperature, constant afterwards
        """
        current_epoch = self._current_epoch()
        if current_epoch >= self.warmup_teacher_temp_epochs:
            return self.teacher_temp
        return self.warmup_teacher_temp + (self.teacher_temp - self.warmup_teacher_temp) * current_epoch / self.warmup_teacher_temp_epochs

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

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """
        Exponential moving average update of the center used to avoid collapse
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (teacher_output.shape[0] * dist.get_world_size())

        self.center = self.center * self.center_momentum + batch_center * (1. - self.center_momentum)

    def distillation_loss(self, student_out, teacher_out, teacher_temp):
        """
        Cross-entropy between the sharpened and centered teacher and the student
        """
        student_out = student_out / self.student_temp
        teacher_out = F.softmax((teacher_out - self.center) / teacher_temp, dim=-1).detach()

        return torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()

    def forward(self, im_1, im_2):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        Output:
            loss: self-distillation loss
        """
        # compute student features: NxC
        feat_1 = self.encoder(im_1)
        proj_1 = self.projector(feat_1)

        feat_2 = self.encoder(im_2)
        proj_2 = self.projector(feat_2)

        # compute teacher features: NxC
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(im_1)
            proj_1_ng = self.projector_k(feat_1_ng)

            feat_2_ng = self.encoder_k(im_2)
            proj_2_ng = self.projector_k(feat_2_ng)

        # calculate symmetric loss across the two views
        teacher_temp = self._get_teacher_temp()
        loss = self.distillation_loss(proj_1, proj_2_ng, teacher_temp) \
            + self.distillation_loss(proj_2, proj_1_ng, teacher_temp)
        loss = loss * 0.5

        self._update_center(torch.cat([proj_1_ng, proj_2_ng], dim=0))

        return loss


class DINOHead(nn.Module):
    def __init__(self, in_dim=384, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, num_layers=3, use_bn=False, norm_last_layer=True):
        super(DINOHead, self).__init__()

        num_layers = max(num_layers, 1)

        # hidden layers
        if num_layers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            mlp = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                mlp.append(nn.BatchNorm1d(hidden_dim))
            mlp.append(nn.GELU())
            for _ in range(num_layers - 2):
                mlp.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    mlp.append(nn.BatchNorm1d(hidden_dim))
                mlp.append(nn.GELU())
            mlp.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*mlp)

        # output layer
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
