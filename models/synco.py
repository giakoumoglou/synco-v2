# --------------------------------------------------------
# SynCo: Synthetic Hard Negatives in Contrastive Learning
# Copyright (c) 2025 Imperial College London
# Licensed under The MIT License [see LICENSE for details]
# Written by Nikolaos Giakoumoglou
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffdist import functional


class SynCo(nn.Module):
    def __init__(self,
                 cfg,
                 encoder,
                 encoder_k,
                 contrast_momentum=0.99,
                 contrast_temperature=0.2,
                 contrast_num_negative=4096,
                 proj_num_layers=2,
                 pred_num_layers=2,
                 n_hard=1024, 
                 n1=128, 
                 n2=128, 
                 n3=128, 
                 n4=64, 
                 n5=64, 
                 n6=64,
                 warmup_epochs=10,
                 cooldown_epochs=100,
                 **kwargs):
        super().__init__()
        
        self.cfg = cfg
        
        self.encoder = encoder
        self.encoder_k = encoder_k
        
        self.contrast_momentum = contrast_momentum
        self.contrast_temperature = contrast_temperature
        self.contrast_num_negative = contrast_num_negative
        
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

        if self.cfg.MODEL.SWIN.NORM_BEFORE_MLP == 'bn':
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.K = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.EPOCHS
        self.k = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() / self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.START_EPOCH

        # create the queue
        self.register_buffer("queue1", torch.randn(256, self.contrast_num_negative))
        self.register_buffer("queue2", torch.randn(256, self.contrast_num_negative))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # synco: https://arxiv.org/abs/2410.02401
        self.hard_alpha = 0.5
        self.hard_beta = 1.5
        self.hard_gamma = 1
        self.sigma = 0.1
        self.delta = 0.01
        self.eta = 0.01
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.n_hard = n_hard
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n5 = n5
        self.n4 = n4
        self.n6 = n6
        self.use_type1 = n1 > 0
        self.use_type2 = n2 > 0
        self.use_type3 = n3 > 0
        self.use_type5 = n4 > 0
        self.use_type4 = n5 > 0
        self.use_type6 = n6 > 0
        assert n_hard >= max(n1, n2, n3, n5, n4, n6), "n_hard should be greater than all n"

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
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        keys1 = dist_collect(keys1)
        keys2 = dist_collect(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.contrast_num_negative % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.contrast_num_negative  # move pointer

        self.queue_ptr[0] = ptr
    
    def find_hard_negatives(self, logits):
        """ 
        Find top-N hard negatives from queue
        """
        _, idxs_hard = torch.topk(logits.clone().detach(), k=self.n_hard, dim=-1, sorted=False)
        return idxs_hard

    def hard_negatives_interpolation(self, q, queue, idxs_hard):
        """
        Type 1 hard negatives: interpolated
        s = a * q + (1-a) * n
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n1), device=device)
        alpha = torch.rand(size=(batch_size, self.n1, 1), device=device) * self.hard_alpha 
        hard_negatives = queue.T[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        hard_negatives = alpha * q.clone().detach()[:, None] + (1 - alpha) * hard_negatives
        return nn.functional.normalize(hard_negatives, dim=-1).detach()

    def hard_negatives_extrapolation(self, q, queue, idxs_hard):
        """
        Type 2 hard negatives: extrapolated
        s = n + b * (n - q)
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n2), device=device)
        beta = 1 + torch.rand(size=(batch_size, self.n2, 1), device=device) * (self.hard_beta - 1)
        hard_negatives = queue.T[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        hard_negatives = q.clone().detach()[:, None] + beta * (hard_negatives - q.clone().detach()[:, None])
        return nn.functional.normalize(hard_negatives, dim=-1).detach()
    
    def hard_negatives_mixup(self, q, queue, idxs_hard):
        """
        Type 3 hard negatives: mixup
        s = g * n1 + (1-g) * n2
        """
        batch_size, device = q.shape[0], q.device
        batch_size, device = q.shape[0], q.device
        idxs1, idxs2 = torch.randint(0, self.n_hard, size=(2, batch_size, self.n3), device=device)
        gamma = torch.rand(size=(batch_size, self.n3, 1), device=device) * self.hard_gamma
        hard_negatives1 = queue.T[torch.gather(idxs_hard, dim=1, index=idxs1)].clone().detach()
        hard_negatives2 = queue.T[torch.gather(idxs_hard, dim=1, index=idxs2)].clone().detach()
        neg_hard = gamma * hard_negatives1 + (1 - gamma) * hard_negatives2
        return nn.functional.normalize(neg_hard, dim=-1).detach()
    
    def hard_negatives_noise_inject(self, q, queue, idxs_hard):
        """
        Type 4 hard negatives: noise injected
        s = n + N(0, var)
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n4), device=device)
        hard_negatives = queue.T[torch.gather(idxs_hard, dim=1, index=idxs)].clone().detach()
        noise = torch.randn_like(hard_negatives) * self.sigma
        return nn.functional.normalize(hard_negatives + noise, dim=-1).detach()
        
    def hard_negatives_autograd_1(self, q, queue, idxs_hard):
        """
        Type 5 hard negatives: perturbed using autograd
        s = n + d * grad(q, n)
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n5), device=device)
        hard_negatives = queue.T[idxs_hard[torch.arange(batch_size).unsqueeze(1), idxs]].detach().clone()
        hard_negatives_list = []
        for i in range(hard_negatives.size(1)):
            neighbor = hard_negatives[:, i, :].detach().clone().requires_grad_(True)
            similarity = torch.einsum('nc,nc->n', [q, neighbor])
            grad = torch.autograd.grad(similarity.sum(), neighbor, create_graph=False)[0]
            perturbed_neighbor = neighbor + self.delta * grad
            hard_negatives_list.append(perturbed_neighbor.detach())
        
        hard_negatives_final = torch.stack(hard_negatives_list, dim=1)
        return nn.functional.normalize(hard_negatives_final, dim=-1).detach()

    def hard_negatives_autograd_2(self, q, queue, idxs_hard):
        """
        Type 6 hard negatives: adversarial using autograd
        s = n + e * sgn(grad(q, n))
        """
        batch_size, device = q.shape[0], q.device
        idxs = torch.randint(0, self.n_hard, size=(batch_size, self.n6), device=device)
        hard_negatives = queue.T[idxs_hard[torch.arange(batch_size).unsqueeze(1), idxs]].clone().detach()
        hard_negatives_list = []
        for i in range(hard_negatives.size(1)):
            neighbor = hard_negatives[:, i, :].detach().clone().requires_grad_(True)
            similarity = torch.einsum('nc,nc->n', [q, neighbor])
            grad = torch.autograd.grad(similarity.sum(), neighbor, create_graph=False)[0]
            perturbed_neighbor = neighbor + self.eta * grad.sign()
            hard_negatives_list.append(perturbed_neighbor.detach())
        hard_negatives_final = torch.stack(hard_negatives_list, dim=1)
        return nn.functional.normalize(hard_negatives_final, dim=-1).detach()

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # get current epoch
        steps_per_epoch = self.cfg.DATA.TRAINING_IMAGES // self.cfg.DATA.BATCH_SIZE
        current_epoch = self.cfg.TRAIN.START_EPOCH + (self.k // steps_per_epoch)

        # create synthetic hard negatives
        if self.warmup_epochs <= current_epoch <= self.cooldown_epochs:
            # N-hardest negatives
            idxs_hard = self.find_hard_negatives(l_neg)
            
            # append negative logits with harder negatives
            if self.use_type1:
                h1 = self.hard_negatives_interpolation(q, queue, idxs_hard)
                l_neg_1 = torch.einsum("nc,nkc->nk", [q, h1])
                l_neg = torch.cat([l_neg, l_neg_1], dim=1)

            if self.use_type2:
                h2 = self.hard_negatives_extrapolation(q, queue, idxs_hard)
                l_neg_2 = torch.einsum("nc,nkc->nk", [q, h2])
                l_neg = torch.cat([l_neg, l_neg_2], dim=1)
                
            if self.use_type3:
                h3 = self.hard_negatives_mixup(q, queue, idxs_hard)
                l_neg_3 = torch.einsum("nc,nkc->nk", [q, h3])
                l_neg = torch.cat([l_neg, l_neg_3], dim=1)

            if self.use_type4:
                h4 = self.hard_negatives_noise_inject(q, queue, idxs_hard)
                l_neg_4 = torch.einsum("nc,nkc->nk", [q, h4])
                l_neg = torch.cat([l_neg, l_neg_4], dim=1)
                
            if self.use_type5:
                h5 = self.hard_negatives_autograd_1(q, queue, idxs_hard)
                l_neg_5 = torch.einsum("nc,nkc->nk", [q, h5])
                l_neg = torch.cat([l_neg, l_neg_5], dim=1)

            if self.use_type6:
                h6 = self.hard_negatives_autograd_2(q, queue, idxs_hard)
                l_neg_6 = torch.einsum("nc,nkc->nk", [q, h6])
                l_neg = torch.cat([l_neg, l_neg_6], dim=1)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)

    def forward(self, im_1, im_2):
        """
        Input:
            im_1: a batch of query images
            im_2: a batch of key images
        Output:
            loss: contrastive loss
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

            feat_1_ng = self.encoder_k(im_1)  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(im_2)
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        # compute loss
        loss = self.contrastive_loss(pred_1, proj_2_ng, self.queue2) \
            + self.contrastive_loss(pred_2, proj_1_ng, self.queue1)

        self._dequeue_and_enqueue(proj_1_ng, proj_2_ng)

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