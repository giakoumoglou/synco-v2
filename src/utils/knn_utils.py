#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.distributed as dist
from timm.utils import AverageMeter


@torch.no_grad()
def extract_features(config, data_loader, model, logger):
    """
    Extract features from the model for KNN evaluation.
    
    args:
        config (config): config
        data_loader (torch.utils.data.DataLoader): data loader
        model (nn.Module): model
        logger (Logger): logger
    returns:
        features (torch.Tensor): extracted features
        labels (torch.Tensor): labels
    """
    model.eval()
    
    features = None
    labels_list = []
    
    logger.info(f"Extracting features from {len(data_loader.dataset)} images...")
    
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        # Forward pass to extract features
        # For models with linear head, we want the features before the head
        if hasattr(model, 'module'):
            if hasattr(model.module, 'forward_features'):
                feats = model.module.forward_features(images)
            elif hasattr(model.module, 'encoder'):
                feats = model.module.encoder(images)
            else:
                # Try to get features before the final classifier
                if hasattr(model.module, 'head'):
                    # Temporarily remove head
                    head = model.module.head
                    model.module.head = nn.Identity()
                    feats = model.module(images)
                    model.module.head = head
                else:
                    feats = model(images)
        else:
            if hasattr(model, 'forward_features'):
                feats = model.forward_features(images)
            elif hasattr(model, 'encoder'):
                feats = model.encoder(images)
            else:
                # Try to get features before the final classifier
                if hasattr(model, 'head'):
                    head = model.head
                    model.head = nn.Identity()
                    feats = model(images)
                    model.head = head
                else:
                    feats = model(images)
        
        # Handle different feature dimensions
        if len(feats.shape) > 2:
            feats = feats.mean(dim=1)  # Global average pooling if needed
        
        # Initialize storage feature matrix
        if features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            features = features.cuda(non_blocking=True)
            logger.info(f"Storing features into tensor of shape {features.shape}")
        
        # Get the actual indices for this batch
        batch_start = idx * config.DATA.BATCH_SIZE
        batch_end = min(batch_start + feats.size(0), len(data_loader.dataset))
        features[batch_start:batch_end] = feats
        
        labels_list.append(target)
        
        if idx % config.PRINT_FREQ == 0:
            logger.info(f"Processed {idx}/{len(data_loader)} batches")
    
    # Concatenate all labels
    labels = torch.cat(labels_list, dim=0)
    
    # Normalize features
    features = nn.functional.normalize(features, dim=1, p=2)
    
    logger.info(f"Feature extraction completed. Feature shape: {features.shape}")
    
    return features, labels


@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes, logger):
    """
    Weighted k-NN classifier.
    
    args:
        train_features (torch.Tensor): training features
        train_labels (torch.Tensor): training labels
        test_features (torch.Tensor): test features
        test_labels (torch.Tensor): test labels
        k (int): number of nearest neighbors
        T (float): temperature for weighting
        num_classes (int): number of classes
        logger (Logger): logger
    returns:
        top1 (float): top-1 accuracy
        top5 (float): top-5 accuracy
    """
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images = test_labels.shape[0]
    num_chunks = 100
    imgs_per_chunk = max(num_test_images // num_chunks, 1)
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    
    logger.info(f"Running {k}-NN classification with temperature {T}")
    
    for idx in range(0, num_test_images, imgs_per_chunk):
        # Get the features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # Calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # Find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()
        total += targets.size(0)
    
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    
    return top1, top5