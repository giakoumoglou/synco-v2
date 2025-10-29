#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torchvision import datasets


class OxfordParisDataset(torch.utils.data.Dataset):
    """Dataset for Oxford and Paris retrieval benchmarks"""
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))
        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)
        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['im_fname'] = config_imname
        cfg['qim_fname'] = config_qimname
        cfg['dataset'] = dataset
        self.cfg = cfg

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)
        return img, index


def config_imname(cfg, i):
    """Get image name from config"""
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])


def config_qimname(cfg, i):
    """Get query image name from config"""
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])


@torch.no_grad()
def extract_features(config, data_loader, model, logger, multiscale=False):
    """
    Extract features from the model for image retrieval.
    
    args:
        config (config): config
        data_loader (torch.utils.data.DataLoader): data loader
        model (nn.Module): model
        logger (Logger): logger
        multiscale (bool): whether to use multiscale feature extraction
    returns:
        features (torch.Tensor): extracted features
    """
    model.eval()
    
    features = None
    
    logger.info(f"Extracting features from {len(data_loader.dataset)} images...")
    
    for idx, (images, index) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        
        if multiscale:
            # Multiscale feature extraction
            feats = extract_multiscale_features(images, model)
        else:
            # Single scale feature extraction
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
        
        # Store features
        features[index] = feats
        
        if idx % config.PRINT_FREQ == 0:
            logger.info(f"Processed {idx}/{len(data_loader)} batches")
    
    # Normalize features
    features = nn.functional.normalize(features, dim=1, p=2)
    
    logger.info(f"Feature extraction completed. Feature shape: {features.shape}")
    
    return features


@torch.no_grad()
def extract_multiscale_features(images, model):
    """
    Extract multiscale features from images.
    
    args:
        images (torch.Tensor): input images
        model (nn.Module): model
    returns:
        features (torch.Tensor): multiscale features
    """
    # Define scales
    scales = [1.0, 0.707, 0.5, 0.354, 0.25]
    
    features = []
    for scale in scales:
        if scale != 1.0:
            h, w = images.shape[2:]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_images = torch.nn.functional.interpolate(
                images, size=(new_h, new_w), mode='bilinear', align_corners=False
            )
        else:
            scaled_images = images
        
        # Extract features
        if hasattr(model, 'module'):
            if hasattr(model.module, 'forward_features'):
                feats = model.module.forward_features(scaled_images)
            else:
                feats = model(scaled_images)
        else:
            if hasattr(model, 'forward_features'):
                feats = model.forward_features(scaled_images)
            else:
                feats = model(scaled_images)
        
        # Handle different feature dimensions
        if len(feats.shape) > 2:
            feats = feats.mean(dim=1)
        
        features.append(feats)
    
    # Average multiscale features
    features = torch.stack(features, dim=0).mean(dim=0)
    
    return features


def compute_map(ranks, gnd, kappas=[]):
    """
    Compute mean average precision (mAP).
    
    args:
        ranks (numpy.ndarray): ranks of retrieved images
        gnd (list): ground truth
        kappas (list): k values for precision at k
    returns:
        map (float): mean average precision
        aps (numpy.ndarray): average precision for each query
        mpr (float): mean precision at k
        prs (numpy.ndarray): precision at k for each query
    """
    # Number of queries
    nq = len(gnd)
    
    # Initialize
    aps = np.zeros(nq)
    prs = np.zeros((nq, len(kappas))) if len(kappas) > 0 else None
    
    for i in range(nq):
        qgnd = np.array(gnd[i]['ok'])
        qgndj = np.array(gnd[i]['junk'])
        
        # Positions of positive and junk images
        pos = np.arange(ranks.shape[0])
        
        # Remove junk images
        junk_idx = np.in1d(ranks[:, i], qgndj)
        pos = pos[~junk_idx]
        
        # Get ranks of positive images
        ranks_pos = ranks[~junk_idx, i]
        
        # Get indices of positive images
        pos_idx = np.in1d(ranks_pos, qgnd)
        
        if np.sum(pos_idx) == 0:
            aps[i] = 0
            if prs is not None:
                prs[i, :] = 0
            continue
        
        # Compute average precision
        pos_positions = pos[pos_idx]
        recall_step = 1.0 / len(qgnd)
        
        ap = 0
        for j, pos_position in enumerate(pos_positions):
            precision = (j + 1) / (pos_position + 1)
            ap += precision * recall_step
        
        aps[i] = ap
        
        # Compute precision at k
        if prs is not None:
            for j, k in enumerate(kappas):
                if k > len(pos):
                    prs[i, j] = 0
                else:
                    prs[i, j] = np.sum(pos_idx[:k]) / k
    
    # Compute mean
    map_value = np.mean(aps)
    mpr = np.mean(prs, axis=0) if prs is not None else None
    
    return map_value, aps, mpr, prs


@torch.no_grad()
def evaluate_retrieval(train_features, query_features, gnd, ks=[1, 5, 10], logger=None):
    """
    Evaluate image retrieval performance.
    
    args:
        train_features (torch.Tensor): database features
        query_features (torch.Tensor): query features
        gnd (list): ground truth
        ks (list): k values for precision at k
        logger (Logger): logger
    returns:
        results (dict): evaluation results
    """
    if logger:
        logger.info("Computing similarity matrix...")
    
    # Compute similarity
    sim = torch.mm(query_features, train_features.T)
    ranks = torch.argsort(-sim, dim=1).cpu().numpy()
    
    if logger:
        logger.info("Evaluating results...")
    
    # Evaluate for Medium (easy + hard)
    gnd_medium = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk']])
        gnd_medium.append(g)
    
    mapM, apsM, mprM, prsM = compute_map(ranks.T, gnd_medium, ks)
    
    # Evaluate for Hard
    gnd_hard = []
    for i in range(len(gnd)):
        g = {}
        g['ok'] = np.concatenate([gnd[i]['hard']])
        g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
        gnd_hard.append(g)
    
    mapH, apsH, mprH, prsH = compute_map(ranks.T, gnd_hard, ks)
    
    results = {
        'mAP_M': mapM * 100,
        'mAP_H': mapH * 100,
        'mP@k_M': mprM * 100 if mprM is not None else None,
        'mP@k_H': mprH * 100 if mprH is not None else None,
        'ks': ks
    }
    
    return results