#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import glob
import queue
from urllib.request import urlopen

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def extract_feature(model, frame, return_h_w=False, original_size=None):
    """
    Extract one frame feature for video segmentation.
    
    args:
        model (nn.Module): model
        frame (torch.Tensor): input frame
        return_h_w (bool): whether to return spatial dimensions
        original_size (tuple): original frame size (h, w)
    returns:
        out (torch.Tensor): extracted features [h*w, dim]
        h (int): feature height (if return_h_w=True)
        w (int): feature width (if return_h_w=True)
    """
    frame_input = frame.unsqueeze(0).cuda()
    
    # Extract features
    if hasattr(model, 'module'):
        if hasattr(model.module, 'forward_features'):
            features = model.module.forward_features(frame_input)
        elif hasattr(model.module, 'get_intermediate_layers'):
            features = model.module.get_intermediate_layers(frame_input, n=1)[0]
        else:
            features = model(frame_input)
    else:
        if hasattr(model, 'forward_features'):
            features = model.forward_features(frame_input)
        elif hasattr(model, 'get_intermediate_layers'):
            features = model.get_intermediate_layers(frame_input, n=1)[0]
        else:
            features = model(frame_input)
    
    # Handle different feature formats
    if len(features.shape) == 3:  # [B, num_patches, dim]
        # Remove batch dimension and CLS token if present
        out = features[0]
        if out.shape[0] > (frame.shape[1] // 16) * (frame.shape[2] // 16):
            # Has CLS token, remove it
            out = out[1:, :]
        
        # Calculate spatial dimensions
        if original_size is not None:
            h, w = original_size[0] // 16, original_size[1] // 16
        else:
            h = frame.shape[1] // 16
            w = frame.shape[2] // 16
        
        dim = out.shape[-1]
        
        # Reshape to spatial grid and back to [h*w, dim]
        out = out.reshape(h, w, dim).reshape(-1, dim)
        
    elif len(features.shape) == 2:  # [num_patches, dim]
        out = features
        
        # Calculate spatial dimensions
        if original_size is not None:
            h, w = original_size[0] // 16, original_size[1] // 16
        else:
            h = frame.shape[1] // 16
            w = frame.shape[2] // 16
    
    else:
        raise ValueError(f"Unexpected feature shape: {features.shape}")
    
    if return_h_w:
        return out, h, w
    return out


def restrict_neighborhood(h, w, size_mask_neighborhood):
    """
    Restrict the set of source nodes to a spatial neighborhood.
    
    args:
        h (int): feature height
        w (int): feature width
        size_mask_neighborhood (int): neighborhood size
    returns:
        mask (torch.Tensor): neighborhood mask [h*w, h*w]
    """
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1

    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def norm_mask(mask):
    """
    Normalize mask to [0, 1] range.
    
    args:
        mask (torch.Tensor): input mask [C, H, W]
    returns:
        mask (torch.Tensor): normalized mask
    """
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if mask_cnt.max() > 0:
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


@torch.no_grad()
def label_propagation(model, frame_tar, list_frame_feats, list_segs, 
                     mask_neighborhood, size_mask_neighborhood, topk, 
                     patch_size, original_size=None, logger=None):
    """
    Propagate segmentation labels from reference frames to target frame.
    
    args:
        model (nn.Module): model
        frame_tar (torch.Tensor): target frame
        list_frame_feats (list): list of reference frame features
        list_segs (list): list of reference segmentations
        mask_neighborhood (torch.Tensor): neighborhood mask
        size_mask_neighborhood (int): neighborhood size
        topk (int): number of top neighbors
        patch_size (int): patch size
        original_size (tuple): original frame size
        logger (Logger): logger
    returns:
        seg_tar (torch.Tensor): propagated segmentation
        return_feat_tar (torch.Tensor): target frame features
        mask_neighborhood (torch.Tensor): updated neighborhood mask
    """
    # Extract features of the target frame
    feat_tar, h, w = extract_feature(model, frame_tar, return_h_w=True, original_size=original_size)
    
    return_feat_tar = feat_tar.T  # dim x h*w
    
    ncontext = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats)  # nmb_context x dim x h*w
    
    # Normalize features
    feat_tar = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)
    
    # Compute affinity
    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / 0.1)  # nmb_context x h*w x h*w
    
    # Apply spatial neighborhood constraint
    if size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = restrict_neighborhood(h, w, size_mask_neighborhood)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood
    
    # Apply top-k filtering
    aff = aff.transpose(2, 1).reshape(-1, h * w)  # nmb_context*h*w x h*w
    tk_val, _ = torch.topk(aff, dim=0, k=topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0
    
    # Normalize affinity
    aff = aff / torch.sum(aff, keepdim=True, axis=0)
    
    # Propagate labels
    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h_seg, w_seg = segs.shape
    segs = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T  # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)
    
    return seg_tar, return_feat_tar, mask_neighborhood


@torch.no_grad()
def eval_video_tracking(config, model, frame_list, video_name, first_seg, seg_ori, 
                       color_palette, n_last_frames, size_mask_neighborhood, 
                       topk, patch_size, logger):
    """
    Evaluate tracking on a video given first frame & segmentation.
    
    args:
        config (config): config
        model (nn.Module): model
        frame_list (list): list of frame paths
        video_name (str): video name
        first_seg (torch.Tensor): first frame segmentation
        seg_ori (np.ndarray): original first frame segmentation
        color_palette (np.ndarray): color palette
        n_last_frames (int): number of preceding frames to use
        size_mask_neighborhood (int): spatial neighborhood size
        topk (int): number of top neighbors
        patch_size (int): patch size
        logger (Logger): logger
    returns:
        None
    """
    video_folder = os.path.join(config.OUTPUT, video_name)
    os.makedirs(video_folder, exist_ok=True)
    
    # The queue stores the n preceding frames
    que = queue.Queue(n_last_frames)
    
    # First frame
    frame1, ori_h, ori_w = read_frame(frame_list[0])
    # Extract first frame feature
    frame1_feat = extract_feature(model, frame1, original_size=(ori_h, ori_w)).T  # dim x h*w
    
    # Save first segmentation
    out_path = os.path.join(video_folder, "00000.png")
    imwrite_indexed(out_path, seg_ori, color_palette)
    
    mask_neighborhood = None
    
    logger.info(f"Processing video {video_name} with {len(frame_list)} frames...")
    
    for cnt in tqdm(range(1, len(frame_list)), desc=f"Processing {video_name}"):
        frame_tar, frame_tar_ori_h, frame_tar_ori_w = read_frame(frame_list[cnt])
        
        # Use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs = [first_seg] + [pair[1] for pair in list(que.queue)]
        
        # Label propagation
        frame_tar_avg, feat_tar, mask_neighborhood = label_propagation(
            model, frame_tar, used_frame_feats, used_segs, 
            mask_neighborhood, size_mask_neighborhood, topk, 
            patch_size, original_size=(frame_tar_ori_h, frame_tar_ori_w)
        )
        
        # Pop out oldest frame if necessary
        if que.qsize() == n_last_frames:
            que.get()
        # Push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])
        
        # Upsampling & argmax
        frame_tar_avg = F.interpolate(
            frame_tar_avg, 
            scale_factor=patch_size, 
            mode='bilinear', 
            align_corners=False, 
            recompute_scale_factor=False
        )[0]
        frame_tar_avg = norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)
        
        # Save to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(
            Image.fromarray(frame_tar_seg).resize((frame_tar_ori_w, frame_tar_ori_h), 0)
        )
        frame_nm = os.path.basename(frame_list[cnt]).replace(".jpg", ".png")
        imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)
    
    logger.info(f"Completed video {video_name}")


def read_frame_list(video_dir):
    """Read list of frames in a video directory."""
    frame_list = [img for img in glob.glob(os.path.join(video_dir, "*.jpg"))]
    frame_list = sorted(frame_list)
    return frame_list


def read_frame(frame_dir, scale_size=[480]):
    """
    Read a single frame & preprocess.
    
    args:
        frame_dir (str): path to frame
        scale_size (list): target size
    returns:
        img (torch.Tensor): preprocessed frame
        ori_h (int): original height
        ori_w (int): original width
    """
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    
    if len(scale_size) == 1:
        if ori_h > ori_w:
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    
    return img, ori_h, ori_w


def read_seg(seg_dir, factor, scale_size=[480]):
    """
    Read segmentation mask.
    
    args:
        seg_dir (str): path to segmentation
        factor (int): downsampling factor
        scale_size (list): target size
    returns:
        small_seg (torch.Tensor): downsampled one-hot segmentation
        seg_ori (np.ndarray): original segmentation
    """
    seg = Image.open(seg_dir)
    _w, _h = seg.size  # note PIL.Image.Image's size is (w, h)
    
    if len(scale_size) == 1:
        if _w > _h:
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 64) * 64)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 64) * 64)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    
    small_seg = np.array(seg.resize((_tw // factor, _th // factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    
    return to_one_hot(small_seg), np.asarray(seg)


def to_one_hot(y_tensor, n_dims=None):
    """
    Convert integer tensor to one-hot representation.
    
    args:
        y_tensor (torch.Tensor): integer tensor
        n_dims (int): number of classes
    returns:
        y_one_hot (torch.Tensor): one-hot tensor
    """
    if n_dims is None:
        n_dims = int(y_tensor.max() + 1)
    _, h, w = y_tensor.size()
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def color_normalize(x, mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225]):
    """Normalize image with ImageNet statistics."""
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def imwrite_indexed(filename, array, color_palette):
    """Save indexed PNG for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")
    
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')


def load_color_palette():
    """Load color palette from URL."""
    color_palette = []
    for line in urlopen("https://raw.githubusercontent.com/Liusifei/UVC/master/libs/data/palette.txt"):
        color_palette.append([int(i) for i in line.decode("utf-8").split('\n')[0].split(" ")])
    color_palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)
    return color_palette