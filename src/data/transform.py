#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from PIL import ImageFilter, ImageOps

from torchvision import transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.transforms import _pil_interp


def build_transform(is_train, config):
    """
    Build data transformation pipeline.
    
    Supports different augmentation strategies:
    - SSL augmentations (BYOL, SimCLR)
    - Linear evaluation augmentations
    - Default supervised augmentations
    
    Args:
        is_train: True for training, False for validation
        config: Configuration object with augmentation settings
        
    Returns:
        transform: Single transform or tuple of transforms
        
    Example:
        >>> # BYOL augmentation
        >>> config.AUG.SSL_AUG = True
        >>> config.AUG.SSL_AUG_TYPE = 'byol'
        >>> transform = build_transform(is_train=True, config=config)
        >>> # Returns (transform_1, transform_2) for two views
    """
    # ================ SSL augmentation ================
    if config.AUG.SSL_AUG:
        return _build_ssl_transform(config)
    
    # ================ Linear eval augmentation ================
    if config.AUG.SSL_LINEAR_AUG:
        return _build_linear_eval_transform(is_train, config)
    
    # ================ Default supervised augmentation ================
    return _build_default_transform(is_train, config)


def _build_ssl_transform(config):
    """
    Build SSL augmentation (BYOL or SimCLR).
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of two transforms for creating two views
        
    Raises:
        NotImplementedError: If SSL augmentation type is not supported
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # ================ BYOL augmentation ================
    if config.AUG.SSL_AUG_TYPE == 'byol':
        # First view: with Gaussian blur
        transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                config.DATA.IMG_SIZE,
                scale=(config.AUG.SSL_AUG_CROP, 1.)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=1.0),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Second view: with solarization instead of strong blur
        transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(
                config.DATA.IMG_SIZE,
                scale=(config.AUG.SSL_AUG_CROP, 1.)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.1),
            transforms.RandomApply([ImageOps.solarize], p=0.2),
            transforms.ToTensor(),
            normalize,
        ])
        
        return (transform_1, transform_2)
    
    # ================ SimCLR augmentation ================
    elif config.AUG.SSL_AUG_TYPE == 'simclr':
        transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(
                config.DATA.IMG_SIZE,
                scale=(0.2, 1.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
            transforms.ToTensor(),
            normalize,
        ])
        
        # Use same transform for both views in SimCLR
        return (transform_1, transform_1)
    
    else:
        raise NotImplementedError(
            f"-----> Unknown SSL augmentation type: {config.AUG.SSL_AUG_TYPE}. "
            f"Supported types: 'byol', 'simclr'"
        )


def _build_linear_eval_transform(is_train, config):
    """
    Build augmentation for linear evaluation.
    
    Lighter augmentation compared to SSL training.
    
    Args:
        is_train: True for training, False for validation
        config: Configuration object
        
    Returns:
        Transform pipeline
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(config.DATA.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(config.DATA.IMG_SIZE + 32),
            transforms.CenterCrop(config.DATA.IMG_SIZE),
            transforms.ToTensor(),
            normalize,
        ])
    
    return transform


def _build_default_transform(is_train, config):
    """
    Build default supervised learning augmentation.
    
    Uses timm's create_transform with AutoAugment, RandAugment, etc.
    
    Args:
        is_train: True for training, False for validation
        config: Configuration object
        
    Returns:
        Transform pipeline
    """
    resize_im = config.DATA.IMG_SIZE > 32
    
    if is_train:
        # Use timm's comprehensive augmentation
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        
        if not resize_im:
            # Replace RandomResizedCrop with RandomCrop for small images
            transform.transforms[0] = transforms.RandomCrop(
                config.DATA.IMG_SIZE,
                padding=4
            )
        
        return transform
    
    # Validation transform
    t = []
    
    if resize_im:
        if config.TEST.CROP:
            # Standard ImageNet validation: resize to 256, center crop to 224
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(
                size,
                interpolation=_pil_interp(config.DATA.INTERPOLATION)
            ))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            # Direct resize without crop
            t.append(transforms.Resize(
                (config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                interpolation=_pil_interp(config.DATA.INTERPOLATION)
            ))
    
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    
    return transforms.Compose(t)


class GaussianBlur:
    """
    Gaussian blur augmentation.
    
    Randomly applies Gaussian blur with sigma in the given range.
    Used in BYOL and SimCLR augmentation strategies.
    
    Args:
        sigma: Sigma range for Gaussian blur. Can be:
            - None: sigma randomly chosen from [0.1, 2.0]
            - List/tuple: [min_sigma, max_sigma]
            
    Example:
        >>> blur = GaussianBlur(sigma=[0.1, 2.0])
        >>> blurred_image = blur(image)
    """
    
    def __init__(self, sigma=None):
        if sigma is None:
            self.sigma = [0.1, 2.0]
        else:
            self.sigma = sigma
    
    def __call__(self, x):
        """
        Apply Gaussian blur to image.
        
        Args:
            x: PIL Image
            
        Returns:
            Blurred PIL Image
        """
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x