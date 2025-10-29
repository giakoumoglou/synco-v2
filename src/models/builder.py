#!/usr/bin/env python

# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# Modified by Nikolaos Giakoumoglou
# --------------------------------------------------------

from functools import partial
from termcolor import colored

from .moby import MoBY
from .byol import BYOL
from .synco import SynCo

# resnet imports
from timm.models import resnet50, resnet101, resnet152, resnet200

# custom vision transformer imports
from .vision_transformer import (
    vit_small_patch16_224,
    vit_base_patch16_224,
    vit_large_patch16_224,
    vit_huge_patch14_224,
)

# custom swin transformer imports
from .swin_transformer import (
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
    swin_large_patch4_window7_224,
)

models = dict(
    # Vision Transformers
    vit_small=vit_small_patch16_224,
    vit_base=vit_base_patch16_224,
    vit_large=vit_large_patch16_224,
    vit_huge=vit_huge_patch14_224,
    
    # Swin Transformers
    swin_tiny=swin_tiny_patch4_window7_224,
    swin_small=swin_small_patch4_window7_224,
    swin_base=swin_base_patch4_window7_224,
    swin_large=swin_large_patch4_window7_224,
    
    # ResNets
    resnet50=resnet50,
    resnet101=resnet101,
    resnet152=resnet152,
    resnet200=resnet200,
)


def build_model(config):
    """
    Build self-supervised learning model.
    
    Supports various SSL methods (BYOL, MoBY, SynCo) and linear evaluation.
    Creates appropriate encoder architecture based on configuration.
    
    Args:
        config: Configuration object with model settings
        
    Returns:
        model: Initialized model ready for training
        
    Raises:
        NotImplementedError: If model type is not supported
    """
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.ENCODER
    stop_grad_conv1 = getattr(config.MODEL, 'STOP_GRAD_CONV1', False)
    
    print(colored(
        f"==============> Building {model_type} with {encoder_type} ....................",
        "red"
    ))

    # ================ encoder factory ================
    def create_encoder(drop_path_rate=0.0):
        """
        Create encoder with consistent parameters for all architectures.
        
        Args:
            drop_path_rate: Stochastic depth rate for ViT/Swin models
            
        Returns:
            encoder: Initialized encoder network
        """
        common_args = {
            'num_classes': 0,  # for feature extraction
            'drop_path_rate': drop_path_rate
        }
        
        if encoder_type.startswith('vit') or encoder_type.startswith('swin'):
            common_args['stop_grad_conv1'] = stop_grad_conv1
        elif encoder_type.startswith('resnet'):
            return models[encoder_type](num_classes=0)  # ResNet doesn't support drop_path
            
        return models[encoder_type](**common_args)

    # ================ build model ================
    if model_type == 'byol':
        model = BYOL(
            cfg=config,
            encoder=create_encoder(config.MODEL.ONLINE_DROP_PATH_RATE),
            encoder_k=create_encoder(config.MODEL.TARGET_DROP_PATH_RATE),
            contrast_momentum=config.MODEL.CONTRAST_MOMENTUM,
            proj_num_layers=config.MODEL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.PRED_NUM_LAYERS,
        )
        
    elif model_type == 'moby':
        model = MoBY(
            cfg=config,
            encoder=create_encoder(config.MODEL.ONLINE_DROP_PATH_RATE),
            encoder_k=create_encoder(config.MODEL.TARGET_DROP_PATH_RATE),
            contrast_momentum=config.MODEL.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.PRED_NUM_LAYERS,
        )
        
    elif model_type == 'synco':
        model = SynCo(
            cfg=config,
            encoder=create_encoder(config.MODEL.ONLINE_DROP_PATH_RATE),
            encoder_k=create_encoder(config.MODEL.TARGET_DROP_PATH_RATE),
            contrast_momentum=config.MODEL.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.PRED_NUM_LAYERS,
            n_hard=config.MODEL.N_HARD,
            warmup_epochs=config.MODEL.WARMUP_EPOCHS,
            cooldown_epochs=config.MODEL.COOLDOWN_EPOCHS,
        )
        
    elif model_type == 'linear':
        linear_args = {'num_classes': config.MODEL.NUM_CLASSES}
        if not encoder_type.startswith('resnet'):
            linear_args['drop_path_rate'] = config.MODEL.DROP_PATH_RATE
        model = models[encoder_type](**linear_args)
        
    else:
        raise NotImplementedError(
            f"-----> Unknown model_type: {model_type}. "
            f"Only 'byol', 'moby', 'synco', and 'linear' are currently supported."
        )

    return model