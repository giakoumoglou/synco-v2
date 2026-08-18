# Copyright (C) 2026.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from functools import partial
from termcolor import colored

from .moby import MoBY
from .byol import BYOL
from .dino import DINO
from .vitamins import ViTAMINS

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
    # vision transformers
    vit_small=vit_small_patch16_224,
    vit_base=vit_base_patch16_224,
    vit_large=vit_large_patch16_224,
    vit_huge=vit_huge_patch14_224,
    
    # swin transformers
    swin_tiny=swin_tiny_patch4_window7_224,
    swin_small=swin_small_patch4_window7_224,
    swin_base=swin_base_patch4_window7_224,
    swin_large=swin_large_patch4_window7_224,
    
    # resnets
    resnet50=resnet50,
    resnet101=resnet101,
    resnet152=resnet152,
    resnet200=resnet200,
)


def build_model(config):
    """
    Build self-supervised learning model
    """
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.ENCODER
    stop_grad_conv1 = getattr(config.MODEL, 'STOP_GRAD_CONV1', False)
    print(colored(f"==============> Building {model_type} with {encoder_type} ....................", "red"))

    # ================ encoder ... ================
    def create_encoder(drop_path_rate=0.0):
        """
        Create encoder with consistent parameters for all architectures
        """
        common_args = {
            'num_classes': 0,  # for feature extraction
            'drop_path_rate': drop_path_rate,
        }
        
        if encoder_type.startswith('vit') or encoder_type.startswith('swin'):
            common_args['stop_grad_conv1'] = stop_grad_conv1
        elif encoder_type.startswith('resnet'):
            return models[encoder_type](num_classes=0)  # resnet does not support drop path
            
        return models[encoder_type](**common_args)

    # ================ model ... ================
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
    elif model_type == 'dino':
        model = DINO(
            cfg=config,
            encoder=create_encoder(config.MODEL.ONLINE_DROP_PATH_RATE),
            encoder_k=create_encoder(config.MODEL.TARGET_DROP_PATH_RATE),
            contrast_momentum=config.MODEL.CONTRAST_MOMENTUM,
            out_dim=config.MODEL.DINO_OUT_DIM,
            hidden_dim=config.MODEL.DINO_HIDDEN_DIM,
            bottleneck_dim=config.MODEL.DINO_BOTTLENECK_DIM,
            proj_num_layers=config.MODEL.PROJ_NUM_LAYERS,
            use_bn=config.MODEL.DINO_USE_BN_IN_HEAD,
            norm_last_layer=config.MODEL.DINO_NORM_LAST_LAYER,
            student_temp=config.MODEL.DINO_STUDENT_TEMP,
            teacher_temp=config.MODEL.DINO_TEACHER_TEMP,
            warmup_teacher_temp=config.MODEL.DINO_WARMUP_TEACHER_TEMP,
            warmup_teacher_temp_epochs=config.MODEL.DINO_WARMUP_TEACHER_TEMP_EPOCHS,
            center_momentum=config.MODEL.DINO_CENTER_MOMENTUM,
            freeze_last_layer_epochs=config.MODEL.DINO_FREEZE_LAST_LAYER_EPOCHS,
        )
    elif model_type == 'vitamins':
        model = ViTAMINS(
            cfg=config,
            encoder=create_encoder(config.MODEL.ONLINE_DROP_PATH_RATE),
            encoder_k=create_encoder(config.MODEL.TARGET_DROP_PATH_RATE),
            contrast_momentum=config.MODEL.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.PRED_NUM_LAYERS,
            n_hard=config.MODEL.N_HARD,
            n1=config.MODEL.N1,
            n2=config.MODEL.N2,
            n3=config.MODEL.N3,
            n4=config.MODEL.N4,
            n5=config.MODEL.N5,
            n6=config.MODEL.N6,
            warmup_epochs=config.MODEL.WARMUP_EPOCHS,
            cooldown_epochs=config.MODEL.COOLDOWN_EPOCHS,
        )
    elif model_type == 'linear':
        linear_args = {'num_classes': config.MODEL.NUM_CLASSES}
        if not encoder_type.startswith('resnet'):
            linear_args['drop_path_rate'] = config.MODEL.DROP_PATH_RATE
        model = models[encoder_type](**linear_args)
    else:
        raise NotImplementedError(f'-----> Unknown model_type: {model_type}, we only support byol, moby, dino, vitamins and linear')

    return model