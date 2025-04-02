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

# the following imports use no tricks
#from timm.models import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224
#from timm.models import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_large_patch4_window7_224
from timm.models import resnet50, resnet101, resnet152, resnet200

# the following imports use tricks
# 1) no tricks here da
from .swin_transformer import swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224, swin_large_patch4_window7_224
# 2) swin transformer with batch norm instead of layer norm before the final classifier
from .swin_transformer import swin_tiny_bn_patch4_window7_224, swin_small_bn_patch4_window7_224, swin_base_bn_patch4_window7_224, swin_large_bn_patch4_window7_224
# 3) vision transformer with stop gradients on the first conv layer
from .vision_transformer import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224
# 4) vision transformer with stop gradients on the first conv layer and conv layer with kernel size 3x3
from .vision_transformer import vit_conv_small_patch16_224, vit_conv_base_patch16_224, vit_conv_large_patch16_224, vit_conv_huge_patch14_224

from .moby import MoBY
from .byol import BYOL
from .synco import SynCo


models = dict(
    # vit
    vit_small=vit_small_patch16_224, # can use stop_grad_conv1=True though
    vit_base=vit_base_patch16_224,
    vit_large=vit_large_patch16_224,
    vit_huge=vit_huge_patch16_224,
    # vit with conv stem
    vit_conv_small=vit_conv_small_patch16_224,
    vit_conv_base=vit_conv_base_patch16_224,
    vit_conv_large=vit_conv_large_patch16_224,
    vit_conv_huge=vit_conv_huge_patch14_224,
    # swin
    swin_tiny=swin_tiny_patch4_window7_224,
    swin_small=swin_small_patch4_window7_224,
    swin_base=swin_base_patch4_window7_224,
    swin_large=swin_large_patch4_window7_224,
    # swin with batch norm
    swin_tiny_bn=swin_tiny_bn_patch4_window7_224,
    swin_small_bn=swin_small_bn_patch4_window7_224,
    swin_base_bn=swin_base_bn_patch4_window7_224,
    swin_large_bn=swin_large_bn_patch4_window7_224,
    # resnet, sweet old resnets
    resnet50=resnet50,
    resnet101=resnet101,
    resnet152=resnet152,
    resnet200=resnet200,
)


def build_model(config):
    """
    Build self-supervised learning model.

    args:
        config (config): config
    returns:
        model (nn.Module): self-supervised or linear model
    """
    model_type = config.MODEL.TYPE
    encoder_type = config.MODEL.SSL.ENCODER

    stop_grad_conv1 = getattr(config.MODEL.SSL, 'STOP_GRAD_CONV1', False)
    print('-----> Using stop grad on conv1:', stop_grad_conv1)

    # ================ encoder ================
    if encoder_type.startswith('resnet'):
        enc = partial(models[encoder_type], num_classes=0)
    elif encoder_type.startswith('vit'):
        # apply stop_grad_conv1 parameter for ViT models
        enc = partial(models[encoder_type], num_classes=0, stop_grad_conv1=stop_grad_conv1)
    elif encoder_type in models:
        enc = partial(models[encoder_type], num_classes=0)
    else:
        raise NotImplementedError(f'-----> Unknown `encoder_type`: {encoder_type}')
    
    # ================ byol ================
    if model_type == 'byol':
        print(colored(f"Building BYOL with {encoder_type} encoder:", "blue"))
        if encoder_type.startswith('resnet'):
            encoder = enc()
            encoder_k = enc()
        else:
            encoder = enc(drop_path_rate=config.MODEL.SSL.ONLINE_DROP_PATH_RATE)
            encoder_k = enc(drop_path_rate=config.MODEL.SSL.TARGET_DROP_PATH_RATE)
        model = BYOL(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.SSL.CONTRAST_MOMENTUM,
            proj_num_layers=config.MODEL.SSL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.SSL.PRED_NUM_LAYERS,
        )
    # ================ moby ================
    elif model_type == 'moby':
        print(colored(f"Building MoBY with {encoder_type} encoder:", "blue"))
        if encoder_type.startswith('resnet'):
            encoder = enc()
            encoder_k = enc()
        else:
            encoder = enc(drop_path_rate=config.MODEL.SSL.ONLINE_DROP_PATH_RATE)
            encoder_k = enc(drop_path_rate=config.MODEL.SSL.TARGET_DROP_PATH_RATE)
        model = MoBY(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.SSL.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.SSL.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.SSL.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.SSL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.SSL.PRED_NUM_LAYERS,
        )
    # ================ synco ================
    elif model_type == 'synco':
        print(colored(f"Building SyncO with {encoder_type} encoder:", "blue"))
        if encoder_type.startswith('resnet'):
            encoder = enc()
            encoder_k = enc()
        else:
            encoder = enc(drop_path_rate=config.MODEL.SSL.ONLINE_DROP_PATH_RATE)
            encoder_k = enc(drop_path_rate=config.MODEL.SSL.TARGET_DROP_PATH_RATE)
        model = SynCo(
            cfg=config,
            encoder=encoder,
            encoder_k=encoder_k,
            contrast_momentum=config.MODEL.SSL.CONTRAST_MOMENTUM,
            contrast_temperature=config.MODEL.SSL.CONTRAST_TEMPERATURE,
            contrast_num_negative=config.MODEL.SSL.CONTRAST_NUM_NEGATIVE,
            proj_num_layers=config.MODEL.SSL.PROJ_NUM_LAYERS,
            pred_num_layers=config.MODEL.SSL.PRED_NUM_LAYERS,
            n_hard=config.MODEL.SSL.N_HARD, 
            n1=config.MODEL.SSL.N1, 
            n2=config.MODEL.SSL.N2, 
            n3=config.MODEL.SSL.N3, 
            n4=config.MODEL.SSL.N4, 
            n5=config.MODEL.SSL.N5, 
            n6=config.MODEL.SSL.N6,
            warmup_epochs=config.MODEL.SSL.WARMUP_EPOCHS,
            cooldown_epochs=config.MODEL.SSL.COOLDOWN_EPOCHS,
        )
    # ================ linear ================
    elif model_type == 'linear':
        print(colored(f"Building linear classifier with {encoder_type} encoder:", "blue"))
        if encoder_type.startswith('resnet'):
            model = enc(num_classes=config.MODEL.NUM_CLASSES)
        else:
            model = enc(num_classes=config.MODEL.NUM_CLASSES, drop_path_rate=config.MODEL.DROP_PATH_RATE)
    else:
        raise NotImplementedError(f'-----> Unknown `model_type`: {model_type}')

    return model
