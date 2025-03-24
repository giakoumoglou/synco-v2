# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

import os
import numpy as np
from PIL import ImageFilter, ImageOps
from termcolor import colored

import torch
import torch.distributed as dist
from torchvision import datasets, transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp

from .cached_image_folder import CachedImageFolder
from .custom_image_folder import CustomImageFolder
from .samplers import SubsetRandomSampler


def build_loader(config, logger):
    """
    Build data loader for training and validation.

    args:
        config (config): config
    returns:
        dataset_train (torch.utils.data.Dataset): training dataset
        dataset_val (torch.utils.data.Dataset): validation dataset
        data_loader_train (torch.utils.data.DataLoader): training data loader
        data_loader_val (torch.utils.data.DataLoader): validation data loader
        mixup_fn: mixup function
    """
    config.defrost()

    # ================ build datasets ================
    dataset_train = None
    if not config.EVAL_MODE:
        dataset_train, _ = build_dataset(is_train=True, config=config)
        config.freeze()
        print(colored(f"Local rank: {config.LOCAL_RANK} / Global rank: {dist.get_rank()} successfully build train dataset", "red"))
        print(colored(f"Total number of images in training dataset: {len(dataset_train)}", "red"))
    dataset_val, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config)
    print(colored(f"Local rank: {config.LOCAL_RANK} / Global rank: {dist.get_rank()} successfully build val dataset", "red"))
    print(colored(f"Total number of images in val dataset: {len(dataset_val)}", "red"))
    print(colored(f"Number of classes: {config.MODEL.NUM_CLASSES}", "red"))
    print(colored(f"Datasets are successfully built", "red", attrs=["bold"]))

    # ================ build samplers ================
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if not config.EVAL_MODE:
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
            indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

        indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)

    # ================ build data loaders ================
    data_loader_train = None
    if not config.EVAL_MODE:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )
    print('Data loaders are successfully built')

    # ================ setup mixup / cutmix ================
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    """
    Build dataset for training and validation. 
    We only support ImageNet. 
    For zip mode, we only support ImageNet ILSVRC2012.

    args:
        is_train (bool): training or validation
        config (config): config
    returns:
        dataset (torch.utils.data.Dataset): dataset
        nb_classes (int): number of classes
    """
    transform = build_transform(is_train, config)
    
    # ================ imagenet ================
    print(colored('Building dataset: ' + config.DATA.DATASET, "red"))
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, 
                                        ann_file, 
                                        prefix, 
                                        transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part',
                                        )
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = CustomImageFolder(root, transform=transform)
        nb_classes = 1000
    # ================ imagenet100 ================
    elif config.DATA.DATASET == 'imagenet100':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            raise NotImplementedError("For zip mode, we only support ImageNet ILSVRC2012 dataset for now")
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = CustomImageFolder(root, transform=transform)
        nb_classes = 100
    # ================ cifar10 ================
    elif config.DATA.DATASET == 'cifar10':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.CIFAR10(root=config.DATA.DATA_PATH, train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR10(root=config.DATA.DATA_PATH, train=False, download=True, transform=transform)
        nb_classes = 10
    # ================ cifar100 ================
    elif config.DATA.DATASET == 'cifar100':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.CIFAR100(root=config.DATA.DATA_PATH, train=True, download=True, transform=transform)
        else:
            dataset = datasets.CIFAR100(root=config.DATA.DATA_PATH, train=False, download=True, transform=transform)
        nb_classes = 100
    # ================ stl10 ================
    elif config.DATA.DATASET == 'stl10':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.STL10(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.STL10(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 10
    # ================ oxford_flowers102 ================
    elif config.DATA.DATASET == 'oxford_flowers102':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.Flowers102(root=config.DATA.DATA_PATH, train=True, download=True, transform=transform)
        else:
            dataset = datasets.Flowers102(root=config.DATA.DATA_PATH, train=False, download=True, transform=transform)
        nb_classes = 102
    # ================ oxford pets ================
    elif config.DATA.DATASET == 'oxford_pets':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.OxfordIIITPet(root=config.DATA.DATA_PATH, image_set='train', download=True, transform=transform)
        else:
            dataset = datasets.OxfordIIITPet(root=config.DATA.DATA_PATH, image_set='test', download=True, transform=transform)
        nb_classes = 37
    # ================ food101 ================
    elif config.DATA.DATASET == 'food101':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.Food101(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.Food101(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 101
    elif config.DATA.DATASET == 'standford_cars':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.StanfordCars(root=config.DATA.DATA_PATH, train=True, download=True, transform=transform)
        else:
            dataset = datasets.StanfordCars(root=config.DATA.DATA_PATH, train=False, download=True, transform=transform)
        nb_classes = 196
    else:
        raise NotImplementedError("-----> Unknown dataset: {}".format(config.DATA.DATASET))
    return dataset, nb_classes


def build_transform(is_train, config):
    """
    Build data transform for training and validation same as BYOL: https://arxiv.org/abs/2006.07733
    or SimCLR: https://arxiv.org/abs/2003.04297

    args:
        is_train (bool): training or validation
        config (config): config
    returns:
        transform (torchvision.transforms.Compose): data transform
    """
    if config.AUG.SSL_AUG:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # ================ augmentation BYOL ================
        if config.AUG.SSL_AUG_TYPE == 'byol':
            transform_1 = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=1.0),
                transforms.ToTensor(),
                normalize,
            ])
            transform_2 = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(config.AUG.SSL_AUG_CROP, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur()], p=0.1),
                transforms.RandomApply([ImageOps.solarize], p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
            
            transform = (transform_1, transform_2)
            return transform
        
        # ================ augmentation SimCLR ================
        elif config.AUG.SSL_AUG_TYPE == 'simclr':
            transform_1 = transforms.Compose([
                transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur(sigma=[0.1, 2.0])], p=0.5),
                transforms.ToTensor(),
                normalize,
            ])
            
            transform = (transform_1, transform_1)
            return transform
        else:
            raise NotImplementedError("-----> Unknown SSL augmentation type: {}. We only support BYOL and SimCLR for now...".format(config.AUG.SSL_AUG_TYPE))
    
    # ================ augmentation lin eval ================
    if config.AUG.SSL_LINEAR_AUG:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
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
    
    # ================ augmentation default ================
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
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
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)))
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), interpolation=_pil_interp(config.DATA.INTERPOLATION)))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class GaussianBlur(object):
    """Gaussian blur"""
    def __call__(self, x):
        sigma = np.random.uniform(0.1, 2.0)
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x