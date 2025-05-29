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


class VOCClassification(torch.utils.data.Dataset):
    """VOC dataset adapted for single-label classification using CELoss"""
    def __init__(self, root, year='2007', image_set='train', transform=None):
        self.voc_dataset = datasets.VOCDetection(
            root=root, 
            year=year, 
            image_set=image_set, 
            download=True
        )
        self.transform = transform
        self.classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
            'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.voc_dataset)
    
    def get_dominant_class(self, target):
        """Get the class with the largest bounding box area"""
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        
        max_area = 0
        dominant_class = 0  # Default to first class if no valid objects found
        
        for obj in objects:
            bbox = obj['bndbox']
            width = int(bbox['xmax']) - int(bbox['xmin'])
            height = int(bbox['ymax']) - int(bbox['ymin'])
            area = width * height
            
            if area > max_area:
                max_area = area
                class_name = obj['name']
                if class_name in self.class_to_idx:
                    dominant_class = self.class_to_idx[class_name]
        
        return dominant_class
    
    def __getitem__(self, idx):
        image, target = self.voc_dataset[idx]
        
        # Get single dominant class label for CELoss
        label = self.get_dominant_class(target)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def build_loader(config):
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
    dataset_train = datasets.FakeData(size=130000, image_size=(3, 224, 224), num_classes=100, transform=build_transform(is_train=True, config=config))  # dummy dataset for training
    if not config.EVAL_MODE:
        dataset_train, _ = build_dataset(is_train=True, config=config)
    dataset_val, config.MODEL.NUM_CLASSES = build_dataset(is_train=False, config=config)
    config.freeze()

    # ================ build samplers ================
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
    sampler_val = SubsetRandomSampler(indices)

    # ================ build data loaders ================
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
    print(colored(f"==============> Building {'training' if is_train else 'validation'} dataset {config.DATA.DATASET} ....................", "red"))

    # ================ imagenet ================
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
    elif config.DATA.DATASET == 'stl10' or config.DATA.DATASET == 'stl':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.STL10(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.STL10(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 10
    # ================ oxford_flowers102 ================
    elif config.DATA.DATASET == 'oxford_flowers102' or config.DATA.DATASET == 'flowers':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.Flowers102(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.Flowers102(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 102
    # ================ oxford pets ================
    elif config.DATA.DATASET == 'oxford_pets' or config.DATA.DATASET == 'pets':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.OxfordIIITPet(root=config.DATA.DATA_PATH, split='trainval', download=True, transform=transform)
        else:
            dataset = datasets.OxfordIIITPet(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 37
    # ================ food101 ================
    elif config.DATA.DATASET == 'food101' or config.DATA.DATASET == 'food':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.Food101(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.Food101(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 101
    # ================ stanford cars ================
    elif config.DATA.DATASET == 'standford_cars' or config.DATA.DATASET == 'cars':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.StanfordCars(root=config.DATA.DATA_PATH, train=True, download=False, transform=transform)
        else:
            dataset = datasets.StanfordCars(root=config.DATA.DATA_PATH, train=False, download=False, transform=transform)
        nb_classes = 196
    # ================ caltech101 ================
    elif config.DATA.DATASET == 'caltech101' or config.DATA.DATASET == 'caltech':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.Caltech101(root=config.DATA.DATA_PATH, target_type='category', download=True, transform=transform)
        else:
            dataset = datasets.Caltech101(root=config.DATA.DATA_PATH, target_type='category', download=True, transform=transform)
        nb_classes = 101
    # ================ dtd ================
    elif config.DATA.DATASET == 'dtd':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.DTD(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.DTD(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 47
    # ================ fgvc aircraft ================
    elif config.DATA.DATASET == 'fgvc_aircraft' or config.DATA.DATASET == 'aircraft':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.FGVCAircraft(root=config.DATA.DATA_PATH, split='train', download=True, transform=transform)
        else:
            dataset = datasets.FGVCAircraft(root=config.DATA.DATA_PATH, split='test', download=True, transform=transform)
        nb_classes = 100
    # ================ sun397 ================
    elif config.DATA.DATASET == 'sun397':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.SUN397(root=config.DATA.DATA_PATH, download=True, transform=transform)
        else:
            dataset = datasets.SUN397(root=config.DATA.DATA_PATH, download=True, transform=transform)
        nb_classes = 397
    # ================ voc2007 classification ================
    elif config.DATA.DATASET == 'voc2007' or config.DATA.DATASET == 'voc':
        config.DATA.DATA_PATH = './data/'
        year = '2007'
        image_set = 'train' if is_train else 'val'
        dataset = VOCClassification(root=config.DATA.DATA_PATH, year=year, image_set=image_set, transform=transform)
        nb_classes = 20  # Single-label classification with 20 classes (compatible with CELoss)
    # ================ places365 ================
    elif config.DATA.DATASET == 'places365' or config.DATA.DATASET == 'places':
        config.DATA.DATA_PATH = './data/'
        if is_train:
            dataset = datasets.Places365(root=config.DATA.DATA_PATH, split='train-standard', small=True, download=True, transform=transform)
        else:
            dataset = datasets.Places365(root=config.DATA.DATA_PATH, split='val', small=True, download=True, transform=transform)
        nb_classes = 365
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
