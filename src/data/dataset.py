#!/usr/bin/env python

# Copyright (C) 2025. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):
    """
    ImageFolder that supports multiple transformations.
    
    Useful for self-supervised learning where we need to apply
    different augmentations to the same image (e.g., BYOL's two views).
    
    Args:
        root: Root directory path
        transform: Single transform or tuple/list of transforms
        target_transform: Optional transform for labels
        
    Returns:
        If transform is tuple/list: [view1, view2, ..., target]
        If transform is single: [image, target]
        
    Example:
        >>> # BYOL-style two views
        >>> transform = (transform_1, transform_2)
        >>> dataset = CustomImageFolder(root, transform=transform)
        >>> view1, view2, target = dataset[0]
    """
    
    def __getitem__(self, index):
        """
        Get item with support for multiple transformations.
        
        Args:
            index: Index of the item
            
        Returns:
            List containing transformed views and target label
        """
        path, target = self.samples[index]
        image = self.loader(path)
        
        ret = []
        
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):
                # Apply multiple transforms (for SSL with multiple views)
                for t in self.transform:
                    ret.append(t(image))
            else:
                # Apply single transform
                ret.append(self.transform(image))
        else:
            ret.append(image)
        
        # Apply target transformation if provided
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        ret.append(target)
        
        return ret