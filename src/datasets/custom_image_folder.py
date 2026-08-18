# Copyright (C) 2026.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from torchvision import datasets


class CustomImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        """
        args:
            index (int): index of the sample
        returns:
            tuple: (image, target) where target is the class index of the target class
        """
        path, target = self.samples[index]
        image = self.loader(path)
        
        ret = []
        
        if self.transform is not None:
            if isinstance(self.transform, (list, tuple)):  # if transform is a list or tuple
                for t in self.transform:
                    ret.append(t(image))
            else:  # if transform is a single callable
                ret.append(self.transform(image))
        else:
            ret.append(image)
        
        # apply the target transform if provided
        if self.target_transform is not None:
            target = self.target_transform(target)
        ret.append(target)

        return ret