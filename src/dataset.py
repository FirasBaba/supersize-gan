import torch
import os
import random
from PIL import Image

from torchvision import transforms

import numpy as np
from scipy.ndimage import convolve
import config

import copy

# size = config.lr_size * 4

class CelebDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, size=None, transform=None, is_train=True, crop_proba=None):
        super().__init__()
        self.is_train = is_train
        self.size = size
        self.transform = transform
        self.data = image_list
        self.crop_proba = crop_proba

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_train:
            img = Image.open(os.path.join(config.train_data_path, self.data[idx]))
            if self.crop_proba and random.random() < self.crop_proba:
                width, height = img.size
                x = random.randint(0, width - self.size)
                y = random.randint(0, height - self.size)
                img = img.crop((x, y, x + self.size, y + self.size))
        else:
            img = Image.open(os.path.join(config.cryptopunk_path, self.data[idx])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.is_train:
            transform_hr = transforms.Compose([
                transforms.Resize((self.size * 4 , self.size*4)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5 ,0.5]),
            ])

            transform_lr = transforms.Compose([
                transforms.Resize((self.size, self.size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ])

            img_hr = transform_hr(img)
            img_lr = transform_lr(img)
            return img_lr, img_hr
        else:
            original = copy.deepcopy(img)
            original_size = img.size
            transform_inference = transforms.Compose([
                transforms.Resize((original_size[0], original_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
            ])
            # import pdb; pdb.set_trace()
            return transform_inference(original), transform_inference(img)
