# code modified from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw, PIL.ImageFilter

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms.transforms import Compose

from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2




"""
SimCLR-style + Pytorch-like augmentations for PIL Images

"""

# probabilistic version

def RandomResizedCrop(img, m):
    assert 0.05 <= m <= 0.95
    h, w = img.size[0], img.size[1]
    t = transforms.RandomResizedCrop((h, w), scale=(1-m, 1))
    return t(img)

def Gray(img, m):
    return TF.to_grayscale(img, 3)

def GaussianBlur(img, m):
    assert 0.1 <= m <= 1.9 # sigma size
    r = int(np.ceil(2 * m)) | 1 # radius
    # faster
    return img.filter(PIL.ImageFilter.GaussianBlur(r))

def ColorJitter(img, m):
    assert 0.1 <= m <= 2.4
    jitter_scale = [0.8 * m, 0.8 * m, 0.8 * m, 0.2 * m]
    t = transforms.ColorJitter(*jitter_scale)
    return t(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)


def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Solarize(img, v):  # [0, 256]
    assert 0 <= v <= 256
    v = np.random.uniform(256 - v, 256)
    v = int(v)
    return PIL.ImageOps.solarize(img, v)


def Posterize(img, v):  # [1, 8]
    assert 1 <= v <= 8
    v = np.random.uniform(8 - v, v)
    v = int(v)
    return PIL.ImageOps.posterize(img, v)



def augment_list():  # SimCLR operations
    eps = 1e-6
    l = [
        (RandomResizedCrop, 0.05, 0.95),
        (Gray, 0.1, 1.9),
        (GaussianBlur, 0.1, 1.9),
        (ColorJitter, 0.1, 2.4),
        # Added
        (Invert, 0, 1),
        (Equalize, 0, 1),
        (Solarize, 0, 256),
        (Posterize, 1, 8)
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}


def get_augment(name):
    return augment_dict[name]


def apply_augment(img, name, level):
    # level is always 0 - 1
    augment_fn, low, high = get_augment(name)
    m = low + (high - low) * level # scale to augmentation levels
    return augment_fn(img.copy(), m)