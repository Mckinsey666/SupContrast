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
#import cv2




"""
SimCLR-style + Pytorch-like augmentations for PIL Images

"""

# probabilistic version

def RandomResizedCrop(img, m):
    assert 0.05 <= m <= 0.95
    h, w = img.size[0], img.size[1]
    t = transforms.RandomResizedCrop((h, w), scale=(1-m, 1))
    return t(img)

def Brightness_p(img, m):
    assert 0.1 <= m <= 1.9
    low = max(0, 1-m)
    high = 1+m
    s = np.random.uniform(low, high)
    return TF.adjust_brightness(img, s)

def Contrast_p(img, m):
    assert 0.1 <= m <= 1.9
    low = max(0, 1-m)
    high = 1+m
    s = np.random.uniform(low, high)
    return TF.adjust_contrast(img, s) 
    
def Saturation_p(img, m):
    assert 0.1 <= m <= 1.9
    low = max(0, 1-m)
    high = 1+m
    s = np.random.uniform(low, high)
    return TF.adjust_saturation(img, s) 

def Hue_p(img, m):
    assert 0.1 <= m <= 0.5
    s = np.random.uniform(-m, m) # -0.5 ~ 0.5
    return TF.adjust_hue(img, s)

def Gray_p(img, m):
    # s doesn't matter
    return TF.to_grayscale(img, 3) # num output channels = 3 for RGB

def GaussianBlur_p(img, m):
    assert 0.1 <= m <= 1.9 # range magnitude
    s = np.random.uniform(2-m, 2) # sigma
    r = int(np.ceil(2 * s)) | 1 # radius
    return img.filter(PIL.ImageFilter.GaussianBlur(r))

# Deterministic version
def ResizedCrop(img, m):
    assert 0.1 <= m <= 0.9
    h, w = img.size[0], img.size[1]
    new_h = int(h * (1-m))
    new_w = int(w * (1-m))
    t = transforms.RandomCrop(size=(new_h, new_w))
    return t(img)

def Brightness(img, m):
    assert 0.1<=m<=1.9
    return TF.adjust_brightness(img, m)

def Contrast(img, m):
    assert 0.1<=m<=1.9
    return TF.adjust_contrast(img, m)

def Saturation(img, m):
    assert 0.1<=m<=1.9
    return TF.adjust_saturation(img, m)

def Hue(img, m):
    assert -0.5<=m<=0.5
    return TF.adjust_hue(img, m)

def Gray(img, m):
    return TF.to_grayscale(img, 3)

def GaussianBlur(img, m):
    assert 0.1 <= m <= 1.9 # sigma size
    r = int(np.ceil(2 * m)) | 1 # radius
    # faster
    return img.filter(PIL.ImageFilter.GaussianBlur(r))
    
# SimCLR style 

def ColorJitter(img, m):
    assert 0.1 <= m <= 2.4
    jitter_scale = [0.8 * m, 0.8 * m, 0.8 * m, 0.2 * m]
    t = transforms.ColorJitter(*jitter_scale)
    return t(img)

def augment_list():  # SimCLR operations
    eps = 1e-6
    l = [
        (RandomResizedCrop, 0.05, 0.95),
        (Brightness_p, 0.1, 1.9),
        (Contrast_p, 0.1, 1.9),
        (Saturation_p, 0.1, 1.9),
        (Hue_p, 0.1, 0.5),
        (Gray_p, 0, 1), # but magnitude doesn't matter, just dummy
        (GaussianBlur_p, 0.1, 1.9), # blur radius
        (ResizedCrop, 0.1, 0.9),
        (Brightness, 0.1, 1.9),
        (Contrast, 0.1,1.9),
        (Saturation, 0.1,1.9),
        (Hue, -0.5, 0.5),
        (Gray, 0.1, 1.9),
        (GaussianBlur, 0.1, 1.9),
        (ColorJitter, 0.1, 2.4)
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



