# code modified from https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/augmentations.py
# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms.transforms import Compose

from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2
from random import uniform




"""
SimCLR-style + Pytorch-like augmentations for PIL Images

"""

# probabilistic version

def RandomResizedCrop_p(img, m):
    assert 0 <= m <= 1
    h, w = img.size[0], img.size[1]
    t = transforms.RandomResizedCrop((h, w), scale=(1-m-1e-6, 1))
    return t(img)

def Brightness_p(img, m):
    assert 0 <= m <= 1
    s = np.random.uniform(1-m, 1+m) # 0 ~ 2
    return TF.adjust_brightness(img, s)

def Contrast_p(img, m):
    assert 0 <= m <= 1
    s = np.random.uniform(1-m, 1+m) # 0 ~ 2
    return TF.adjust_contrast(img, s) 
    
def Saturation_p(img, m):
    assert 0 <= m <= 1
    s = np.random.uniform(1-m, 1+m) # 0 ~ 2
    return TF.adjust_saturation(img, s) 

def Hue_p(img, m):
    assert 0 <= m <= 0.5
    s = np.random.uniform(-m, m) # -0.5 ~ 0.5
    return TF.adjust_hue(img, s)

def Gray_p(img, m):
    # s doesn't matter
    return TF.to_grayscale(img, 3) # num output channels = 3 for RGB

def GaussianBlur_p(img, m):
    assert 0 <= m <= 2 # sigma size
    img = np.asarray(img)
    kernel_size = int(0.1 * img.shape[0])
    kernel_size |= 1 # make odd size
    s = np.random.uniform(2-m-1e-6, 2) # m ~ 2
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), s)
    return blurred

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
    img = np.asarray(img)
    kernel_size = int(0.1 * img.shape[0])
    kernel_size |= 1 # make odd size
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), m)
    return blurred
    
# Modules

class BrightnessModule(object):
    def __init__(self, m, p):
        self.m = m
        self.p = p
    def __call__(self, img):
        p = uniform(0, 1)
        return TF.adjust_brightness(img, self.m) if p < self.p else img

class ContrastModule(object):
    def __init__(self, m, p):
        self.m = m
        self.p = p
    def __call__(self, img):
        p = uniform(0, 1)
        return TF.adjust_contrast(img, self.m) if p < self.p else img

class SaturationModule(object):
    def __init__(self, m, p):
        self.m = m
        self.p = p
    def __call__(self, img):
        p = uniform(0, 1)
        return TF.adjust_saturation(img, self.m) if p < self.p else img

class HueModule(object):
    def __init__(self, m, p):
        self.m = m
        self.p = p
    def __call__(self, img):
        p = uniform(0, 1)
        return TF.adjust_hue(img, self.m) if p < self.p else img

class GaussianBlurModule(object):
    def __init__(self, m, p):
        self.m = m
        self.p = p
    def __call__(self, img):
        p = uniform(0, 1)
        if p > self.p:
            return img
        img = np.asarray(img)
        kernel_size = int(0.1 * img.shape[0])
        kernel_size |= 1 # make odd size
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), self.m)
        return blurred

    


def augment_list():  # SimCLR operations
    eps = 1e-6
    l = [
        (RandomResizedCrop_p, eps, 1),
        (Brightness_p, eps, 1),
        (Contrast_p, eps, 1),
        (Saturation_p, eps, 1),
        (Hue_p, 0, 0.5),
        (Gray_p, 0, 1), # but magnitude doesn't matter, just dummy
        (GaussianBlur_p, eps, 2),
        (ResizedCrop, 0.1, 0.9),
        (Brightness, 0.1, 1.9),
        (Contrast, 0.1,1.9),
        (Saturation, 0.1,1.9),
        (Hue, -0.5, 0.5),
        (Gray, 0.1, 1.9),
        (GaussianBlur, 0.1, 1.9)
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

module_dict = {
    'Brightness': BrightnessModule,
    'Contrast': ContrastModule,
    'Saturation': SaturationModule,
    'Hue': HueModule,
    'GaussianBlur': GaussianBlurModule,
    'Gray': transforms.Grayscale(3) # 3 output channels
}


def get_augment(name):
    return augment_dict[name]

def get_module(policy):
    name, pr, level = policy
    if name == 'Gray':
        return transforms.RandomApply([transforms.Grayscale(3)], pr)
    else:
        _, low, high = get_augment(name)
        m = low + (high - low) * level
        return module_dict[name](m, pr)

def get_composed(policies):
    return transforms.Compose([get_module(pol) for pol in policies])


def apply_augment(img, name, level):
    # level is always 0 - 1
    augment_fn, low, high = get_augment(name)
    m = low + (high - low) * level # scale to augmentation levels
    return augment_fn(img.copy(), m)