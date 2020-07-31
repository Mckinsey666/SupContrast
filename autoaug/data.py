import torch
import torchvision
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
#from .augmentations import *
from .default_aug import *
from .archive import get_policies#arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet
import random

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def get_data_transform(dataset, name, opt):
    policies, probs = get_policies()
    if dataset == 'cifar10':
        if opt.use_resized_crop:
            print("Use resize crop")
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.0)), # pre transform
                transforms.RandomHorizontalFlip(), # pre transform
                transforms.ToTensor(), # after transform
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD), # after transform
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4), # pre transform
                transforms.RandomHorizontalFlip(), # pre transform
                transforms.ToTensor(), # after transform
                
                transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD), # after transform
            ])
        transform_train.transforms.insert(2, Augmentation(policies[name], probs[name] if name in probs else None))
    return transform_train

i = 0

class Augmentation(object):
    def __init__(self, policies, probs):
        self.policies = policies#[get_composed(pol) for pol in policies]
        print(self.policies)
        self.probs = probs
        if self.probs is None:
            self.probs = np.ones(len(self.policies)) / len(self.policies)
        else:
            self.probs = np.array(self.probs) / np.sum(self.probs)
        self.pid = np.arange(len(self.policies))
        self.i=0
        
    def __call__(self, img):
        #policy_id = np.random.choice(len(self.policies), p=self.probs) # weighted policies
        policy_id = random.choices(self.pid, self.probs, k=1)[0]
        policy = self.policies[policy_id]
        
        for name, pr, level in policy:
            if random.random() > pr:
                img = apply_augment(img, name, level)
        return img