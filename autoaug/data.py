import torch
import torchvision
from PIL import Image
from torchvision.transforms import transforms
from .augmentations import *
from .archive import policies#arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10, fa_reduced_svhn, fa_resnet50_rimagenet

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def get_data_transform(dataset, name, opt):
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
        transform_train.transforms.insert(2, Augmentation(policies[name]))
    return transform_train


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img