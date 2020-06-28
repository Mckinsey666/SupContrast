from __future__ import print_function

import math
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
from PIL import Image

class CIFAR10_Transform:
    """
        Randomly augment CIFAR10.
        
        Original: fixed probability

        1. RandomResizedCrop(size=32, scale=(0.2, 1.)), p = 1
        2. RandomHorizontalFlip, p = 0.5
        3. ColorJitter(0.4, 0.4, 0.4, 0.1), p = 0.8
        4. RandomGrayscale, p = 0.2
        5. Random gaussian blur, p = 0.5

        Note that torchvision.transforms is performed on PIL Images, so we'll need to
        transform tensors back to PIL Images first.

    """

    def __init__(self, aug_classes, normalize):
        self.aug_classes = aug_classes
        self.normalize = normalize

        self.augment_func = {
            0: transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            1: transforms.RandomHorizontalFlip(),
            2: transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            3: transforms.Grayscale(num_output_channels = 3), # always 3 channels
            4: GaussianBlur(kernel_size = 3)
        }

        assert(self.aug_classes == len(self.augment_func))
    
    def get_composed(self, aug_vec):
        return transforms.Compose([self.augment_func[i] for i in range(self.aug_classes) if aug_vec[i] == 1])
    
    def augment(self, x, transform):
        x = F.to_pil_image(x)
        x = transform(x)
        x = F.to_tensor(x)
        x = self.normalize(x)
        return x.unsqueeze(0)

    def __call__(self, x, aug_vec):
        """
            Transfrom image based on n-hot augmentation vector.

            x: N * C * H * W
            aug_vec: N * aug_classes
        """

        augmented_x = []
        for i, x_i in enumerate(x):
            # C * H * W
            augmented_x.append(self.augment(x_i, self.get_composed(aug_vec[i])))
        
        augmented_x = torch.cat(augmented_x, 0)
        return augmented_x

class TensorTransform:
    def __init__(self, normalize):
        self.normalize = normalize
    
    def __call__(self, x):
        return [F.to_tensor(x), self.normalize(F.to_tensor(x))]

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


class GaussianBlur:
    """ Gaussian Blur defined in the SimCLR paper. """

    def __init__(self, kernel_size, min = 0.1, max = 2.0):
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.min = min
        self.max = max
    
    def __call__(self, x):
        x = np.array(x)
        sigma = (self.max - self.min) * np.random.random_sample() + self.min
        blurred = cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), sigma)
        return Image.fromarray(blurred)

class Denormalize:
    def __init__(self, mean, std):
        self.denorm = transforms.Normalize(
            mean=[-m*1.0/s for m, s in zip(mean, std)],
            std=[1.0/s for s in std]
        )
    def __call__(self, batch):
        res = torch.cat([self.denorm(batch[i]).unsqueeze(0) for i in range(len(batch))], dim = 0)
        return res

if __name__ == '__main__':
    x = Image.open('dog.jpg')
    kernel_size = 0.1 * min(x.size)
    blur = GaussianBlur(kernel_size = kernel_size)
    x = blur(x)
    x.save('blur.jpg')

