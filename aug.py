import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distributions
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from torch.utils.data import DataLoader

import cv2
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
    For each augmentation, there will be two differential variables:
    1. Probability (p). 
    2. Magnitude (m)

    The objective is -L_nce (maximize InfoNCE for augmentation policies)

    We need to compute (partial)
    1. dL/dp
    2. dL/dm

    We'll update on the unclamped probability p. p needs to go through 
    sigmoid before sampling.
"""

def augment_batch(aug_vec, augmentations, batch, normalize = None):
    h = Variable(torch.tensor(1.0, device = device), requires_grad = True)
    transforms_list = []
    for i in range(len(aug_vec)):
        if aug_vec[i] == 1:
            h = h * aug_vec[i]
            transforms_list.append(augmentations[i])
        else:
            h = h * (1 - aug_vec[i])
    
    transforms_list.append(transforms.ToTensor())
    if normalize:
        transforms_list.append(normalize)
    transform = transforms.Compose(transforms_list)

    transformed_batch = []
    for x in batch:
        x = TF.to_pil_image(x.cpu())
        x = transform(x).unsqueeze(0)
        transformed_batch.append(x)
    transformed_batch = torch.cat(transformed_batch, dim = 0).to(device)
    return h * transformed_batch
        
def relaxed_bernoulli(p, temperature = 1, eps = 1e-20):
    """
    Args:
        p: [N] tensor indicating bernoulli probability
    Returns:
        c: [N] with 0, 1 values indicating bernoulli sample (hard)
    """
    p = torch.clamp(p, 0, 1)
    #p = torch.sigmoid(p) # raw to (0, 1) prob
    u = torch.empty(p.shape).uniform_(0, 1).to(device)
    q = torch.log(p/(1 - p + eps) + eps) + torch.log(u / (1 - u + eps) + eps)
    y_soft = torch.sigmoid(q / temperature)
    y_hard = torch.where(y_soft > 0.5, torch.ones(y_soft.shape).to(device), torch.zeros(y_soft.shape).to(device))
    return y_hard - y_soft.detach() + y_soft # forward pass hard label, backward soft grad

class GaussianBlurFix:
    def __init__(self, kernel_size, sigma):
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        self.sigma = sigma
    
    def __call__(self, x):
        x = np.array(x)
        blurred = cv2.GaussianBlur(x, (self.kernel_size, self.kernel_size), self.sigma)
        return Image.fromarray(blurred)

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

class CIFAR10_Transform:
    def __init__(self, blur = False):

        self.aug_num = 4
        if blur:
            self.aug_num += 1

        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.augmentations = {
            0: transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            1: transforms.RandomHorizontalFlip(),
            2: transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            3: transforms.Grayscale(num_output_channels = 3), # always 3 channels
        }
        if blur:
            self.augmentations[4] = GaussianBlur(kernel_size = 3)
    
    def __call__(self, batch, aug_vec):
        assert(len(aug_vec) == self.aug_num)
        return augment_batch(aug_vec, self.augmentations, batch, self.normalize)

if __name__ == "__main__":
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)

    augmentations = {
        0: transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        1: transforms.RandomHorizontalFlip(),
        2: transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        3: transforms.Grayscale(num_output_channels = 3), # always 3 channels
        4: GaussianBlur(kernel_size = 3)
    }

    aug_num = len(augmentations)

    p_raw = Variable(torch.zeros(aug_num), requires_grad = True) # raw probabilities, need to update this
    p = torch.sigmoid(p_raw) # clamp to real probability
    optimizer = optim.SGD([p_raw], lr = 0.1)

    s = relaxed_bernoulli(p, temperature = 0.8)
    print(s)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CIFAR10('../datasets', train = True, transform = transform)
    dataloader = DataLoader(dataset, batch_size = 64)
    batch, _ = next(iter(dataloader))
    augmented_batch = augment_batch(s, augmentations, batch)

    criterion = torch.nn.MSELoss()
    loss = criterion(batch, augmented_batch)
    print(torch.sigmoid(p_raw))
    loss.backward()
    optimizer.step()
    print(torch.sigmoid(p_raw))