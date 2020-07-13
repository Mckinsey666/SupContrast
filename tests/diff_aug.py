import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np

def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x

class TruncatedCIFAR10:
    def __init__(self, root, transform = None, fraction = 0.1):
        self.root = root
        self.transform = transform
        self.dataset = np.load(root)
        
        self.fraction = fraction
        self.orig_num_per_class = self.dataset.shape[1]
        self.truncate_num_per_class = int(self.orig_num_per_class * fraction)
        
        
        np.random.seed(12345)
        self.indices = np.random.choice(self.orig_num_per_class, self.truncate_num_per_class, replace = False)
        self.data = []
        for c in range(10):
            for idx in self.indices:
                self.data.append((torch.tensor(self.dataset[c][idx]), c))
        np.random.shuffle(self.data)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
        
    
        
if __name__ == '__main__':
    
    dataset = TruncatedCIFAR10('cifar10.npy')
    dataloader = DataLoader(dataset, batch_size=64, shuffle = True)
    x, _ = next(iter(dataloader))
    save_image(x, 'orig.png')
    
    #x = rand_brightness(x)
    #x = rand_saturation(x)
    #x = rand_contrast(x)
    
    #save_image(x, 'aug.png')
    
    
    
    