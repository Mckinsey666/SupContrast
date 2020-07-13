import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np
import torchvision.transforms.functional as F


class TruncatedCIFAR10:
    def __init__(self, root, transform = None, fraction = 0.1):
        self.root = root
        self.transform = transform
        self.dataset = np.load(root)
        
        self.fraction = fraction
        self.multiple = int(1.0/fraction)
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
        return len(self.data) * self.multiple

    def __getitem__(self, idx):
        idx //= int(self.multiple)
        x, y = self.data[idx]
        x = F.to_pil_image(x)
        if self.transform:
            x = self.transform(x)
        return x, y

if __name__ == '__main__':
    dataset = TruncatedCIFAR10('datasets/cifar10.npy', transform = transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=64, shuffle = True)
    x, _ = next(iter(dataloader))
    save_image(x, 'orig.png')