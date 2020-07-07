import torch 
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from util import TwoCropTransform

from PIL import Image
import numpy as np

def test_rotate():
    
    transform = transforms.ToTensor()
    img = Image.open('tests/house.jpg')
    t = transform(img)
    t = torch.rot90(t, k = 2, dims = [1, 2])
    print(t.shape)

    save_image(t, 'tests/rot.jpg')
    
class CIFAR10_PretextRotate:
    """ CIFAR10 dataset with rotation pretext labels """

    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.dataset = CIFAR10(root=self.root, transform=self.transform, download=True)
    
    def __len__(self):
        return self.dataset.__len__()
    
    def __getitem__(self, idx):
        """ Also return rotation label for pretext learning. """
        
        images, labels = self.dataset.__getitem__(idx)
        rot = np.random.randint(4)
        if type(images) == list:
            for i in range(len(images)):
                images[i] = torch.rot90(images[i], k = rot, dims = [1, 2])
        else:
            images = torch.rot90(images, k = rot, dims = [1, 2])
        return images, labels, rot
    
if __name__ == '__main__':
    
    transform = TwoCropTransform(
        transforms.Compose([
            transforms.ToTensor()
        ])
    )
    dataset = CIFAR10_PretextRotate(root = './datasets', transform = transform)
    dataloader = DataLoader(dataset, 5)
    batch, labels, rot = next(iter(dataloader))
    print(labels)
    print(rot)
    batch = torch.cat(batch, dim = 0)
    save_image(batch, 'tests/rot.jpg')