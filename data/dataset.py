import os
import numpy as np
import torchvision

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
#CLIP official transform
BICUBIC = InterpolationMode.BICUBIC
my_preprocess = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_dataset(data_name, train=True, path="./data"):
    if data_name == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(path, "CIFAR10"), train=train, 
                                                download=True, transform=my_preprocess)
    
    elif data_name == "cifar100coarse":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train,
                                                 download=True, transform=my_preprocess)
        trainset.targets = sparse2coarse(trainset.targets) 
    
    elif data_name == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=os.path.join(path, "CIFAR100"), train=train,
                                                 download=True, transform=my_preprocess)
    
    elif data_name == "imagenet":
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(path, "ImageNet/train"), transform=my_preprocess)


    return trainset


def sparse2coarse(targets):
    """CIFAR100 Coarse Labels. """
    coarse_targets = [ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  3, 14,  9, 18,  7, 11,  3,
                       9,  7, 11,  6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  0, 11,  1, 10,
                      12, 14, 16,  9, 11,  5,  5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 16,
                       4, 17,  4,  2,  0, 17,  4, 18, 17, 10,  3,  2, 12, 12, 16, 12,  1,
                       9, 19,  2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 16, 19,  2,  4,  6,
                      19,  5,  5,  8, 19, 18,  1,  2, 15,  6,  0, 17,  8, 14, 13]
    return np.array(coarse_targets)[targets]