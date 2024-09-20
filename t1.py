'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''


import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST
import core

# if global_seed = 666, the network will crash during training on MNIST. Here, we set global_seed = 555.
global_seed = 555
deterministic = True
torch.manual_seed(global_seed)

def gen_grid(height, k):
    """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
    according to the input height ``height`` and the uniform grid size ``k``.
    """
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
    noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
    array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
    x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
    identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

    return identity_grid, noise_grid


########################CIFAR10#######################
# Define Benign Training and Testing Dataset
dataset = torchvision.datasets.CIFAR10



transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip()
])
trainset = dataset('../datasets', train=True, transform=transform_train, download=True)

transform_test = Compose([
    ToTensor()
])
testset = dataset('../datasets', train=False, transform=transform_test, download=True)


# Show an Example of Benign Training Samples
index = 44

identity_grid,noise_grid=gen_grid(32,4)
torch.save(identity_grid, 'ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
torch.save(noise_grid, 'ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
wanet = core.WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.ResNet(18),
    # model=core.models.PreActResNet18(),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=False,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()


# Train Infected Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '1',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 4,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 300,
    'test_epoch_interval': 1,
    'save_epoch_interval': 1,

    'save_dir': 'experiments',
    'experiment_name': 'ResNet-18_CIFAR-10_WaNet'
}

wanet.train(schedule)

