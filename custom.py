import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torchvision.transforms as transforms

import core

transform_train = Compose([
    ToTensor(),
    transforms.RandomRotation(10),
    RandomHorizontalFlip()
])
transform_test = Compose([
    ToTensor()
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
pattern[0, -3:, -3:] = 255
weight = torch.zeros((1, 32, 32), dtype=torch.float32)
weight[0, -3:, -3:] = 1.0

badnets = core.BadNets(
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.PreActResNet18(),
    # model=core.models.BaselineMNISTNetwork(),
    loss=nn.CrossEntropyLoss(),
    y_target=1,
    poisoned_rate=0.05,
    pattern=pattern,
    weight=weight,
    # poisoned_transform_index=0,
    poisoned_target_transform_index=0,
    schedule=None,
    seed=666
)

poisoned_train_dataset, poisoned_test_dataset = badnets.get_poisoned_dataset()

# train attacked model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 16,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 50,

    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_CIFAR10'
}

badnets.train(schedule)