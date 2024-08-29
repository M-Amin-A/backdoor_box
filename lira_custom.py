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

poison_class = core.LIRA(
    dataset_name='cifar10',
    train_dataset=trainset,
    test_dataset=testset,
    model=core.models.PreActResNet18(),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    eps=0.1,
    alpha=0.5,
    tune_test_eps=0.01,
    tune_test_alpha=0.5,
    best_threshold=0.01,
    schedule=None,
    seed=42
)

poisoned_train_dataset, poisoned_test_dataset = poison_class.get_poisoned_dataset()

# train attacked model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '0',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 1,

    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100],

    'tune_test_lr': 0.01,
    'tune_momentum': 0.9,
    'tune_weight_decay': 5e-4,
    # 'tune_test_epochs': 500,
    # 'schedulerC_milestones': '100,200,300,400',
    'tune_test_epochs': 50,
    'schedulerC_milestones': '10,20,30,40',
    'schedulerC_lambda': 0.1,

    'lr_atk': 0.003,

    'epochs': 50,
    'train_epoch': 1,

    'cls_test_epoch': 5,

    'log_iteration_interval': 300,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_CIFAR10'
}

poison_class.train(schedule)
