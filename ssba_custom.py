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

encoder_schedule = {
    'secret_size': 20,
    'enc_height': 32,
    'enc_width': 32,
    'enc_in_channel': 3,
    'enc_total_epoch': 30,
    'enc_use_dis': False,
    'enc_secret_only_epoch': 30

}

poison_class = core.ISSBA(
    dataset_name='cifar10',
    train_dataset=trainset,
    test_dataset=testset,
    train_steg_set=trainset,
    model=core.models.PreActResNet18(),
    loss=nn.CrossEntropyLoss(),
    poisoned_rate=0.1,
    encoder_schedule=encoder_schedule,
    y_target=0,
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
    # 0.001
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100],
    # 'schedule': [15, 20],

    'epochs': 50,
    # 'epochs': 30,

    'log_iteration_interval': 300,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': 'experiments',
    'experiment_name': 'train_poisoned_CIFAR10'
}

poison_class.train(schedule)
