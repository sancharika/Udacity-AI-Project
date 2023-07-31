import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
from torch.autograd import Variable

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from collections import OrderedDict

arch = {
    "densenet121": 1024,
    "vgg16": 25088
}

def load_data(root="./flowers"):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define common settings for transforms and dataloaders
    batch_size = 32
    shuffle = True
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std)
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normalization_mean, normalization_std)
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return trainloader, validloader, testloader, train_data
