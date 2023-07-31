import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image

import futility
from collections import OrderedDict

import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

def setup_network(structure='vgg16', dropout=0.1, hidden_units=4096, lr=0.001, device='gpu'):
    ''' Set up architecture based on input, device and create model,
        returns model and loss
    '''

    device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else 'cpu')

    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)

    for para in model.parameters():
        para.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)

    return model, criterion


def save_checkpoint(train_data, model, path='checkpoint.pth', structure='vgg16', hidden_units=4096, dropout=0.3, lr=0.001, epochs=1):
    ''' save checkpoint of the PyTorch model
    '''

    model.class_to_idx = train_data.class_to_idx
    torch.save({
        'structure': structure,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': lr,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }, path)


def load_checkpoint(path='checkpoint.pth'):
    ''' Create and load checkpoint for a PyTorch model,
        returns the PyTorch model
    '''

    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']
    model, _ = setup_network(structure, dropout, hidden_units, lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def predict(image_path, model, topk=5, device='gpu'):
    ''' Evaluate the a PyTorch model
    '''

    device = torch.device("cuda" if torch.cuda.is_available() and device == 'gpu' else 'cpu')
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = process_image(image_path).unsqueeze(0).to(device)
        output = model.forward(img)
        probs = torch.exp(output).cpu().data
        top_probs, top_idx = probs.topk(topk)
        top_probs = top_probs.numpy().squeeze()
        top_idx = top_idx.numpy().squeeze()
        class_to_idx_inv = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [class_to_idx_inv[idx] for idx in top_idx]

    return top_probs, top_classes


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''

    img_pil = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = img_transforms(img_pil)
    return image

# Usage example:
if __name__ == "__main__":
    model, criterion = setup_network(structure='vgg16', dropout=0.3, hidden_units=512, lr=0.001, device='gpu')
    print(model)
