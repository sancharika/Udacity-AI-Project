import numpy as np
import argparse
import matplotlib.pyplot as plt
import json
from PIL import Image

import futility
import fmodel

import torch
from torch import nn, optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for training.py')
    parser.add_argument('data_dir', action='store', default='./flowers/')
    parser.add_argument('--arch', action='store', default='vgg16')
    parser.add_argument('--save_dir', action='store', default='./checkpoint.pth')
    parser.add_argument('--hidden_units', action='store', type=int, default=512)
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001)
    parser.add_argument('--dropout', action='store', type=float, default=0.2)
    parser.add_argument('--epochs', action='store', default=3, type=int)
    parser.add_argument('--gpu', action='store_true', default=False)  # Use --gpu to enable GPU

    return parser.parse_args()


def setup_model(struct, dropout, hidden_units, lr, power):
    model, criterion = fmodel.setup_network(struct, dropout, hidden_units, lr, power)
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    return model, criterion, optimizer


def train(model, criterion, optimizer, trainloader, validloader, epochs, device, print_every=5):
    model.to(device)
    steps = 0
    running_loss = 0

    print("--Training starting--")
    for epoch in range(epochs):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss = 0
                    accuracy = 0
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        batch_loss = criterion(log_ps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()

    print("Training completed!")


def save_checkpoint(model, struct, hidden_units, dropout, lr, epochs, class_to_idx, path):
    model.class_to_idx = class_to_idx
    checkpoint = {
        'structure': struct,
        'hidden_units': hidden_units,
        'dropout': dropout,
        'learning_rate': lr,
        'no_of_epochs': epochs,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    torch.save(checkpoint, path)
    print("Saved checkpoint!")


def main():
    args = parse_arguments()

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    trainloader, validloader, testloader, train_data = futility.load_data(args.data_dir)
    model, criterion, optimizer = setup_model(args.arch, args.dropout, args.hidden_units, args.learning_rate, args.gpu)

    train(model, criterion, optimizer, trainloader, validloader, args.epochs, device)

    save_checkpoint(model, args.arch, args.hidden_units, args.dropout, args.learning_rate, args.epochs,
                    train_data.class_to_idx, args.save_dir)


if __name__ == "__main__":
    main()
