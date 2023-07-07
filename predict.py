import numpy as np
import argparse
import matplotlib.pyplot as plt

from collections import OrderedDict
import json
from PIL import Image

import futility
import fmodel

import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable

parser = argparse.ArgumentParser(description = 'Parser for prediction.py')
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')


args = parser.parse_args()
path_image = args.input
device = args.gpu
number_of_outputs = args.top_k
path = args.checkpoint
json_name = args.category_names


def main():
    ''' Predict images from a PyTorch model
    '''

    model=fmodel.load_checkpoint(path)
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
        
    probabilities = fmodel.predict(path_image, model, number_of_outputs, device)
    probability = np.array(probabilities[0][0])
    labels = [name[str(index + 1)] for index in np.array(probabilities[1][0])]
    i = 0
    while i < number_of_outputs:
        print("{} with a probability of {}".format(labels[i], probability[i]))
        i += 1
    print("Finished Predicting!")

    
if __name__== "__main__":
    main()
