import numpy as np
import argparse
import json
from PIL import Image

import futility
import fmodel

import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for prediction.py')
    parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
    parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type=str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type=str)
    parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')

    return parser.parse_args()


def load_json(json_name):
    with open(json_name, 'r') as json_file:
        name = json.load(json_file)
    return name


def main():
    args = parse_arguments()
    device = torch.device("cuda:0" if args.gpu == "gpu" and torch.cuda.is_available() else "cpu")

    model = fmodel.load_checkpoint(args.checkpoint)
    class_to_idx = model.class_to_idx
    idx_to_name = load_json(args.category_names)

    probabilities = fmodel.predict(args.input, model, args.top_k, device)

    probability = np.array(probabilities[0][0])
    labels = [idx_to_name[str(class_idx)] for class_idx in np.array(probabilities[1][0])]

    print("Top {} predictions:".format(args.top_k))
    for label, prob in zip(labels, probability):
        print("{} with a probability of {:.2f}".format(label, prob))

    print("Finished Predicting!")


if __name__ == "__main__":
    main()
