import argparse
import sys
import numpy as np

import torch
import click

from src.data.mnist import mnist
from model import MyAwesomeModel
from tqdm import tqdm

from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("images_filepath")
def evaluate(model_checkpoint,images_filepath):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    
    Xtest = np.load(images_filepath)["images"]
    Ytest = np.load(images_filepath)["labels"]

    # normalize images
    for image in Xtest:
        image -= np.mean(image)
        image /= np.std(image)

    testset = torch.utils.data.TensorDataset(torch.from_numpy(Xtest).float(),torch.from_numpy(Ytest).long())
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for images,labels in testloader:
            ps = torch.exp(model(images))

            _,top_class = ps.topk(1,dim=1)

            equals = top_class == labels.view(*top_class.shape)
            
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    
    accuracy /= len(testloader)

    print(f'Accuracy: {accuracy.item()*100}%')
    model.train()

cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    