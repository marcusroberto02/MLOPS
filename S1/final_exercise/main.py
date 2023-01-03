import argparse
import sys

import torch
import click

from data import mnist
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
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs",default = 5)
@click.option("--model_checkpoint",default = "base")
def train(lr,epochs,model_checkpoint):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    trainset, _ = mnist()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.003)

    epochs = epochs

    train_losses = []
    for e in tqdm(range(epochs)):
        running_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        else:
            train_losses.append(running_loss / len(trainloader))
    
    plt.plot(train_losses,label="Training loss")
    plt.legend(frameon=True)
    plt.show()
    torch.save(model,model_checkpoint)



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, testset = mnist()
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


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    