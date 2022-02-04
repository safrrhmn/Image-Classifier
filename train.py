import os
from collections import OrderedDict

import matplotlib.pyplot as plot
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets as torch_datasets
from torchvision import models as torch_models
from torchvision import transforms as torch_transforms
import torchvision
from torchvision import datasets, models, transforms
import argparse
import utility
from datetime import datetime

argumentParser = argparse.ArgumentParser(description = 'argumentParser for train.py')
argumentParser.add_argument('--data_dir', action="store", default="./flowers/")
argumentParser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
argumentParser.add_argument('--epochs', action="store", default=5, type=int)
argumentParser.add_argument('--arch', action="store",dest='arch', default="vgg16",type=str)
argumentParser.add_argument('--learning_rate', action="store",dest='learning_rate', default=0.01)
argumentParser.add_argument('--hidden_units', action="store",dest='hidden_units', default=512)
argumentParser.add_argument('-gpu', default=True,dest='gpu')

args = argumentParser.parse_args()
data_dir = args.data_dir
checkpoint_dir = args.save_dir
epochs = args.epochs
arch = args.arch
learning_rate=args.learning_rate
hidden_units=args.hidden_units
gpu=args.gpu
device = torch.device("cuda:0" if (torch.cuda.is_available() & gpu) else "cpu")
#print(device)
def main():
    class_names, train_loader, valid_loader, test_loader = utility.load_data(data_dir)
    model = eval(f'torch_models.{arch}(pretrained=True)')
    model.to(device)
    #print(model)
    for param in model.parameters():
        param.requires_grad = False

    orderDict = nn.Sequential(
        nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    )

    model.classifier = orderDict
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    epochs = 5
    batch_loss = 0

    start_time = datetime.now()

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            # shortcut for forward pass
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)
            log_ps = model(images.float())
            loss = criterion(log_ps, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")
            end_time = datetime.now()
            print(f"total time taken: {end_time - start_time} for iteration: {epoch}")
    
    utility.save_checkpoint(path = checkpoint_dir, structure = arch, hidden_units = hidden_units, lr = learning_rate, epochs = epochs)
if __name__ == "__main__":
    main()