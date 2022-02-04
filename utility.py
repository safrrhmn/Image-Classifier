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
from matplotlib import pyplot as plt


def load_data(data_dir):
    TRAIN = 'train'
    VALIDATE = 'valid'
    TEST = 'test'
    train_dir = os.path.join(data_dir, TRAIN)
    valid_dir = os.path.join(data_dir, VALIDATE)
    test_dir = os.path.join(data_dir, TEST)
    

    data_transforms = {
            TRAIN: torch_transforms.Compose([torch_transforms.Resize(256),
                                             torch_transforms.RandomResizedCrop(224),
                                             torch_transforms.RandomHorizontalFlip(p=0.5),
                                             torch_transforms.ToTensor(),
                                             torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])]),

            TEST: torch_transforms.Compose([torch_transforms.Resize(256),
                                            torch_transforms.RandomResizedCrop(224),
                                            torch_transforms.RandomHorizontalFlip(p=0.5),
                                            torch_transforms.ToTensor(),
                                            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])]),

            VALIDATE: torch_transforms.Compose([torch_transforms.Resize(256),
                                                torch_transforms.RandomResizedCrop(224),
                                                torch_transforms.RandomHorizontalFlip(p=0.5),
                                                torch_transforms.ToTensor(),
                                                torch_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                           std=[0.229, 0.224, 0.225])])
    }

    # Data loading..
    image_datasets = {
        dataset: torch_datasets.ImageFolder(
            root=os.path.join(data_dir, dataset),
            transform=data_transforms[dataset]
        )
        for dataset in [TRAIN, VALIDATE, TEST]
    }

    # TODO: Using the image datasets and the transforms, define the data_loaders
    data_loaders = {
        loader: DataLoader(image_datasets[loader], batch_size=64, shuffle=True)
        for loader in [TRAIN, VALIDATE, TEST]
    }

    '''
    for x in [TRAIN, VALIDATE, TEST]:
        print("Loaded {} images under {}".format(data_loaders[x], x))
    '''
    class_names = image_datasets[TRAIN].classes
    train_loader = data_loaders[TRAIN]
    valid_loader = data_loaders[VALIDATE]
    test_loader = data_loaders[TEST]
    class_to_idx=image_datasets[TRAIN].class_to_idx

    return class_names, train_loader, valid_loader, test_loader,class_to_idx

def map_lable(file='cat_to_name.json'):
    import json

    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    #print(cat_to_name)
    return cat_to_name

def save_checkpoint(data_dir='./flowers/',path = 'checkpoint.pth', structure = 'vgg16', hidden_units = 4096, dropout = 0.3, lr = 0.001, epochs = 1):
    model = eval(f'torch_models.{structure}()')
    class_names, train_loader, valid_loader, test_loader,class_to_idx=load_data(data_dir)
    model.class_to_idx = class_to_idx
    orderDict = nn.Sequential(
        nn.Sequential(
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )
    )
    model.classifier = nn.Sequential(orderDict)

    torch.save({'no_of_epochs': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}, path)
    #print(model.class_to_idx)
    
def load_checkpoint(filepath = 'checkpoint.pth', arch="vgg16"):
    check_point = torch.load(filepath)
    model = eval(f'torch_models.{arch}(pretrained=True)')

    orderDict = nn.Sequential(
        nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 102),
            nn.LogSoftmax(dim=1)
        )
    )

    model.classifier = orderDict
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.state_dict = check_point['state_dict']
    model.class_to_idx = check_point['class_to_idx']
    return model

def image_processor(image_loc):
    size = 224, 224
    with Image.open(image_loc) as im:
        im.rotate(45)  #.show()
        im.thumbnail(size)
        img_transforms = torch_transforms.Compose([
            torch_transforms.Resize(256),
            torch_transforms.CenterCrop(224),
            torch_transforms.ToTensor(),
            torch_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return img_transforms(im)

def imshow(image_loc,title=None):
    image_loc=image_processor(image_loc)
    image_loc = image_loc.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * image_loc + mean
    image_loc = np.clip(inp, 0, 1)
    plt.imshow(image_loc)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    return plt