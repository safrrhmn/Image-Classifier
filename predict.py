import numpy as np
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
import matplotlib.pyplot as plt
import json
from PIL import Image
import utility

parser = argparse.ArgumentParser(description = 'Parser for predict.py')

parser.add_argument('input', default='./flowers/test/1/image_06752.jpg', nargs='?', action="store", type = str)
parser.add_argument('--dir', action="store",dest="data_dir", default="./flowers/")
parser.add_argument('checkpoint', default='./checkpoint.pth', nargs='?', action="store", type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

args = parser.parse_args()
image_path = args.input
top_k = args.top_k
device = args.gpu
checkpoint = args.checkpoint

def main():
    model=utility.load_checkpoint(checkpoint)
    name=utility.map_lable()
        
    model=model.to('cpu')
    model=model.eval()
    with torch.no_grad():
        output = model(utility.image_processor(image_path).unsqueeze_(0))

    ps = torch.exp(output).data

    #According to Validation Solution of the Class
    top_p, top_class = ps.topk(top_k, dim=1)

    top_p = top_p.squeeze().detach().cpu().numpy()
    top_class = top_class.detach().squeeze().cpu().numpy()

    #print(f'cat_to_name: {cat_to_name}')
    #print(f'top_p: {top_p}')
    #print(f'top_class: {top_class}')

    class_to_idx = {key: val for key, val in
                    model.class_to_idx.items()}

    #print(f'class_to_idx: {class_to_idx}')
    cat_to_name=utility.map_lable()
    flower_names=[]
    for index in top_class:
        for k,v in model.class_to_idx.items():
            if index == v:
                flower_names.append(cat_to_name[str(index)])

    #print(flower_names)
    #top_p, top_class,flower_names
    #axs = utility.imshow(image_path,cat_to_name[str(top_class[0])])
    for i in range(top_k):
        print("{} with a probability of {}".format(flower_names[i], np.array(ps[0][0])))        
    
    
if __name__== "__main__":
    main()