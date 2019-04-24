import distiller 
import numpy as np
import os, collections
import bitstring 
import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import models 
from matplotlib import pyplot as plt
import multiprocessing 
#i%matplotlib inline

print('using GPU:', torch.cuda.is_available())

test_batch_size = 256
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                   ])),
    batch_size=test_batch_size, shuffle=False, **kwargs)


def test():
    model.eval()
    model.cuda()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').data # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))


# load model and its parameters 
model_path = "/home/hguan2/workspace/fault-tolerance/rethinking-network-pruning/cifar/l1-norm-pruning"+ \
    "/logs/vgg16/cifar10/model_best.pth.tar"
arch = 'vgg'
depth = '16'
dataset = 'cifar10'

checkpoint = torch.load(model_path)
model = models.__dict__[arch](dataset=dataset, depth=depth, cfg=checkpoint['cfg'])
model.load_state_dict(checkpoint['state_dict'])
best_prec1 = checkpoint['best_prec1']
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(model_path, checkpoint['epoch'], best_prec1))


# post train quantization 
quantizer = distiller.quantization.PostTrainLinearQuantizer(model)
quantizer.prepare_model()

# test accuracy after quantization
prec1 = test()

# copy bits per layer 
 
