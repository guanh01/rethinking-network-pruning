from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time 

import models


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar100)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')


parser.add_argument('--warmstart', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint that are not regulated trained(default: none)')
parser.add_argument('--l2', type=float, default=1, metavar='M',
                    help='the hyperparameter (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('using GPU:', args.cuda)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# create logs folder 
args.save = os.path.join(args.save, args.arch+str(args.depth), args.dataset, 
                         'regulated_training', 
                         'l2_'+str(args.l2)+'_lr_'+str(args.lr))
if not os.path.exists(args.save):
    os.makedirs(args.save)
print('train logs will save to:', args.save)

print(args)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
else:
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(4),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# priority is to resume regulated training 
def load_checkpoint():
    print("=> loading checkpoint to continue training'{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
          .format(args.resume, checkpoint['epoch'], best_prec1))

def warmstart():
    if args.warmstart:
        if os.path.isfile(args.warmstart):
            print("=> loading checkpoint to warmstart'{}'".format(args.warmstart))
            checkpoint = torch.load(args.warmstart)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.resume, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(args.warmstart))
    
if args.resume:
    if os.path.isfile(args.resume):
        load_checkpoint()   
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        warmstart()
else:
    warmstart() 
                
            
        
# def train(epoch):
#     model.train()
#     avg_loss = 0.
#     train_acc = 0.

#     for batch_idx, (data, target) in enumerate(train_loader):
#         if args.cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = F.cross_entropy(output, target)
#         avg_loss += loss.item()
#         pred = output.data.max(1, keepdim=True)[1]
#         train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.log_interval == 0:
#             print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.data))
#####################
# regulated training
#####################
def check_large_weights_count():
    count = 0 # number of large weights 
    total = 0 # total number of weights 
    for name, param in model.named_parameters():
        if len(param.data.size()) < 2:
            continue 
        tensor1d = param.data.view(-1)
        N = len(tensor1d)
        thr = max(torch.max(tensor1d).item(), torch.abs(torch.min(tensor1d)).item())/2
        total += N 
        
        indexes = [i for i in range(N) if i%8 != 0]
        tensor = tensor1d[indexes]
        
        large_indexes = torch.nonzero((tensor > thr) + (tensor < -thr))
        count += large_indexes.nelement()
    return count, total, count/total*100        
        
def regularization_loss():
    '''
     regularized training is then just to add a term into the loss function to penalize the size of the weights on other   
     positions (but not the 8th position). That penalty term can be as simple as 
      \lambda/n * \sum_{squares of weights at other positions}. 
      where, \lambda is a parameter and $n$ is the number of weights at other positions.
    '''
    summ = 0 
    count = 0 
    
    # precalculate a general mask 
    maxN = 0 
    for name, param in model.named_parameters():
        if len(param.data.size()) < 2:
            continue 
        maxN = max(maxN, param.data.nelement())
    mask = torch.tensor([1.0] * maxN)
    mask[[i for i in range(maxN) if i%8 == 0]] = 0
    mask = mask.cuda()
    
    # apply the mask 
    for name, param in model.named_parameters():
        # add regularization only on weights in conv/FC layers
        if len(param.data.size()) < 2:
            continue 
        # find every weight in the non-8-th positions
        tensor1d = param.data.view(-1)
        N = len(tensor1d)
        
        squares = torch.pow(torch.mul(tensor1d, mask[:N]), 2)
        summ += torch.sum(squares)
        count += len(squares)
#     print('args.l2', args.l2)
    return summ / count * args.l2
    
def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    avg_time = time.time() 

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
#         data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        
        loss = F.cross_entropy(output, target)
        reg_loss = regularization_loss()
        total_loss = loss + reg_loss 
        avg_loss += loss.item()
        
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
#         loss.backward()
        total_loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tloss: {:.6f}, reg_loss: {:.6f}, total_loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, reg_loss.data, total_loss.data))
    
    avg_time = time.time() - avg_time
    num_large, total, per_large = check_large_weights_count()
    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tloss: {:.6f}, reg_loss: {:.6f}, total_loss: {:.6f}, epoch_time: {:.6f}, #large: {}({}, {})'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data, reg_loss.data, total_loss.data, avg_time, num_large, total, per_large))

            
            
def test():
    model.eval()
    test_loss = 0
    correct = 0
    test_time = time.time() 
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').data # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(test_loader.dataset)
    test_time = time.time() - test_time  

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    # print(correct, len(test_loader.dataset))
    return correct*1.0 / len(test_loader.dataset)

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))

best_prec1 = 0.
num_large, total, per_large = check_large_weights_count()
print('start to train, #large: {}({}, {})'.format(num_large, total, per_large))
      
for epoch in range(args.start_epoch, args.epochs):
    if epoch in [args.epochs*0.5, args.epochs*0.75]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train_time = time.time()
    train(epoch)
    train_time = time.time() - train_time 
    prec1 = test()
    # print(prec1)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
        'cfg': model.cfg
    }, is_best, filepath=args.save)
    print('Epoch: %d, train_time: %.2f (min), prec1: %f, best_prec1: %f\n' %(epoch, train_time/60.0, prec1, best_prec1))

with open(os.path.join(args.save, 'train.txt'), 'w') as f:
    f.write('Epoch: %d, train_time: %.2f (min), prec1: %f, best_prec1: %f\n' %(epoch, train_time/60.0, prec1, best_prec1))