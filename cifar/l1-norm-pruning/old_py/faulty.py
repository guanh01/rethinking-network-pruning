# each bit has the probability of fault_rate to flip

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models 
from fault_injection import * 
import pickle 

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--arch', default='vgg', type=str, 
                    help='architecture to use')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

parser.add_argument('--fault-rate', type=float, default=1e-8,
                    help='bit fault rate')
parser.add_argument('--fault-type', default='faults', type=str,
                    help='fault type: {faults, zero_bit_masking, faults_param}')
parser.add_argument('--start-trial-id', type=int, default=0,
                    help='start trial id')
parser.add_argument('--end-trial-id', type=int, default=49,
                    help='end trial id')
parser.add_argument('--mimic-stats', default=None, type=str, metavar='PATH',
                    help='path to save stats to have the same fault pattern')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print('using GPU:', args.cuda)

if 'mimic' in args.fault_type and args.mimic_stats == None:
    raise ValueError('Need to specify mimic_stats!')

args.save = os.path.join('/'.join(args.model.split('/')[:-1]), args.fault_type, str(args.fault_rate)) 
if not os.path.exists(args.save):
    os.makedirs(args.save)
print('prune log will save to:', args.save)


def check_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def save_pickle(save_path, save_name, save_object):
    check_directory(save_path)
    filepath = os.path.join(save_path, save_name)
    pickle.dump(save_object, open(filepath,"wb" ))
    print('File saved to:', filepath)

def load_pickle(load_path, load_name=None, verbose=False):
    if load_name:
        filepath =  os.path.join(load_path, load_name)
    else:
        filepath = load_path 
    if verbose:
        print('Load pickle file:', filepath)
    return pickle.load( open(filepath, "rb" ))

def load_checkpoint(model):
    if args.model:
        if os.path.isfile(args.model):
            print("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.model))
    else:
        raise ValueError('args.model cannot be empty!')
    return best_prec1 



def test(model):
    if args.cuda:
        model.cuda()
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#             print(pred) 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

    
    
checkpoint = torch.load(args.model)
model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
acc = checkpoint['best_prec1']     
print('Before fault injection, accuracy: %.4f\n' %(acc))


# start to perturb model 
fault_fn = select_fault_fn(args.fault_type)
total_params = sum([param.nelement() for param in model.parameters()])
total_bits = total_params*32 

def get_key_with_param_id(param_id, mimic_stats):
    for key in mimic_stats.keys():
        if key[0] == param_id:
            return key 
    return None 

def perturb_model(trial_id):
    model.cpu()
    load_checkpoint(model)
    np.random.seed(trial_id)
    random = np.random 
    
    mimic_stats = None 
    if 'mimic' in args.fault_type: 
        # load the stats with the fault_rate the trial_id
        filepath = os.path.join(args.mimic_stats, 
                                               str(args.fault_rate), 
                                               'stats',
                                               str(trial_id)+'.pkl')
        if os.path.isfile(filepath):
            mimic_stats = load_pickle(filepath, verbose=True)
#             print('mimic_stats', mimic_stats) 
            
    flipped_bits, changed_params, var_to_stats= 0, 0, {} 
    for param_id, param in enumerate(model.parameters()):
        tensor = param.data.numpy()
        if 'mimic' in args.fault_type:
            if mimic_stats == None:
                continue
            key = get_key_with_param_id(param_id, mimic_stats)
            if key == None:
                continue 
            stats = fault_fn(tensor, mimic_stats[key] , random)
        else:
            stats = fault_fn(tensor, args.fault_rate, random)
        
        if stats:
            var_to_stats[(param_id, tensor.shape)] = stats
            flipped_bits += sum([len(arr) for x, arr in stats.items()])
            changed_params += len(stats)

            
    info = 'trial: %d, fault_rate: %e, total_params: %d' %(trial_id, args.fault_rate, total_params)
    info += ', flipped_bits: %d (%e)' %(flipped_bits, flipped_bits*1.0/total_bits)
    info += ', changed_params: %d (%e)' %(changed_params, changed_params*1.0/total_params)

    if flipped_bits:
        if 'mimic' in args.fault_type:
            check_mimic_correctness(var_to_stats, mimic_stats) 
        save_path = os.path.join(args.save, 'stats')
        save_name = str(trial_id) + '.pkl'
        save_pickle(save_path, save_name, var_to_stats)
#         print('var_to_stats', var_to_stats) 
    return flipped_bits > 0, info  

def check_mimic_correctness(stats, mimic_stats):
    param_ids = [x[0] for x in stats]
    param_ids_mimic = [x[0] for x in mimic_stats]
    assert param_ids == param_ids_mimic 
    
    changed_params = [len(x) for x in stats.values()]
    changed_params_mimic = [len(x) for x in mimic_stats.values()]
    assert changed_params == changed_params_mimic 

def write_detailed_info(info):
    with open(os.path.join(args.save, 'logs.txt'), 'a') as f:
        f.write(info+'\n')
        
        
for trial_id in range(args.start_trial_id, args.end_trial_id+1):
    isDiff, info = perturb_model(trial_id)
    acc_with_fault = acc 
    if isDiff:
        acc_with_fault = test(model)
    info += ', test_accuracy: %f' %(acc_with_fault)
    print(info, '\n')
    write_detailed_info(info)
        



    

