# each bit has the probability of fault_rate to flip
# assume different params have different number of significant bits 
# each trial has:
# bit position: 0 - 31
# param_id: 0 - #params-1
# #bits: 1 - min{#values in the param, 20} 


import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models 
from fault_injection import * 
import pickle, time 
import distiller 

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

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--fault-type', default='faults_layer', type=str,
                    help='fault type: {faults_layer, faults_layer_masking}')
parser.add_argument('--start-trial-id', type=int, default=0,
                    help='start trial id')
parser.add_argument('--end-trial-id', type=int, default=2,
                    help='end trial id (included)')
parser.add_argument('--data-type', type=str, default='float32',
                    help='data type used for weights: {float32, int8}')

args = parser.parse_args()
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print('using GPU:', args.cuda)

if args.data_type == 'float32':
    args.save = os.path.join('/'.join(args.model.split('/')[:-1]), args.fault_type) 
else:
    args.save = os.path.join('/'.join(args.model.split('/')[:-1]), args.data_type, args.fault_type) 
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
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
#             print(pred) 
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    print('Test set: Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

def quantize_model(model):
    # use the default setting
    # https://github.com/NervanaSystems/distiller/blob/master/distiller/quantization/range_linear.py#L573
    # __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32, bits_overrides=None,
    #            mode=LinearQuantMode.SYMMETRIC, clip_acts=False, no_clip_layers=None, per_channel_wts=False,
    #            model_activation_stats=None):
    quantizer = distiller.quantization.PostTrainLinearQuantizer(model)
    quantizer.prepare_model()
    
    # test the accuracy of the quantized model
    prec1 = test(model)
    # write the accuracy 
    save_path = "/".join(args.save.split('/')[:-1])
    with open(os.path.join(save_path, "quantize.txt"), "w") as fp:
        fp.write("Test accuracy: \n"+str(prec1)+"\n")
    # save quantized model     
    torch.save({ 'cfg': model.cfg, 
                'state_dict': model.state_dict(), 
                'prec1': prec1
               }, os.path.join(save_path, 'quantized.pth.tar'))
    
    
    
# load the model parameters 
checkpoint = torch.load(args.model)
# specify the model 
model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
model.load_state_dict(checkpoint['state_dict'])
acc = checkpoint['best_prec1']     
print('Before fault injection, accuracy: %.4f\n' %(acc))

# if the model needs to be quantized (support int 8 quantization only)
if args.data_type == 'int8':
    print('quantize model using data type:', args.data_type)
    quantize_model(model)

def select_fault_injection_function():
    fn = {'float32': {'faults_layer': inject_faults_float32_fixed_bit_position_and_number}, 
          'int8': {
              'faults_layer': inject_faults_int8_fixed_bit_position_and_number,
              'faults_layer_masking': inject_faults_int8_fixed_bit_position_and_number_with_masking}
         }

    return fn[args.data_type][args.fault_type]

fault_injection_fn = select_fault_injection_function()


def perturb_model(param, bit_position, n_bits, trial_id, log_path): 
    # use trial_id to setup random seed 
    np.random.seed(trial_id)
    random = np.random 
    
    # get the param   
    flipped_bits, changed_params= 0, 0
    tensor = param.data.cpu().numpy()
    
    # flip n_bits number of values from tensor in bit_position
    stats = fault_injection_fn(tensor, random, bit_position, n_bits)
        
    flipped_bits += sum([len(arr) for x, arr in stats.items()])
    changed_params += len(stats)
    
    assert flipped_bits == n_bits
    assert changed_params == n_bits 
    
    total_params = param.data.nelement()
    total_bits = total_params * 32 
    info = 'trial: %d, bit_position: %d, n_faults: %d, total_params: %d' %(trial_id, bit_position, n_bits, total_params)
    info += ', flipped_bits: %d (%e)' %(flipped_bits, flipped_bits*1.0/total_bits)
    info += ', changed_params: %d (%e)' %(changed_params, changed_params*1.0/total_params)

    
    save_path = os.path.join(log_path, 'stats')
    save_name = str(trial_id) + '.pkl'
    save_pickle(save_path, save_name, stats)

    return info  

def write_detailed_info(log_path, info):
    with open(os.path.join(log_path, 'logs.txt'), 'a') as f:
        f.write(info+'\n')
        
if args.data_type == 'float32':
    start_bit_position, end_bit_position = 0, 20
elif args.fault_type == 'faults_layer':
    start_bit_position, end_bit_position = 0, 8
else:
    start_bit_position, end_bit_position = 1, 8
    

for param_id, param in enumerate(model.parameters()):
    # don't do simulation on bias and batch normalization layer
    if len(param.size()) < 2:
        continue 
    
    # prepare to preturb param, keep a clone of param
    param_tensor = param.data.cpu().clone()
    max_n_bits = param_tensor.nelement()
      
    for bit_position in range(start_bit_position, end_bit_position):    

        for n_bits in [10**i for i in range(10)]:
            if n_bits > max_n_bits:
                break 
            folder = 'param-%d/bit-%d/nbits-%d' %(param_id, bit_position, n_bits)
            log_path = os.path.join(args.save, folder)
            
            for trial_id in range(args.start_trial_id, args.end_trial_id+1):
                
                info = perturb_model(param, bit_position, n_bits, trial_id, log_path) 
                
                test_time = time.time()
                acc_with_fault = test(model)
                test_time = time.time() - test_time  
                
                info += ', test_time: %d' %(test_time)
                info += ', test_accuracy: %f' %(acc_with_fault)
                print(info, '\n')
                
                write_detailed_info(log_path, info)
                
                # reset the value of that param after each trial 
                param.data = param_tensor.clone() 
            
        
        



    

