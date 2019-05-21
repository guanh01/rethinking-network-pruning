# each bit has the probability of fault_rate to flip
# assume different params have different number of significant bits 
# each trial has:
# fault_rate: ~ 
# the faults can happen at any layer. It is network-wise fault injection 


import argparse
import numpy as np
import os, shutil 

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision 
from fault_injection import * 
import pickle, time, collections
from datetime import datetime  
import distiller 
from eval_util import test_imagenet 

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dataset', type=str, default='imagenet', 
                    help='imagenet')
parser.add_argument('--valdir', type=str, default='/home/hguan2/datasets/imagenet/val',
                    help='test dataset')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--data-type', type=str, default='int8',
                    help='data type used for weights: {int8}')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

# reconfigure 
parser.add_argument('--model-name', default='vgg16', type=str, 
                    help='architecture to use')

parser.add_argument('--fault-type', default='faults_network_rb_inplace', type=str,
                    help='fault type: {faults_network_rb_inplace}')
parser.add_argument('--start-trial-id', type=int, default=0,
                    help='start trial id')
parser.add_argument('--end-trial-id', type=int, default=5,
                    help='end trial id (included)')
parser.add_argument('--clean-dir', action='store_true', default=False,
                    help='clean directory')



args = parser.parse_args()
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print('using GPU:', args.cuda)

args.save = os.path.join(args.save, args.model_name, args.dataset, args.data_type, args.fault_type) 
if os.path.exists(args.save):
    if args.clean_dir:
        shutil.rmtree(args.save)
        print('path already exist! remove path:', args.save)
else:
    os.makedirs(args.save)
print('log will save to:', args.save)
print(args)

def select_fault_injection_function():
    fn = {
          'int8': {
              'faults_network_rb': inject_faults_int8_random_bit_position, 
              'faults_network_rb_ps1': inject_faults_int8_random_bit_position_ps1,
              'faults_network_rb_parity_zero': inject_faults_int8_random_bit_position_parity_zero,
              'faults_network_rb_parity_avg': inject_faults_int8_random_bit_position_parity_avg,
              'faults_network_rb_inplace': inject_faults_int8_random_bit_position_inpace_secded,
          }
         }
    return fn[args.data_type][args.fault_type]  



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

def load_checkpoint(model_path):

    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        best_prec1 = checkpoint['prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' Prec1: {:f}"
          .format(model_path, best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(model_path))
    return best_prec1 



def quantize_model(model, test=False):
    # use the default setting
    # https://github.com/NervanaSystems/distiller/blob/master/distiller/quantization/range_linear.py#L573
    # __init__(self, model, bits_activations=8, bits_parameters=8, bits_accum=32, bits_overrides=None,
    #            mode=LinearQuantMode.SYMMETRIC, clip_acts=False, no_clip_layers=None, per_channel_wts=False,
    #            model_activation_stats=None):
    
    quantizer = distiller.quantization.PostTrainLinearQuantizer(model)
    quantizer.prepare_model()
    
    # test the accuracy of the quantized model
    save_path = "/".join(args.save.split('/')[:-1])
    model_path = os.path.join(save_path, 'quantized.pth.tar')
    
    if not os.path.exists(model_path):
        prec1 = test_imagenet(model, args.valdir, num_batches = 50)
        # write the accuracy 
        
        with open(os.path.join(save_path, "quantize.txt"), "w") as fp:
            fp.write("Test accuracy: %.2f, num_batches: %d\n" %(prec1, num_batches))
        
        # save quantized model 
        torch.save({'state_dict': model.state_dict(), 
                    'prec1': prec1.item()}, model_path)
        print('Quantized model saved to:', model_path)        
    return model_path 
  


def write_detailed_info(log_path, info):
    with open(os.path.join(log_path, 'logs.txt'), 'a') as f:
        f.write(info+'\n')
        
def get_weights(model):
    # get weights stat    
    weights = [] 
    weights_names = [] 
    for name, param in model.named_parameters():
        # don't do simulation on bias and batch normalization layer
        if len(param.size()) >= 2:
            weights.append(param) 
            weights_names.append(name)
    return weights, weights_names
    

def load_steps():
    steps = [] 
    steps_path = os.path.join('./logs', args.model_name, args.dataset, args.data_type, 'gradual_encoding_absolute')
    print('load lossy steps from path:', steps_path) 
    with open(os.path.join(steps_path, 'steps.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split(':')
            accuracy = float(items[0])
            weights_ids = [int(x) for x in items[-1].split(',')] if items[-1].strip() else [] 
            steps.append((accuracy, weights_ids))
    return steps 

def encode_wrapper(named_param):
    name, param = named_param
    tensor = param.data 
    start = time.time()
    secded_encode(tensor)
    end = time.time() - start
    print('encode tensor name: %s, size: %s, time(s): %s' %(name, tensor.nelement(), end))

def perturb_weights(model, n_faults, trial_id, log_path, fault_injection_fn, lossy_ids): 
    # use trial_id to setup random seed 
    start = time.time()
    np.random.seed(trial_id)
    random = np.random  
    flipped_bits, changed_params, stats = 0, 0, {}
      
    
    # get the n_bits for each weight 
    weights, weights_names = get_weights(model)
    weights_sizes = [param.nelement() for param in weights]
    total_values = sum(weights_sizes)
    p = [size/total_values for size in weights_sizes]
    samples = random.choice(len(weights), size=n_faults, p=p)
    counter = collections.Counter(samples)
    print('samples:', sorted(counter.items())) 
    
    # first apply lossy encoding to the weights with lossy_ids
    print('lossy_weight_ids:', lossy_ids)
    for weight_id in lossy_ids:
        encode_wrapper((weights_names[weight_id], weights[weight_id]))
        
    
    for weight_id in sorted(counter.keys()):
        if weight_id in lossy_ids:
            lossy_encoding = True
        else:
            lossy_encoding = False 
            
        param = weights[weight_id]
        tensor = param.data.view(-1)
#         tensor_copy = tensor.clone() 
        
        # flip n_bits number of values from tensor
        n_bits = counter[weight_id]
        res = fault_injection_fn(tensor, random, n_bits, lossy_encoding=lossy_encoding)
        stats[weight_id] = res 
        
        if isinstance(res, tuple):
            flipped_bits += sum([len(arr) for x, arr in stats[weight_id][0].items()])
            changed_params += len(stats[weight_id][0])
#             print('nonzero', torch.nonzero(tensor_copy.view(-1) - tensor.view(-1)).size()[0], len(stats[weight_id][0]))
        else:
            flipped_bits += sum([len(arr) for x, arr in stats[weight_id].items()])
            changed_params += len(stats[weight_id])
#             print('nonzero', torch.nonzero(tensor_copy.view(-1) - tensor.view(-1)).size()[0], len(stats[weight_id]))
    
    assert flipped_bits == n_faults and changed_params <= n_faults, '%d, %d, %d' %(flipped_bits, changed_params, n_faults) 
    
    total_bits = total_values* 8
    info = 'trial: %d, n_faults: %d, total_params: %d' %(trial_id, n_faults, total_values)
    info += ', flipped_bits: %d (%.2e)' %(flipped_bits, flipped_bits*1.0/total_bits)
    info += ', changed_params: %d (%.2e)' %(changed_params, changed_params*1.0/total_values)
    
    end = time.time() - start
    print('Finish fault injection, time (s):', end) 
    
    save_path = os.path.join(log_path, 'stats')
    save_name = str(trial_id) + '.pkl'
    save_pickle(save_path, save_name, stats)
    
    return info  




        
# load the model parameters 
pretrained_models = {'resnet18': torchvision.models.resnet18(pretrained=True),
                     'resnet34': torchvision.models.resnet34(pretrained=True),
                     'alexnet': torchvision.models.alexnet(pretrained=True),
                     'squeezenet': torchvision.models.squeezenet1_0(pretrained=True),
                     'vgg16':  torchvision.models.vgg16(pretrained=True), 
                      'vgg16_bn':  torchvision.models.vgg16_bn(pretrained=True), 
#                      'densenet':  torchvision.models.densenet161(pretrained=True),
#                      'inception_v3':  torchvision.models.inception_v3(pretrained=True),
                    }
model = pretrained_models[args.model_name]
fault_injection_fn = select_fault_injection_function()

# if the model needs to be quantized (support int 8 quantization only)
if args.data_type == 'int8':
    print('quantize model using data type:', args.data_type)
    model_path = quantize_model(model)

    
weights, weights_names = get_weights(model)
weights_sizes = [param.nelement() for param in weights]
total_values = sum(weights_sizes)
print('# weights params:', len(weights), ', total_values:', total_values)
for i, item in enumerate(zip(weights_names, weights_sizes)):
    print('\t', i, item[0], item[1], '(%f)' %(item[1]/total_values))

lossy_steps = load_steps()
for item in lossy_steps:
     print("%.2f" %(item[0]) + ': '+ ', '.join([str(x) for x in item[1]]))
        
##########################
## start simulation ######
##########################
# for each fault_rate, use fault rate to get the number of faults
print('\nSimulation start: ', datetime.now())
simulation_start = time.time()

fault_rates = [10**x for x in range(-9, -2, 1)]
# fault_rates = [1e-3] 

for fault_rate in fault_rates:
    
    n_faults = int(total_values * 8 * fault_rate)
    if n_faults <= 0: 
        continue
    
    for lossy_step in lossy_steps:
        target_acc = lossy_step[0]
        lossy_ids = set(lossy_step[1])
        
        folder = 'r%s/%.2f' %(fault_rate, target_acc)
        log_path = os.path.join(args.save, folder)

        # for each trial, initialize the model  
        for trial_id in range(args.start_trial_id, args.end_trial_id):
            print('\nfault_rate:', fault_rate, ', n_faults:', n_faults, ', target_acc:', target_acc, ', trial_id:', trial_id)
            start = time.time()

            load_checkpoint(model_path)
            model.cpu()
    #         tensor_before = list(model.parameters())[0].clone()

            info = perturb_weights(model, n_faults, trial_id, log_path, fault_injection_fn, lossy_ids) 
            acc1 = test_imagenet(model, args.valdir, num_batches = 50)

    #         tensor_after = list(model.parameters())[0].cpu()
    #         print('tensor_after - tensor_before', torch.nonzero(tensor_after - tensor_before).size())

            duration = time.time() - start  

            info += ', trial_time: %d' %(duration)
            info += ', test_accuracy: %f' %(acc1)
            print(info, '\n')
            write_detailed_info(log_path, info)
        
        
#         break 
#     break 
simulation_time = time.time() - simulation_start
print('Simulation ends:', datetime.now(), ', duration(s):%.2f' %(simulation_time)) 
                 
    

    
    



        
    


        

        
        



    

