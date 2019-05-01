import distiller 
import numpy as np
import os, collections, sys, shutil
import bitstring 
import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import torchvision 
import models 
import matplotlib
from matplotlib import pyplot as plt
from eval_util import test_imagenet 
# import multiprocessing 
# %matplotlib inline
import argparse
from fault_injection import * 
from datetime import datetime 


parser = argparse.ArgumentParser(description='quantized net exploration')
parser.add_argument('--quantize', action='store_true', default=False,
                    help='quantize model')
parser.add_argument('--valdir', type=str, default='/home/hguan2/datasets/imagenet/val',
                    help='test dataset')
parser.add_argument('--dataset', type=str, default='imagenet', 
                    help='imagenet')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--data-type', type=str, default='int8',
                    help='data type used for weights: {int8}')

# reconfigure 
parser.add_argument('--model-name', default='vgg16', type=str, 
                    help='architecture to use')
parser.add_argument('--num-batches', default=50, type=int, 
                    help='number of batches for testing')
parser.add_argument('--test-quantized', action='store_true', default=False,
                    help='test quantized model accuracy')
parser.add_argument('--plot-dist', action='store_true', default=False,
                    help='plot quantized model distribution')
parser.add_argument('--encode', action='store_true', default=False,
                    help='lossy encoding with SEC-DED')

matplotlib.rcParams['pdf.fonttype'] = 42


args = parser.parse_args()
torch.manual_seed(args.seed)
print('using GPU:', torch.cuda.is_available())
print(args)

def prepare_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('path already exist! remove path:', path)
    os.makedirs(path)

def plot_model_dist(model):
    
    # check the distribution of parameters all weights
    thr = 32
    total_values, num_weights = 0, 0 
    counter = collections.Counter()
    for param_name, param in model.named_parameters():
        total_values += param.nelement()
        if len(param.size()) < 2:
            continue
        num_weights += param.nelement()
        counter.update(collections.Counter(np.abs(param.data.cpu().numpy().ravel())//thr + 1))

    tmp = sorted(counter.items(), key=lambda x: x[0])
    values, counts = zip(*tmp)
    
    # merge the interval [64, 96] and [96, 128]
    values = list(values)[:-1]
    counts = list(counts)
    counts[-2] += counts[-1]
    counts.pop() 

    total_weights = sum(list(counts))
    assert total_weights == num_weights
    print('#weights:', total_weights, ', #params:', total_values, 'percentage:', '%.6f' %(num_weights/total_values))
    
    fontsize = 16
    percentages = [count*100/total_weights for count in counts]
    bar = plt.bar(values, percentages)
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.2f%%' %(height), ha='center', va='bottom', fontsize = 16)
    #     print(['%.2f' %(p) for p in percentages])
    #plt.hist(param.data.cpu().numpy().ravel(), bins=10, density=True)
    # plt.xticks(values, [str(int(v)*thr) for v in values])
    plt.xticks(values, ['[0, 32)', '[32, 64)', '[64, 128]'], fontsize = 16)
    plt.title(args.model_name, fontsize = 16)
    #     plt.grid()
    plt.ylim(0, 110)
#     plt.show()
    plt.yticks(fontsize = 16)
    plt.xlabel('value ranges', fontsize = 16)
    plt.ylabel('percentage (%)', fontsize = 16)
    figdir = './figures/weight_distribution/'
    figname = args.model_name+'_int8_weight_distribution.pdf'
    check_directory(figdir)
    plt.savefig(os.path.join(figdir, figname), bbox_inches='tight')
    
def encode_wrapper(named_param):
    name, param = named_param
    tensor = param.data 
    start = time.time()
    secded_encode(tensor)
    end = time.time() - start
    print('encode tensor name: %s, size: %s, time(s): %s' %(name, tensor.nelement(), end))

def get_named_weights(model):
    named_params = [] 
    for name, param in model.named_parameters():
        if len(param.size()) >= 2:
            named_params.append((name, param)) 
    return named_params 

def large_value_percentage(tensor):
    size = tensor.nelement()
    num_large_values = torch.nonzero((tensor > 63) +(tensor < -64)).size()[0]
    return num_large_values*1.0/size   
    
def write_detailed_info(log_path, info):
    with open(os.path.join(log_path, 'logs.txt'), 'a') as f:
        f.write(info+'\n')

def gradual_encoding(model):
    log_path = os.path.join(args.save, args.model_name, args.dataset, args.data_type, 'lossy_encoding') 
    prepare_directory(log_path) 
    
    named_weights = get_named_weights(model)
    named_weights = [(name, weight, large_value_percentage(weight.data)) for name, weight in named_weights]
    named_weights = sorted(named_weights, key=lambda x: x[-1], reverse=True)
    loop_id = 1
    for name, weight, percentage in named_weights:
        # need to move tensor to CPU to enable multiprocessing  
        model.cpu()
#         print('encode param name:', name, 'tensor type:', weight.data.dtype, ', thr:', percentage)
        encode_wrapper((name, weight))
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
#         print('After encode with thr=%s, accuracy: %.2f' %(percentage, acc1))
        
        info = 'loop: %d/%d, thr: %s, name: %s, accuracy: %.2f' %(loop_id, len(named_weights), percentage, name, acc1)
        print(info) 
        write_detailed_info(log_path, info)
        loop_id += 1
    
    
    

pretrained_models = {'resnet18': torchvision.models.resnet18(pretrained=True),
                     'resnet34': torchvision.models.resnet34(pretrained=True),
                     'alexnet': torchvision.models.alexnet(pretrained=True),
                     'squeezenet': torchvision.models.squeezenet1_0(pretrained=True),
                     'vgg16':  torchvision.models.vgg16(pretrained=True), 
                      'vgg16_bn':  torchvision.models.vgg16_bn(pretrained=True), 
                     'densenet':  torchvision.models.densenet161(pretrained=True),
                     'inception_v3':  torchvision.models.inception_v3(pretrained=True),
                    }

print('use model:', args.model_name) 
model = pretrained_models[args.model_name]
# print(model)
start = time.time()
print('Start network exploration at time:', datetime.now())

if not args.quantize:
    print('test float model accuracy ...')
    acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
    print('Before quantization, accuracy: %.2f' %(acc1))
    
else:
    # post train quantization 
    quantizer = distiller.quantization.PostTrainLinearQuantizer(model)
    quantizer.prepare_model()
    # print(model)
    if args.test_quantized:
        print('test quantized model accuracy ...')
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
        print('After quantization, accuracy: %.2f' %(acc1))
    
    if args.plot_dist:
        print('plot model distribution...')
        plot_model_dist(model) 
    
    # sec-ded encoding gradually 
    if args.encode:
        print('start to encode model weights ...')
        gradual_encoding(model)
        
elapsed = time.time() - start
print('Finished at:', datetime.now(), ', elapsed time (s):', elapsed) 



#     if args.encode:
#         print('start to encode model weights ...')
#         named_params = [] 
#         for name, param in model.named_parameters():
#             if len(param.size()) >= 2:
#                 named_params.append((name, param)) 
                
#         model.cpu() # move parameters to cpu for modification 
#         with Pool(5) as p:
#             p.map(encode_wrapper, named_params)
#         acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
#         print('After encode, accuracy: %.2f' %(acc1))
     



