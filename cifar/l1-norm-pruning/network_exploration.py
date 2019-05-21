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
from eval_util import *  
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
parser.add_argument('--test-accuracy', action='store_true', default=False,
                    help='test model accuracy')
parser.add_argument('--plot-dist', action='store_true', default=False,
                    help='plot quantized model distribution')
parser.add_argument('--gradual-encode', action='store_true', default=False,
                    help='encoding with SEC-DED gradually with decreasing percentage')
parser.add_argument('--gradual-encode-absolute', action='store_true', default=False,
                    help='encoding with SEC-DED gradually with decreasing absolute number of large values')
parser.add_argument('--single-encode', action='store_true', default=False,
                    help='encoding with SEC-DED for a single layer')
parser.add_argument('--gradual-encode-adaptive', action='store_true', default=False,
                    help='gradually encode with lossy SEC-DEC with decreasing overhead')

parser.add_argument('--save-weights', action='store_true', default=False,
                    help='save weights including their large value indexes (v > 32)')
parser.add_argument('--save-parity', action='store_true', default=False,
                    help='save the parity encode to file')

parser.add_argument('--test-inference-speed', action='store_true', default=False,
                    help='test the inference speed of a network')
parser.add_argument('--num-images', default=500, type=int, 
                    help='number of images for testing inference speed')

matplotlib.rcParams['pdf.fonttype'] = 42


args = parser.parse_args()
torch.manual_seed(args.seed)
print('using GPU:', torch.cuda.is_available())
print(args)

def clean_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
        print('path already exist! remove path:', path)
    os.makedirs(path)

def check_directory(path):
    if not os.path.isdir(path):
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
    num_large_values = large_value_number(tensor)
    return num_large_values*1.0/size   

def large_value_number(tensor):
    return torch.nonzero((tensor > 63) +(tensor < -64)).size()[0]

def large_value_indexes(tensor, thr=64):
    '''tensor is torch tensor, thr is the large value threshold'''
    tensor = tensor.view(-1)
    indexes = torch.nonzero((tensor > thr-1) + (tensor < -thr)).view(-1)
    return indexes 

   

####################
# parity encoding
####################
def _parity_bit_numpy(v, width=8):
    bits = np.binary_repr(v, width=width) # 8 bits
    code = (sum(x=='1' for x in bits)%2 ==1)
    return code 

def _parity_bits(values):
    return [_parity_bit_numpy(int(v)) for v in values]


def factors(n):    
    return sorted(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
def _parity_encode(tensor):
    size = len(tensor)
    if size >= 128:
        fact = factors(size)
        index = bisect.bisect(fact, 128)
#         print('size:', size, ', factors:', fact, 'index:', index)
        split = fact[index]
#         print('split:', split) 
        tensor = tensor.view(split, -1)
        with Pool(32) as p:
            codes = p.map(_parity_bits, tensor)
        codes = torch.tensor(codes).view(-1)
    else:
        codes = torch.tensor(_parity_bits(tensor))
    return codes 
####################
# end parity encoding
####################

def write_detailed_info(log_path, info):
    with open(os.path.join(log_path, 'logs.txt'), 'a') as f:
        f.write(info+'\n')

def load_checkpoint(model_path):
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        best_prec1 = checkpoint['prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' Prec1: {:.2f}"
          .format(model_path, best_prec1))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(model_path))
    return best_prec1 


def gradual_encoding(model):
    '''sort layer in decreasing order based on the percentage of large values'''
    log_path = os.path.join(args.save, args.model_name, args.dataset, args.data_type, 'gradual_encoding') 
    clean_directory(log_path) 
    
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

        
def gradual_encoding_absolute(model):
    '''sort layer in decreasing order based on the absolute number of large values'''
    log_path = os.path.join(args.save, args.model_name, args.dataset, args.data_type, 'gradual_encoding_absolute') 
    clean_directory(log_path) 
    
    named_weights = get_named_weights(model)
    named_weights = [(name, weight, large_value_number(weight.data)) for name, weight in named_weights]
    named_weights = sorted(named_weights, key=lambda x: x[-1], reverse=True)
    loop_id = 1
    model.cpu()
    for name, weight, n_large in named_weights:
        # need to move tensor to CPU to enable multiprocessing  
        
#         print('encode param name:', name, 'tensor type:', weight.data.dtype, ', thr:', percentage)
        encode_wrapper((name, weight))
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
        model.cpu()
        
        info = 'loop: %d/%d, n_large: %d, name: %s, accuracy: %.2f' %(loop_id, len(named_weights), n_large, name, acc1)
        print(info) 
        write_detailed_info(log_path, info)
        loop_id += 1

def single_encoding(model):
    ''' apply lossy encoding on a single layer '''
    log_path = os.path.join(args.save, args.model_name, args.dataset, args.data_type, 'single_encoding') 
    clean_directory(log_path) 
    
    named_weights = get_named_weights(model)
    loop_id = 1
    model.cpu()
    for name, param in named_weights:
        
        tensor_copy = param.data.clone()
        number = large_value_number(tensor_copy)
        percentage = number * 1.0 / tensor_copy.nelement() 
        encode_wrapper((name, param))
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
         
        info = 'loop: %d/%d, n_large: %d (%f), name: %s, accuracy: %.2f' %(loop_id, len(named_weights), number, percentage, name, acc1)
        print(info) 
        write_detailed_info(log_path, info)
        loop_id += 1
        
        model.cpu()
        param.data = tensor_copy 
        

def gradual_encoding_adaptive(model):
    '''sort layer in decreasing order based on the encoding overhead'''
    log_path = os.path.join(args.save, args.model_name, args.dataset, args.data_type, 'gradual_encoding_adaptive') 
    named_weights = get_named_weights(model)
    
    # load the sorted weight ids
    sorted_weight_ids = [] 
    with open(os.path.join(log_path, 'tier1.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                items = line.split(',')
                weight_id = int(items[0])
                sorted_weight_ids.append(weight_id)
                
                # check it is correct
                weight_name = items[1].strip()
                weight_name2 = named_weights[weight_id][0]
                assert weight_name == weight_name2, 'weight name not equal: %s, %s' %(weight_name, weight_name2)
    # first k weights uses lossless encoding, last n-k weights use lossy encoding
    # use increasing n-k ==> use decreasing k
    assert len(sorted_weight_ids) == len(named_weights), 'weights number not equal: %d, %d' %(len(sorted_weight_id), len(named_ewights))
    sorted_weight_ids = sorted_weight_ids[::-1]
    layers = [(i, named_weights[i]) for i in sorted_weight_ids]
    
    # start to evaluate the accuracy of models with different encoding 
    model.cpu()
    loop_id = 1
    for weight_id, named_weight in layers:
        name, weight = named_weight 
        encode_wrapper((name, weight))
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
        model.cpu()
        
        info = 'loop: %d/%d, weight_id: %d, name: %s, accuracy: %.2f' %(loop_id, len(named_weights), weight_id, name, acc1)
        print(info) 
        write_detailed_info(log_path, info)
        loop_id += 1
    
def save_weights(model):
    metapath = os.path.join('./weights', args.model_name) 
    datapath = os.path.join('./weights', args.model_name, 'weights') 
    indexpath = os.path.join('./weights', args.model_name, 'indexes') 
    
    named_params = get_named_weights(model)
    clean_directory(datapath)
    clean_directory(indexpath)
    print('weights will save to dir:', datapath) 
    
    weight_id =  0 
    meta = '' 
    for name, param in named_params:

        tensor = param.data
        shape = tuple(tensor.size())
        size = tensor.nelement()
        
        # becuase majority vote work for data less than 32, so only store the indexes of larger than 32
        indexes = large_value_indexes(param.data, thr=32)
        largerThan32 = len(indexes) 

        # record data meta info 
        info = '%d, %s, %s, %d, %d' %(weight_id, name, shape, size, largerThan32)
        meta += info+'\n' 
        print(info)

        # save tensor to file 
        tensor1d = tensor.view(-1).numpy().astype(np.int8)
        np.savetxt(os.path.join(datapath, '%d.txt' %(weight_id)), tensor1d, fmt='%d')
        
        # save indexes to file 
        indexes = indexes.numpy()
        np.savetxt(os.path.join(indexpath, '%d.txt' %(weight_id)), indexes, fmt='%d')

        weight_id += 1
    
    # save meta to file 
    with open(os.path.join(metapath, 'meta.txt'), 'w') as f:
        f.write('weight_id, name, shape, size, largerThan32\n')
        f.write(meta)

        
def test_inference_speed(model):
    valdir = args.valdir
    batch_size = 1
    workers = 1
    # define loss function (criterion) and optimizer
#     criterion = nn.CrossEntropyLoss().cuda()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

    # switch to evaluate mode
    model.eval()
    model.cuda()

    with torch.no_grad():
        num_images = args.num_images 
        total_time = 0 
        for i, (input, target) in enumerate(val_loader):
            
            # take average over 100 images 
            if i >= num_images:
                break 
            
            input = input.cuda()
            target = target.cuda()
            
            # measure pure GPU inference time
            end = time.time()  
            output = model(input)
            end = time.time() - end 
            total_time += end 
    avg_time = total_time / num_images 
    print('num_images: %d, total time(s): %f, time/image(s): %f' %(num_images, total_time, avg_time))

    return avg_time 
    
def save_parity(model):
    datapath = os.path.join('./weights', args.model_name, 'parity') 
    named_params = get_named_weights(model)
    clean_directory(datapath)
    print('parity will save to dir:', datapath) 
    
    weight_id =  0 
    for name, param in named_params:

        tensor = param.data
        shape = tuple(tensor.size())
        size = tensor.nelement()
        
        codes = _parity_encode(tensor.view(-1))

        # save tensor to file 
        tensor1d = codes.view(-1).numpy().astype(np.int8)
        np.savetxt(os.path.join(datapath, '%d.txt' %(weight_id)), tensor1d, fmt='%d')
        
        info = '%d, %s, %s, %d' %(weight_id, name, shape, size)
        print(info)

        weight_id += 1
    
    
    

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
    if args.test_accuracy:
        print('test float model accuracy ...')
        acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
        print('Before quantization, accuracy: %.2f' %(acc1))
        
    if args.test_inference_speed:
        print('test inference speed by averaging over %d images...' %(args.num_images))
        test_inference_speed(model)
        
    
else:
    # post train quantization 
    quantizer = distiller.quantization.PostTrainLinearQuantizer(model)
    quantizer.prepare_model()
    save_path = os.path.join(args.save, args.model_name, args.dataset, args.data_type) 
    check_directory(save_path) 
    model_path = os.path.join(save_path, 'quantized.pth.tar')
    
    if not os.path.exists(model_path):
        prec1 = test_imagenet(model, args.valdir, num_batches = args.num_batches)
        # write the accuracy 
        with open(os.path.join(save_path, "quantize.txt"), "w") as fp:
            fp.write("Test accuracy: %.2f, num_batches: %d\n" %(prec1, args.num_batches))
        
        # save quantized model 
        torch.save({'state_dict': model.state_dict(), 
                    'prec1': prec1.item()}, model_path)
        print('Quantized model saved to:', model_path)  
    
#     # print(model)
#     if args.test_quantized:
#         print('test quantized model accuracy ...')
#         acc1 = test_imagenet(model, args.valdir, num_batches=args.num_batches)
#         print('After quantization, accuracy: %.2f' %(acc1))
    
    if args.plot_dist:
        print('plot model distribution...')
        plot_model_dist(model) 
    
    # sec-ded encoding gradually 
    if args.gradual_encode:
        print('start to gradually encode model weights according to large value percentage ...')
        load_checkpoint(model_path)
        gradual_encoding(model)
        
    if args.gradual_encode_absolute:
        print('start to gradually encode model weights according to absolute number of large values...')
        load_checkpoint(model_path)
        gradual_encoding_absolute(model)
    
    if args.single_encode:
        print('start to encode single model layer ...')
        load_checkpoint(model_path)
        single_encoding(model)
        
    if args.gradual_encode_adaptive:
        print('start to gradually encode model weights according to encoding overhead adaptively ...')
        load_checkpoint(model_path)
        gradual_encoding_adaptive(model)
        
    if args.save_weights:
        print('save weights and large value indexes to file ...')
        load_checkpoint(model_path)
        save_weights(model)
        
    if args.save_parity:
        print('save parity encoding bit to file ...')
        load_checkpoint(model_path)
        save_parity(model)
        
    if args.test_inference_speed:
        print('test inference speed by averaging over %d images...' %(args.num_images))
        load_checkpoint(model_path)
        test_inference_speed(model)
        
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
     



