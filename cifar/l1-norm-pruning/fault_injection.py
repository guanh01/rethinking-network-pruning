# inject faults into weights 
import numpy as np 
import bitstring 
from  collections import defaultdict  
import itertools 

# 1, extract weights from checkpoint 
# 2, randomly pertub weight

def select_fault_fn(fault_type):
    d = {
        'faults': inject_faults_float32,
        'zero_bit_masking': inject_faults_float32_with_zero_bit_masking,
        'faults_param': inject_faults_float32_param,
        'faults_param_mimic': inject_faults_float32_param_mimic,
    }
    return d[fault_type]

def inject_faults_float32(tensor, fault_rate, random, debug_mode=False):

    mask_shape = [x for x in tensor.shape] + [32]
    mask = random.rand(*mask_shape)
    indexes = np.argwhere(mask < fault_rate)

    stats = defaultdict(list)
    for index in indexes:
        vid, bid = tuple(index[:-1]), index[-1]
        value = tensor[vid]
        bits = bitstring.pack('>f', value)
        if debug_mode:
            print('before flip, value:', value, ', bits:', bits) 

        bits[bid] ^=1
        tensor[vid] = bits.float  
        if debug_mode:
            print('after flip, value:', bits.float, ', bits:', bits , ', flipped bit id:', index[-1]) 

        stats[vid].append((value, bid, bits[bid], bits.float))

    del mask, mask_shape, indexes  
    return stats 

def inject_faults_float32_param(tensor, fault_rate, random, debug_mode=False):

    mask_shape = [x for x in tensor.shape] 
    mask = random.rand(*mask_shape)
    indexes = np.argwhere(mask < fault_rate)

    stats = defaultdict(list)
    for index in indexes:
        vid, bid = tuple(index), random.randint(32) 
        value = tensor[vid]
        bits = bitstring.pack('>f', value)
        if debug_mode:
            print('before flip, value:', value, ', bits:', bits) 

        bits[bid] ^=1
        tensor[vid] = bits.float  
        if debug_mode:
            print('after flip, value:', bits.float, ', bits:', bits , ', flipped bit id:', index[-1]) 

        stats[vid].append((value, bid, bits[bid], bits.float))

    del mask, mask_shape, indexes  
    return stats 

def inject_faults_float32_param_mimic(tensor, mimic_stats, random, debug_mode=False):
    shape = tensor.shape 
    ranges = [range(x) for x in shape] 
    k = len(mimic_stats)
    all_indexes = list(itertools.product(*ranges))
    indexes = [all_indexes[i] for i in random.choice(len(all_indexes), size=k, replace=False)]
    mimic_stats = list(mimic_stats.values())
    stats = defaultdict(list)
    for i, index in enumerate(indexes):
        vid, bid = tuple(index), mimic_stats[i][0][1] # 0: first tuple; second 1 means the bid 
        value = tensor[vid]
        bits = bitstring.pack('>f', value)
        if debug_mode:
            print('before flip, value:', value, ', bits:', bits) 

        bits[bid] ^=1
        tensor[vid] = bits.float  
        if debug_mode:
            print('after flip, value:', bits.float, ', bits:', bits , ', flipped bit id:', index[-1]) 

        stats[vid].append((value, bid, bits[bid], bits.float))

    del mimic_stats, indexes  
    return stats


def inject_faults_float32_fixed_bit_position_and_number(tensor, random, bit_position, n_bits, debug_mode=False):
    shape = tensor.shape 
    ranges = [range(x) for x in shape] 
    all_indexes = list(itertools.product(*ranges))
    indexes = [all_indexes[i] for i in random.choice(len(all_indexes), size=n_bits, replace=False)]
    stats = defaultdict(list)
    for i, index in enumerate(indexes):
        vid, bid = tuple(index), bit_position # 0: first tuple; second 1 means the bid 
        value = tensor[vid]
        bits = bitstring.pack('>f', value)
        if debug_mode:
            print('before flip, value:', value, ', bits:', bits) 

        bits[bid] ^=1
        tensor[vid] = bits.float  
        if debug_mode:
            print('after flip, value:', bits.float, ', bits:', bits , ', flipped bit id:', bid) 

        stats[vid].append((value, bid, bits[bid], bits.float))

    del all_indexes, indexes  
    return stats


def inject_faults_int8_fixed_bit_position_and_number(tensor, random, bit_position, n_bits, debug_mode=False):
    shape = tensor.shape 
    ranges = [range(x) for x in shape] 
    all_indexes = list(itertools.product(*ranges))
    indexes = [all_indexes[i] for i in random.choice(len(all_indexes), size=n_bits, replace=False)]
    stats = defaultdict(list)
    for i, index in enumerate(indexes):
        vid, bid = tuple(index), bit_position # 0: first tuple; second 1 means the bid 
        value = int(tensor[vid])
        
        assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])
        
        bits = bitstring.pack('>b', value)
        
        if debug_mode:
            print('before flip, value:', value, 'bits:', bits) 
        
        bits[bid] ^= 1 
        value_after_flip = bits.int 
        
        tensor[vid] = value_after_flip   
        if debug_mode:
            print('after flip, value:', value_after_flip, 'bits:', bits, ',flipped bit id:', bid) 

        stats[vid].append((value, bid, bits[bid], value_after_flip))

    del all_indexes, indexes  
    return stats

def inject_faults_int8_fixed_bit_position_and_number_with_masking(tensor, random, bit_position, n_bits, debug_mode=False):
    shape = tensor.shape 
    ranges = [range(x) for x in shape] 
    all_indexes = list(itertools.product(*ranges))
    indexes = [all_indexes[i] for i in random.choice(len(all_indexes), size=n_bits, replace=False)]
    stats = defaultdict(list)
    for i, index in enumerate(indexes):
        vid, bid = tuple(index), bit_position # 0: first tuple; second 1 means the bid 
        value = int(tensor[vid])
        
        assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])
        
        bits = bitstring.pack('>b', value)
        
        if debug_mode:
            print('before flip, value:', value, 'bits:', bits) 
        
        bits[bid] = bits[0] # copy the sign bit  
        value_after_flip = bits.int 
        
        tensor[vid] = value_after_flip   
        if debug_mode:
            print('after flip, value:', value_after_flip, 'bits:', bits, ',flipped bit id:', bid) 

        stats[vid].append((value, bid, bits[bid], value_after_flip))

    del all_indexes, indexes  
    return stats


def inject_faults_float32_with_zero_bit_masking(tensor, fault_rate, random, debug_mode=False):

    mask_shape = [x for x in tensor.shape] + [32]
    mask = random.rand(*mask_shape)
    indexes = np.argwhere(mask < fault_rate)

    stats = defaultdict(list)
    for index in indexes:
        vid, bid = tuple(index[:-1]), index[-1]
        value = tensor[vid]
        bits = bitstring.pack('>f', value)
        if debug_mode:
            print('before flip, value:', value, ', bits:', bits) 

        bits[bid] = 0
        tensor[vid] = bits.float  
        if debug_mode:
            print('after flip, value:', bits.float, ', bits:', bits , ', flipped bit id:', index[-1]) 

        stats[vid].append((value, bid, bits[bid], bits.float))

    del mask, mask_shape, indexes  
    return stats  


if __name__ == '__main__':
    tensor = np.asarray([-11, -120, -1, 1, 20, 123])
    random = np.random
    bit_position = 1 
    n_bits = 1
    inject_faults_int8_fixed_bit_position_and_number_with_masking(tensor, random, bit_position, n_bits, debug_mode=True)