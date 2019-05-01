# inject faults into weights 
import numpy as np 
import bitstring, time
from  collections import defaultdict  
import itertools, torch , collections 
from multiprocessing import Pool 
from functools import reduce
import bisect 

# __all__ = [inject_faults_int8_random_bit_position, inject_faults_int8_random_bit_position_ps1,
#           inject_faults_int8_random_bit_position_parity]

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

# ---------------------------------------------------------
# ---------------------------------------------------------

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
            print('before flip, value:', value, 'bits:', bits.bin) 
        
        bits[bid] = bits[0] # copy the sign bit  
        value_after_flip = bits.int 
        
        tensor[vid] = value_after_flip   
        if debug_mode:
            print('after flip, value:', value_after_flip, 'bits:', bits.bin, ',flipped bit id:', bid) 

        stats[vid].append((value, bid, bits[bid], value_after_flip))

    del all_indexes, indexes  
    return stats


def inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=False):
    """ For the tensor, randomly choose n_bits number of bits to flip. Total number of bits is num_values * 8 
    input tensor should be 1-d torch tensor. """
    start = time.time()
#     tensor = tensor.view(-1)
#     indexes = random.choice(tensor.nelement(), size=n_bits)
    num_values = tensor.nelement()
    indexes = random.choice(num_values*8, size=n_bits, replace=False)
    sample_time = time.time() - start
    
    start = time.time() 
    stats = defaultdict(list)
    for index in indexes:
        vid, bid = index>>3, index&0b111
        value = int(tensor[vid])

        assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])

        bits = bitstring.pack('>b', value)

#         if debug_mode:
#             print('befor flip, value: %10d (%s)' %( value, bits.bin))

        bits[bid] ^= 1 
        value_after_flip = bits.int 

        tensor[vid] = value_after_flip   
        if debug_mode:
            print('vid: %5d, before: %5d, bid: %d => %s, after: %5d (%s)' 
                  %(vid, value, bid, bits[bid], value_after_flip, bits.bin)) 

        stats[vid].append((value, bid, bits[bid], value_after_flip))
    del indexes
    injection_time = time.time() - start
    print('sample time (s):', '%.4f' %(sample_time), ', injection_time (s):', '%.4f' %(injection_time)) 
    return stats 


# def _one_bit_encode(tensor):
#     codes = [False] * len(tensor)
#     for i, v in enumerate(tensor):
#         bits = bitstring.pack('>b', v)
#         codes[i] = (bits[0] == bits[1] == bits[2]) # encoding = 1 if first three bits are the same 
#     return codes 

##################################################################
## use 1-bit encoding for |v| < 32 and majority vote to correction 
##################################################################

def _one_bit_encode(tensor):
    # return type: uint8 
    return (tensor >= -32) & (tensor < 32) 
def _correct_error_majority_vote_ps1(tensor, codes, flipped):
    corr = {} 
    new_codes = _one_bit_encode(tensor)
    # correct the value whose original code is 1, current code is 0 
    indexes = torch.nonzero((codes==1) & (new_codes==0))
   
    for index in indexes:
        value = int(tensor[index].item())
        bits = bitstring.pack('>b', value) # 8 bits
        if bits[0]+bits[1]+bits[2] >= 2:
            bits[0] = bits[1] = bits[2] = 1 
        else:
            bits[0] = bits[1] = bits[2] = 0 
        if index.item() in flipped:
            corr[index.item()] = (value, bits.int, False)
        else:
            corr[index.item()] = (value, bits.int, True)
        tensor[index] = bits.int
    return corr 
        
def inject_faults_int8_random_bit_position_ps1(tensor, random, n_bits, debug_mode=False):
    # step 1: 1-bit encoding 
    start = time.time()
    codes = _one_bit_encode(tensor)
    if debug_mode:
        print('codes:', codes)
        print(tensor)
    encode_time = time.time() - start
    
    # step 2: inject faults to weights 
    stats = inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=debug_mode)
    
    # step 3: inject faults to codes
    start = time.time()
    num_values = tensor.nelement()
    indexes = random.choice(num_values, size=int(n_bits//8), replace=False)
    for i in indexes:
        codes[i] = 0 if codes[i]==1 else 1
    injection_time = time.time() - start 
    
    # step 4: error correction
    start = time.time()
    corr = _correct_error_majority_vote_ps1(tensor, codes, set(indexes))
    if debug_mode:
        print('correction:', corr)
    correction_time = time.time() - start
    del indexes
    
    print('encoding time(s): %.4f, sample+injection time(s): %.4f, correction time(s): %.4f' 
          %( encode_time, injection_time, correction_time))
    return stats, corr 

##############################################################
# use parity bit to detect error and then set the value to zero 
##############################################################

def _parity_bit(v):
    bits = bitstring.pack('>b', v) # 8 bits
#     code = (sum(bits)%2 == 1)
    code = reduce(lambda x, y: x^y, bits)
    return code 
def _parity_bit_sum(v):
    bits = bitstring.pack('>b', v) # 8 bits
    code = (sum(bits)%2 == 1)
    return code 
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
# def _parity_encode(tensor):
#     codes = [_parity_bit(v) for v in tensor]
#     with Pool() as p:
#         codes = p.map(_parity_bit, tensor)
#     return np.asarray(codes) 

def _correct_error_parity_zero(tensor, codes, flipped):
    new_codes = _parity_encode(tensor)
    # check when the two codes are different
    indexes = torch.nonzero(codes - new_codes)
    corr = {i.item(): (int(tensor[i].item()), 0, i.item() not in flipped) for i in indexes}
    tensor[indexes] = 0 
    return corr

def _correct_error_parity_avg(tensor, codes, flipped):
    new_codes = _parity_encode(tensor)
    # check when the two codes are different
    indexes = torch.nonzero(codes - new_codes)
    corr = {}
    for index in indexes:
        value = int(tensor[index].item())
        left = int(tensor[index-1].item())
        right = int(tensor[(index+1)%len(tensor)].item())
        new_value = int((left+right)/2)
        tensor[index] = new_value 
        corr[index.item()] = (value, new_value, index.item() not in flipped)
    return corr  

def inject_faults_int8_random_bit_position_parity_zero(tensor, random, n_bits, debug_mode=False):
    # step 1: parity encoding 
    start = time.time()
    codes = _parity_encode(tensor)
    if debug_mode:
        print('codes:', codes)
#         print(tensor)
    encode_time = time.time() - start
    
    # step 2: inject faults to weights 
    stats = inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=debug_mode)
    
    # step 3: inject faults to codes
    start = time.time()
    num_values = tensor.nelement()
    indexes = random.choice(num_values, size=int(n_bits//8), replace=False)
    for i in indexes:
        codes[i] = 0 if codes[i]==1 else 1
    injection_time = time.time() - start 
    
    # step 4: error correction
    start = time.time()
    corr = _correct_error_parity_zero(tensor, codes, set(indexes))
    if debug_mode:
        print('stats:', sorted(stats.items()))
        print('faulty #codes:', len(indexes), sorted(indexes)) 
        print('correction:', sorted(corr.items()))
    correction_time = time.time() - start
    del indexes
    
    print('encoding time(s): %.4f, sample+injection time(s): %.4f, correction time(s): %.4f' 
          %( encode_time, injection_time, correction_time))
    return stats, corr 

def inject_faults_int8_random_bit_position_parity_avg(tensor, random, n_bits, debug_mode=False):
    # step 1: parity encoding 
    start = time.time()
    codes = _parity_encode(tensor)
    if debug_mode:
        print('codes:', codes)
#         print(tensor)
    encode_time = time.time() - start
    
    # step 2: inject faults to weights 
    stats = inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=debug_mode)
    
    # step 3: inject faults to codes
    start = time.time()
    num_values = tensor.nelement()
    indexes = random.choice(num_values, size=int(n_bits//8), replace=False)
    for i in indexes:
        codes[i] = 0 if codes[i]==1 else 1
    injection_time = time.time() - start 
    
    # step 4: error correction
    start = time.time()
    corr = _correct_error_parity_avg(tensor, codes, set(indexes))
    if debug_mode:
        print('faulty #codes:', len(indexes)) 
        print('correction:', corr)
    correction_time = time.time() - start
    del indexes
    
    print('encoding time(s): %.4f, sample+injection time(s): %.4f, correction time(s): %.4f' 
          %( encode_time, injection_time, correction_time))
    return stats, corr 

##############################################################
# use SEC-DCD to detect error and correct error 
##############################################################

def _secded_parity_bits(tensor):
    # use tensor to generate the parity bits for the data/tensor  
    assert len(tensor) < 9, '#values should be less than 9, current #values:%d' %(len(tensor))
    data = [] 
    LSBs = [] 
    for v in tensor:
        bits = np.binary_repr(int(v), width=8) # 8 bits
        data.append(bits[:-1]) # don't use the last bit
        LSBs.append(int(bits[-1])) # record the least significant bits 
#         print(v, data[-1])
    data = "".join(data) 
    assert len(data) == len(tensor)*7, 'length of data is not:%d' %(len(tensor)*7)
#     print('data to be encoded:', data)
#     data = "".join(data[::-1])
#     print('data reversed:', data)
    # generate parity bits for the data 
    P0 = [0, 1, 3, 4, 6, 8, 10, 11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
    P1 = [0, 2, 3, 5, 6, 9, 10, 12, 13, 16, 17, 20, 21, 24, 25, 27, 28, 31, 32, 35, 36, 39, 40, 43, 44, 47, 48, 51, 52, 55]
    P2 = [1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 17, 22, 23, 24, 25, 29, 30, 31, 32, 37, 38, 39, 40, 45, 46, 47, 48, 53, 54, 55]
    P3 = [4, 5, 6, 7, 8, 9, 10, 18, 19, 20, 21, 22, 23, 24, 25, 33, 34, 35, 36, 37, 38, 39, 40, 49, 50, 51, 52, 53, 54, 55]
    P4 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    P5 = [26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    # need totally 7-bit as check bits + parity bit 
    parity_bits = [0]*7
    hamming = [P5, P4, P3, P2, P1, P0]
    
    # first six are hamming check bits
    for i in range(6):
        parity_bits[i] = sum(data[x] == '1' for x in hamming[i])%2
    
    # the last one is the added parity bit
    parity_bits[-1] = (sum(d == '1' for d in data) + sum(parity_bits[:-1])) % 2
    
    # make parity_bits the same size as LSBs
    parity_bits =  parity_bits + [LSBs[-1]]
    
    return parity_bits, LSBs
        

def _test_secded_parity_bits():
    tensor = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    parity_bits = _secded_parity_bits(tensor)
    print('parity bits:', parity_bits)

def _secded_batch(tensor):
    # the input tensor is a batch of tensor to be encoded
    # input tensor of size: nx8
    # output of size: nx8, nx8 
    res = [_secded_parity_bits(v) for v in tensor]
    parity_bits, LSBs = zip(*res)
    return parity_bits, LSBs 
    
    
def secded_encode(tensor, debug=False):
    # the input argument tensor is modified in place, so no return value 
    tensor = tensor.view(-1)
    length = tensor.size()[0]
    if debug:
        print('before encode:', tensor)
        print('length:', length) 
    
    # make sure tensor is of length that is a multiple of eight. Otherwise, need to pad the value. TODO later 
    assert length%8 == 0, 'length of the weights is not a multiple of 8: %d' %(length)
    
    # view tensor as an mxnx8 matrix, where nx8 is a batch 
    if length/8 >= 128:
        fact = factors(length/8)
        index = bisect.bisect(fact, 128)
        split = int(fact[index])
#         print('split:', split) 
        tensor = tensor.view(split, -1, 8)
        with Pool(32) as p:
            codes = p.map(_secded_batch, tensor)
        # codes is a list of tupes 
        parity_bits, LSBs = zip(*codes)
    else:
        parity_bits, LSBs = _secded_batch(tensor.view(-1, 8))
                                          
    parity_bits = torch.tensor(parity_bits, dtype=torch.float32).view(-1)
    LSBs = torch.tensor(LSBs, dtype=torch.float32).view(-1)
    
    # use parity_bits and LSBs to change tensor values 
    tensor = tensor.view(-1)
    tensor.add_(parity_bits).sub_(LSBs)
    if debug:
        print('after encode:', tensor)
                                          

# previous sequential version        
def secded_encode_sequential(tensor, debug=False):
    # the input argument tensor is modified in place 
    tensor = tensor.view(-1)
    length = tensor.size()[0]
    if debug:
        print('before encode:', tensor)
        print('length:', length) 
    
    # make sure tensor is of length that is a multiple of eight. 
    assert length%8 == 0, 'length of the weights is not a multiple of 8: %d' %(length)
    
    # change tensor shape 
    tensor = tensor.view(-1, 8)
    # calculate the encoding for each row of tensor 
    for row in tensor:
        parity_bits, LSBs = _secded_parity_bits(row)
        # modify the LSB of the first 7 values in tensor in place
        if debug:
            print('row:', row)
            
            print('LSBs:', LSBs)
            print('pari:', parity_bits)
            
        for i in range(len(parity_bits)):
            # two methods to change the value
            pre_value = int(row[i])
            
            # option 1
#             bits = bitstring.pack('>b', row[i])
#             bits[-1] = parity_bits[i]
#             new_value = bits.int 
            
            # option 2
            if LSBs[i] != parity_bits[i]:
                if parity_bits[i] == 1:
                    row[i] += 1
                else:
                    row[i] -= 1
              
            # compare option 1 and option 2
#             assert new_value == row[i].item(), "pre_value: %d, bit operation value:%d (%s), manual value:%d" %(pre_value, new_value, bits.bin, row[i].item())
    if debug:
        print('after encode:', tensor)
                    
def _test_secded_encode():
    tensor = torch.randint(-10, 10, size=(2000, 80))
    tensor_clone = tensor.clone()
    # method 1: 
    start = time.time()
    secded_encode(tensor, debug=True)
    end1 = time.time() - start 
    # method 2:
    start = time.time()
    secded_encode_sequential(tensor_clone, debug=False)
    end2 = time.time() - start 
    # compare results from the two methods
    print('time(s): %f, time(s):%f, nonzero: %d'%(end1, end2, np.nonzero(tensor-tensor_clone).size()[0]))
    
#     before encode: tensor([ 6, -6, -9,  ..., -7,  9,  7])
#     length: 1600000
#     after encode: tensor([  6,  -6, -10,  ...,  -7,   8,   7])
#     time(s): 1.749301, time(s):35.786590, nonzero: 0

    

def inject_faults_int8_random_bit_position_secded(tensor, random, n_bits, debug_mode=False):
    """ Given the tenor as an the weight matrix of a layer. Use SEC-DED to encode data and check fault injection results.
    It uses the LSB to record the check bits and is able to correct 1-bit error."""
    pass 
    
    
    

# def inject_faults_int8_random_bit_position_ps1(tensor, random, n_bits, debug_mode=False):
#     """ For the tensor, randomly choose n_bits number of bits to flip. Total number of bits is num_values * 8 
#     input tensor should be torch tensor. 
#     protection: 1-bit encoding + majority vote, 
#     the encoding bit may also flipped. """
#     tensor = tensor.view(-1)
    
#     # step 1: 1-bit encoding 
#     start = time.time()
#     codes = _one_bit_encode(tensor)
#     if debug_mode:
#         print('codes:', codes)
#         print(tensor)
#     encode_time = time.time() - start
    
#     # step 2: sample faults, include the encoding bit  
#     start = time.time()
#     num_values = tensor.nelement()
#     indexes = random.choice(num_values*9, size=n_bits, replace=False)
#     sample_time = time.time() - start
    
#     # step 3: inject faults 
#     start = time.time() 
#     stats = defaultdict(list)
#     for index in indexes:
#         vid, bid = index//9, index%9
#         value = int(tensor[vid])

# #         assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])
#         bits = bitstring.pack('>b', value) # 8 bits

#         if debug_mode:
#             print('befor flip, vid: %3d, value: %5d (%s)' %(vid, value, bits.bin))
#         if bid < 8:
#             bits[bid] = not bits[bid] 
#         else:
#             codes[vid] = 0 if codes[vid] == 1 else 1 
            
#         value_after_flip = bits.int 
#         tensor[vid] = value_after_flip   
#         if debug_mode:
#             print('after flip, vid: %3d, value: %5d (%s), bid: %d' %(vid, value_after_flip, bits.bin, bid)) 

#         stats[vid].append((value, bid, bits[bid] if bid < 8 else codes[vid], value_after_flip))
#     injection_time = time.time() - start   
#     if debug_mode:
#         print(tensor)
        
#     # step 4: error correction
    
#     start = time.time()
#     corr = _correct_error_majority_vote(tensor, codes)
        
#     if debug_mode:
#         print('correction:', corr)
#     correction_time = time.time() - start
#     del indexes
    
#     print('encoding time(s): %.4f, sample time(s): %.4f, injection time(s): %.4f, correction time(s): %.4f' 
#           %( encode_time, sample_time, injection_time, correction_time))
#     return stats, corr 




    
# this version is too slow. 
# def correct_error_majority_vote(tensor, codes):
#     corr = {}
#     for i, value in enumerate(tensor):
#         code = codes[i] 
#         if code:
#             # first three bits should be the same
#             bits = bitstring.pack('>b', value) # 8 bits
#             if bits[0] == bits[1] == bits[2]:
#                 continue 
            
#             if bits[0]+bits[1]+bits[2] >= 2:
#                 bits[0] = bits[1] = bits[2] = 1 
#             else:
#                 bits[0] = bits[1] = bits[2] = 0 
            
#             corr[i] = (value.item(), bits.int)
#             tensor[i] = bits.int 
            
#         else:
#             pass # dothing is the first three bits are not supposed to be the same 
#     return corr 
    


# def inject_faults_int8_random_bit_position(tensor, random, n_bits, debug_mode=False):
#     """ For the tensor, randomly choose n_bits number of bits to flip. Total number of bits is num_values * 8 """
#     start = time.time()
#     shape = tensor.shape 
#     shape = list(shape) + [8]
#     ranges = [range(x) for x in shape] 
#     all_indexes = list(itertools.product(*ranges))
#     indexes = [all_indexes[i] for i in random.choice(len(all_indexes), size=n_bits, replace=False)]
#     sample_time = time.time() - start
    
#     start = time.time() 
#     stats = defaultdict(list)
#     for i, index in enumerate(indexes):
#         vid, bid = tuple(index[:-1]), index[-1] 
#         value = int(tensor[vid])
        
#         assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])
        
#         bits = bitstring.pack('>b', value)
        
#         if debug_mode:
#             print('before flip, value:', value, 'bits:', bits.bin) 
        
#         bits[bid] ^= 1 
#         value_after_flip = bits.int 
        
#         tensor[vid] = value_after_flip   
#         if debug_mode:
#             print('after flip, value:', value_after_flip, 'bits:', bits.bin, ',flipped bit id:', bid) 

#         stats[vid].append((value, bid, bits[bid], value_after_flip))

#     del all_indexes, indexes
#     injection_time = time.time() - start
#     print('sample time (s):', sample_time, ', injection_time (s):', injection_time) 
#     return stats


        
        

# def inject_faults_int8_random_bit_position_ps1(tensor, random, n_bits, debug_mode=False):
#     """ For the tensor, the fault model is: 
#     randomly choose n_bits number of bits to flip. Total number of bits is num_values * 8 
#     The protection strategy 1: 
#     1. |v| < 32, the first three bits are the same. Use majority vote to correct error.  
#     2. |v| >= 32, do nothing. """
    
#     shape = tensor.shape 
#     shape = list(shape) + [8]
#     ranges = [range(x) for x in shape] 
#     all_indexes = list(itertools.product(*ranges))
#     indexes = [all_indexes[i] for i in random.choice(len(all_indexes), size=n_bits, replace=False)]
#     stats = defaultdict(list)
    
#     for i, index in enumerate(indexes):
#         vid, bid = tuple(index[:-1]), index[-1] 
#         value = int(tensor[vid])
        
#         assert value == tensor[vid], "value is not an integer," + str(value) + ', '+ str(tensor[vid])
        
#         bits = bitstring.pack('>b', value)
        
#         if debug_mode:
#             print('\nbefore flip, value:', value, 'bits:', bits.bin) 
        
#         same_bits = False
#         if bits[0] == bits[1] == bits[2]:
#             same_bits = True 
            
#         bits[bid] ^= 1 
#         # When one the first three bits flipped, use majority vote to correct error. 
#         if same_bits and 0 <= bid <= 2:
#             bits[bid] = (sum([bits[0], bits[1], bits[2]]) >= 2)
#             if debug_mode:
#                 print('use majority vote to protect first three bits.')
                    
#         value_after_flip = bits.int 
        
#         tensor[vid] = value_after_flip   
#         if debug_mode:
#             print('after flip, value:', value_after_flip, 'bits:', bits.bin, ',flipped bit id:', bid) 

#         stats[vid].append((value, bid, bits[bid], value_after_flip))

#     del all_indexes, indexes  
#     return stats

def _test_parity_bits():
    tensor = torch.randint(-40, 40, size=(100000,))
    
    s1 = time.time()
    t1 = torch.tensor(_parity_bits(tensor))
    e1 = time.time() - s1
    
    s2 = time.time()
    t2 = _parity_encode(tensor)
    e2 = time.time() - s2
    
    print(e1, e2, sum(t1-t2).item())

def _test_parity_bit():
    tensor = torch.randint(-40, 40, size=(5000,))
    s1 = time.time()
    t1 = [_parity_bit(v) for v in tensor]
    s1 = time.time() - s1
    
    s2 = time.time()
    t2 = [_parity_bit_sum(v) for v in tensor]
    s2 = time.time() - s2
    
    s3 = time.time()
    t3 = [_parity_bit_numpy(v) for v in tensor]
    s3 = time.time() - s3
    
    print(s1, s2, s3, sum(torch.tensor(t1) - torch.tensor(t2)).item(), sum(torch.tensor(t1) - torch.tensor(t3)).item())
    # 0.40941762924194336 0.40434861183166504 0.12994694709777832 0 0

if __name__ == '__main__':
    
#     _test_secded_parity_bits()
    _test_secded_encode()
    
#     _test_parity_bit() 
#     tensor = torch.randint(-40, 40, size=(50,))
#     random = np.random
    
#     n_bits = 10
#     print('before:', tensor)
#     inject_faults_int8_random_bit_position_parity_zero(tensor, random, n_bits, debug_mode=True)
#     print('after: ', tensor)