
# coding: utf-8

# # Quantizing RNN Models

# In this example, we show how to quantize recurrent models.  
# Using a pretrained model `model.RNNModel`, we convert the built-in pytorch implementation of LSTM to our own, modular implementation.  
# The pretrained model was generated with:  
# ```time python3 main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --tied --wd=1e-6```  
# The reason we replace the LSTM that is because the inner operations in the pytorch implementation are not accessible to us, but we still want to quantize these operations. <br />
# Afterwards we can try different techniques to quantize the whole model.  
# 
# _NOTE_: We use `tqdm` to plot progress bars, since it's not in `requirements.txt` you should install it using 
# `pip install tqdm`.

# In[1]:


from model import DistillerRNNModel, RNNModel
from data_wikitext.data import Corpus
import torch
from torch import nn
import distiller
from distiller.modules import DistillerLSTM as LSTM
from tqdm import tqdm # for pretty progress bar
import numpy as np
from copy import deepcopy


# ### Preprocess the data

# In[2]:


corpus = Corpus('./data_wikitext/data/wikitext-2/')


# In[3]:


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)
device = 'cuda:0'
batch_size = 20
eval_batch_size = 10
train_data = batchify(corpus.train, batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)


# ### Loading the model and converting to our own implementation

# In[4]:


rnn_model = torch.load('./logs/lstm/checkpoint.pth.tar.best')
rnn_model = rnn_model.to(device)
rnn_model


# Here we convert the pytorch LSTM implementation to our own, by calling `LSTM.from_pytorch_impl`:

# In[5]:


def manual_model(pytorch_model_: RNNModel):
    nlayers, ninp, nhid, ntoken, tie_weights =         pytorch_model_.nlayers,         pytorch_model_.ninp,         pytorch_model_.nhid,         pytorch_model_.ntoken,         pytorch_model_.tie_weights

    model = DistillerRNNModel(nlayers=nlayers, ninp=ninp, nhid=nhid, ntoken=ntoken, tie_weights=tie_weights).to(device)
    model.eval()
    model.encoder.weight = nn.Parameter(pytorch_model_.encoder.weight.clone().detach())
    model.decoder.weight = nn.Parameter(pytorch_model_.decoder.weight.clone().detach())
    model.decoder.bias = nn.Parameter(pytorch_model_.decoder.bias.clone().detach())
    model.rnn = LSTM.from_pytorch_impl(pytorch_model_.rnn)

    return model

man_model = manual_model(rnn_model)
torch.save(man_model, './logs/lstm/manual.checkpoint.pth.tar')
man_model


# ### Batching the data for evaluation

# In[6]:


sequence_len = 35
def get_batch(source, i):
    seq_len = min(sequence_len, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

hidden = rnn_model.init_hidden(eval_batch_size)
data, targets = get_batch(test_data, 0)


# ### Check that the convertion has succeeded

# In[7]:


rnn_model.eval()
man_model.eval()
y_t, h_t = rnn_model(data, hidden)
y_p, h_p = man_model(data, hidden)

print("Max error in y: %f" % (y_t-y_p).abs().max().item())


# ### Defining the evaluation

# In[8]:


criterion = nn.CrossEntropyLoss()
def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    

def evaluate(model, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        with tqdm(range(0, data_source.size(0), sequence_len)) as t:
            # The line below was fixed as per: https://github.com/pytorch/examples/issues/214
            for i in t:
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                output_flat = output.view(-1, ntokens)
                total_loss += len(data) * criterion(output_flat, targets).item()
                hidden = repackage_hidden(hidden)
                avg_loss = total_loss / (i + 1)
                t.set_postfix((('val_loss', avg_loss), ('ppl', np.exp(avg_loss))))
    return total_loss / len(data_source)


# # Quantizing the Model

# ## Collect activation statistics

# The model uses activation statistics to determine how big the quantization range is. The bigger the range - the larger the round off error after quantization which leads to accuracy drop.  
# Our goal is to minimize the range s.t. it contains the absolute most of our data.  
# After that, we divide the range into chunks of equal size, according to the number of bits, and transform the data according to this scale factor.  
# Read more on scale factor calculation [in our docs](https://nervanasystems.github.io/distiller/algo_quantization.html).
# 
# The class `QuantCalibrationStatsCollector` collects the statistics for defining the range $r = max - min$.  
# 
# Each forward pass, the collector records the values of inputs and outputs, for each layer:
# - absolute over all batches min, max (stored in `min`, `max`)
# - average over batches, per batch min, max (stored in `avg_min`, `avg_max`)
# - mean
# - std
# - shape of output tensor  
# 
# All these values can be used to define the range of quantization, e.g. we can use the absolute `min`, `max` to define the range.

# In[9]:


import os
from distiller.data_loggers import QuantCalibrationStatsCollector, collector_context

man_model = torch.load('./logs/lstm/manual.checkpoint.pth.tar')
distiller.utils.assign_layer_fq_names(man_model)
collector = QuantCalibrationStatsCollector(man_model)

if not os.path.isfile('/logs/lstm/manual_lstm_pretrained_stats.yaml'):
    with collector_context(collector) as collector:
        val_loss = evaluate(man_model, val_data)
        collector.save('./logs/lstm/manual_lstm_pretrained_stats.yaml')


# ## Prepare the Model For Quantization
#   
# We quantize the model after the training has completed.  
# Here we check the baseline model perplexity, to have an idea how good the quantization is.

# In[10]:


from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from copy import deepcopy

# Load and evaluate the baseline model.
man_model = torch.load('./logs/lstm/manual.checkpoint.pth.tar')
# val_loss = evaluate(man_model, val_data)
# print('val_loss:%8.2f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# Now we do our magic - __Preparing the model for quantization__.  
# The quantizer replaces the layers in out model with their quantized versions.  

# In[11]:


# Define the quantizer
# quantizer = PostTrainLinearQuantizer(
#     deepcopy(man_model),
#     model_activation_stats='./logs/lstm/manual_lstm_pretrained_stats.yaml')

# Quantizer magic
# stats_before_prepare = deepcopy(quantizer.model_activation_stats)
dummy_input = (torch.zeros(1,1).to(dtype=torch.long), man_model.init_hidden(1))
# quantizer.prepare_model(dummy_input)


# ### Net-Aware Quantization
# 
# Note that we passes a dummy input to `prepare_model`. This is required for the quantizer to be able to create a graph representation of the model, and to infer the connectivity between the modules.  
# Understanding the connectivity of the model is required to enable **"Net-aware quantization"**. This term (coined in [\[1\]](#references), section 3.2.2), means we can achieve better quantization by considering sequences of operations.  
# In the case of LSTM, we have an element-wise add operation whose output is split into 4 and fed into either Tanh or Sigmoid activations. Both of these ops saturate at relatively small input values - tanh at approximately $|4|$, and sigmoid saturates at approximately $|6|$. This means we can safely clip the output of the element-wise add operation between $[-6,6]$. `PostTrainLinearQuantizer` detects this patterm and modifies the statistics accordingly.

# In[12]:


# import pprint
# pp = pprint.PrettyPrinter(indent=1)
# print('Stats BEFORE prepare_model:')
# pp.pprint(stats_before_prepare['rnn.cells.0.eltwiseadd_gate']['output'])

# print('\nStats AFTER to prepare_model:')
# pp.pprint(quantizer.model_activation_stats['rnn.cells.0.eltwiseadd_gate']['output'])


# Note the value for `avg_max` did not change, since it was already below the clipping value of $6.0$.

# ### Inspecting the Quantized Model
# 
# Let's see how the model has after being prepared for quantization:

# In[13]:


# quantizer.model


# Note how `encoder` and `decoder` have been replaced with wrapper layers (for the relevant module type), which handle the quantization. The same holds for the internal layers of the `DistillerLSTM` module, which we don't print for brevity sake. To "peek" inside the `DistillerLSTM` module, we need to access it directly. As an example, let's take a look at a couple of the internal layers:

# In[14]:


# print(quantizer.model.rnn.cells[0].fc_gate_x)
# print(quantizer.model.rnn.cells[0].eltwiseadd_gate)


# In[15]:


# print(quantizer.model.encoder)


# ## Running the Quantized Model
# 
# ### Try 1: Initial settings - simple symmetric quantization
# 
# Finally, let's go ahead and evaluate the quantized model:

# In[16]:


# val_loss = evaluate(quantizer.model.to(device), val_data)
# print('val_loss:%8.2f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# As we can see here, the perplexity has increased much - meaning our quantization has damaged the accuracy of our model.

# ### Try 2: Assymetric, per-channel
# 
# Let's try quantizing each channel separately, and making the range of the quantization asymmetric.

# In[17]:


# quantizer = PostTrainLinearQuantizer(
#     deepcopy(man_model),
#     model_activation_stats='./manual_lstm_pretrained_stats.yaml',
#     mode=LinearQuantMode.ASYMMETRIC_SIGNED,
#     per_channel_wts=True
# )
# quantizer.prepare_model(dummy_input)
# quantizer.model


# In[18]:


# val_loss = evaluate(quantizer.model.to(device), val_data)
# print('val_loss:%8.2f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# A tiny bit better, but still no good.

# ### Try 3: Mixed FP16 and INT8
# 
# Let us try the half precision (aka FP16) version of the model:

# In[19]:


# model_fp16 = deepcopy(man_model).half()
# val_loss = evaluate(model_fp16, val_data)
# print('val_loss: %8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# The result is very close to our original model! That means that the roundoff when quantizing linearly to 8-bit integers is what hurts our accuracy.
# 
# Luckily, `PostTrainLinearQuantizer` supports quantizing some/all layers to FP16 using the `fp16` parameter. In light of what we just saw, and as stated in [\[2\]](#References), let's try keeping element-wise operations at FP16, and quantize everything else to 8-bit using the same settings as in try 2.

# In[20]:


# overrides_yaml = """
# .*eltwise.*:
#     fp16: true
# encoder:
#     fp16: true
# decoder:
#     fp16: true
# """
# overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
# quantizer = PostTrainLinearQuantizer(
#     deepcopy(man_model),
#     model_activation_stats='./manual_lstm_pretrained_stats.yaml',
#     mode=LinearQuantMode.ASYMMETRIC_SIGNED,
#     overrides=overrides,
#     per_channel_wts=True
# )
# quantizer.prepare_model(dummy_input)
# quantizer.model


# In[21]:


# val_loss = evaluate(quantizer.model.to(device), val_data)
# print('val_loss:%8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# The accuracy is still holding up very well, even though we quantized the inner linear layers!  

# ### Try 4: Clipping Activations
# 
# Now, lets try to choose different boundaries for `min`, `max`.  
# Instead of using absolute ones, we take the average of all batches (`avg_min`, `avg_max`), which is an indication of where usually most of the boundaries lie. This is done by specifying the `clip_acts` parameter to `ClipMode.AVG` or `"AVG"` in the quantizer ctor:

# In[22]:


# overrides_yaml = """
# encoder:
#     fp16: true
# decoder:
#     fp16: true
# """
# overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
# quantizer = PostTrainLinearQuantizer(
#     deepcopy(man_model),
#     model_activation_stats='./manual_lstm_pretrained_stats.yaml',
#     mode=LinearQuantMode.ASYMMETRIC_SIGNED,
#     overrides=overrides,
#     per_channel_wts=True,
#     clip_acts="AVG"
# )
# quantizer.prepare_model(dummy_input)
# val_loss = evaluate(quantizer.model.to(device), val_data)
# print('val_loss:%8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# Great! Even though we quantized all of the layers except the embedding and the decoder - we got almost no accuracy penalty. Lets try quantizing them as well:

# In[23]:


quantizer = PostTrainLinearQuantizer(
    deepcopy(man_model),
    model_activation_stats='./logs/lstm/manual_lstm_pretrained_stats.yaml',
    mode=LinearQuantMode.ASYMMETRIC_SIGNED,
    per_channel_wts=True,
    clip_acts="AVG"
)
quantizer.prepare_model(dummy_input)
# val_loss = evaluate(quantizer.model.to(device), val_data)
# print('val_loss:%8.6f\t|\t ppl:%8.2f' % (val_loss, np.exp(val_loss)))


# # fault injection experiments

# In[24]:



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

class Args(object):
    pass 
args = Args() 
args.no_cuda = False 
args.seed = 1 
args.save='./logs'
torch.manual_seed(args.seed)

args.cuda = not args.no_cuda and torch.cuda.is_available()
print('using GPU:', args.cuda)


# In[25]:


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
def write_detailed_info(log_path, info):
    with open(os.path.join(log_path, 'logs.txt'), 'a') as f:
        f.write(info+'\n')


# In[26]:


args.model_name = 'lstm'
args.dataset = 'wikitext'
args.data_type = 'int8'
args.fault_type = 'faults_network_rb_ecc'

args.save = os.path.join(args.save, args.model_name, args.dataset, args.data_type, args.fault_type) 
if os.path.exists(args.save):
#     shutil.rmtree(args.save)
    print('path already exist! remove path:', args.save)
else:
    os.makedirs(args.save)
print('log will save to:', args.save)
print(args.__dict__)

def select_fault_injection_function():
    fn = {
          'int8': {
              'faults_network_rb': inject_faults_int8_random_bit_position, 
              'faults_network_rb_ps1': inject_faults_int8_random_bit_position_ps1,
              'faults_network_rb_parity_zero': inject_faults_int8_random_bit_position_parity_zero,
              'faults_network_rb_parity_avg': inject_faults_int8_random_bit_position_parity_avg,
              'faults_network_rb_ecc': inject_faults_int8_random_bit_position_ecc,
              'faults_network_rb_bch': inject_faults_int8_random_bit_position_bch,
          }
         }
    return fn[args.data_type][args.fault_type]  
fault_injection_fn = select_fault_injection_function()


# In[27]:


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
weights, weights_names = get_weights(quantizer.model)
# weights, weights_names = get_weights(model)
weights_sizes = [param.nelement() for param in weights]
total_values = sum(weights_sizes)
print('# weights params:', len(weights), ', total_values:', total_values)
for i, item in enumerate(zip(weights_names, weights_sizes)):
    print('\t', i, item[0], item[1], '(%f)' %(item[1]/total_values))


# In[28]:


def perturb_weights(model, n_faults, trial_id, log_path, fault_injection_fn): 
    # use trial_id to setup random seed 
    start = time.time()
    np.random.seed(trial_id)
    random = np.random  
    flipped_bits, changed_params, stats = 0, 0, {}
    
    # get the n_bits for each weight 
    weights, _ = get_weights(model)
    weights_sizes = [param.nelement() for param in weights]
    total_values = sum(weights_sizes)
    p = [size/total_values for size in weights_sizes]
    samples = random.choice(len(weights), size=n_faults, p=p)
    counter = collections.Counter(samples)
    
    print('samples:', sorted(counter.items())) 
    
    for weight_id in sorted(counter.keys()):
        param = weights[weight_id]
        tensor = param.data.view(-1)
#         tensor_copy = tensor.clone() 
        
        # flip n_bits number of values from tensor
        n_bits = counter[weight_id]
        res = fault_injection_fn(tensor, random, n_bits)
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

        


# In[30]:


##########################
## start simulation ######
##########################
args.start_trial_id = 0
args.end_trial_id = 10
# for each fault_rate, use fault rate to get the number of faults
print('\nSimulation start: ', datetime.now())
simulation_start = time.time()
# fault_rates = [10**x for x in range(-8, -2, 1)]
fault_rates = [10**-4] 
for fault_rate in fault_rates:
    
    n_faults = int(total_values * 8 * fault_rate)
    if n_faults <= 0: 
        continue
    
    folder = 'r%s' %(fault_rate)
    log_path = os.path.join(args.save, folder)
    check_directory(log_path)
    
    # for each trial, initialize the model  
    for trial_id in range(args.start_trial_id, args.end_trial_id):
        print('\nfault_rate:', fault_rate, ', n_faults:', n_faults, ', trial_id:', trial_id)
        start = time.time()
        quantized_model = deepcopy(quantizer.model)
        quantized_model.cpu()
        
        info = perturb_weights(quantized_model, n_faults, trial_id, log_path, fault_injection_fn) 
        val_loss = evaluate(quantized_model.to(device), val_data)
        ppl = np.exp(val_loss)
        duration = time.time() - start  

        info += ', test_time(s): %d' %(duration)
        info += ', test_loss: %f, perplexity: %f' %(val_loss, ppl)
        print(info, '\n')
        write_detailed_info(log_path, info)
        
        
#         break 
#     break 
simulation_time = time.time() - simulation_start
print('Simulation ends:', datetime.now(), ', duration(s):%.2f' %(simulation_time)) 


# Here we see that sometimes quantizing with the right boundaries gives better results than actually using floating point operations (even though they are half precision). 

# ## Conclusion
# 
# Choosing the right boundaries for quantization  was crucial for achieving almost no degradation in accrucay of LSTM.  
#   
# Here we showed how to use the Distiller quantization API to quantize an RNN model, by converting the PyTorch implementation into a modular one and then quantizing each layer separately.

# ## References
# 
# 1. **Jongsoo Park, Maxim Naumov, Protonu Basu, Summer Deng, Aravind Kalaiah, Daya Khudia, James Law, Parth Malani, Andrey Malevich, Satish Nadathur, Juan Miguel Pino, Martin Schatz, Alexander Sidorov, Viswanath Sivakumar, Andrew Tulloch, Xiaodong Wang, Yiming Wu, Hector Yuen, Utku Diril, Dmytro Dzhulgakov, Kim Hazelwood, Bill Jia, Yangqing Jia, Lin Qiao, Vijay Rao, Nadav Rotem, Sungjoo Yoo, Mikhail Smelyanskiy**. Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications. [arxiv:1811.09886](https://arxiv.org/abs/1811.09886)
# 
# 2. **Qinyao He, He Wen, Shuchang Zhou, Yuxin Wu, Cong Yao, Xinyu Zhou, Yuheng Zou**. Effective Quantization Methods for Recurrent Neural Networks. [arxiv:1611.10176](https://arxiv.org/abs/1611.10176)
