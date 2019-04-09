#!/home/hguan2/anaconda2/envs/torch/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 23:02:49 2019

@author: hguan2
"""

# import distiller 
import numpy as np
import os
import bitstring 
import time 
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F

import models 
from matplotlib import pyplot as plt


print('using GPU:', torch.cuda.is_available())