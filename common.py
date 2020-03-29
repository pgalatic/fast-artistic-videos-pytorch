
# STD LIB
import os
import time
import logging

# EXTERNAL LIB
import torch

import cv2
import numpy as np

# LOCAL LIB
from const import *

def preprocess(img):
    # in: (h, w, 3)
    # out: (1, 3, h, w)
    assert(len(img.shape) == 3)
    assert(img.shape[2] == 3)
    
    # Swap RGB to BGR (this appears unnecessary as cv2 is already BGR)
    #bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Swap axes
    tmp = np.swapaxes(img, 0, 2)
    
    # Unsqueeze
    usq = torch.Tensor(tmp).unsqueeze(0)
    
    # Subtract mean
    mean = torch.FloatTensor(VGG_MEAN).view((1, 3, 1, 1)).expand_as(usq)
    sub = usq - mean
    
    return sub

def deprocess(img):
    # in: (1, 3, h, w)
    # out: (h, w, 3)
    assert(len(img.shape) == 4)
    assert(img.shape[1] == 3)
    
    # Add mean
    mean = torch.FloatTensor(VGG_MEAN).view((1, 3, 1, 1)).expand_as(img)
    add = img + mean
    # Squeeze
    sqz = torch.squeeze(add).detach().numpy()
    
    # Swap axes
    tmp = np.swapaxes(sqz, 0, 2)
    
    # Swap BGR to RGB (this appears unnecessary in general)
    # rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    
    return tmp

def warp(img, flow):
    '''
    Warp an image or feature map with optical flow
    Args:
        img (np.ndarray): shape (3, h, w), values range from 0 to 255
        flow (np.ndarray): shape (2, h, w), values range from -1 to 1
    Returns:
        Tensor: warped image or feature map
    '''
    height, width = flow.shape[:2]
    flow[:, :, 0] += np.arange(width)
    flow[:, :, 1] += np.arange(height)[:, np.newaxis]
    out = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return out

def wait_for(fname):
    '''
    given:
        fname -> (str) a filename
    
    halts the program until that file exists. NOTE: the file might not yet be complete, and other
        measures are used to account for that
    '''
    # If you wish upon a star...
    logging.debug('Waiting for {}...'.format(fname))
    while not os.path.exists(fname):
        time.sleep(1)
    logging.debug('...{} found!'.format(fname))
    return fname