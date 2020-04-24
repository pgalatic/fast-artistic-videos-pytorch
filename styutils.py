
# STD LIB
import os
import sys
import glob
import time
import logging

# EXTERNAL LIB
import torch

import cv2
import numpy as np

# LOCAL LIB
try:
    from sconst import *
except:
    from .sconst import *

def start_logging():
    '''
    Begins Python's logging capabilities in a default configuration. Logs are appended to a 
    logfile, and all logs are also printed to console.
    '''
    logging.basicConfig(filename=LOGFILE, filemode='a', format=LOGFORMAT, level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    def my_handler(type, value, tb):
        logger.exception("Uncaught exception: {0}".format(str(value)))
    sys.excepthook = my_handler

def preprocess(img):
    # in: (h, w, 3)
    # out: (1, 3, h, w)
    assert(len(img.shape) == 3)
    assert(img.shape[2] == 3)
    
    # Swap RGB to BGR (this appears unnecessary as cv2 is already BGR)
    #bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Swap axes
    tmp = np.swapaxes(np.swapaxes(img, 0, 1), 0, 2)
    
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
    tmp = np.swapaxes(np.swapaxes(sqz, 0, 2), 0, 1)
    
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
    # Don't do in-place modifications!
    flow_copy = np.copy(flow)
    flow_copy[:, :, 0] += np.arange(width)
    flow_copy[:, :, 1] += np.arange(height)[:, np.newaxis]
    out = cv2.remap(img, flow_copy, None, cv2.INTER_LINEAR)
    return out

def count_files(dir, extension):
    '''
    given:
        dir         -> (pathlib.Path) a directory
        extension   -> (str) an extention (.png, .ppm, etc)
    
    returns:
        (int) the number of files in dir that end in that extention
    '''
    return len(glob.glob1(str(dir), '*{}'.format(extension)))

def makedirs(dirname):
    '''
    given:
        dirname -> (pathlib.Path) the path to a directory
    
    attempts to create the directory, accounting for various situations when such an attempt will 
        fail
    '''
    # Convert from pathlib.Path to string if necessary.
    dirname = str(dirname)
    if not os.path.isdir(dirname):
        try:
            os.makedirs(dirname)
        except FileExistsError:
            pass # The directory was created just before we tried to create it.
        except PermissionError:
            logging.warning('Directory {} creation failed! Are you sure that the common folder is accessible/mounted?'.format(dirname))
            raise

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