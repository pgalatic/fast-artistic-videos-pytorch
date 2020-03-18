# author: Paul Galatic
#
# FIXME: cv2.remap() is probably not equivalent to lua_torch:image:warp()


# STD LIB
import re
import os
import pdb
import time
import logging
import pathlib
import functools

# EXTERNAL LIB
import torch
import torch.nn as nn

import cv2
import flowiz
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

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func, self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return functools.reduce(self.lambda_func, self.forward_prepare(input))
        
class Interpolate(nn.Module):
    '''
    As UpsamplingNearest2d is deprecated, yet still important to include as part of the sequential model,
    this class serves as a workaround.
    '''
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x):
        return nn.functional.interpolate(x, 
            size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class StylizationModel():
    def __init__(self, weights_fname=None):
    
        # Main stylization network
        self.model = nn.Sequential( # Sequential,
            nn.ReflectionPad2d((40, 40, 40, 40)),
            nn.Conv2d(7,32,(9, 9),(1, 1),(4, 4)),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                    ),
                    nn.ConstantPad2d(-2, 0),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                    ),
                    nn.ConstantPad2d(-2, 0),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                    ),
                    nn.ConstantPad2d(-2, 0),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                    ),
                    nn.ConstantPad2d(-2, 0),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
            nn.Sequential( # Sequential,
                LambdaMap(lambda x: x, # ConcatTable,
                    nn.Sequential( # Sequential,
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                        nn.ReLU(),
                        nn.Conv2d(128,128,(3, 3)),
                        nn.InstanceNorm2d(128, affine=True),
                    ),
                    nn.ConstantPad2d(-2, 0),
                ),
                LambdaReduce(lambda x,y: x+y), # CAddTable,
            ),
            Interpolate(scale_factor=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.Conv2d(128,64,(3, 3),(1, 1),(1, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            Interpolate(scale_factor=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            nn.Conv2d(64,3,(9, 9),(1, 1),(4, 4)),
            nn.Tanh(),
            Lambda(lambda x: x * 150),
            Lambda(lambda x: x), # nn.TotalVariation,
        )
        
        # Min filter used to limit signal of consistency check
        self.min_filter = nn.Sequential(
            Lambda(lambda x: x * -1),
            Lambda(lambda x: x + 1),
            nn.MaxPool2d((7, 7), (1, 1), (3, 3)),
            Lambda(lambda x: x * -1),
            Lambda(lambda x: x + 1)
        )
        
        self.model.eval()
        self.min_filter.eval()
        
        if weights_fname:
            self.set_fname(weights_fname)
    
    def run_image(self, img):
        start = time.time()
        # Preprocess the current input image
        pre = preprocess(img)
        # All optical flow fields are blank for the first image
        blanks = torch.zeros((1, 4, pre.shape[-2], pre.shape[-1]))
        # Concatenate everything into a tensor of shape (1, 7, height, width)
        tmp = torch.cat((pre, blanks), dim=1)
        # Run the tensor through the model
        out = self.model.forward(tmp)
        # Deprocess and return the result
        dep = deprocess(out)
        logging.info(
            'Elapsed time for stylizing frame independently: {}'.format(round(time.time() - start, 3)))
        return dep
        
    def run_next_image(self, img, prev, flow, cert):
        start = time.time()
        # Consistency check preprocessing: Apply min filter and swap axes
        pre_cert = self.min_filter.forward(torch.FloatTensor(np.swapaxes(cert / 255, 0, 1)).unsqueeze(0).unsqueeze(0))
        # Warp the previous output with the optical flow between the new image and previous image
        prev_warped = warp(prev, flow)
        # Apply preprocessing to the warped image
        prev_warped_pre = preprocess(prev_warped)
        # Mask the warped image with the consistency check
        prev_warped_masked = prev_warped_pre * pre_cert.expand_as(prev_warped_pre)
        # Preprocess the current input image
        pre = preprocess(img)
        # Concatenate everything into a tensor of shape (1, 7, height, width)
        tmp = torch.cat((pre, prev_warped_masked, pre_cert), dim=1)
        # Run the tensor through the model
        out = self.model.forward(tmp)
        # Deprocess and return the result
        dep = deprocess(out)
        logging.info(
            'Elapsed time for stylizing frame: {}'.format(round(time.time() - start, 3)))
        # round output to prevent drifting over time
        return np.round(dep)

    def set_fname(self, weights_fname):
        self.model.load_state_dict(torch.load(weights_fname))
    
    def set_weights(self, weights):
        # Assumes torch.load() has already been called
        self.model.load_state_dict(weights)

    def stylize(self, framefiles, flowfiles, certfiles, out_dir='.', out_format=OUTPUT_FORMAT):
        # Flowfiles and certfiles lists must have a None at the start, which is skipped
        for idx, (framefile, flowfile, certfile) in enumerate(zip(framefiles, flowfiles, certfiles)):
            # img shape is (h, w, 3), range is [0-255], uint8
            assert(os.path.exists(framefile))
            img = cv2.imread(framefile)
            
            if idx == 0:
                # Independent style transfer is equivalent to Fast Neural Style by Johnson et al.
                out = self.run_image(img)
            else:
                assert(os.path.exists(flowfile))
                assert(os.path.exists(certfile))
                logging.debug('Using: {} {}'.format(flowfile, certfile))
                # flow shape is (h, w, 2), range is [0-1], float32
                flow = flowiz.read_flow(flowfile)
                # cert shape is (h, w, 1), range is [0 | 255], float32
                cert = cv2.imread(certfile, cv2.IMREAD_UNCHANGED)
                # out shape is (h, w, 3), range is [0-255], float32
                out = self.run_next_image(img, out, flow, cert)
            
            idy = int(re.findall(r'\d+', os.path.basename(framefile))[0])
            out_fname = str(pathlib.Path(out_dir) / (OUTPUT_FORMAT % (idy)))
            logging.info('Writing to {}...'.format(out_fname))
            cv2.imwrite(out_fname, out)
            