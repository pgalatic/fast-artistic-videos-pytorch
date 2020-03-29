# author: Paul Galatic
#

# STD LIB
import re
import os
import pdb
import time
import logging
import functools
import threading

# EXTERNAL LIB
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import flowiz
import numpy as np

# LOCAL LIB
try:
    import loss
    import common
    import optflow
    from const import *
except:
    from . import loss
    from . import common
    from . import optflow
    from .const import *

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
        return F.interpolate(x, 
            size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class StylizationModel():
    def __init__(self, weights_fname=None, style_fname=None):
    
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
        
        self.eval = False
        if weights_fname:
            self.set_fname(weights_fname)
        if style_fname:
            assert(os.path.exists(style_fname))
            self.style_fname = style_fname
            self.eval = True
    
    def run_image(self, img):
        start = time.time()
        # Preprocess the current input image
        pre = common.preprocess(img)
        # All optical flow fields are blank for the first image
        blanks = torch.zeros((1, 4, pre.shape[-2], pre.shape[-1]))
        # Concatenate everything into a tensor of shape (1, 7, height, width)
        tmp = torch.cat((pre, blanks), dim=1)
        # Run the tensor through the model
        out = self.model.forward(tmp)
        # Deprocess and return the result
        dep = common.deprocess(out)
        logging.info(
            'Elapsed time for stylizing frame independently: {}'.format(round(time.time() - start, 3)))
        return dep
        
    def run_next_image(self, img, prev, flow, cert):
        start = time.time()
        # Preprocess the current input image
        pre = common.preprocess(img)
        # Consistency check preprocessing: Apply min filter and swap axes
        pre_cert = self.min_filter.forward(torch.FloatTensor(np.swapaxes(cert / 255, 0, 1)).unsqueeze(0).unsqueeze(0))
        # Warp the previous output with the optical flow between the new image and previous image
        prev_warped = common.warp(prev, flow)
        # Apply preprocessing to the warped image
        prev_warped_pre = common.preprocess(prev_warped)
        # Mask the warped image with the consistency check
        prev_warped_masked = prev_warped_pre * pre_cert.expand_as(prev_warped_pre)
        # Concatenate everything into a tensor of shape (1, 7, height, width)
        tmp = torch.cat((pre, prev_warped_masked, pre_cert), dim=1)
        # Run the tensor through the model
        out = self.model.forward(tmp)
        # Deprocess and return the result
        dep = common.deprocess(out)
        logging.info(
            'Elapsed time for stylizing frame: {}'.format(round(time.time() - start, 3)))
        # round output to prevent drifting over time
        return np.round(dep)

    def set_fname(self, weights_fname):
        self.model.load_state_dict(torch.load(weights_fname))
        logging.info('...{} loaded.'.format(weights_fname))
    
    def set_weights(self, weights):
        # Assumes torch.load() has already been called
        self.model.load_state_dict(weights)
        logging.info('...Weights loaded.')

    def stylize(self, start, frames, remote):
        crit = None
        if self.eval: crit = loss.StyleTransferVideoLoss(self.style_fname)
        threading.Thread(target=optflow.optflow, args=(start, frames, remote)).start()
        # Flowfiles and certfiles lists must have a None at the start, which is skipped
        for idx, fname in enumerate(frames):
            # img shape is (h, w, 3), range is [0-255], uint8
            img = cv2.imread(fname)
            
            if idx == 0:
                # Independent style transfer is equivalent to Fast Neural Style by Johnson et al.
                out = self.run_image(img)
                if self.eval: crit.eval(img, out, None)
            else:
                flowname = str(remote / 'backward_{}_{}.flo'.format(idx + start + 1, idx + start))
                certname = str(remote / 'reliable_{}_{}.pgm'.format(idx + start + 1, idx + start))
                # flow shape is (h, w, 2)
                flow = flowiz.read_flow(common.wait_for(flowname))
                # cert shape is (h, w, 1)
                cert = cv2.imread(common.wait_for(certname), cv2.IMREAD_UNCHANGED)
                # out shape is (h, w, 3)
                out = self.run_next_image(img, out, flow, cert)
                if self.eval: crit.eval(img, out, (pout, flow, cert))
                # Remove unnecessary files to save space.
                os.remove(flowname)
                os.remove(certname)
            os.remove(fname)
            
            idy = int(re.findall(r'\d+', os.path.basename(fname))[0])
            out_fname = str(remote / (OUTPUT_FORMAT % (idy)))
            logging.info('Writing to {}...'.format(out_fname))
            cv2.imwrite(out_fname, out)
            pout = out
            