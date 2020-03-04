
# STD LIB
import os
import pdb
import sys
import glob
import time
import logging
import pathlib
import functools

# EXTERNAL LIB
import torch
import torch.nn as nn

from PIL import Image

import cv2
import flowiz
import numpy as np
#from torchsummary import summary

# LOCAL LIB

# CONSTANTS
VGG_MEAN = [103.939, 116.779, 123.68]
OUTPUT_FORMAT = 'out-%05d.png'
LOGFILE = 'thesis.log'
LOGFORMAT = '%(asctime)s %(name)s:%(levelname)s -- %(message)s'

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
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return functools.reduce(self.lambda_func,self.forward_prepare(input))

def get():
    model = nn.Sequential( # Sequential,
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
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.InstanceNorm2d(128, affine=True),
        nn.ReLU(),
        nn.Conv2d(128,64,(3, 3),(1, 1),(1, 1)),
        nn.InstanceNorm2d(64, affine=True),
        nn.ReLU(),
        nn.UpsamplingNearest2d(scale_factor=2),
        nn.InstanceNorm2d(64, affine=True),
        nn.ReLU(),
        nn.Conv2d(64,3,(9, 9),(1, 1),(4, 4)),
        nn.Tanh(),
        Lambda(lambda x: x * 150),
        Lambda(lambda x: x), # nn.TotalVariation,
    )

    model.eval()
    return model

def set(model, weightfile):
    model.load_state_dict(torch.load(weightfile))

def min_filter(img):
    net = nn.Sequential(
        Lambda(lambda x: x * -1),
        Lambda(lambda x: x + 1),
        nn.MaxPool2d((7, 7), (1, 1), (3, 3)),
        Lambda(lambda x: x * -1),
        Lambda(lambda x: x + 1)
    )
    return net.forward(img)

def preprocess(img):
    # in: (h, w, 3)
    # out: (1, 3, h, w)
    assert(len(img.shape) == 3)
    assert(img.shape[2] == 3)
    
    # Swap RGB to BGR (this appears unnecessary)
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
    
    # Swap BGR to RGB (this appears unnecessary)
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

def run_image(model, img):
    start = time.time()
    # Preprocess the current input image
    pre = preprocess(img)
    # All optical flow fields are blank for the first image
    blanks = torch.zeros((1, 4, pre.shape[-2], pre.shape[-1]))
    # Concatenate everything into a tensor of shape (1, 7, height, width)
    tmp = torch.cat((pre, blanks), dim=1)
    # Run the tensor through the model
    out = model.forward(tmp)
    # Deprocess and return the result
    dep = deprocess(out)
    logging.info('Elapsed time for stylizing frame independently: {}'.format(round(time.time() - start, 3)))
    return dep

def run_next_image(model, img, prev, flow, cert):
    start = time.time()
    # Warp the previous output with the optical flow between the new image and previous image
    prev_warped_pre = warp(prev, flow)
    # Apply preprocessing to the warped image
    prev_warped = preprocess(prev_warped_pre)
    # Mask the warped image with the consistency check
    prev_warped_masked = prev_warped * torch.FloatTensor(cert).expand_as(prev_warped)
    # Preprocess the current input image
    pre = preprocess(img)
    # Concatenate everything into a tensor of shape (1, 7, height, width)
    tmp = torch.cat((pre, prev_warped_masked, cert.unsqueeze(0)), dim=1)
    # Run the tensor through the model
    out = model.forward(tmp)
    # Deprocess and return the result
    dep = deprocess(out)
    logging.info('Elapsed time for stylizing frame: {}'.format(round(time.time() - start, 3)))
    return dep

def main():
    logging.basicConfig(filename=LOGFILE, filemode='a', format=LOGFORMAT, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    model = get()
    set(model, 'styles/schlief.pth')

    #input_size=(7, 180, 180)
    #summary(model, input_size)
    
    data = pathlib.Path('data')
    ppms = [str(data / name) for name in glob.glob1(str(data), '*.ppm')]

    for idx, ppm in enumerate(ppms):
        if idx == 0:
            img = cv2.imread(ppms[idx])
            out = run_image(model, img)
        else:
            flowfile = str(data / 'backward_{}_{}.flo'.format(idx + 1, idx))
            certfile = str(data / 'reliable_{}_{}.pgm'.format(idx + 1, idx))
            assert(os.path.exists(flowfile))
            assert(os.path.exists(certfile))
            flow = flowiz.read_flow(flowfile)
            cert = torch.FloatTensor(np.asarray(Image.open(certfile)) / 255).unsqueeze(0)
            pre_cert = min_filter(cert)
            img = cv2.imread(ppms[idx])
            out = run_next_image(model, img, out, flow, pre_cert)
        
        cv2.imwrite(OUTPUT_FORMAT % (idx + 1), out)

if __name__ == '__main__':
    main()