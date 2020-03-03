
# STD LIB
import os
import pdb
import sys
import glob
import pathlib

# EXTERNAL LIB
import torch
import torch.nn as nn

from PIL import Image

import cv2
import flowiz
import numpy as np
from functools import reduce
from torchsummary import summary

# LOCAL LIB

# CONSTANTS
VGG_MEAN = [103.939, 116.779, 123.68]
OUTPUT_FORMAT = 'out-%05d.png'

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
        return reduce(self.lambda_func,self.forward_prepare(input))

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
    
    # Swap RGB to BGR
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Swap axes
    tmp = np.swapaxes(bgr, 0, 2)
    
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
    #rgb = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    
    return tmp

def run_image(model, img):
    height, width = img.shape[0], img.shape[1]
    pre = preprocess(img)
    # All optical flow fields are blank for the first image
    blanks = torch.zeros((1, 4, pre.shape[-2], pre.shape[-1]))
    tmp = torch.cat((pre, blanks), dim=1)
    out = model.forward(tmp)
    return deprocess(out)

def warp_frame(img, flow):
    """Warp an image or feature map with optical flow
    Args:
        img (Tensor): size (n, c, h, w)
        flow (Tensor): size (n, 2, h, w), values range from -1 to 1 (relevant to image width or height)

    Returns:
        Tensor: warped image or feature map
    """
    # This function is WRONG.
    assert img.size()[-2:] == flow.size()[-2:]
    n, _, h, w = img.size()
    x_ = torch.arange(w).view(1, -1).expand(h, -1)
    y_ = torch.arange(h).view(-1, 1).expand(-1, w)
    grid = torch.stack([x_, y_], dim=0).float()
    grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
    grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1
    grid += 2 * flow
    grid = grid.permute(0, 2, 3, 1)
    out = torch.nn.functional.grid_sample(img, grid, padding_mode='zeros')
    return out

def run_next_image(model, img, prev, flow, cert, idx):    
    # Apply some preprocessing before applying optical flow warp
    usq = torch.FloatTensor(np.swapaxes(prev, 0, 2)).unsqueeze(0)
    usf = torch.FloatTensor(np.swapaxes(flow, 0, 2)).unsqueeze(0)
    
    # Deprocess output before preprocessing, again
    # prev_warped_pre = warp_frame(usq, usf)
    prev_warped_pre = cv2.imread('flow_{}.png'.format(idx))
    #cv2.imwrite('{}_flow.png'.format('out-%05d' % idx), 
    #    np.swapaxes(torch.squeeze(prev_warped_pre).detach().numpy(), 0, 2))
    prev_warped = preprocess(prev_warped_pre)
    prev_warped_masked = prev_warped * torch.FloatTensor(cert).expand_as(prev_warped)

    #prev_warped_masked = torch.zeros(prev_warped_masked.shape)
    #cert = torch.zeros(cert.shape)
        
    pre = preprocess(img)
    tmp = torch.cat((pre, prev_warped, cert.unsqueeze(0)), dim=1)
    out = model.forward(tmp)
    return deprocess(out)

def main():
    model = get()
    set(model, 'styles/candy.pth')

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
            print('{} + {} + {} -> {}'.format(
                flowfile, certfile, OUTPUT_FORMAT % idx, OUTPUT_FORMAT % (idx + 1)))
            flow = flowiz.read_flow(flowfile)
            cert = torch.FloatTensor(np.asarray(Image.open(certfile)) / 255).unsqueeze(0)
            pre_cert = min_filter(cert)
            img = cv2.imread(ppms[idx])
            out = run_next_image(model, img, out, flow, pre_cert, idx + 1)
        cv2.imwrite(OUTPUT_FORMAT % (idx + 1), out)

if __name__ == '__main__':
    main()