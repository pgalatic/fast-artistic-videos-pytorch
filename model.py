
# STD LIB
import pdb

# EXTERNAL LIB
import torch
import torch.nn as nn

import cv2
import numpy as np
from functools import reduce
from torchsummary import summary

# LOCAL LIB

# CONSTANTS
VGG_MEAN = [103.939, 116.779, 123.68]

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

def main():
    model = get()
    set(model, 'styles/checkpoint_candy_video.pth')

    input_size=(7, 180, 180)
    #summary(model, input_size)

    img = cv2.imread('data/frame_00001.ppm')
    out = run_image(model, img)

    cv2.imwrite('out.png', out)

if __name__ == '__main__':
    main()