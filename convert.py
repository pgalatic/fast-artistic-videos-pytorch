from __future__ import print_function

import os
import pdb
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.legacy.nn as lnn
import torch.nn.functional as F

from functools import reduce
from torch.autograd import Variable
from torch.utils.serialization import load_lua

MUL_CONSTANT = 150

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
        # result is Variables list [Variable1, Variable2, ...]
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func,self.forward_prepare(input))


def copy_param(old,new):
    if old.weight is not None: new.weight.data.copy_(old.weight)
    if old.bias is not None: new.bias.data.copy_(old.bias)
    pdb.set_trace()
    try: 
        new.running_mean.copy_(old.running_mean)
    except AttributeError:
        pass
    try:
        new.running_var.copy_(old.running_var)
    except AttributeError:
        pass

def add_submodule(seq, *args):
    for new in args:
        seq.add_module(str(len(seq._modules)),new)

def lua_recursive_model(module,seq):
    for old in module.modules:
        name = type(old).__name__
        real = old
        if name == 'TorchObject':
            name = old._typename.replace('cudnn.','')
            old = old._obj

        if name == 'SpatialConvolution' or name == 'nn.SpatialConvolutionMM':
            if not hasattr(old,'groups') or old.groups is None: old.groups=1
            new = nn.Conv2d(old.nInputPlane,old.nOutputPlane,(old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),1,old.groups,bias=(old.bias is not None))
            copy_param(old,new)
            add_submodule(seq,new)
        elif name == 'SpatialBatchNormalization':
            new = nn.BatchNorm2d(old.running_mean.size(0), old.eps, old.momentum, old.affine)
            copy_param(old,new)
            add_submodule(seq,new)
        elif name == 'VolumetricBatchNormalization':
            new = nn.BatchNorm3d(old.running_mean.size(0), old.eps, old.momentum, old.affine)
            copy_param(old, new)
            add_submodule(seq, new)
        elif name == 'ReLU':
            new = nn.ReLU()
            add_submodule(seq,new)
        elif name == 'Tanh':
            new = nn.Tanh()
            add_submodule(seq, new)
        elif name == 'Sigmoid':
            new = nn.Sigmoid()
            add_submodule(seq,new)
        elif name == 'SpatialMaxPooling':
            new = nn.MaxPool2d((old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),ceil_mode=old.ceil_mode)
            add_submodule(seq,new)
        elif name == 'SpatialAveragePooling':
            new = nn.AvgPool2d((old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),ceil_mode=old.ceil_mode)
            add_submodule(seq,new)
        elif name == 'SpatialUpSamplingNearest':
            new = nn.UpsamplingNearest2d(scale_factor=old.scale_factor)
            add_submodule(seq,new)
        elif name == 'View':
            new = Lambda(lambda x: x.view(x.size(0),-1))
            add_submodule(seq,new)
        elif name == 'Reshape':
            new = Lambda(lambda x: x.view(x.size(0),-1))
            add_submodule(seq,new)
        elif name == 'Linear':
            # Linear in pytorch only accept 2D input
            n1 = Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x )
            n2 = nn.Linear(old.weight.size(1),old.weight.size(0),bias=(old.bias is not None))
            copy_param(old,n2)
            new = nn.Sequential(n1,n2)
            add_submodule(seq,new)
        elif name == 'Dropout':
            old.inplace = False
            new = nn.Dropout(old.p)
            add_submodule(seq,new)
        elif name == 'SoftMax':
            new = nn.Softmax()
            add_submodule(seq,new)
        elif name == 'MulConstant':
            new = Lambda(lambda x: x * MUL_CONSTANT)
            add_submodule(seq, new)
        elif name == 'Identity' or name == 'nn.TotalVariation':
            new = Lambda(lambda x: x) # do nothing
            add_submodule(seq,new)
        elif name == 'nn.ShaveImage' or name == 'SpatialZeroPadding':
            new = nn.ConstantPad2d(-old.size, 0)
            add_submodule(seq, new)
        elif name == 'SpatialFullConvolution':
            new = nn.ConvTranspose2d(old.nInputPlane,old.nOutputPlane,(old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),(old.adjW,old.adjH))
            copy_param(old,new)
            add_submodule(seq,new)
        elif name == 'VolumetricFullConvolution':
            new = nn.ConvTranspose3d(old.nInputPlane,old.nOutputPlane,(old.kT,old.kW,old.kH),(old.dT,old.dW,old.dH),(old.padT,old.padW,old.padH),(old.adjT,old.adjW,old.adjH),old.groups)
            copy_param(old,new)
            add_submodule(seq, new)
        elif name == 'SpatialReplicationPadding':
            new = nn.ReplicationPad2d((old.pad_l,old.pad_r,old.pad_t,old.pad_b))
            add_submodule(seq,new)
        elif name == 'SpatialReflectionPadding':
            new = nn.ReflectionPad2d((old.pad_l,old.pad_r,old.pad_t,old.pad_b))
            add_submodule(seq,new)
        elif name == 'nn.InstanceNormalization':
            new = nn.InstanceNorm2d(old.nOutput, affine=True)
            copy_param(old, new)
            add_submodule(seq, new)
        elif name == 'Copy':
            new = Lambda(lambda x: x) # do nothing
            add_submodule(seq,new)
        elif name == 'Narrow':
            new = Lambda(lambda x,a=(old.dimension,old.index,old.length): x.narrow(*a))
            add_submodule(seq,new)
        elif name == 'SpatialCrossMapLRN':
            lrn = lnn.SpatialCrossMapLRN(old.size,old.alpha,old.beta,old.k)
            new = Lambda(lambda x,lrn=lrn: Variable(lrn.forward(x.data)))
            add_submodule(seq,new)
        elif name == 'Sequential':
            new = nn.Sequential()
            lua_recursive_model(old,new)
            add_submodule(seq,new)
        elif name == 'ConcatTable': # output is list
            new = LambdaMap(lambda x: x)
            lua_recursive_model(old,new)
            add_submodule(seq,new)
        elif name == 'CAddTable': # input is list
            new = LambdaReduce(lambda x,y: x+y)
            add_submodule(seq,new)
        elif name == 'Concat':
            dim = old.dimension
            new = LambdaReduce(lambda x,y,dim=dim: torch.cat((x,y),dim))
            lua_recursive_model(old,new)
            add_submodule(seq,new)
        else:
            print('Not Implement',name)


def lua_recursive_source(module):
    s = []
    for old in module.modules:
        name = type(old).__name__
        real = old
        if name == 'TorchObject':
            name = old._typename.replace('cudnn.','')
            old = old._obj

        if name == 'SpatialConvolution' or name == 'nn.SpatialConvolutionMM':
            if not hasattr(old,'groups') or old.groups is None: old.groups=1
            s += ['nn.Conv2d({},{},{},{},{},{},{},bias={}),#Conv2d'.format(old.nInputPlane,
                old.nOutputPlane,(old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),1,old.groups,old.bias is not None)]
        elif name == 'SpatialBatchNormalization':
            s += ['nn.BatchNorm2d({},{},{},{}),#BatchNorm2d'.format(old.running_mean.size(0), old.eps, old.momentum, old.affine)]
        elif name == 'VolumetricBatchNormalization':
            s += ['nn.BatchNorm3d({},{},{},{}),#BatchNorm3d'.format(old.running_mean.size(0), old.eps, old.momentum, old.affine)]
        elif name == 'ReLU':
            s += ['nn.ReLU()']
        elif name == 'Tanh':
            s += ['nn.Tanh()']
        elif name == 'Sigmoid':
            s += ['nn.Sigmoid()']
        elif name == 'SpatialMaxPooling':
            s += ['nn.MaxPool2d({},{},{},ceil_mode={}),#MaxPool2d'.format((old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),old.ceil_mode)]
        elif name == 'SpatialAveragePooling':
            s += ['nn.AvgPool2d({},{},{},ceil_mode={}),#AvgPool2d'.format((old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),old.ceil_mode)]
        elif name == 'SpatialUpSamplingNearest':
            s += ['nn.UpsamplingNearest2d(scale_factor={})'.format(old.scale_factor)]
        elif name == 'View':
            s += ['Lambda(lambda x: x.view(x.size(0),-1)), # View']
        elif name == 'Reshape':
            s += ['Lambda(lambda x: x.view(x.size(0),-1)), # Reshape']
        elif name == 'Linear':
            s1 = 'Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x )'
            s2 = 'nn.Linear({},{},bias={})'.format(old.weight.size(1),old.weight.size(0),(old.bias is not None))
            s += ['nn.Sequential({},{}),#Linear'.format(s1,s2)]
        elif name == 'Dropout':
            s += ['nn.Dropout({})'.format(old.p)]
        elif name == 'SoftMax':
            s += ['nn.Softmax()']
        elif name == 'MulConstant':
            s += ['Lambda(lambda x: x * {})'.format(MUL_CONSTANT)]
        elif name == 'Identity'  or name == 'nn.TotalVariation':
            s += ['Lambda(lambda x: x), # {}'.format(name)]
        elif name == 'nn.ShaveImage' or name == 'SpatialZeroPadding':
            s += ['nn.ConstantPad2d({}, 0)'.format(-old.size)]
        elif name == 'SpatialFullConvolution':
            s += ['nn.ConvTranspose2d({},{},{},{},{},{})'.format(old.nInputPlane,
                old.nOutputPlane,(old.kW,old.kH),(old.dW,old.dH),(old.padW,old.padH),(old.adjW,old.adjH))]
        elif name == 'VolumetricFullConvolution':
            s += ['nn.ConvTranspose3d({},{},{},{},{},{},{})'.format(old.nInputPlane,
                old.nOutputPlane,(old.kT,old.kW,old.kH),(old.dT,old.dW,old.dH),(old.padT,old.padW,old.padH),(old.adjT,old.adjW,old.adjH),old.groups)]
        elif name == 'SpatialReplicationPadding':
            s += ['nn.ReplicationPad2d({})'.format((old.pad_l,old.pad_r,old.pad_t,old.pad_b))]
        elif name == 'SpatialReflectionPadding':
            s += ['nn.ReflectionPad2d({})'.format((old.pad_l,old.pad_r,old.pad_t,old.pad_b))]
        elif name == 'nn.InstanceNormalization':
            s += ['nn.InstanceNorm2d({}, affine=True)'.format((old.nOutput))]
        elif name == 'Copy':
            s += ['Lambda(lambda x: x), # Copy']
        elif name == 'Narrow':
            s += ['Lambda(lambda x,a={}: x.narrow(*a))'.format((old.dimension,old.index,old.length))]
        elif name == 'SpatialCrossMapLRN':
            lrn = 'lnn.SpatialCrossMapLRN(*{})'.format((old.size,old.alpha,old.beta,old.k))
            s += ['Lambda(lambda x,lrn={}: Variable(lrn.forward(x.data)))'.format(lrn)]

        elif name == 'Sequential':
            s += ['nn.Sequential( # Sequential']
            s += lua_recursive_source(old)
            s += [')']
        elif name == 'ConcatTable':
            s += ['LambdaMap(lambda x: x, # ConcatTable']
            s += lua_recursive_source(old)
            s += [')']
        elif name == 'CAddTable':
            s += ['LambdaReduce(lambda x,y: x+y), # CAddTable']
        elif name == 'Concat':
            dim = old.dimension
            s += ['LambdaReduce(lambda x,y,dim={}: torch.cat((x,y),dim), # Concat'.format(old.dimension)]
            s += lua_recursive_source(old)
            s += [')']
        else:
            s += '# ' + name + ' Not Implement,\n'
    s = map(lambda x: '\t{}'.format(x),s)
    return s

def simplify_source(s):
    s = map(lambda x: x.replace(',(1, 1),(0, 0),1,1,bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',(0, 0),1,1,bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',1,1,bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',bias=True),#Conv2d',')'),s)
    s = map(lambda x: x.replace('),#Conv2d',')'),s)
    s = map(lambda x: x.replace(',1e-05,0.1,True),#BatchNorm2d',')'),s)
    s = map(lambda x: x.replace('),#BatchNorm2d',')'),s)
    s = map(lambda x: x.replace(',(0, 0),ceil_mode=False),#MaxPool2d',')'),s)
    s = map(lambda x: x.replace(',ceil_mode=False),#MaxPool2d',')'),s)
    s = map(lambda x: x.replace('),#MaxPool2d',')'),s)
    s = map(lambda x: x.replace(',(0, 0),ceil_mode=False),#AvgPool2d',')'),s)
    s = map(lambda x: x.replace(',ceil_mode=False),#AvgPool2d',')'),s)
    s = map(lambda x: x.replace(',bias=True)),#Linear',')), # Linear'),s)
    s = map(lambda x: x.replace(')),#Linear',')), # Linear'),s)

    s = map(lambda x: '{},\n'.format(x),s)
    s = map(lambda x: x[1:],s)
    s = reduce(lambda x,y: x+y, s)
    return s

def torch_to_pytorch(t7_filename,outputname=None):
    model = load_lua(t7_filename,unknown_classes=True)
    if type(model).__name__=='hashable_uniq_dict': model=model.model
    model.gradInput = None
    slist = lua_recursive_source(lnn.Sequential().add(model))
    s = simplify_source(slist)
    header = '''
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

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
'''
    varname = t7_filename.replace('.t7','').replace('.','_').replace('-','_')
    s = '{}\n\n{} = {}'.format(header,varname,s[:-2])

    if outputname is None: outputname=varname
    with open(outputname+'.py', "w") as pyfile:
        pyfile.write(s)

    new = nn.Sequential()
    lua_recursive_model(model,new)
    torch.save(new.state_dict(),outputname+'.pth')


parser = argparse.ArgumentParser(description='Convert torch t7 model to pytorch')
parser.add_argument('--model','-old', type=str, required=True,
                    help='torch model file in t7 format')
parser.add_argument('--output', '-o', type=str, default=None,
                    help='output file name prefix, xxx.py xxx.pth')
args = parser.parse_args()

torch_to_pytorch(args.model,args.output)
