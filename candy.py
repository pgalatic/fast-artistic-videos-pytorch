
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


old_styles/checkpoint_candy_video = nn.Sequential( # Sequential,
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