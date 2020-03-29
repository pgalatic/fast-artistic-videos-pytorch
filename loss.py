
# STD LIB
import pdb
import logging
from collections import namedtuple

# EXERNAL LIB
import torch
import torch.nn as nn
from torchvision import models

import cv2
import numpy as np

# LOCAL LIB
try:
    import styutils
except:
    from . import styutils

class Vgg16(torch.nn.Module):
    # Source: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out

class StyleTransferVideoLoss():
    def __init__(self, style_fname):
        style_img = cv2.imread(style_fname)

        self.vgg = Vgg16(requires_grad=False)
        style_features = self.vgg(styutils.preprocess(style_img))
        self.style_gram = [self.gram_matrix(feature) for feature in style_features]
        self.mse_loss = nn.MSELoss()
    
    def gram_matrix(self, tensor):
        n, c, h, w = tensor.size()
        features = tensor.view(n, c, h*w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c*h*w)
        return gram

    def eval(self, img, out, multi=None):
        img_features = self.vgg(styutils.preprocess(img))
        out_features = self.vgg(styutils.preprocess(out))
        
        content_loss = self.mse_loss(img_features.relu2_2, out_features.relu2_2)
        
        style_loss = 0
        for feature, target_gram in zip(out_features, self.style_gram):
            out_gram = self.gram_matrix(feature)
            style_loss += self.mse_loss(out_gram, target_gram)
        
        temporal_loss = 0
        if multi:
            prev, flow, cert = multi
            pre_cert = torch.FloatTensor(np.swapaxes(cert / 255, 0, 1)).unsqueeze(0).unsqueeze(0)
            prev_warped = torch.FloatTensor(np.swapaxes(styutils.warp(prev, flow), 0, 2)).unsqueeze(0)
            # The masked portions do not influence the temporal loss.
            # FIXME: Is it a bug that the pre_cert does not use the min filter?
            prev_warped_masked = prev_warped * pre_cert.expand_as(prev_warped)
            out_masked = torch.FloatTensor(np.swapaxes(out, 0, 2)) * pre_cert.expand_as(prev_warped)
            temporal_loss = self.mse_loss(prev_warped_masked, out_masked)
        
        logging.debug('CONTENT:\t{}\tSTYLE:\t{}\tTEMPORAL:\t{}'.format(
            content_loss, style_loss, temporal_loss))
        score = content_loss + style_loss + temporal_loss
        logging.info('Loss:\t{}'.format(score))
        return 