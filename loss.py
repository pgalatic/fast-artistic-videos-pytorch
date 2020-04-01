
# STD LIB
import os
import pdb
import glob
import logging
import pathlib
import argparse
from collections import namedtuple

# EXERNAL LIB
import torch
import torch.nn as nn
from torchvision import models

import cv2
import flowiz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        
        style_basename = os.path.splitext(os.path.basename(style_fname))[0]
        style_weights = {'WomanHat': 20, 'picasso': 10, 'candy': 10, 'mosaic': 10, 'scream': 20}
        self.style_weight = style_weights[style_basename]

        self.vgg = Vgg16(requires_grad=False)
        style_features = self.vgg(styutils.preprocess(style_img))
        self.style_gram = [self.gram_matrix(feature) for feature in style_features]
        self.mse_loss = nn.MSELoss()
    
    def gram_matrix(self, tensor):
        n, c, h, w = tensor.size()
        flat = tensor.view(n*c, -1)
        out = torch.mm(flat, flat.t()) / (c*h*w)
        return out 

    def eval(self, img, out, multi=None):
        h, w = img.shape[0], img.shape[1]
        img_features = self.vgg(styutils.preprocess(img))
        out_features = self.vgg(styutils.preprocess(out))
        
        content_loss = self.mse_loss(img_features.relu3_3, out_features.relu3_3)
        
        style_loss = 0
        for feature, target_gram in zip(out_features, self.style_gram):
            out_gram = self.gram_matrix(feature)
            style_loss += self.mse_loss(out_gram, target_gram)
        style_loss *= self.style_weight
        
        temporal_loss = 0
        if multi:
            # The masked portions do not influence the temporal loss.
            # FIXME: Is it a bug that the pre_cert does not use the min filter?
            prev, flow, cert = multi
            prev_warped = torch.FloatTensor(styutils.warp(prev, flow)).view(1, 3, h, w)
            pre_cert = torch.FloatTensor(cert / 255).unsqueeze(0).unsqueeze(0)
            prev_warped_masked = prev_warped * pre_cert.expand_as(prev_warped)
            out_pre = torch.FloatTensor(out).view(1, 3, h, w)
            out_masked = out_pre * pre_cert.expand_as(out_pre)
            temporal_loss = self.mse_loss(prev_warped_masked, out_masked)
        
        logging.debug('CONTENT:\t{}\tSTYLE:\t{}\tTEMPORAL:\t{}'.format(
            content_loss, style_loss, temporal_loss))
        score = content_loss + style_loss + temporal_loss
        logging.debug('Loss:\t{}'.format(score))
        return content_loss, style_loss, temporal_loss

def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    # Required arguments
    ap.add_argument('style', type=str,
        help='The style image to evaluate against')
    ap.add_argument('src_dir', type=str,
        help='The path to the original .ppm, .flo, .pgm files')
    ap.add_argument('fav_dir', type=str,
        help='The path to the output of FAV by Ruder et al.')
    ap.add_argument('dav_dir', type=str,
        help='The path to the output of DAV')
    
    return ap.parse_args()

def main():
    args = parse_args()
    styutils.start_logging()
    
    assert(os.path.exists(args.style))
    l = StyleTransferVideoLoss(args.style)
    
    ppm_names = [str(pathlib.Path(args.src_dir) / name) for name in glob.glob1(args.src_dir, '*.ppm')]
    flo_names = [str(pathlib.Path(args.src_dir) / name) for name in glob.glob1(args.src_dir, 'backward*.flo')]
    pgm_names = [str(pathlib.Path(args.src_dir) / name) for name in glob.glob1(args.src_dir, '*.pgm')]
    fav_names = [str(pathlib.Path(args.fav_dir) / name) for name in glob.glob1(args.fav_dir, '*.png')]
    dav_names = [str(pathlib.Path(args.dav_dir) / name) for name in glob.glob1(args.dav_dir, '*.png')]
    
    flo_names = sorted(flo_names, key=lambda x: len(x))
    pgm_names = sorted(pgm_names, key=lambda x: len(x))
    
    ppms = [cv2.imread(fname) for fname in ppm_names]
    flos = [None] + [flowiz.read_flow(fname) for fname in flo_names]
    pgms = [None] + [cv2.imread(fname, cv2.IMREAD_UNCHANGED) for fname in pgm_names]
    favs = [cv2.imread(fname) for fname in fav_names]
    davs = [cv2.imread(fname) for fname in dav_names]
    
    assert(len(ppms) > 0)
    logging.info('Images loaded...')
    
    fc_ls, fs_ls, ft_ls, f_tls = [], [], [], []
    dc_ls, ds_ls, dt_ls, d_tls = [], [], [], []
    
    for i in range(len(davs)):
        if i == 0:
            fc_l, fs_l, ft_l = map(int, l.eval(ppms[i], favs[i]))
            dc_l, ds_l, dt_l = map(int, l.eval(ppms[i], davs[i]))
            
            fc_ls.append(fc_l)
            fs_ls.append(fs_l)
            f_tls.append(int(fc_l + fs_l))
            dc_ls.append(dc_l)
            ds_ls.append(ds_l)
            d_tls.append(int(dc_l + ds_l))
        else:
            fc_l, fs_l, ft_l = map(int, l.eval(ppms[i], favs[i], (favs[i-1], flos[i], pgms[i])))
            dc_l, ds_l, dt_l = map(int, l.eval(ppms[i], davs[i], (davs[i-1], flos[i], pgms[i])))
            
            fc_ls.append(fc_l)
            fs_ls.append(fs_l)
            ft_ls.append(ft_l)
            f_tls.append(int(fc_l + fs_l + ft_l))
            dc_ls.append(dc_l)
            ds_ls.append(ds_l)
            dt_ls.append(dt_l)
            d_tls.append(int(dc_l + ds_l + dt_l))
    
    df = pd.DataFrame({ 
        'fav_content': fc_ls, 'fav_style': fs_ls, 'fav_temporal': [0] + ft_ls, 
        'dav_content': dc_ls, 'dav_style': ds_ls, 'dav_temporal': [0] + dt_ls, 
        'fav_total': f_tls, 'dav_total': d_tls
    }).astype(int)
    logging.info(df)
    df.to_csv('_' + os.path.basename(args.src_dir) + '_loss.csv')

if __name__ == '__main__':
    main()