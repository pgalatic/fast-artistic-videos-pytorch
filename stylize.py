# author: Paul Galatic
#
# Program to

# STD LIB
import os
import pdb
import sys
import glob
import pathlib
import argparse

# EXTERNAL LIB

# LOCAL LIB
import model
import video
import styutils

def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    # Required arguments
    ap.add_argument('video', type=str,
        help='The path to the stylization target, e.g. foo/bar.mp4')
    ap.add_argument('style', type=str,
        help='The path to the model used for stylization, e.g. foo/bar.pth')
        
    # Optional arguments
    ap.add_argument('eval_fname', nargs='?', default=None,
        help='The path to the style image used for evaluating the model. Specifying this activates evaluation.')
    ap.add_argument('--fast', action='store_true',
        help='Use Farneback optical flow, which is faster than the default, DeepFlow2.')
    
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    styutils.start_logging()
    
    # Ensure the destination directory is created
    remote = pathlib.Path('out') / os.path.splitext(os.path.basename(args.video))[0]
    styutils.makedirs(remote)
    
    # Split the video into frames
    video.split_frames(args.video, remote)
    frames = sorted([str(remote / frame) for frame in glob.glob1(str(remote), '*.ppm')])
    
    # Stylize the video
    model = model.StylizationModel(args.style, args.eval_fname)
    model.stylize(0, frames, remote, args.fast)
    
    # Combine the stylized frames
    video.combine_frames(args.video, remote)
    
    # Clean up any lingering files.
    for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.pkl')]:
        os.remove(fname)
    for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.ppm')]:
        os.remove(fname)
    #for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.plc')]:
    #    os.remove(fname)
    #for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.png')]:
    #    os.remove(fname)