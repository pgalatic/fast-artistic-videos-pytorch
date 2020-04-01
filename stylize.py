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
from const import *

def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    # Required arguments
    ap.add_argument('video', type=str,
        help='The path to the stylization target, e.g. foo/bar.mp4')
    ap.add_argument('style', type=str,
        help='The path to the model used for stylization, e.g. foo/bar.pth')
        
    # Optional arguments
    ap.add_argument('--test', action='store_true',
        help='Test the algorithm by stylizing only a few frames of the video, rather than all of the frames.')
    ap.add_argument('--seed', nargs='?', default=None,
        help='Use an image to seed stylization instead of random initialization')
    ap.add_argument('--eval_fname', nargs='?', default=None,
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
    
    if args.test:
        to_remove = frames[NUM_FRAMES_FOR_TEST:]
        frames = frames[:NUM_FRAMES_FOR_TEST]
        for frame in to_remove:
            os.remove(frame)
    
    # Stylize the video
    model = model.StylizationModel(args.style, args.seed, args.eval_fname)
    model.stylize(0, frames, remote, args.fast)
    
    # Combine the stylized frames
    if not args.test:
        video.combine_frames(args.video, remote)
    
    # Clean up any lingering files.
    for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.pkl')]:
        os.remove(fname)
    #for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.ppm')]:
    #    os.remove(fname)
    #for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.plc')]:
    #    os.remove(fname)
    #for fname in [str(remote / name) for name in glob.glob1(str(remote), '*.png')]:
    #    os.remove(fname)