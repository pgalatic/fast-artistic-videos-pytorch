# author: Paul Galatic
#
# Program to

# STD LIB
import sys
import glob
import logging
import pathlib
import argparse

# EXTERNAL LIB

# LOCAL LIB
from .const import *
import main

def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    ap.add_argument('src_dir',
        help='The directory from which to source .ppm images, .flo flow files, and .pgm consistency checks.')
    ap.add_argument('style',
        help='The full path to the pyTorch style model.')
    
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(filename=LOGFILE, filemode='a', format=LOGFORMAT, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    data = pathlib.Path(args.src_dir)
    model = main.StylizationModel(args.style)
    
    # Gather all the frames for stylization
    frames = sorted([str(data / name) for name in glob.glob1(str(data), '*.ppm')])
    # First flow/cert doesn't exist, so use None as a placeholder
    flows = [None] + sorted([str(data / name) for name in glob.glob1(str(data), 'backward*.flo')])
    certs = [None] + sorted([str(data / name) for name in glob.glob1(str(data), 'reliable*.pgm')])
    # Sanity checks
    assert(len(frames) > 0 and len(flows) > 0 and len(certs) > 0 
            and len(frames) == len(flows) and len(flows) == len(certs))
    
    model.stylize(frames, flows, certs)