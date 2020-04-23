# author: Paul Galatic
#
# This module handles optical flow calculations. These calculations are made using DeepMatching and
# DeepFlow. Future versions may use FlowNet2 or other, faster, better-quality optical flow 
# measures. The most important criteria is accuracy, and after that, speed.

# STD LIB
import os
import pdb
import glob
import time
import logging
import pathlib
import argparse
import platform
import threading
import subprocess

# EXTERNAL LIB
import cv2
import torch
import numpy as np

# LOCAL LIB
try:
    import styutils
    from sconst import *
    from spynet import spynet
    from flownet import flownet
except ImportError:
    from . import styutils
    from .sconst import *
    from .spynet import spynet
    from .flownet import flownet

def write_flow(fname, flow):
    '''
    Write optical flow to a .flo file
    Args:
        fname: Path where to write optical flow
        flow: an ndarray containing optical flow data
    '''
    # Save optical flow to disk
    with open(fname, 'wb') as f:
        np.array(202021.25, dtype=np.float32).tofile(f) # Write magic number for .flo files
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)      # Write width
        np.array(height, dtype=np.uint32).tofile(f)     # Write height
        flow.astype(np.float32).tofile(f)               # Write data

def claim_job(idx, frames, dst):
    '''
    All nodes involved are assumed to share a common directory. In this directory, placeholders
    are created so that no two nodes work compute the same material. 
    '''
    # Try to create a placeholder.
    placeholder = str(dst / (os.path.splitext(FRAME_NAME)[0] % idx + '.plc'))
    try:
        # This will only succeed if this program successfully created the placeholder.
        with open(placeholder, 'x') as handle:
            handle.write('PLACEHOLDER CREATED BY {name}'.format(name=platform.node()))
        
        logging.debug('Job claimed: {}'.format(idx))
        start_name = str(dst / (FRAME_NAME % idx))
        end_name = str(dst / (FRAME_NAME % (idx + 1)))
        return start_name, end_name
    except FileExistsError:
        # We couldn't claim that job.
        return None, None

def farneback_flow(start_name, end_name):
    start = cv2.cvtColor(cv2.imread(start_name), cv2.COLOR_BGR2GRAY)
    end = cv2.cvtColor(cv2.imread(end_name), cv2.COLOR_BGR2GRAY)
    
    forward = cv2.calcOpticalFlowFarneback(start, end, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    backward = cv2.calcOpticalFlowFarneback(end, start, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    return forward, backward

def spynet_flow(start_name, end_name):
    start = torch.Tensor(cv2.imread(start_name).transpose(2, 0, 1) * (1.0 / 255.0))
    end = torch.Tensor(cv2.imread(end_name).transpose(2, 0, 1) * (1.0 / 255.0))

    forward = spynet.estimate(start, end).detach().numpy().transpose(1, 2, 0)
    backward = spynet.estimate(end, start).detach().numpy().transpose(1, 2, 0)
    
    return forward, backward

def net_flow(start_name, end_name):
    start = torch.Tensor(cv2.imread(start_name).transpose(2, 0, 1) * (1.0 / 255.0))
    end = torch.Tensor(cv2.imread(end_name).transpose(2, 0, 1) * (1.0 / 255.0))

    forward = flownet.estimate(start, end).squeeze().detach().numpy().transpose(1, 2, 0)
    backward = flownet.estimate(end, start).squeeze().detach().numpy().transpose(1, 2, 0)
    
    return forward, backward

def deep_flow(start_name, end_name, forward_name, backward_name):
    # Compute forward optical flow.
    root = pathlib.Path(__file__).parent.absolute()
    forward_dm = subprocess.Popen([
        str('.' / root / DEEPMATCHING), start_name, end_name, '-nt', '0', '-downscale', '2'
    ], stdout=subprocess.PIPE)
    subprocess.run([
        str('.' / root / DEEPFLOW2), start_name, end_name, forward_name, '-match'
    ], stdin=forward_dm.stdout)
    
    # Compute backward optical flow.
    backward_dm = subprocess.Popen([
        str('.' / root / DEEPMATCHING), end_name, start_name, '-nt', '0', '-downscale', '2'
    ], stdout=subprocess.PIPE)
    subprocess.run([
        str('.' / root / DEEPFLOW2), end_name, start_name, backward_name, '-match'
    ], stdin=backward_dm.stdout)

def run_job(idx, start_name, end_name, dst, method):
    logging.info('Computing optical flow for job {}.'.format(idx))
    
    forward_name = str(dst / 'forward_{}_{}.flo'.format(idx, idx+1))
    backward_name = str(dst / 'backward_{}_{}.flo'.format(idx+1, idx))
    reliable_name = str(dst / 'reliable_{}_{}.pgm'.format(idx+1, idx))
    
    if method == 'farneback':
        forward, backward = farneback_flow(start_name, end_name)
        # Write flows to disk so that they can be used in the consistency check.
        write_flow(forward_name, forward)
        write_flow(backward_name, backward)
    elif method == 'spynet':
        forward, backward = spynet_flow(start_name, end_name)
        # Write flows to disk so that they can be used in the consistency check.
        write_flow(forward_name, forward)
        write_flow(backward_name, backward)
    elif method == 'flownet':
        forward, backward = net_flow(start_name, end_name)
        write_flow(forward_name, forward)
        write_flow(backward_name, backward)
    elif method == 'deepflow2': # TODO: options
        deep_flow(start_name, end_name, forward_name, backward_name)
    else:
        raise Exception('Bad flow method: {}'.format(method))
    
    # The absolute path accounts for if this file is being run as part of a submodule.
    root = pathlib.Path(__file__).parent.absolute()
    # Compute consistency check for backwards optical flow.
    subprocess.run([
        str('.' / root / CONSISTENCY_CHECK),
        backward_name, forward_name, reliable_name, end_name
    ])
    
    # Remove forward optical flow to save space, as it is only needed for the consistency check.
    os.remove(forward_name)

def optflow(start, frames, dst, method):
    logging.info('Starting optical flow calculations...')
        
    running = []

    for idx in range(start, start + len(frames)-1):
        # If there isn't room in the jobs list, wait for a thread to finish.
        while len(running) >= MAX_OPTFLOW_JOBS:
            running = [thread for thread in running if thread.isAlive()]
            time.sleep(1)
        # Optical flow files are 1-indexed.
        start_name, end_name = claim_job(idx + 1, frames, dst)
        if start_name:
            # Spawn a thread to complete that job, then get the next one.
            running.append(threading.Thread(target=run_job, 
                args=(idx + 1, start_name, end_name, dst, method)))
            running[-1].start()
    
    # Join all remaining threads.
    logging.info('Wrapping up threads for optical flow calculation...')
    for thread in running:
        thread.join()

    logging.info('...optical flow calculations are finished.')
    
def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    ap.add_argument('dst', type=str,
        help='The directory in which the .ppm files are stored and in which to place the .flo, .pgm files.')
    
    # Optional arguments
    ap.add_argument('--method', nargs='?', 
        choices=['farneback', 'spynet', 'flownet', 'deepflow2'], 
        default='deepflow2',
        help='Choice of optical flow calculation. Farneback is the fastest, but least accurate. Deepflow2 is the slowest, but most accurate. The others are somewhere in-between.')
    ap.add_argument('--test', action='store_true',
        help='Compute optical flow over only a few frames to test functionality.')

    
    return ap.parse_args()

def main():
    args = parse_args()
    styutils.start_logging()
    
    dst = pathlib.Path(args.dst)
    
    frames = glob.glob1(str(dst), '*.ppm')
    if args.test:
        frames = frames[:NUM_FRAMES_FOR_TEST]
    
    optflow(0, frames, dst, args.method)

if __name__ == '__main__':
    main()