# author: Paul Galatic
#
# This module handles optical flow calculations. These calculations are made using DeepMatching and
# DeepFlow. Future versions may use FlowNet2 or other, faster, better-quality optical flow 
# measures. The most important criteria is accuracy, and after that, speed.

# STD LIB
import os
import re
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
import numpy as np

# LOCAL LIB
import common
from const import *

def most_recent_optflo(remote):
    # Check to see if the optical flow folder exists.
    if not os.path.isdir(str(remote)):
        # If it doesn't exist, then there are no optflow files, and we start from scratch.
        return 1

    # The most recent optflow is the most recent placeholder plus 1.
    placeholders = glob.glob1(str(remote), 'frame_*.plc')
    if len(placeholders) == 0: return 1
    
    return max(map(int, [re.findall(r'\d+', plc)[0] for plc in placeholders])) + 1

def claim_job(idx, frames, remote):
    '''
    All nodes involved are assumed to share a common directory. In this directory, placeholders
    are created so that no two nodes work compute the same material. 
    '''
    # Try to create a placeholder.
    placeholder = str(remote / (os.path.splitext(FRAME_NAME)[0] % idx + '.plc'))
    try:
        # This will only succeed if this program successfully created the placeholder.
        with open(placeholder, 'x') as handle:
            handle.write('PLACEHOLDER CREATED BY {name}'.format(name=platform.node()))
        
        logging.debug('Job claimed: {}'.format(idx))
        start_name = str(remote / (FRAME_NAME % idx))
        end_name = str(remote / (FRAME_NAME % (idx + 1)))
        return start_name, end_name
    except FileExistsError:
        # We couldn't claim that job.
        return None, None

def write_flow(fname, flow):
    """Write optical flow to a .flo file
    Args:
        flow: optical flow
        dst_file: Path where to write optical flow
    """
    # Save optical flow to disk
    with open(fname, 'wb') as f:
        np.array(202021.25, dtype=np.float32).tofile(f)
        height, width = flow.shape[:2]
        np.array(width, dtype=np.uint32).tofile(f)
        np.array(height, dtype=np.uint32).tofile(f)
        flow.astype(np.float32).tofile(f)

def farneback(start_name, end_name, forward_name, backward_name):
    start = cv2.cvtColor(cv2.imread(start_name), cv2.COLOR_BGR2GRAY)
    end = cv2.cvtColor(cv2.imread(end_name), cv2.COLOR_BGR2GRAY)
    
    # Compute forward optical flow.
    forward = cv2.calcOpticalFlowFarneback(start, end, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute backward optical flow.
    backward = cv2.calcOpticalFlowFarneback(end, start, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    write_flow(forward_name, forward)
    write_flow(backward_name, backward)

def deepflow(start_name, end_name, forward_name, backward_name, downsamp_factor='2'):
    # Compute forward optical flow.
    forward_dm = subprocess.Popen([
        './core/deepmatching-static', start_name, end_name, '-nt', '0', '-downscale', downsamp_factor
    ], stdout=subprocess.PIPE)
    subprocess.run([
        './core/deepflow2-static', start_name, end_name, forward_name, '-match'
    ], stdin=forward_dm.stdout)
    
    # Compute backward optical flow.
    backward_dm = subprocess.Popen([
        './core/deepmatching-static', end_name, start_name, '-nt', '0', '-downscale', downsamp_factor, '|',
    ], stdout=subprocess.PIPE)
    subprocess.run([
        './core/deepflow2-static', end_name, start_name, backward_name, '-match'
    ], stdin=backward_dm.stdout)

def run_job(idx, start_name, end_name, remote, fast=True):
    logging.info('Computing optical flow for job {}.'.format(idx))
    
    forward_name = str(remote / 'forward_{}_{}.flo'.format(idx, idx+1))
    backward_name = str(remote / 'backward_{}_{}.flo'.format(idx+1, idx))
    reliable_name = str(remote / 'reliable_{}_{}.pgm'.format(idx+1, idx))
    
    if fast:
        farneback(start_name, end_name, forward_name, backward_name)
    else:
        deepflow(start_name, end_name, forward_name, backward_name)
    
    # Compute consistency check for backwards optical flow.
    subprocess.run([
        './core/consistencyChecker/consistencyChecker',
        backward_name, forward_name, reliable_name, end_name
    ])
    
    # Remove forward optical flow to save space, as it is only needed for the consistency check.
    os.remove(forward_name)

def optflow(start, frames, remote):
    logging.info('Starting optical flow calculations...')
        
    running = []

    for idx in range(start, start + len(frames)-1):
        # If there isn't room in the jobs list, wait for a thread to finish.
        while len(running) >= MAX_OPTFLOW_JOBS:
            running = [thread for thread in running if thread.isAlive()]
            time.sleep(1)
        # Optical flow files are 1-indexed.
        start_name, end_name = claim_job(idx + 1, frames, remote)
        if start_name:
            # Spawn a thread to complete that job, then get the next one.
            running.append(threading.Thread(target=run_job, 
                args=(idx + 1, start_name, end_name, remote)))
            running[-1].start()
    
    # Join all remaining threads.
    logging.info('Wrapping up threads for optical flow calculation...')
    for thread in running:
        thread.join()

    logging.info('...optical flow calculations are finished.')
    
def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    ap.add_argument('remote', type=str,
        help='The directory in which the .ppm files are stored and in which to place the .flo, .pgm files.')
    
    # Optional arguments
    ap.add_argument('--test', action='store_true',
        help='Compute optical flow over only a few frames to test functionality.')
    
    return ap.parse_args()

def main():
    args = parse_args()
    common.start_logging()
    
    remote = pathlib.Path(args.remote)
    
    frames = glob.glob1(str(remote), '*.ppm')
    if args.test:
        frames = frames[:NUM_FRAMES_FOR_TEST]
    
    optflow(frames, remote)

if __name__ == '__main__':
    main()