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

# EXTERNAL LIB

# LOCAL LIB
try:
    import styutils
    from sconst import *
    from flowcalc import flowcalc
except ImportError:
    from . import styutils
    from .sconst import *
    from .flowcalc import flowcalc

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
            running.append(threading.Thread(target=flowcalc.estimate, 
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
        choices=['farneback', 'spynet', 'deepflow2'], 
        default='deepflow2',
        help='Choice of optical flow calculation. Farneback is the fastest, but least accurate. Deepflow2 is the slowest, but most accurate. Spynet is the best balance on the CPU.')
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