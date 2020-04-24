# author: Paul Galatic
#
# Handles initial task of splitting video and uploading frames to the shared system.

# STD LIB
import os
import pdb
import sys
import shutil
import logging
import pathlib
import argparse
import subprocess

# EXTERNAL LIB
import ffprobe3

# LOCAL LIB
try:
    import styutils
    from sconst import *
except:
    from . import styutils
    from .sconst import *

def check_deps():
    # Preliminary operations to make sure that the environment is set up properly.
    check = shutil.which('ffmpeg')
    if not check:
        sys.exit('ffmpeg not installed. Aborting')

def split_frames(target, dst, extension='ppm'):
    check_deps()
    
    styutils.makedirs(dst)
    # Split video into a collection of frames. 
    # Don't split the video if we've already done so.
    probe = ffprobe3.FFProbe(str(target))
    num_frames = int(probe.streams[0].nb_frames)
    files_present = styutils.count_files(dst, '.{}'.format(extension))
    
    if num_frames <= files_present:
        logging.info('Video is already split into {} frames'.format(num_frames))
        return num_frames
    
    logging.info('{} frames in video != {} frames on disk; splitting frames...'.format(
        num_frames, files_present))
    
    # This line is to account for extensions other than the default.
    frame_name = '{}.{}'.format(os.path.splitext(FRAME_NAME)[0], extension)
    subprocess.run(['ffmpeg', '-i', str(target), str(dst / frame_name)])

def combine_frames(target, src, dst=pathlib.Path('out'), format=None):
    check_deps()
    
    styutils.makedirs(dst)
    if not format: format = OUTPUT_FORMAT
    basename = os.path.splitext(os.path.basename(str(target)))[0]
    no_audio = str(src / ('{}_no_audio.mp4'.format(basename)))
    audio = str(dst / ('{}_stylized.mp4'.format(basename)))

    # Don't try to combine frames if the destination already exists.
    # FFMPEG checks for this, but if we should preemptively avoid it if we can.
    if os.path.exists(no_audio):
        logging.info('{} already exists -- quitting'.format(no_audio))
        return
    if os.path.exists(audio):
        logging.info('{} already exists -- quitting'.format(audio))
        return

    # Get the original video's length. This will be necessary to properly reconstruct it.
    # TODO: Check to see if a video contains audio before attempting to add audio.
    probe = ffprobe3.FFProbe(str(target))
    duration = probe.streams[-1].duration
    num_frames = str(probe.streams[0].nb_frames)
    
    # Combine stylized frames into a video.
    subprocess.run([
        'ffmpeg', '-i', str(src / format),
        '-c:v', 'libx264', '-preset', 'veryslow',
        '-pix_fmt', 'yuv420p',
        '-filter:v', 'setpts={}/{}*N/TB'.format(duration, num_frames),
        '-r', '{}/{}'.format(num_frames, duration),
        no_audio
    ])
    
    # Add audio to that video.
    subprocess.run([
        'ffmpeg', '-i', no_audio, '-i', str(target),
        '-c', 'copy', '-map', '0:0', '-map', '1:1',
        audio
    ])
    
    # Remove the version without audio.
    os.remove(no_audio)

def parse_args():
    '''Parses arguments.'''
    ap = argparse.ArgumentParser()
    
    # Required arguments
    ap.add_argument('mode', type=str, choices=['c', 's', 'n'],
        help='Whether or not to split or combine frames. Options: (s)plit, (c)ombine, (n)um_frames')
    ap.add_argument('target', type=str,
        help='The name of the video (locally, on disk).')
    
    # Optional arguments
    ap.add_argument('--src', type=str, nargs='?', default=None,
        help='The path to the folder in which the frames are contained, if combining them.')
    ap.add_argument('--format', type=str, default=None,
        help='The format of image filenames, when combining frames, e.g. out-%%05d.png [None].')
    ap.add_argument('--extension', choices=['ppm', 'png'], nargs='?', default='png',
        help='The extension used for the frames of a split video [.png].')
    
    return ap.parse_args()

def main():
    args = parse_args()
    styutils.start_logging()
    
    if args.mode == 'n':
        probe = ffprobe3.FFProbe(str(args.target))
        num_frames = str(probe.streams[0].nb_frames)
        print(num_frames)
        return
    
    dst = pathlib.Path('out') / os.path.splitext(os.path.basename(args.target))[0]
    styutils.makedirs(dst)

    if args.mode == 's':
        split_frames(args.target, dst, args.extension)
    elif args.mode == 'c':
        if not args.src:
            sys.exit('Please specify a source directory.')
        combine_frames(args.target, pathlib.Path(args.src), format=args.format)

if __name__ == '__main__':
    main()
