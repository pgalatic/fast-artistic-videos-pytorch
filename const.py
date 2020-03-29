# author: Paul Galatic
# 
# Stores constants for the program.
#

# The name of the file in which logs are stored, and the format of displaying messages.
LOGFILE = 'thesis.log'
LOGFORMAT = '%(asctime)s %(name)s:%(levelname)s -- %(message)s'

# The format name to properly split frames.
FRAME_NAME = 'frame_%05d.ppm'

# When testing the algorithm, it is not immediately necessary to test it on a
# full video; instead, it can be tested with the first few frames of a video.
NUM_FRAMES_FOR_TEST = 15

# The prefixes used by the core stylization procedure to denote the names of 
# output images and upload them correctly to the common directory.
OUTPUT_PREFIX = 'out'
OUTPUT_FORMAT = 'out-%05d.png'

# A small constant useful to avoid dividing by zero.
EPSILON = 0.01

# Any key pair of frames should be several times more different from another as
# the average pair of frames. This is to guard against keyframes being chosen 
# to fill the knee-point quota, which can perform terribly on mostly contiguous
# videos.
MIN_DIST_FACTOR = 5

# The threshold at which to stop accepting new keyframes. Increase this threshold 
# for fewer keyframes. Decrease it to add more. Set to 0 to force every frame to 
# be its own partition (untested).
KNEE_THRESHOLD = 0.05

# The maximum number of threading jobs to run simultaneously.
MAX_OPTFLOW_JOBS = 8
MAX_UPLOAD_JOBS = 8
MAX_STYLIZATION_JOBS = 1 # experimental

# This mean is used for preprocessing input to the stylization neural network.
VGG_MEAN = [103.939, 116.779, 123.68]

# These are names of functions that only one computer should complete while the 
# others wait.
UPLOAD_VIDEO_TAG = 'upload_video.plc'
DIVIDE_TAG = 'divide.pkl'
