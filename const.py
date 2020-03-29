# author: Paul Galatic
# 
# Stores constants for the program.
#

# The name of the file in which logs are stored, and the format of displaying messages.
LOGFILE = 'thesis.log'
LOGFORMAT = '%(asctime)s %(name)s:%(levelname)s -- %(message)s'

# The the paths to run various commands.
DEEPMATCHING = './deepmatching-static'
DEEPFLOW2 = './deepflow2-static'
CONSISTENCY_CHECK = './consistencyChecker/consistencyChecker'

# The format name to properly split frames.
FRAME_NAME = 'frame_%05d.ppm'

# When testing the algorithm, it is not immediately necessary to test it on a
# full video; instead, it can be tested with the first few frames of a video.
NUM_FRAMES_FOR_TEST = 15

# The prefixes used by the core stylization procedure to denote the names of 
# output images and upload them correctly to the common directory.
OUTPUT_PREFIX = 'out'
OUTPUT_FORMAT = 'out-%05d.png'

# The maximum number of threading jobs to run simultaneously.
MAX_OPTFLOW_JOBS = 4
MAX_STYLIZATION_JOBS = 1 # experimental, will cause huge slowdown on normal computers

# This mean is used for preprocessing input to the stylization neural network.
VGG_MEAN = [103.939, 116.779, 123.68]
