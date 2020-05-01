# Fast Artistic Videos in pyTorch
An implementation of [Fast Artistic Videos](https://github.com/manuelruder/fast-artistic-videos) by Ruder et al. in pyTorch instead of Lua Torch.

## Background

Neural Style Transfer is the process of redrawing one image in the style of another image. While there are many implementation of neural style transfer for single images, there are few that can do the same for video without introducing undesirable flickering effects and other artifacts. Even fewer can stylize a video in a reasonable amount of time.

This work addresses several weaknesses of its predecessor, [Fast Artistic Videos](https://github.com/manuelruder/fast-artistic-videos) by Ruder et al. (2018) while preserving all of its strengths. Installation is far faster and easier, and runtime is cut by over 50%. 

That said, runtime will still run into the hours on any video longer than a few seconds. If you are capable of running stylization across two or more computers, considering using [this repository](https://github.com/pgalatic/thesis) instead.

## Installation

1. Ensure all computers involved have at least Ubuntu >= 16.04 (see [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) if using Windows), Python >= 3.5.6, and that they have an FFMPEG installation compiled with libx264.
1. Clone this repository
1. Run `install.sh`

## Usage Guide

There are two required arguments in order to run the program.
1. `video` -- the path where the node can find the video
1. `style` -- the path where the node can find the stylization model

Consider the following example: I want to stylize a video called `eggdog.mp4` with the neural model `candy.pth`.

```
python stylize.py out/eggdog.mp4 out/candy.pth
```

Other options can be shown by running
```
python stylize.py -h
```
By default, this algorithm uses the [SPyNet](https://arxiv.org/abs/1611.00850) architecture to calculate optical flow, which is the best balance between speed and quality when running on a CPU. Please read the help message carefully so that you are aware of all the options available in the latest release, as the choice of optical flow calculator will dramatically affect both the quality of the final stylized video and the total processing time.

## Credits

This work is based on [Fast Artistic Videos](https://github.com/manuelruder/fast-artistic-videos). It relies on static binaries of [DeepMatching](https://thoth.inrialpes.fr/src/deepmatching/) and [Deepflow2](https://thoth.inrialpes.fr/src/deepflow/), among other external libraries.

If you use my code, please include a link back to this repository. If you use it for research, please include this citation.

```
@mastersthesis{Galatic2020Divide
  author  = "Paul Galatic",
  title   = "Divide and Conquer in Video Style Transfer",
  school  = "Rochester Institute of Technology - RIT",
  year    = "2020",
  url     = "https://github.com/pgalatic/thesis"
}
```

## Known Issues

Known issues:
* Currently, if a computation is interrupted and later began again, not all files will be successfully computed. It would be helpful to have a preprocessing step where, if placeholder files are observed to already exist, it will comb through and delete those that do not have a corresponding optical flow file.
* If the filesystem runs out of memory, the program may loop endlessly trying to create the same optical flow or stylization files, because Python's open(fname, 'x') will execute successfully even though the file is not created. This should not happen anymore now that memory use has been drastically reduced, but could theoretically still happen in certain scenarios or hyperparameter configurations.
* Sometimes OpenCV has trouble opening the occlusion `.pgm` files. This creates a scary traceback, but seems benign as far as I can tell—it doesn't cause a crash or affect the output, as the program simply tries to load the file again.

## Future Work

* [ ] Convert consistencyChecker to Python
* [x] Implement SPyNet
* [ ] Enable CUDA/CUDNN
* [ ] Implment LiteFlowNet

1. Enabling CUDA/CUDNN is an important priority, and now that the repository uses pyTorch instead of Torch, this upgrade should be relatively simple, though it must wait until my thesis is published.
1. Migrating from using external binaries for calculating optical flow to using Python-integrated optical flow calculations will allow the optical flow files to be stored in memory instead of being written to disk. This is especially possible for the certs, which are quick enough to be computed on the fly.
1. Fully recreating the training mechanisms of the core program, or replacing it with an alternative, is the next most important step, so that new models can be trained.
