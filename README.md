# Fast Artistic Videos in pyTorch
An implementation of [Fast Artistic Videos](https://github.com/manuelruder/fast-artistic-videos) by Ruder et al. in pyTorch instead of Lua Torch.

# NOTICE

This is the **reproducibility** branch. If you use this code, know that it is much less efficient than the current master branch. If you are trying to reproduce my results and you are having trouble, raise an issue. Otherwise, please try to use the current master before you point out any flaws.

## Background

Neural Style Transfer is the process of redrawing one image in the style of another image. While there are many implementation of neural style transfer for single images, there are few that can do the same for video without introducing undesirable flickering effects and other artifacts. Even fewer can stylize a video in a reasonable amount of time.

This work addresses several weaknesses of its predecessor, [Fast Artistic Videos](https://github.com/manuelruder/fast-artistic-videos) by Ruder et al. (2018) while preserving all of its strengths. Installation is far faster and easier, and runtime is cut by over 50%. 

That said, runtime in this implementation will still run into the hours on any video longer than a few seconds due to inefficiencies in the code that are left unfixed intentionally for the purpose of reproducibility.

## Installation and Usage Guide

This repository cannot stand on its own. It includes testing and debugging utilities which I used to verify its functionality, but unlike the current master cannot be used to stylize videos. You will need to install and use [parent repository](https://github.com/pgalatic/thesis/tree/repro) for that. 

## Credits

This work is based on [Fast Artistic Videos](https://github.com/manuelruder/fast-artistic-videos). It relies on static binaries of [DeepMatching](https://thoth.inrialpes.fr/src/deepmatching/) and [Deepflow2](https://thoth.inrialpes.fr/src/deepflow/).

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
* If the filesystem runs out of memory, which is very likely with any HD video larger than a minute, the program may loop endlessly trying to create the same optical flow or stylization files, because Python's open(fname, 'x') will execute successfully even though the file is not created.

## Future Work

* Because this is the reproducibility repository, it will never be updated beyond fixing crashes and updating documentation.
