Processing
==========

Processing
The core module of FORCE L3PS is force-level3, which generates composites and/or spectral temporal statistics. The processed images are written to the output directory given in the Level 3 parameter file (see VII.D for all available options).
Module	|	force-level3

Usage	|	force-level3     par-file

The parameter file needs to be given as 1st argument. All options need to be specified in the parameter file, e.g. tiled or mosaicked output, spatial extent, spatial resolution, temporal target / compositing period, phenology-adaptive or static method etc. (see VII.D for all available options).
At the end of processing, histograms about number of observations, sensors, days used for compositing, etc. is printed to the screen.

