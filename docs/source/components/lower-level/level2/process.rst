Processing
==========

Image archives
The core module of FORCE L2PS is force-level2, which handles the mass processing of the Level 1 input. In principle, the same algorithm is applied to all sensors – although specific processing options are triggered or are only available for some sensors (e.g. Sentinel-2 resolution merge). Each image enqueued in the file queue (see section VII.B) is processed to Level 2 according to the specifications in the parameter file (see section VII.A); the file queue is specified in the parameter file too (the filename of the queue, not the content of the file). After processing, the image is dequeued, and as such it is possible to schedule processing in near-real time, e.g. if called from a cronjob. If reprocessing is required, the image needs to be enqueued again (see section VII.B). The processed images, as well as a logfile are written to the output directory given in the parameter file.
Module	|	force-level2

Usage	|	force-level2     par-file     ncpu     delay

The parameter file needs to be given as 1st argument. The number of CPUs used for parallel processing needs to be given as 2nd argument. Each processor handles one input image. Note that GNU parallel spawns jobs by using multiple ssh logins; the number of allowed ssh logins per user must be sufficiently large (ask your system administrator to increase this number if necessary). It is advised to monitor the workload. Swapping should be prevented – in this case, decrease the number of parallel jobs. The number of CPUs can be re-adjusted while(!) force-level2 is running. A file ‘cpu-$TIME’ is temporarily created in DIR_TEMP. This file can be modified. Note that the effect is not immediate, as the load is only adjusted after one of the running jobs (images) is finished.
To prevent an I/O jam at startup (by reading / extracting a lot of data simultaneously), a delay (in seconds) needs to given as 3rd argument. Each ‘delay’ seconds, the processing of a new image is started until ‘ncpu’ parallel jobs are running. Depending on processing speed per image plus ‘ncpu’ and ‘delay’ settings, it is possible that ‘ncpu’ parallel processes won’t be reached.
The necessary delay is dependent on your system’s architecture (I/O speed etc), and on sensor to be processed. Our recommendations are as follows (but note that these values might not be fitting for you, each system is different): 
Sentinel-2: large delay (large data volume + fairly long processing time). Try 20 seconds.
Landsat: delay should approximately reflect the time necessary for extracting a single *.tar.gz archive. Small values for Landsat 5-7 are reasonable (small data volume + short processing time). Try 3–5 seconds. Larger delays may be necessary for Landsat 8 (fairly high data volume + short processing time). Try 10–15 seconds.

Single images
The workhorse of FORCE L2PS is force-l2ps, which is a lower-level routine called from within force-level2. It processes one single image. For the majority of users, it is recommended to use force-level2 instead of directly calling force-l2ps. However, for specific purposes (e.g. testing / debugging), the expert user may want to use this program directly (or if you want to implement your own job scheduler).
Module	|	force-l2ps

Usage	|	force-l2ps     L1-image     par-file

The 1st argument is the directory that contains the image data. In case of Landsat, the *.tar.gz archive needs to be extracted before processing. In case of Sentinel-2, the *.zip archive needs to extracted before processing and one tile (directory) within the ‘GRANULE’ directory must be given as input. Note that the extraction of Landsat images is automatically performed within force-level2, and the extraction of Sentinel-2 data is performed within force-level1-sentinel2 (see VI.A.2). The parameter file needs to be given as 2nd argument.
The direct usage of force-l2ps is recommended for debugging or for detailed output. The debugging mode also features extensive output where images for most processing steps are saved. Note that these images are intended for software development and do not necessarily have intuitive file names; metadata or projections are also not appended. If debug output is required, the software needs to be re-compiled, which should only be done by expert users. If DEBUG is activated, force-level2 does not allow you to process multiple images or to use parallel processing (your system will be unresponsive because too much data is simultaneously written to the disc, and parallel calls to force-l2ps would overwrite the debugging images). For debugging, follow these steps:
	Modify force/src/force.h
	After the includes, there is a ‘compiler options’ section. Remove the comments before DEBUG and DEBUGPATH and define an existing path for DEBUGPATH. The debug images will be saved in this directory.
	Re-compile the software
