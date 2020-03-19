.. _faq:

Frequently Asked Questions
==========================


Something is wrong with the text files (including the parameter file), but I cannot see a mistake.
--------------------------------------------------------------------------------------------------

Make sure that the End-of-Line character is in Unix format (\n). Do not use the standard Windows text editors as they will automatically change EOL to Windows standard (\r\n).
One of the programs crashed.
Make sure to use full file names and avoid relative filenames containing characters like ‘.’, ‘..’, ‘~’; avoid special characters like spaces ‘ ‘. You may have found a bug…
The tile IDs of the processed Level 2 data have negative numbers, e.g. X-100_Y0100.
Make sure that the origin of the target grid (ORIGIN LAT / ORIGIN LON) is in the North-West of your study area.
Potentially, you have accidentally swapped latitude and longitude. Note that a geographic location in the North-West is not necessarily North-West in the output coordinate system, too (for an example see Fig. 14). Although not recommended, higher-level FORCE functions should be able to digest negative tile numbers (note that we did not test this exhaustively).


Is it possible to have a look at all the temporary layers that are created in the L2PS internals?
-------------------------------------------------------------------------------------------------

Theoretically yes, but this option should only be used by experts. You can re-compile the code in DEBUG mode, which features extensive output where images for most processing steps are saved. Note that these images are intended for software development and do not necessarily have intuitive file names; metadata or projections are also not appended. If DEBUG is activated, force-level2 does not allow you to process multiple images or to use parallel processing (your system will be unresponsive because too much data is simultaneously written to the disc, and parallel calls to force-l2ps would overwrite the debugging images). For debugging, follow the steps summarized on page 23.


Transformation failed.Computing tile origin in dst_srs failed. Error in geometric module Following error appears (L2PS) in the Level 2 logfile: Transformation failed. Computing tile origin in dst_srs failed...
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This is most probably due to a bug in force-parameter-level2 (see VI.B.1), which can be solved by adding following line before the projection definition:

PROJECTION =


Following error appears (L2PS): 'grep: QUEUED: No such file or directory. No images in …
-----------------------------------------------------------------------------------------

Most probably, you have pasted the content of the file queue into the FILE_QUEUE parameter in the parameter file. FILE_QUEUE expects the file name (full path) of the file queue.


Following error appears (L2PS) in the Level 2 logfile: Unacceptable tier.
-------------------------------------------------------------------------

The acceptable tier level can be specified in the parameter file. Landsat Level-1T data (tier = 1) feature the best geometric correction and geolocation. We recommend to not use Level-1G or Level-1Gt images (tier = 2), unless you know what you are doing. See section VII.A.


Following error appears (L2PS) in the Level 2 logfile: DEM out of bounds
------------------------------------------------------------------------
There is a problem with the DEM. The DEM must be provided in meters a.s.l., and nodata needs to be -32767. There is a security query, which ensures that the DEM is within -500 m and +9000 m.


Following error appears (L2PS) in the logfile: zero-filled temperature.
-----------------------------------------------------------------------

Our Landsat cloud detection is in need of temperature data. The Landsat-8 TIRS was reconfigured due to anomalous current levels, and for a certain period of time, data were distributed by USGS with zero-filled temperature data. The USGS has fixed the problem and the data are/were reprocessed in a phased processing strategy. You should regularly check on updates from USGS, re-download the failed images and process them again. This might happen again, though.


Following error appears (L2PS) in the logfile: Unable to open MTL file. Parsing metadata failed.
------------------------------------------------------------------------------------------------
FORCE can only handle Landsat data processed by the Level 1 Product Generation System (LPGS) of USGS, which comes with an MTL text file (contains the metadata). Data processed with the outdated National Land Archive Production System (NLAPS) are not supported. Sorry.


Following error appears (L2PS) in the logfile: tar.gz container is corrupt.
---------------------------------------------------------------------------

There are two possible reasons: 

1) the file downloaded from USGS is corrupt, incomplete, etc. In this case, delete the image, remove it from the file queue and download/process again. 

2) force-level2 checks for non-zero exit code when extracting the images. On some systems, the tar/gzip programs throws a warning each time it extracts an image; this is probably related to some write permissions or mount settings. 

There is not much to do about this from our side. You need to fix your settings, speak with your admin. Alternatively, you can disable the exit-code check by changing, removing or commenting following lines in bash/force-level2.sh. If doing this, follow-up errors will occur if there really was a problem with the file.

.. code-block:: bash
  if [ ! $? -eq 0 ]; then
    echo "$BASE: tar.gz container is corrupt."
    FAIL=1
  fi


An error like this appears (L2PS) in the logfile: L1C_T21MXM_A007643_null: unknown Satellite Mission. Parsing metadata failed.
------------------------------------------------------------------------------------------------------------------------------
This can happen when Sentinel-2 data downloads are incomplete. Delete the image, remove it from the file queue and download/process again.


Following error appears (L2PS) in the logfile: L2PS is already running. Exit.
-----------------------------------------------------------------------------
FORCE L2PS has a built-in safeguard, which was implemented to allow safe operational and scheduled processing. FORCE L1AS and FORCE L2PS can be used for NRT processing, i.e. data can be downloaded and processed with n CPUs at given intervals. As the processing can take longer than these intervals, the safeguard protects your system from launching another n processing jobs, which may exceed the N CPUs available on your machine. You can disable the safeguard by changing, removing or commenting following lines in bash/force-level2.sh:

.. code-block:: bash
  # protect against multiple calls
  if [ $(ps aux | grep 'L2PS' | wc -l) -gt 1 ]; then
    echo "L2PS is already running. Exit." > $OD/FORCE-L2PS_$TIME.log
    exit
  fi


Following error appears (L2PS) in the logfile: Unable to lock file. Error in writing products! Tiling images failed! Error in geometric module.
-----------------------------------------------------------------------------------------------------------------------------------------------

There is a write problem. 

1) If L2PS was aborted in a previous run, some left-over lockfiles might exist (*.lock). In this case, FORCE cannot lock the file as it is already ‘locked’. Temporary locking the files is important as we’ll have write conflicts from parallel calls if not doing this. You need to remove the lock files. 

2) The lockfile generation timed out. This may happen if there is too much I/O activity on your system, such that FORCE is not allowed to write data for quite some time. Reduce I/O from other processes/users. Try to use fewer parallel processes. Try to increase the delay. Try writing to a disc that can handle the I/O, preferably directly attached to the server.


Following warning appears on the screen: 'lockfile creation failed: exceeded maximum number of lock attempts' 
-------------------------------------------------------------------------------------------------------------

There is a known problem with CIFS mounted network drives. You can ignore these warnings; they are no fatal errors. But you might want to inspect the file queue after Level 2 processing, as there is a minor possibility that there were some conflicts due to parallel write attempts: a few images might not have been switched from QUEUED to DONE status. This does not imply that the image was not processed (check the logfile as well).


There are holes in my processed Level 2 images. Why?
----------------------------------------------------

Nodata values in the DEM are masked. Impulse Noise is attempted to be detected and is masked out. The image border (including SLC-off stripes) is buffered by one pixel as these pixels are often erroneous. The masks are applied all output products.


The programs don’t run and there are strange symbols on the screen.
-------------------------------------------------------------------

You have probably copied text from this document to your shell. This might be an encoding issue. Try to manually type the commands.
