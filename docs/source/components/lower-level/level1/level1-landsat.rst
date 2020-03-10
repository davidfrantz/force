.. _level1-landsat:

level1-landsat
==============

FORCE can process Level 1 Landsat data, generated using the Level 1 Product Generation System (LPGS) of the U.S. Geological Survey (USGS). 

At the time of writing, pre-collection, collection 1, and collection 2 data were successfully digested by FORCE. Following Landsat sensors are supported:

* Landsat 4 Thematic Mapper
* Landsat 5 Thematic Mapper
* Landsat 7 Enhanced Thematic Mapper+
* Landsat 8 Operational Land Imager

Before getting started, the full resolution Landsat images must be acquired from the USGS archive, e.g. through EarthExplorer or GloVis. FORCE currently does not provide functionality to download Landsat data. 

It is recommended to store the compressed images in a consistent data pool (without duplicates or different processing versions). 

A file queue needs to be prepared (see section VII.B). 

Both tasks can be handled by ``force-level1-landsat``. Extraction of the \*.tar.gz archives is not necessary at this point as this is done on the fly during Level 2 processing.

.. code-block:: bash

  force-level1-landsat

  Usage: force-level1-landsat from to queue cp/mv [dry]

* from
  
  | The input directory (1st argument) is recursively scanned for \*.tar.gz files. The Path/Row is extracted from the file paths. 
  | Note that the input directory should not be a parent of the target directory. The input directory is scanned recursively, thus files already in the L1 data pool will be moved again if the target is a child of the input (this is unnecessary and might take a while).
  | *White-space characters are not allowed in the file path, e.g. when using the Bulk Download Application from USGS. This needs to be taken care of before running this program.* 
  
* to

  | A subdirectory for every Path/Row is created in the target directory. 
  | The files from the input directoty are imported in the corresponding Path/Row folders. 
  | Files are not imported if they are duplicates of already existing files. 
  | Existing files are replaced by files with a newer production number.

* queue

  | A file queue (e.g. a file named ``level1-landsat-germany.txt``) needs to be given. 
  | If it does not exist, it will be created. 
  | If it exists, new imports are appended to this file. 
  | Outdated files (older production number) are removed from this queue, and the new imports are appended to the end. 
  | This queue is needed for Level 2 processing. All images with ``QUEUED`` status will be processed, then set to ``DONE``.

* cp/mv

  The Level 1 archives are moved (``mv``, recommended) or copied (``cp``) from input to target directory.

* [dry]

  This argument is optional, and if ``dry`` is specified, no files are moved/copied. The program only prints what it would do.

