.. _queue:

File queue
==========

The file queue is mandatory for ``force-level2``.
It specifies the input images that are to be processed.

One image per line should be given.
The full file paths must be given.
No white spaces should be present in the file paths.
The file is specified with ``FILE_QUEUE`` in the Level 2 :ref:`l2-param`.

Each image is followed by ``QUEUED``, ``FAIL``, or ``DONE``, which indicates the queue status.
Queued images will be processed, and the queue status will be changed to ``DONE`` after Level 2 processing.
If a reprocessing is required, the queue status needs to be changed to ``QUEUED``, e.g. using

.. code-block:: bash

  sed -i 's/DONE/QUEUED/' queue.txt

  
Although not specifically required, we recommend to use a consistent and clean data pool that contains all (and nothing else) input images without duplicates or different processing versions. 


The queues look like below. 
Note that this examples shows all the different options to specify the input images. 
If you are using the same parameterization for Landsat and Sentinel-2 (highly recommended), you can use the same file queue for Landsat and Sentinel-2 to process all images at once.
You can give the compressed or extracted images. 
For Sentinel-2, it is possible to only give the filepath of the top directory (``.SAFE``).

.. code-block:: bash

  /data/level1/landsat/177072/LT51770722008065JSA00.tar.gz QUEUED
  /data/level1/landsat/177072/LC81770722014129LGN00 QUEUED
  /data/level1/sentinel/T33LZE/S2A_MSIL1C_20170706T083601_N0205_R064_T33LZE_20170706T090107.zip QUEUED
  /data/level1/sentinel/T33LZF/S2A_MSIL1C_20170706T083601_N0205_R064_T33LZF_20170706T090107.SAFE QUEUED

