.. _queue:

File queue
==========

The file queue is mandatory for ``force-level2``.
It specifies the input images that are to be processed.

One image per line should be given.
The full file paths must be given.
No white spaces should be present in the file paths.
The file is specified with ``FILE_QUEUE`` in the Level 2 :ref:`l2-param`.

Each image is followed by ``QUEUED`` or ``DONE``, which indicates the queue status.
Queued images will be processed, and the queue status will be changed to ``DONE`` after Level 2 processing.
If a reprocessing is required, the queue status needs to be changed to ``QUEUED``, e.g. using

.. code-block:: bash

  sed -i ‘s/DONE/QUEUED/’ queue.txt

  
File queues can be generated – and updated with new acquisitions – using the :ref:`l1as`.

Although not specifically required, we recommend to use a consistent and clean data pool that contains all (and nothing else) input images without duplicates or different processing versions. 
:ref:`l1as` assists in generating and maintaining clean data pools.


The queues look like below. 
Note that this examples shows all the different options to specify the input images. 
If you are using the same parameterization for Landsat and Sentinel-2 (highly recommended), you can use the same file queue for Landsat and Sentinel-2 to process all images at once.
You can give the compressed or extracted images. 
For Sentinel-2, it is possible to only give the filepath of the top directory (``.SAFE``).
However, if the image follows the outdated file structure with multiple granules, only the first granule will be processed. 
For the sake of backward compatibility, it is still possible to give the filepath of the granule (last line below).


.. code-block:: bash

  /data/level1/landsat/177072/LT51770722008065JSA00.tar.gz QUEUED
  /data/level1/landsat/177072/LC81770722014129LGN00 QUEUED
  /data/level1/sentinel/T33LZE/S2A_MSIL1C_20170706T083601_N0205_R064_T33LZE_20170706T090107.zip QUEUED
  /data/level1/sentinel/T33LZF/S2A_MSIL1C_20170706T083601_N0205_R064_T33LZF_20170706T090107.SAFE QUEUED
  /data/level1/sentinel/T33LYC/S2A_MSIL1C_20170706T083601_N0205_R064_T33LYC_20170706T090107.SAFE/GRANULE/L1C_T33LYC_A010643_20170706T090107 QUEUED

