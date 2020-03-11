.. _l1as:

Level 1 Archiving Suite
=======================

Add more about information.

The FORCE Level 1 Archiving Suite (FORCE L1AS) is intended to assist in organizing and maintaining a clean and consistent Level 1 data pool, as well as downloading of Sentinel-2 data.
It is attempted to reduce redundancy and supports versioning, e.g. by removing old data if new processing versions are available.
In addition, FORCE L1AS assists in building and updating the file queues needed for :ref:`l2ps`:

+-----------------------+-------------------------+
+ :ref:`level1-landsat` + :ref:`level1-sentinel2` +
+-----------------------+-------------------------+


.. image:: L1AS.jpg

**Figure.** FORCE Level 1 Archiving Suite (L1AS) workflow. 

The main difference is that Landsat data need to be downloaded manually, while Sentinel-2 images are automatically retrieved by FORCE. 
On successful ingestion, the image is appended to a file queue, which controls Level 2 processing. 
The file queue is a text file that holds the full path to the image, as well as a processing-state flag. 
This flag is either ``QUEUED`` or ``DONE``, which means that it is enqueued for Level 2 processing or was already processed and will be ignored next time.


.. toctree::
   :maxdepth: 1
   :hidden:

   level1-landsat.rst
   level1-sentinel2.rst
