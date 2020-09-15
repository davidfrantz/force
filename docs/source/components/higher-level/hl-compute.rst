.. _hl-compute:

Compute model
=============

The conceptual figures below explain the general concept of the higher-level processing strategy, compute model, and nested parallelism. 
The individual sub-figures can be enlarged by clicking on them.

.. |hl-compute1-text| replace:: The cubed data are stored in a grid system. Each tile has a unique tile ID, which consists of an X-ID, and a Y-ID. The numbers increase from left ro right, and from top to bottom. In a first step, a rectangular extent needs to be specified using the tile X- and Y-IDs. In this example, we have selected the extent covering Belgium, i.e. 9 tiles.
.. |hl-compute2-text| replace:: If you do not want to process all tiles, you can use a :ref:`tilelist`. The allow-list is intersected with the analysis extent, i.e. only tiles included in both the analysis extent AND the allow-list will be processed. This is optional.
.. |hl-compute3-text| replace:: The image chips in each tile have an internal block structure for partial image access. These blocks are strips that are as wide as the ``TILE_SIZE`` and as high as the ``BLOCK_SIZE``. The blocks are the main processing units (PU), and are processed sequentially, i.e. one after another.
.. |hl-compute4-text| replace:: FORCE uses a streaming strategy, where three teams take care of reading, computing and writing data. The teams work simultaneously, e.g. input data for PU 19 is read, pre-loaded data for PU 18 is processed, and processed results for PU 17 are written - at the same time. If processing takes longer than I/O, this streaming strategy avoids idle CPUs waiting for delivery of input data. Optionally, :ref:`processing-masks` can be used, which restrict processing and analysis to certain pixels of interest. Processing units, which do not contain any active pixels, are skipped (in this case, the national territory of Belgium).
.. |hl-compute5-text| replace:: Each team can use several threads to further parallelize the work. In the input team, multiple threads read multiple input images simultaneously, e.g. different dates of ARD. In the computing team, the pixels are distributed to different threads (please note that the actual load distribution may differ from the idealized figure due to load balancing etc.). In the output team, multiple threads write multiple output products simultaneously, e.g. different Spectral Temporal Metrics.

.. |hl-compute1-image| image:: hl-1.jpg
   :width: 70%
.. |hl-compute2-image| image:: hl-2.jpg
   :width: 70%
.. |hl-compute3-image| image:: hl-3.jpg
   :width: 70%
.. |hl-compute4-image| image:: hl-4.jpg
   :width: 70%
.. |hl-compute5-image| image:: hl-5.jpg
   :width: 70%

+----+--------------------+---------------------+
+ 1. + |hl-compute1-text| + |hl-compute1-image| +
+----+--------------------+---------------------+
+ 2. + |hl-compute2-text| + |hl-compute2-image| +
+----+--------------------+---------------------+
+ 3. + |hl-compute3-text| + |hl-compute3-image| +
+----+--------------------+---------------------+
+ 4. + |hl-compute4-text| + |hl-compute4-image| +
+----+--------------------+---------------------+
+ 5. + |hl-compute5-text| + |hl-compute5-image| +
+----+--------------------+---------------------+

