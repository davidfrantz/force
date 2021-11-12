.. _hl-aux:

Auxiliary data
==============

FORCE HLPS supports the optional usage of several auxiliary datasets as explained below.

.. _tilelist:

Tile allow-list
^^^^^^^^^^^^^^^

Tile allow-lists are optional, and can be used to limit the analysis extent to non-square extents.
The allow-list is intersected with the analysis extent, i.e. only tiles included in both the analysis extent AND the allow-list will be processed.
It is specified via ``FILE_TILE`` in the parameter files.

As an example, if the extent of your study area (e.g. country) is 4x5 tiles, but the study area is actually only covered by 10 tiles (X-Signature), you will save 50% processing time when using a tile allow-list.

+---+---+---+---+---+
+ / + X + X + / + / +
+---+---+---+---+---+
+ X + X + X + X + / +
+---+---+---+---+---+
+ / + / + X + X + X +
+---+---+---+---+---+
+ / + / + X + / + / +
+---+---+---+---+---+

The file must be prepared as follows: the 1st line must give the number of tiles for which output should be created.
The corresponding tile IDs must be given in the following lines, one ID per line; end with an empty line.
The sorting does not matter.
Truncated example:

.. code-block:: bash

  4524
  X0044_Y0014
  X0044_Y0015
  X0045_Y0013
  X0045_Y0014
  X0045_Y0015
  X0045_Y0016
  X0045_Y0017
  X0045_Y0018
  ...

.. seealso::

  Check out the `datacube tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains how to visualize the tiling grid using :ref:`tabulate-grid`.
  

.. _processing-masks:

Processing masks
^^^^^^^^^^^^^^^^

Processing masks can be used to restrict processing and analysis to certain pixels of interest. 
The masks need to be in datacube format, i.e. they need to be raster images in the same grid as all the other data. 
The masks can - but don’t need to - be in the same directory as the other data. 
The masks should be binary images. 
The pixels that have a mask value of 0 will be skipped.

.. seealso::

  Check out the `processing mask tutorial <https://davidfrantz.github.io/tutorials/force-masks/masks/>`_, which explains what processing masks are, why they are super-useful, how to generate them, and how to use them in the FORCE Higher Level Processing System.


