.. _higher-level:

Higher Level
============

About
^^^^^

The FORCE Higher Level Processing System (HLPS) provides functionality for Higher Level Processing of cubed ARD and feature datasets.

Some more text and description on how it works.

Tiles, Blocks = Processing UNits.
Prepare/Update figure for this.

Read/compute/output teams, streaming..
Prepare Figure for this?

force-higher-level for all submodules.


**Table** Submodules.

+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| Submodule     | Level | Short description                                                                                                                           +
+===============+=======+=============================================================================================================================================+
| :ref:`level3` | 3     | Generate temporal aggregations of Level 2 ARD, i.e. pixel-based composites                                                                  +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`cso`    | 3     | Statistics for Level 2-3 ARD data availability mining                                                                                       +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`tsa`    | 3-4   | Multitemporal analysis and processing based on Level 2-3 ARD                                                                                +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`ml`     | 4     | Model predictions based on any cubed features                                                                                               +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`txt`    | 4     | Morphological transformations based on any cubed features                                                                                   +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`lsm`    | 4     | Quantification of spatial patterns based on any cubed features                                                                              +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`smp`    | /     | Point-based extraction of features for training/validation purposes                                                                         +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`cfi`    | 4     | Increase the spatial resolution of coarse continuous fields (like Land Surface Phenology) to Level 2 ARD resolution using the ImproPhe code +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+
| :ref:`l2i`    | 2     | Increase the spatial resolution of lower resolution Level 2 ARD using higher resolution Level 2 ARD using the ImproPhe code                 +
+---------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------+


Usage
^^^^^

.. code-block:: bash

  force-higher-level

  Usage: force-higher-level parameter-file

* parameter-file

  | Any higher-level parameter file needs to be given as sole argument.
  | Depending on the parameter file, the program will figure out which submodule to execute, e.g. time series analysis or machine learning predictions.


.. _tilelist:
  
Tile white-list
^^^^^^^^^^^^^^^  

Tile white-lists are optional, and can be used to limit the analysis extent to non-square extents.
The white list is intersected with the analysis extent, i.e. only tiles included in both the analysis extent AND the white-list will be processed.
It is specified via ``FILE_TILE`` in the parameter files.

As an example, if the extent of your study area (e.g. country) is 4x5 tiles, but the study area is actually only covered by 10 tiles (X-Signature), you will save 50% processing time when using a tile white-list.

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

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains how to visualize the tiling grid using :ref:`tabulate-grid`.
  

.. _processing-masks:

Processing masks
^^^^^^^^^^^^^^^^

Processing masks can be used to restrict processing and analysis to certain pixels of interest. 
The masks need to be in datacube format, i.e. they need to be raster images in the same grid as all the other data. 
The masks can - but don’t need to - be in the same directory as the other data. 
The masks should be binary images. 
The pixels that have a mask value of 0 will be skipped.

.. seealso::

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-masks/masks/>`_, which explains what processing masks are, why they are super-useful, how to generate them, and how to use them in the FORCE Higher Level Processing System.


+---------------+------------+------------+
+ :ref:`level3` + :ref:`cso` + :ref:`tsa` +
+---------------+------------+------------+
+ :ref:`ml`     + :ref:`txt` + :ref:`lsm` +
+---------------+------------+------------+
+ :ref:`l2i`    + :ref:`cfi` + :ref:`l2i` +
+---------------+------------+------------+
 
 
.. toctree::
   :hidden:
   :maxdepth: 2

   l3/index.rst
   tsa/index.rst
   cso/index.rst
   ml/index.rst
   txt/index.rst
   lsm/index.rst
   smp/index.rst
   l2i/index.rst
   cfi/index.rst

