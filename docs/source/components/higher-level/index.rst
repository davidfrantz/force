.. _hlps:

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


**Table 1** Higher Level module.

+--------+-------------------------+-------+-----------------------------------------------------------------------------------------------------+
| Module | Program                 | Level | Short description                                                                                   |
+========+=========================+=======+=====================================================================================================+
+--------+-------------------------+-------+-----------------------------------------------------------------------------------------------------+
| HLPS   | The FORCE Higher Level Processing System (HLPS) provides functionality for Higher Level Processing of cubed ARD and feature datasets. |
+        +-------------------------+-------+-----------------------------------------------------------------------------------------------------+
|        | :ref:`higher-level`     | 3-4   | Higher Level processing                                                                             |
+--------+-------------------------+-------+-----------------------------------------------------------------------------------------------------+


**Table 2** Submodules of HLPS.

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

 
.. toctree::
   :hidden:
   :maxdepth: 2

   higher-level.rst
   l3/index.rst
   tsa/index.rst
   cso/index.rst
   ml/index.rst
   txt/index.rst
   lsm/index.rst
   smp/index.rst
   l2i/index.rst
   cfi/index.rst

