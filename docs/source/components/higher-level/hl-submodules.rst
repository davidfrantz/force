.. _hl-submodules:

Submodules
==========

There are multiple submodules available, which implement different workflows.
They all share the higher-level :ref:`hl-compute`.
There is one executable only, i.e. ``force-higher-level``. 
Depending on the parameter file, the program will figure out which submodule to execute.


**Table 2** Submodules of HLPS.

+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ Submodule     + Level + :ref:`hl-input` + Short description                                                                                                                           +
+===============+=======+=================+=============================================================================================================================================+
+ :ref:`level3` + 3     + ARD             + Generate temporal aggregations of Level 2 ARD, i.e. pixel-based composites                                                                  +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`cso`    + 3     + ARD             + Statistics for Level 2-3 ARD data availability mining                                                                                       +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`tsa`    + 3-4   + ARD             + Multitemporal analysis and processing based on Level 2-3 ARD                                                                                +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`ml`     + 4     + feature         + Model predictions based on any cubed features                                                                                               +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`txt`    + 3-4   + feature         + Morphological transformations based on any cubed features                                                                                   +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`lsm`    + 3-4   + feature         + Quantification of spatial patterns based on any cubed features                                                                              +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`smp`    + /     + feature         + Point-based extraction of features for training/validation purposes                                                                         +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`cfi`    + 4     + ARD             + Increase the spatial resolution of coarse continuous fields (like Land Surface Phenology) to Level 2 ARD resolution using the ImproPhe code +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`l2i`    + 2     + ARD             + Increase the spatial resolution of lower resolution Level 2 ARD using higher resolution Level 2 ARD using the ImproPhe code                 +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+
+ :ref:`udf`    + 3-4   + ARD             + Custom processing using User-Defined Functions                                                                                              +
+---------------+-------+-----------------+---------------------------------------------------------------------------------------------------------------------------------------------+


.. toctree::
   :hidden:
   :maxdepth: 2

   l3/index.rst
   cso/index.rst
   tsa/index.rst
   ml/index.rst
   txt/index.rst
   lsm/index.rst
   smp/index.rst
   cfi/index.rst
   l2i/index.rst

