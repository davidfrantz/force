.. _higher-level:

Higher Level
============

**Table 1.** Components.

+--------+------------------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
| Module | Program                | Level | Short description                                                                                                                                       +
+========+========================+=======+=========================================================================================================================================================+
| HLPS   | The FORCE Higher Level Processing System provides functionality for Higher Level Processing of cubed ARD and feature datasets                                                            +
+        +------------------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        | force-higher-level     | 3     | :ref:`level3`: Generate temporal aggregations of Level 2 ARD, i.e. pixel-based composites                                                               +
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 3     | :ref:`cso`: Statistics for Level 2-3 ARD data availability mining                                                                                       +
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 3-4   | :ref:`tsa`: Multitemporal analysis and processing based on Level 2-3 ARD                                                                                +
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 4     | :ref:`ml`: Model predictions based on any cubed features                                                                                                +
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 4     | :ref:`txt`: Morphological transformations based on any cubed features                                                                                   |
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 4     | :ref:`lsm`: Quantification of spatial patterns based on any cubed features                                                                              |
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | /     | :ref:`l2i`: Point-based extraction of features for training/validation purposes                                                                         |
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 4     | :ref:`cfi`: Increase the spatial resolution of coarse continuous fields (like Land Surface Phenology) to Level 2 ARD resolution using the ImproPhe code |
+        +                        +-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+
|        |                        | 2     | :ref:`l2i`: Increase the spatial resolution of lower resolution Level 2 ARD using higher resolution Level 2 ARD using the ImproPhe code                 |
+--------+------------------------+-------+---------------------------------------------------------------------------------------------------------------------------------------------------------+


Usage
^^^^^

.. code-block:: bash

  force-higher-level

  Usage: force-higher-level parameter-file

* parameter-file

  | Any higher-level parameter file needs to be given as sole argument.
  | Depending on the parameter file, the program will figure out which submodule to execute, e.g. time series analysis or machine learning predictions.


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

