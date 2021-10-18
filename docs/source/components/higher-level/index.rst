.. _hlps:

Higher Level
============

The FORCE Higher Level Processing System (HLPS) provides functionality for Higher Level Processing.

HLPS consists of one executable only, i.e. ``force-higher-level``. 
Multiple :ref:`hl-submodules` are available, which either process ARD or feature datasets (see :ref:`hl-input`). 


**Table 1** Higher Level module.

+--------+-------------------------+-------+-----------------------------------------------------------------------------------------------------+
+ Module + Program                 + Level + Short description                                                                                   +
+========+=========================+=======+=====================================================================================================+
+ HLPS   + The FORCE Higher Level Processing System (HLPS) provides functionality for Higher Level Processing of cubed ARD and feature datasets. +
+        +-------------------------+-------+-----------------------------------------------------------------------------------------------------+
+        + :ref:`higher-level`     + 3-4   + Higher Level processing                                                                             +
+--------+-------------------------+-------+-----------------------------------------------------------------------------------------------------+


.. higher-level:

**Usage**

.. code-block:: bash

  force-higher-level

  Usage: force-higher-level parameter-file

* parameter-file

  | Any higher-level parameter file needs to be given as sole argument.
  | Depending on the parameter file, the program will figure out which submodule to execute, e.g. time series analysis or machine learning predictions.


.. toctree::
   :hidden:
   :maxdepth: 3

   hl-input.rst
   hl-compute.rst
   hl-aux.rst
   hl-submodules.rst

