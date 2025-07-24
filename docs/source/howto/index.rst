.. _howto:

Tutorials
=========

There are several tutorials available that showcase how to use the FORCE.

Please note that all users are warmly welcome to share their own tutorials!


+--------------+--------+---------------------+----------------------------------------------------------------------------+
+ Level        + Module + Tutorial            + Learning Objective                                                         +
+==============+========+=====================+============================================================================+
+ Essential    +        + :ref:`tut-datacube` + How to handle cubed data                                                   +
+--------------+--------+---------------------+----------------------------------------------------------------------------+
+ Lower Level  + L1AS   + :ref:`tut-l1csd`    + How to download Landsat and Sentinel-2 Level-1 data from cloud services    +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-s2l1c`    + How to download and manage Sentinel-2 Level 1C data from the ESA hub       +
+              +--------+---------------------+----------------------------------------------------------------------------+
+              + L2PS   + :ref:`tut-ard`      + How to generate Analysis Ready Data                                        +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-coreg`    + How to coregister Sentinel-2 with Landsat                                  +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-qai`      + How to deal with the Level 2 quality masks                                 +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-dem`      + How to prepare a DEM for Level 2 Processing                                +
+              +--------+---------------------+----------------------------------------------------------------------------+
+              + WVDB   + :ref:`tut-wvdb`     + How to prepare the Water Vapor Database for Level 2 Processing             +
+--------------+--------+---------------------+----------------------------------------------------------------------------+
+ Higher Level + HLPS   + :ref:`tut-tsi`      + How to interpolate and animate time series                                 +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-stm`      + How to aggregate time series                                               +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-mask`     + Speed up Higher Level Processing using masks                               +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-udf_py`   + How to customize your processing using Python User-Defined Functions (UDF) +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-udf_r`    + How to customize your processing using R User-Defined Functions (UDF)      +
+              +        +---------------------+----------------------------------------------------------------------------+
+              +        + :ref:`tut-lcf`      + How to map land cover fractions with synthetically mixed training data     +
+--------------+--------+---------------------+----------------------------------------------------------------------------+


.. toctree::
   :maxdepth: 1
   :hidden:

   datacube.rst
   level1-csd.rst
   sentinel2-l1c.rst
   l2-ard.rst
   coreg.rst
   qai.rst
   dem.rst
   wvdb.rst
   tsi.rst
   stm.rst
   masks.rst
   udf_py.rst
   udf_r.rst
   lcf.rst
   