.. _howto:

Tutorials
=========

There are several tutorials available that showcase how to use the FORCE.


|--------------|--------|---------------------|----------------------------------------------------------------|
| Level        | Module | Tutorial            | Learning Objective                                             |
|==============|========|=====================|================================================================|
| Essential    |        | :ref:`tut-datacube` | How to handle cubed data                                       |
|--------------|--------|---------------------|----------------------------------------------------------------|
| Lower Level  | L1AS   | :ref:`tut-s2l1c`    | How to download and manage Sentinel-2 Level 1C data            |
|              |--------|---------------------|----------------------------------------------------------------|
|              | L2PS   | :ref:`tut-ard`      | How to generate Analysis Ready Data                            |
|              |        |---------------------|----------------------------------------------------------------|
|              |        | :ref:`tut-coreg`    | How to coregister Sentinel-2 with Landsat                      |
|              |        |---------------------|----------------------------------------------------------------|
|              |        | :ref:`tut-qai`      | How to deal with the Level 2 quality masks                     |
|              |        |---------------------|----------------------------------------------------------------|
|              |        | :ref:`tut-dem`      | How to prepare a DEM for Level 2 Processing                    |
|              |--------|---------------------|----------------------------------------------------------------|
|              | WVDB   | :ref:`tut-wvdb`     | How to prepare the Water Vapor Database for Level 2 Processing |
|--------------|--------|---------------------|----------------------------------------------------------------|
| Higher Level | HLPS   | :ref:`tut-tsi`      | How to interpolate and animate time series                     |
|              |        |---------------------|----------------------------------------------------------------|
|              |        | :ref:`tut-stm`      | How to aggregate time series                                   |
|              |        |---------------------|----------------------------------------------------------------|
|              |        | :ref:`tut-mask`     | Speed up Higher Level Processing using masks                   |
|--------------|--------|---------------------|----------------------------------------------------------------|


.. toctree::
   :maxdepth: 1
   :hidden:

   sentinel2-l1c.rst

   datacube.rst
   l2-ard.rst
   coreg.rst
   qai.rst
   wvdb.rst
   dem.rst

   masks.rst
   tsi.rst
   stm.rst
