.. _wvdb:

Water Vapor Database
====================


The FORCE Water Vapor Database (FORCE WVDB) component can be used to generate and maintain a water vapor database used for atmospheric correction of Landsat data (within FORCE L2PS). Instead of running this component on your own, an application-ready global water vapor database (2000–July 2018) can be downloaded from https://doi.org/10.1594/PANGAEA.893109. The water vapor database is not used when processing Sentinel-2 data because water vapor can be estimated from the images themselves. 



A glimpse of what you get:

 
Fig. 10. Water vapor climatology.
Global, and monthly water vapor climatology for each land-intersecting Landsat WRS-2 scene. 
[The daily water vapor database, and the climatology were generated with force-lut-modis.
This dataset is freely available here: https://doi.org/10.1594/PANGAEA.893109]


.. toctree::
   :maxdepth: 2

   process.rst

