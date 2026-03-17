.. _eco_overview:

Additional Tools and Resources
==============================

This page provides provides a curated list of software tools and resources 
that are related to or can be used in conjunction with FORCE.


Download tools
--------------

landsatlinks
++++++++++++

`landsatlinks <https://github.com/ernstste/landsatlinks>`_ by Stefan Ernst offers a simple command line interface 
to retrieve download links for Landsat Collection 2 Level 1 product bundles through the USGS/EROS machine-to-machine API.
This tool has been shipped under the name ``force-level1-landsat`` in past FORCE releases, 
but is available as a standalone tool with the same usage as before.
It provides built-in support for checking download links against FORCE logfiles and only retrieve unprocessed images.
It can also produce file queues for Level 2 processing with ``force-level2``.
A docker image is available.

CDSE_Sentinel2_downloader
+++++++++++++++++++++++++

The `CDSE_Sentinel2_downloader <https://github.com/vudongpham/CDSE_Sentinel2_downloader>`_ by Vu-Dong Pham is a 
Python tool to retrieve Sentinel-2 Level 1C products from the Copernicus Data Space Ecosystem through the OData API.
It provides built-in support for checking download links against FORCE logfiles and only retrieve unprocessed images.

QGIS Plugins
------------

Other
-----

https://github.com/vudongpham/FORCE-Live

https://github.com/leonsnill/geeo

https://github.com/felixlobert/force-sar

https://github.com/Florian-Katerndahl/haze

https://github.com/maxfreu/ForceCubeAccess.jl

