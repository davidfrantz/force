.. _cso-format:

Output Format
=============

Data organization
^^^^^^^^^^^^^^^^^

The data are organized in a gridded data structure, i.e. data cubes.
The tiles manifest as directories in the file system, and the images are stored within. For mosaicking, use force-mosaic.

.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains what a datacube is, how it is parameterized, how you can find a POI, how to visualize the tiling grid, and how to conveniently display cubed data.


Data Cube definition
^^^^^^^^^^^^^^^^^^^^

The spatial data cube definition is appended to each data cube, i.e. to each directory containing tiled datasets, see :ref:`datacube-def`.

Naming convention
^^^^^^^^^^^^^^^^^

Following 41-digit naming convention is applied to all output files:

Example filename: 2000-2010_001-365-03_HL_CSO_LNDLG_NUM.tif

There are several product types available, all are optional. 
The Number of Observations (``NUM``) product contains the number of clear sky observations for each pixel and bin. 
The other products are statistical summaries of the temporal difference (dt) between consecutive observations within these bins.


**Table 1:** Naming convention

+----------------+---------+---------------------------------------------------------+
+ Digits         + Description                                                       +
+================+=========+=========================================================+
+ 1–9            + Temporal range for the years as YYYY–YYYY                         +
+----------------+---------+---------------------------------------------------------+
+ 11-17          + Temporal range for the DOY as DDD–DDD                             +
+----------------+---------+---------------------------------------------------------+
+ 19-20          + Temporal binning in months                                        +
+----------------+---------+---------------------------------------------------------+
+ 22-23          + Product Level                                                     +
+----------------+---------+---------------------------------------------------------+
+ 25-27          + Submodule                                                         +
+----------------+---------+---------------------------------------------------------+
+ 29-33          + Sensor ID                                                         +
+                +---------+---------------------------------------------------------+
+                + LNDLG   + Landsat legacy bands                                    +
+                +---------+---------------------------------------------------------+
+                + SEN2L   + Sentinel-2 land bands                                   +
+                +---------+---------------------------------------------------------+
+                + SEN2H   + Sentinel-2 high-res bands                               +
+                +---------+---------------------------------------------------------+
+                + R-G-B   + Visible bands                                           +
+                +---------+---------------------------------------------------------+
+                + VVVHP   + VV/VH Dual Polarized                                    +
+----------------+---------+---------------------------------------------------------+
+ 35-37          + Product Type                                                      +
+                +---------+---------------------------------------------------------+
+                + NUM     + Number of Observations                                  +
+                +---------+---------------------------------------------------------+
+                + AVG     + Average of days between observations (dt)               +
+                +---------+---------------------------------------------------------+
+                + STD     + Standard Deviation of dt                                +
+                +---------+---------------------------------------------------------+
+                + MIN     + Minimum of dt                                           +
+                +---------+---------------------------------------------------------+
+                + MAX     + Maximum of dt                                           +
+                +---------+---------------------------------------------------------+
+                + RNG     + Range of dt                                             +
+                +---------+---------------------------------------------------------+
+                + SKW     + Skewness of dt                                          +
+                +---------+---------------------------------------------------------+
+                + KRT     + Kurtosis of dt                                          +
+                +---------+---------------------------------------------------------+
+                + QXX     + Quantile of dt (e.g. 0.25 quantile = Q25)               +
+                +---------+---------------------------------------------------------+
+                + IQR     + Interquartile Range of dt                               +
+----------------+---------+---------------------------------------------------------+
+ 39-41          + File extension                                                    +
+                +---------+---------------------------------------------------------+
+                + tif     + image data in compressed GeoTiff format                 +
+                +---------+---------------------------------------------------------+
+                + dat     + image data in flat binary ENVI format                   +
+                +---------+---------------------------------------------------------+
+                + hdr     + metadata for ENVI format                                +
+----------------+---------+---------------------------------------------------------+


File format
^^^^^^^^^^^

The images are provided with signed 16bit datatype and band sequential (BSQ) interleaving in one of the following formats:

* GeoTiff 
  
  This is the recommended output option. 
  Images are compressed GeoTiff images using LZW compression with horizontal differencing.
  The images are generated with internal blocks for partial image access.
  These blocks are strips that are as wide as the ``TILE_SIZE`` and as high as the ``BLOCK_SIZE``.
  
* ENVI Standard format

  This produces flat binary images without any compression.
  This option might seem tempting as there is no overhead in cracking the compression when reading these data.
  However, the transfer of the larger data volume from disc to CPU often takes longer than cracking the compression.
  Therefore, we recommend to use the GeoTiff option.


Metadata
^^^^^^^^

Metadata are written to all output products.
For ENVI format, the metadata are written to the ENVI header (``.hdr`` extension).
For GeoTiff format, the metadata are written into the GeoTiff file.
If the metadata is larger than allowed by the GeoTiff driver, the excess metadata will be written to an "auxiliary metadata" file with ``.aux.xml`` extension.
FORCE-specific metadata will be written to the FORCE domain, and thus are probably not visible unless the FORCE domain (or all domains) are specifically printed:

.. code-block:: bash

  gdalinfo -mdd all 20160823_LEVEL2_SEN2A_BOA.tif


Product type
^^^^^^^^^^^^

* Clear Sky Observation Statistcs

  Currently available statistics are the number of observations, and aggregate statistics of the temporal difference between observations 
  (available are average, standard deviation, minimum, maximum, range, skewness, kurtosis, any quantile from 1-99%, and interquartile range.
