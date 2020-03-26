.. _level3-format:

Output Format
=============

Data organization
^^^^^^^^^^^^^^^^^

The data are organized in a gridded data structure, i.e. data cubes.
The tiles manifest as directories in the file system, and the images are stored within.

.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains what a datacube is, how it is parameterized, how you can find a POI, how to visualize the tiling grid, and how to conveniently display cubed data.


Data Cube definition
^^^^^^^^^^^^^^^^^^^^

The spatial data cube definition is appended to each data cube, i.e. to each directory containing tiled datasets, see :ref:`datacube-def`.


Naming convention
^^^^^^^^^^^^^^^^^

Following 29-digit naming convention is applied to all output files:

Example filename: 20160701_LEVEL3_LNDLG_BAP.tif

+--------+-------+-----------------------------------------+
+ Digits + Description                                     +
+========+=======+=========================================+
+ 1–8    + Target date as YYYYMMDD                         +
+--------+-------+-----------------------------------------+
+ 10–15  + Product Level                                   +
+--------+-------+-----------------------------------------+
+ 17–21  + Sensor ID                                       +
+        +-------+-----------------------------------------+
+        + LNDLG + Landsat legacy bands                    +
+        +-------+-----------------------------------------+
+        + SEN2L + Sentinel-2 land bands                   +
+        +-------+-----------------------------------------+
+        + SEN2H + Sentinel-2 high-res bands               +
+        +-------+-----------------------------------------+
+        + R-G-B + Visible bands                           +
+        +-------+-----------------------------------------+
+        + VVVHP + VV/VH Dual Polarized                    +
+--------+-------+-----------------------------------------+
+ 23–25  + Product Type                                    +
+        +-------+-----------------------------------------+
+        + BAP   + Best Available Pixel composite          +
+        +-------+-----------------------------------------+
+        + INF   + Compositing Information                 +
+        +-------+-----------------------------------------+
+        + SCR   + Compositing Score                       +
+--------+-------+-----------------------------------------+
+ 27–29  + File extension                                  +
+        +-------+-----------------------------------------+
+        + tif   + image data in compressed GeoTiff format +
+        +-------+-----------------------------------------+
+        + dat   + image data in flat binary ENVI format   +
+        +-------+-----------------------------------------+
+        + hdr   + metadata for ENVI format                +
+--------+-------+-----------------------------------------+


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

* Best Available Pixel composite

  A Best Available Pixel (BAP) composite is a reflectance product.
  It holds the reflectance of the most suitable observation - for each pixel.
  The scale is 10000, and nodata value is -9999.
  The product contains multiple bands, which represent wavelengths.
  The number of bands is dependent on the ``SENSOR`` used.
  All overlapping bands are used, and the output product is named according to this band set (see table above).

* Compositing Information

  The Compositing Information (INF) product contains information about the selected observation in the BAP product. It is a multi-band image:
  
  +------+----------------------------------------------------------------------------+
  + Band + Description                                                                +
  +======+============================================================================+
  + 1    + Quality Assurance Information of best observation (see :ref:`qai`)         +
  +------+----------------------------------------------------------------------------+
  + 2    + Number of cloud-free observations within compositing period                +
  +------+----------------------------------------------------------------------------+
  + 3    + Acquisition DOY of best observation                                        +
  +------+----------------------------------------------------------------------------+
  + 4    + Acquisition Year of best observation                                       +
  +------+----------------------------------------------------------------------------+
  + 5    + Difference between band 3 and Target DOY                                   +
  +------+----------------------------------------------------------------------------+
  + 6    + Sensor ID  of best observation (in the order given in the :ref:`l3-param`) +
  +------+----------------------------------------------------------------------------+

* Compositing Score

  The Compositing Score (SCR) product contains the scores of the selected observation in the BAP product.
  The score is between 0 and 1, the scaling factor is 10000. 
  It is a multi-band image:
  
  +------+---------------------------------+
  + Band + Description                     +
  +======+=================================+
  + 1    + Total score                     +
  +------+---------------------------------+
  + 2    + DOY score (intra-annual score)  +
  +------+---------------------------------+
  + 3    + Year score (inter-annual score) +
  +------+---------------------------------+
  + 4    + Cloud distance score            +
  +------+---------------------------------+
  + 5    + Haze score                      +
  +------+---------------------------------+
  + 6    + Correlation score               +
  +------+---------------------------------+
  + 7    + View angle score                +
  +------+---------------------------------+


Quicklooks
^^^^^^^^^^

If ``OUTPUT_OVV = TRUE``, small quicklooks images are generated,
The quicklooks are fixed-stretch images.
For optical images, they are RGB quicklooks.
For radar images, RGB refer to VV, VH, and VV/VH ratio.

