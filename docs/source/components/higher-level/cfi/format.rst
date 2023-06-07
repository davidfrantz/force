.. _cfi-format:

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

Following 21-digit naming convention is applied to all output files:

Example filename: 2017_IMPROPHE_IGS.tif

**Table 1:** Naming convention

+----------------+---------+---------------------------------------------------------+
+ Digits         + Description                                                       +
+================+=========+=========================================================+
+ 1–4	           + Year                                                              +
+----------------+---------+---------------------------------------------------------+
+ 6–13	         + Processing Type                                                   +
+                +---------+---------------------------------------------------------+
+                + IMPROPHE                                                          +
+----------------+---------+---------------------------------------------------------+
+ 15–17	         + Product Tag                                                       +
+                +---------+---------------------------------------------------------+
+                + XXX	   + These 3-digit tags are specified in the parameter file  +
+----------------+---------+---------------------------------------------------------+
+ 19–21	         + File extension                                                    +
+                +---------+---------------------------------------------------------+
+                + tif		 + image data in compressed GeoTiff format                 +
+                +---------+---------------------------------------------------------+
+                + dat		 + image data in flat binary ENVI format                   +
+                +---------+---------------------------------------------------------+
+                + hdr		 + metadata                                                +
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

* High resolution continuous fields
