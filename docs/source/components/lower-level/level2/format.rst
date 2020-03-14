.. _level2-format:

Output Format
=============


Data organization
^^^^^^^^^^^^^^^^^

Depending on parameterization (``DO_REPROJ`` / ``DO_TILE``), the output data are organized according to their original spatial reference system (WRS-2 frames / MGRS zones) or are provided in a gridded data structure as ARD (strongly recommended!), i.e. data cubes.
The tiles (or original reference system) manifest as directories in the file system, and the images are stored within.
The user can choose to keep the original projection (UTM) or to reproject all data to one consistent projection (strictly recommended for ARD!).


.. warning::

  If you are not using the datacube options, i.e. ``DO_REPROJ = FALSE`` or ``DO_TILE = FALSE``, you are running into a *dead end* for FORCE. In this case, the data cannot be further processed or analysed with any :ref:`higher-level`.


.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains what a datacube is, how it is parameterized, how you can find a POI, how to visualize the tiling grid, and how to conveniently display cubed data.


Data Cube definition
^^^^^^^^^^^^^^^^^^^^

The spatial data cube definition is appended to each data cube, i.e. to each directory containing tiled datasets.
The file ``datacube-definition.prj`` is a 7-line text file that contains the (1) projection as WKT string, (2) origin of the tile system as geographic Longitude, (3) origin of the tile system as geographic Latitude, (4) origin of the tile system as projected X-coordinate, (5) origin of the tile system as projected Y-coordinate, (6) size of the tiles in projection units, and (7) block size within each tile.

.. warning::

  Do not modify or delete any of these files!

The datacube definition file looks like this:

.. code-block:: bash

  PROJCS["ETRS89 / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","3035"]]
  -25.000000
  60.000000
  2456026.250000
  4574919.500000
  30000.000000
  3000.0000000


Naming convention
^^^^^^^^^^^^^^^^^

Following 29-digit naming convention is applied to all output files:

Example filename: 20160823_LEVEL2_SEN2A_BOA.tif

+--------+----------------------------------------------+
+ Digits + Description                                  +
+========+==============================================+
+ 1–8    + Acquisition date as YYYYMMDD                 +
+--------+----------------------------------------------+
+ 10–15  + Product Level                                +
+--------+----------------------------------------------+
+ 17–21  + Sensor ID                                    +
+        + LND04: Landsat 4 Thematic Mapper             +
+        + LND05: Landsat 5 Thematic Mapper             +
+        + LND07: Landsat 7 Enhanced Thematic Mapper    +
+        + LND08: Landsat 8 Operational Land Imager     +
+        + SEN2A: Sentinel-2A MultiSpectral Instrument  +
+        + SEN2B: Sentinel-2B MultiSpectral Instrument  +
+--------+----------------------------------------------+
+ 23–25  + Product Type                                 +
+        + BOA: Bottom-of-Atmosphere Reflectance        +
+        + TOA: Top-of-Atmosphere Reflectance           +
+        + QAI: Quality Assurance Information           +
+        + AOD: Aerosol Optical Depth                   +
+        + DST: Cloud / Cloud shadow /Snow distance     +
+        + WVP: Water vapor                             +
+        + VZN: View zenith                             +
+        + HOT: Haze Optimized Transformation           +
+--------+----------------------------------------------+
+ 27–29  + File extension                               +
+        + tif: image data in compressed GeoTiff format +
+        + dat: image data in flat binary ENVI format   +
+        + hdr: metadata for ENVI format                +
+        + jpg: quicklooks                              +
+--------+----------------------------------------------+


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

Reflectance data (BOA / TOA) and Quality Assurance Information (QAI) are standard output and cannot be disabled.
All other products are optional.


* Reflectance

  Bottom-of-Atmosphere (BOA) reflectance is standard output if atmospheric correction is used.
  Top-of-Atmosphere (TOA) reflectance is standard output if atmospheric correction is not used.
  The scale is 10000, and nodata value is -9999.
  BOA / TOA data contain multiple bands, which represent wavelengths, see metadata and following tables).
  All bands are provided at the same spatial resolution (see :ref:`l2-param`).
  Bands intended for atmospheric characterization are not output (e.g. ultra-blue, water vapor or cirrus bands).
  Following tables summarize the output bands for each sensor.

  * Landsat 4–5 Thematic Mapper (TM):

    +-------------------+----------------------+------------------------+------------------+--------------------+
    + USGS Level 1 band + Wavelength name      + Wavelength range in µm + Resolution in m  + FORCE Level 2 band +
    +===================+======================+========================+==================+====================+
    + 1                 + Blue                 + 0.45–0.52              + 30               + 1                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 2                 + Green                + 0.52–0.60              + 30               + 2                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 3                 + Red                  + 0.63–0.69              + 30               + 3                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 4                 + Near Infrared        + 0.76–0.90              + 30               + 4                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 5                 + Shortwave Infrared 1 + 1.55–1.75              + 30               + 5                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 6                 + Thermal Infrared     + 10.40–12.50            + 30 (120 :sup:`1`)+ - :sup:`2`         +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 7                 + Shortwave Infrared 2 + 2.08–2.35              + 30               + 6                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    
    | :sup:`1` Band is acquired at 120m resolution, but USGS products are resampled and provided at 30m.
    | :sup:`2` Thermal band is used internally for cloud / cloud shadow detection, but not output.


  * Landsat 7 Enhanced Thematic Mapper Plus (ETM+):

    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + USGS Level 1 band + Wavelength name      + Wavelength range in µm + Resolution in m + FORCE Level 2 band +
    +===================+======================+========================+=================+====================+
    + 1                 + Blue                 + 0.45–0.52              + 30              + 1                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 2                 + Green                + 0.52–0.60              + 30              + 2                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 3                 + Red                  + 0.63–0.69              + 30              + 3                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 4                 + Near Infrared        + 0.77–0.90              + 30              + 4                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 5                 + Shortwave Infrared 1 + 1.55–1.75              + 30              + 5                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 6                 + Thermal Infrared     + 10.40–12.50            + 30 (60 :sup:`1`)+ - :sup:`2`         +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 7                 + Shortwave Infrared 2 + 2.09–2.35              + 30              + 6                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+
    + 8                 + Panchromatic         + 0.52–0.90              + 15              + -                  +
    +-------------------+----------------------+------------------------+-----------------+--------------------+

    | :sup:`1` Band is acquired at 60m resolution, but USGS products are resampled and provided at 30m.
    | :sup:`2` Thermal band is used internally for cloud / cloud shadow detection, but not output.


  * Landsat 8 Operational Land Imager (OLI) / Thermal Infrared Sensor (TIRS):
  
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + USGS Level 1 band + Wavelength name      + Wavelength range in µm + Resolution in m  + FORCE Level 2 band +
    +===================+======================+========================+==================+====================+
    + 1                 + Ultra-Blue           + 0.435–0.451            + 30               + - :sup:`2`         +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 2                 + Blue                 + 0.452–0.512            + 30               + 1                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 3                 + Green                + 0.533–0.590            + 30               + 2                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 4                 + Red                  + 0.636–0.673            + 30               + 3                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 5                 + Near Infrared        + 0.851–0.879            + 30               + 4                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 6                 + Shortwave Infrared 1 + 1.566–1.651            + 30               + 5                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 7                 + Shortwave Infrared 2 + 2.107–2.294            + 30               + 6                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 8                 + Panchromatic         + 0.503–0.676            + 15               + -                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 9                 + Cirrus               + 1.363–1.384            + 30               + - :sup:`3`         +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 10                + Thermal Infrared 1   + 10.60–11.19            + 30 (100 :sup:`1`)+ - :sup:`4`         +
    +-------------------+----------------------+------------------------+------------------+--------------------+
    + 11                + Thermal Infrared 2   + 11.50–12.51            + 30 (100 :sup:`1`)+ -                  +
    +-------------------+----------------------+------------------------+------------------+--------------------+

    | :sup:`1` Bands are acquired at 100m resolution, but USGS products are resampled and provided at 30m.
    | :sup:`2` Ultra-Blue band is used internally for aerosol retrieval, but not output.
    | :sup:`3` Cirrus band is used internally for cirrus cloud detection, but not output.
    | :sup:`4` Thermal band is used internally for cloud / cloud shadow detection, but not output.


  * Sentinel-2 A/B MultiSpectral Instrument (MSI):

    +------------------+----------------------+------------------------+-----------------+--------------------+
    + ESA Level 1 band + Wavelength name      + Wavelength range in µm + Resolution in m + FORCE Level 2 band +
    +==================+======================+========================+=================+====================+
    + 1                + Ultra-Blue           + 0.430–0.457            + 60              + - :sup:`1`         +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 2                + Blue                 + 0.440–0.538            + 10              + 1                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 3                + Green                + 0.537–0.582            + 10              + 2                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 4                + Red                  + 0.646–0.684            + 10              + 3                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 5                + Red Edge 1           + 0.694–0.713            + 20              + 4                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 6                + Red Edge 2           + 0.731–0.749            + 20              + 5                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 7                + Red Edge 3           + 0.769–0.797            + 20              + 6                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 8                + Broad Near Infrared  + 0.760–0.908            + 10              + 7                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 8A               + Near Infrared        + 0.848–0.881            + 20              + 8                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 9                + Water Vapor          + 0.932–0.958            + 60              + - :sup:`2`         +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 10               + Cirrus               + 1.337–1.412            + 60              + - :sup:`3`         +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 11               + Shortwave Infrared 1 + 1.539–1.682            + 20              + 9                  +
    +------------------+----------------------+------------------------+-----------------+--------------------+
    + 12               + Shortwave Infrared 2 + 2.078–2.320            + 20              + 10                 +
    +------------------+----------------------+------------------------+-----------------+--------------------+

    | :sup:`1` Ultra-Blue band is used internally for aerosol retrieval, but not output.
    | :sup:`2` Water vapor band is used internally for water vapor retrieval, but not output.
    | :sup:`3` Cirrus band is used internally for cirrus cloud detection, but not output.


* Quality Assurance Information

  This product contains all per-pixel quality information, including the cloud masks.
  
  .. warning:
  
    Quality Assurance Information (QAI product) are key for any higher-level analysis of ARD. Use QAI rigourosuly! If not, your analyses will be crap.
 
  .. seealso:: 

    Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-qai/qai/>`_, which explains what quality bits are, how quality bits are implemented in FORCE, how to visualize them, and how to deal with them in Higher Level Processing..

  QAI are provided bit-wise for each pixel, thus the 16-bit integers have to be parsed using following conventions.
  As an example, integer 28672 would be a poorly illuminated, sloped pixel where water vapor could not have been estimated.

  +-------+----+----+----+----+----+----+---+---+---+---+---+---+---+---+---+---+-----------+
  + Bit:  + 15 + 14 + 13 + 12 + 11 + 10 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1 + 0 +           +
  +-------+----+----+----+----+----+----+---+---+---+---+---+---+---+---+---+---+-----------+
  + Flag: + 0  + 1  + 1  + 1  + 0  + 0  + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + ∑ = 28672 +
  +-------+----+----+----+----+----+----+---+---+---+---+---+---+---+---+---+---+-----------+


  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + Bit No. + Parameter name       + Bit comb. + Integer + State                                                              +
  +=========+======================+===========+=========+====================================================================+
  + 0       + Valid data           + 0         + 0       + valid                                                              +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + no data                                                            +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 1–2     + Cloud state          + 00        + 0       + clear                                                              +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 01        + 1       + less confident cloud (i.e., buffered cloud 300 m)                  +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 10        + 2       + confident, opaque cloud                                            +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 11        + 3       + cirrus                                                             +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 3       + Cloud shadow flag    + 0         + 0       + no                                                                 +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes                                                                +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 4       + Snow flag            + 0         + 0       + no                                                                 +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes                                                                +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 5       + Water flag           + 0         + 0       + no                                                                 +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes                                                                +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 6–7     + Aerosol state        + 00        + 0       + estimated (best quality)                                           +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 01        + 1       + interpolated (mid quality)                                         +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 10        + 2       + high (aerosol optical depth > 0.6, use with caution)               +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 11        + 3       + fill (global fallback, low quality)                                +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 8       + Subzero flag         + 0         + 0       + no                                                                 +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes (use with caution)                                             +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 9       + Saturation flag      + 0         + 0       + no                                                                 +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes (use with caution)                                             +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 10      + High sun zenith flag + 0         + 0       + no                                                                 +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes (sun elevation < 15°, use with caution)                        +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 11–12   + Illumination state   + 00        + 0       + good (incidence angle < 55°, best quality for top. correction)     +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 01        + 1       + medium (incidence angle 55°–80°, good quality for top. correction) +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 10        + 2       + poor (incidence angle > 80°, low quality for top. correction)      +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 11        + 3       + shadow (incidence angle > 90°, no top. correction applied)         +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 13      + Slope flag           + 0         + 0       + no (cosine correction applied)                                     +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + yes (enhanced C-correction applied)                                +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 14      + Water vapor flag     + 0         + 0       + measured (best quality, only Sentinel-2)                           +
  +         +                      +-----------+---------+--------------------------------------------------------------------+
  +         +                      + 1         + 1       + fill (scene average, only Sentinel-2)                              +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+
  + 15      + Empty                + 0         + 0       + TBD                                                                +
  +---------+----------------------+-----------+---------+--------------------------------------------------------------------+

  * Nodata values are values where nothing was observed, where auxiliary data was not given (e.g. nodata in DEM), or where data is substantially corrupt (e.g. impulse noise, or when the surface reflectance estimate is > 2.0 or < -1.0)

  * Clouds are given in three categories, i.e. opaque clouds (confident cloud), buffered clouds (300m; less confident cloud), and cirrus clouds.

  * Cloud shadows are detected on the basis of the cloud layer. If a cloud is missed, the cloud shadow is missed, too. If a false positive cloud is detected, false positive cloud shadows follow.

  * Aerosol Optical Depth is estimated for fairly coarse grid cells. If there is no valid AOD estimation in any cell, values are interpolated. If there is no valid AOD estimation for the complete image, a fill value is assigned (AOD is guessed). If AOD @550nm is higher than 0.6, it is flagged as high aerosol; this is not necessarily critical, but should be used with caution (see subzero flag).

  * If the surface reflectance estimate in any band is < 0, the subzero flag is set. This can point to overestimation of AOD.

  * If DNs were saturated, or if the surface reflectance estimate in any band is > 1, the saturation flag is set.

  * If sun elevation is smaller than 15°, the high sun zenith flag is set. Use this data with caution, radiative transfer computations might be out of specification.

  * The illumination state is related to the quality of the topographic correction. If the incidence angle is smaller than 55°, quality is best. If the incidence angle is larger than 80°, the quality of the topographic correction is low, and data artefacts are possible. If the area is not illuminated at all, no topographic correction is done (values are the same as without topographic correction).

  * The slope flag indicates whether a simple cosine correction (slope ≤ 2°) was used for topographic correction, or if the enhanced C-correction was used (slope > 2°).

  * The water vapor flag indicates whether water vapor was estimated, or if the scene average was used to fill. Water vapor is not estimated over water and cloud shadow pixels. This flag only applies to Sentinel-2 images.


* Aerosol Optical Depth

  The Aerosol Optical Depth (AOD) product is optional output.
  It contains the AOD of the green band (~550 nm).
  The scale is 1000, and nodata value is -9999.
  This product is not used by any of the higher-level FORCE modules.

  
* Cloud / cloud shadow / snow distance

  The Cloud / cloud shadow / snow distance (CLD) product is optional output.
  The cloud distance gives the distance to the next opaque cloud, buffered cloud, cirrus cloud, cloud shadow or snow.
  The unit is in projection units (commonly in meters), and nodata value is -9999.
  This product can be used in :ref:`level3` to generate Best Available Pixel (BAP) composites.

  .. note:: 
  
    This is not the actual cloud mask! For cloud masks and quality screening, rather use the QAI product.

    
* Water vapor

  The Water vapor (WVP) product is optional output.
  It contains the atmospheric water vapor (as derived from the Sentinel-2 data on pixel level, or as ingested with the water vapor database for Landsat).
  The scale is 1000, and nodata value is -9999.
  This product is not used by any of the higher-level FORCE modules.

* View zenith

  The View zenith (VZN) product is optional output.
  It contains the view zenith (the average view zenith for Sentinel-2, and an approximated view zenith for Landsat).
  The scale is 100, and nodata value is -9999.
  This product can be used in :ref:`level3` to generate Best Available Pixel (BAP) composites.

  
* Haze Optimized Transformation

  The Haze Optimized Transformation (HOT) product is optional output.
  It contains the HOT index, which is computed on TOA reflectance (and therefore cannot be computed on Level 2 ARD).
  The HOT is useful to avoid hazy and residual cloud contamination.
  The scale is 10000, and nodata value is -9999.
  This product can be used in :ref:`level3` to generate Best Available Pixel (BAP) composites.


Logfile
^^^^^^^

*This part needs updating*

A logfile is created by force-level2 in the output directory.
Following 29-digit naming convention is applied:
FORCE-L2PS_20170712040001.log
Digits 1–10 Processing module
Digits 12–25 Processing time (start time) as YYYYMMDDHHMMSS
Digits 27–29 File extension

Typical entries look like this:
LC08_L1TP_195023_20180110_20180119_01_T1: sc:   0.10%. cc:  89.59%. AOD: 0.2863. # of targets: 0/327.  4 product(s) written. Success! Processing time: 32 mins 37 secs
LC08_L1TP_195023_20170328_20170414_01_T1: sc:   0.00%. cc:   2.56%. AOD: 0.0984. # of targets: 394/6097.  6 product(s) written. Success! Processing time: 19 mins 03 secs
LC08_L1TP_195023_20170312_20170317_01_T1: sc:   0.29%. cc:  91.85%. Skip. Processing time: 13 mins 22 secs 

The first entry indicates the image ID, followed by overall snow and cloud cover, aerosol optical depth @ 550 nm (scene average), the number of dark targets for retrieving aerosol optical depth (over water/vegetation), the number of products written (number of tiles, this is dependent on tile cloud cover, and FILE_TILE), and a supportive success indication.
In the case the overall cloud coverage is higher than allowed, the image is skipped.
The processing time (real time) is appended at the end.


Quicklooks
^^^^^^^^^^

If ``OUTPUT_OVV = TRUE``, small jpeg quicklooks images are generated,
The quicklooks are fixed-stretch RGB images with overlays of key quality indicators:

+---------------------+----------+
+ quality indicator   + color    +
+=====================+==========+
+ cirrus              + red      +
+---------------------+----------+
+ opaque cloud        + pink     +
+---------------------+----------+
+ cloud shadow        + cyan     +
+---------------------+----------+
+ snow                + yellow   +
+---------------------+----------+
+ saturated pixels    + orange   +
+---------------------+----------+
+ subzero reflectance + greenish +
+---------------------+----------+

