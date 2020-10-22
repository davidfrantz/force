.. _tut-dem:

Digital Elevation Model
=======================

**How to prepare a DEM for Level 2 Processing**

This tutorial shows how to prepare a Digital Elevation Model (DEM) for the FORCE Level 2 Processing System (FORCE L2PS).

.. admonition:: Info

   *This tutorial uses FORCE v. 3.0*


Background
----------

FORCE L2PS uses a DEM for 

- enhanced cloud and cloud shadow detection, 
- atmospheric correction, and to 
- perform the topographic correction.

In the cloud shadow detection, the DEM is primarily used to distinguish cloud shadows from water and topographic shadows.
In the atmospheric correction, the DEM is used to scale the optical depths with altitude.
The topographic correction is of course relying on the DEM.
In principle, FORCE L2PS can be used without a DEM (FILE_DEM = NULL).
In this case, the surface is assumed to be flat at z = 0m a.s.l.
The topographic correction, however, can only be used if a DEM is given (surprise).

In any case, it is strongly advised to use a DEM.
Plus, it is not complicated to acquire it, free options are available.
You probably already have a DEM for your study area anyway.


Data format
-----------

There are little requirements on the data format:

- The unit must be meters.
- The Nodata value shouldn't be 0, which is a valid elevation.
- The DEM must cover the complete image(s) to be processed.

Thus, a mosaic that covers your complete study area needs to be prepared.
The DEM is warped and cropped to the projection and extent of the Level 1 image, which is processed with FORCE L2PS.
This is done on-the-fly.
Therefore, data type, data format, projection, extent etc.
can be chosen freely - as long as GDAL is able to handle it (GDAL can handle pretty much anything).

Please note, that pixels with nodata values in the DEM will have nodata values in the Level 2 products, too.
Thus, make sure your DEM covers the complete area of interest.


Which DEM?
----------

The DEM should match the resolution of the Level 1 image data as closely as possible.
If possible, it is advised to use a finer resolution.
However, as it is hard to acquire high spatial resolution DEMs, especially for larger areas, lower resolution works too.
Often, we use the 30m SRTM DEM or 30m ASTER DEM, or a combination thereof, e.g. SRTM filled with ASTER (SRTM is a bit better, but there are holes in mountainous regions, and coverage is only 60°N-60°S).

The SRTM DEM can be obtained from `EarthExplorer <https://earthexplorer.usgs.gov/>`_.
The ASTER DEM can be obtained from `EarthData <https://search.earthdata.nasa.gov/search/`_ or `Japan Space Systems <https://ssl.jspacesystems.or.jp/ersdac/GDEM/E/>`_.
Both are free of charge.


Prepare the mosaic
------------------

The following steps illustrate how to build a virtual mosaic from SRTM data.
Generally, DEM data come in tiles (datacube style), e.g. each SRTM tile covers 1°.
The `GDAL Virtual Format <gdal.org/drivers/raster/vrt.html>`_ allows to mosaick data without producing a physical representation, i.e. the virtual mosaic only holds links to the original tiled data, plus some rules on how to combine them into the mosaic.

Assuming you have downloaded some SRTM tiles, we first prepare a text file that holds all the filepaths:

.. code-block:: bash

   find /data/Dagobah/global/dem/srtm -name '*.tif' > /data/Earth/global/dem/srtm.txt
   cat /data/Dagobah/global/dem/srtm.txt

   /data/Dagobah/global/dem/srtm/n35_e027_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n35_e026_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n37_e026_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n36_e025_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n37_e027_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n37_e025_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n36_e024_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n35_e023_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n37_e024_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n36_e026_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n37_e023_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n35_e024_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n35_e025_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n36_e023_1arc_v3.tif
   /data/Dagobah/global/dem/srtm/n36_e027_1arc_v3.tif


Then, we use the ``gdalbuildvrt`` command to generate the virtual mosaic.

.. code-block:: bash

   gdalbuildvrt -input_file_list /data/Dagobah/global/dem/srtm.txt /data/Earth/global/dem/srtm.vrt

   0...10...20...30...40...50...60...70...80...90...100 - done.


The VRT file is a simple xml file:

.. code-block:: bash

   head -n 14 /data/Dagobah/global/dem/srtm.vrt

   <VRTDataset rasterXSize="18001" rasterYSize="10801">
     <SRS>GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]</SRS>
     <GeoTransform>  2.2999861111111112e+01,  2.7777777777777794e-04,  0.0000000000000000e+00,  3.8000138888888891e+01,  0.0000000000000000e+00, -2.7777777777777794e-04</GeoTransform>
     <VRTRasterBand dataType="Int16" band="1">
       <NoDataValue>-32767</NoDataValue>
       <ColorInterp>Gray</ColorInterp>
       <ComplexSource>
         <SourceFilename relativeToVRT="1">srtm/n35_e027_1arc_v3.tif</SourceFilename>
         <SourceBand>1</SourceBand>
         <SourceProperties RasterXSize="3601" RasterYSize="3601" DataType="Int16" BlockXSize="3601" BlockYSize="1" />
         <SrcRect xOff="0" yOff="0" xSize="3601" ySize="3601" />
         <DstRect xOff="14400" yOff="7200" xSize="3601" ySize="3601" />
         <NODATA>-32767</NODATA>
       </ComplexSource>


Any software that is based on GDAL is able to read this file, e.g. QGIS - and FORCE.
The filepath of this file needs to given in the FORCE L2PS parameter file:

``FILE_DEM = /data/Dagobah/global/dem/srtm.vrt``


------------

.. |author-pic| image:: profile/dfrantz.jpg

+--------------+--------------------------------------------------------------------------------+
+ |author-pic| + This tutorial was written by                                                   +
+              + `David Frantz <https://davidfrantz.github.io>`_,                               +
+              + main developer of **FORCE**,                                                   +
+              + postdoc at `EOL <https://www.geographie.hu-berlin.de/en/professorships/eol>`_. +
+              + *Views are his own.*                                                           +
+--------------+--------------------------------------------------------------------------------+
+ **EO**, **ARD**, **Data Science**, **Open Science**                                           +
+--------------+--------------------------------------------------------------------------------+
