.. _tsa-format:

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

Following 42 to 46-digit naming convention is applied to all output files:

Example filename: 1984-2020_182-274_HL_TSA_LNDLG_TCG_STM.tif

Digits 36–38	3-digit index compact name
+                + BLU   + Blue band
+                + GRN   + Green band
+                + RED   + Red band
+                + NIR   + Near Infrared band
+                + SW1   + Shortwave Infrared band 1
+                + SW2   + Shortwave Infrared band 2
+                + RE1   + Red Edge band 1
+                + RE2   + Red Edge band 2
+                + RE3   + Red Edge band 3
+                + BNR   + Broad Near Infrared band
+                + NDV   + Normalized Difference Vegetation Index
+                + EVI   + Enhanced Vegetation Index
+                + NBR   + Normalized Burn Ratio
+                + ARV   + Atmospherically Resistant Vegetation Index
+                + SAV   + Soil Adjusted Vegetation Index
+                + SRV   + Soil and Atmospherically Resistant Vegetation Index
+                + TCB   + Tasseled Cap Brightness
+                + TCG   + Tasseled Cap Greenness
+                + TCW   + Tasseled Cap Wetness
+                + TCD   + Tasseled Cap-based Disturbance Index
+                + SMA   + Spectral Mixture Analysis abundance

Digits 59–61	Product Type
+                + TSS   + Time Series Stack
+                + TSI   + Time Series Interpolation
+                + RMS   + RMSE Time Series of SMA
+                + STM   + Spectral Temporal Metrics
+                + TRD   + Trend Analysis
+                + CAT   + Extended CAT Analysis
+                + FBY   + Fold-by-Year Stack
+                + FBM   + Fold-by-Month Stack
+                + FBW   + Fold-by-Week Stack
+                + FBD   + Fold-by-Day Stack

1984-2020_182-274_HL_TSA_LNDLG_TCG_STM.tif

**Table 1:** Naming convention

+----------------+---------+------------------------------------------+
+ Digits         + Description                                        +
+================+=========+==========================================+
+ 1–9            + Temporal range for the years as YYYY–YYYY          +
+----------------+---------+------------------------------------------+
+ 11-17          + Temporal range for the DOY as DDD–DDD              +
+----------------+---------+------------------------------------------+
+ 19-20          + Product Level                                      +
+----------------+---------+------------------------------------------+
+ 22-24          + Submodule                                          +
+----------------+---------+------------------------------------------+
+ 26-30          + Sensor ID                                          +
+                +---------+------------------------------------------+
+                + LNDLG   + Landsat legacy bands                     +
+                +---------+------------------------------------------+
+                + SEN2L   + Sentinel-2 land bands                    +
+                +---------+------------------------------------------+
+                + SEN2H   + Sentinel-2 high-res bands                +
+                +---------+------------------------------------------+
+                + R-G-B   + Visible bands                            +
+                +---------+------------------------------------------+
+                + VVVHP   + VV/VH Dual Polarized                     +
+----------------+---------+------------------------------------------+
+ 32–34          + Index Short Name                                   +
+                +---------+------------------------------------------+
+----------------+---------+------------------------------------------+
+ 36-38          + Product Type                                       +
+                +---------+------------------------------------------+
+----------------+---------+------------------------------------------+
+ 36-42          + Product Type: Phenology (replace XXX with Table 3) +
+                +---------+------------------------------------------+
+                + XXX-LSP + Phenometrics                             +
+                +---------+------------------------------------------+
+                + XXX-TRP + Trend Analysis on Phenometrics           +
+                +---------+------------------------------------------+
+                + XXX-CAP + Extended CAT Analysis on Phenometrics    +
+----------------+---------+------------------------------------------+
+ 40-42 / 44-46  + File extension                                     +
+                +---------+------------------------------------------+
+                + tif     + image data in compressed GeoTiff format  +
+                +---------+------------------------------------------+
+                + dat     + image data in flat binary ENVI format    +
+                +---------+------------------------------------------+
+                + hdr     + metadata for ENVI format                 +
+----------------+---------+------------------------------------------+


**Table 2:** Folding tags

+--------+-----------------+
+ Letter + Description     +
+========+=================+
+ Y      + Fold by Year    +
+--------+-----------------+
+ Q      + Fold by Quarter +
+--------+-----------------+
+ M      + Fold by Month   +
+--------+-----------------+
+ W      + Fold by Week    +
+--------+-----------------+
+ D      + Fold by Day     +
+--------+-----------------+


**Table 3:** Phenology name tags

+-----+-----------------------------+
+ Tag + Description                 +
+=====+=============================+
+ DEM + Date of Early Minimum       +
+-----+-----------------------------+
+ DSS + Date of Start of Season     +
+-----+-----------------------------+
+ DRI + Date of Rising Inflection   +
+-----+-----------------------------+
+ DPS + Date of Peak of Season      +
+-----+-----------------------------+
+ DFI + Date of Falling Inflection  +
+-----+-----------------------------+
+ DES + Date of End of Season       +
+-----+-----------------------------+
+ DLM + Date of Late Minimum        +
+-----+-----------------------------+
+ LTS + Length of Total Season      +
+-----+-----------------------------+
+ LGS + Length of Green Season      +
+-----+-----------------------------+
+ VEM + Value of Early Minimum      +
+-----+-----------------------------+
+ VSS + Value of Start of Season    +
+-----+-----------------------------+
+ VRI + Value of Rising Inflection  +
+-----+-----------------------------+
+ VPS + Value of Peak of Season     +
+-----+-----------------------------+
+ VFI + Value of Falling Inflection +
+-----+-----------------------------+
+ VES + Value of End of Season      +
+-----+-----------------------------+
+ VLM + Value of Late Minimum       +
+-----+-----------------------------+
+ VBL + Value of Base Level         +
+-----+-----------------------------+
+ VSA + Value of Seasonal Amplitude +
+-----+-----------------------------+
+ IST + Integral of Total Season    +
+-----+-----------------------------+
+ IBL + Integral of Base Level      +
+-----+-----------------------------+
+ IBT + Integral of Base+Total      +
+-----+-----------------------------+
+ IGS + Integral of Green Season    +
+-----+-----------------------------+
+ RAR + Rate of Average Rising      +
+-----+-----------------------------+
+ RAF + Rate of Average Falling     +
+-----+-----------------------------+
+ RMR + Rate of Maximum Rising      +
+-----+-----------------------------+
+ RMF + Rate of Maximum Falling     +
+-----+-----------------------------+


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







Product type
Time Series
Time Series products have as many bands as there are available or requested time steps. If no temporal subset was specified:
the TSS product contains one band per available acquisition (this may vary between the tiles), 
the RMS product contains one band per available acquisition (this may vary between the tiles), 
the TSI product contains one band per interpolation step,
the FBY product contains one band per year (do not overdo YEAR_MIN/MAX, this will give many useless bands), 
the FBM product contains one band per month (up to 12, depends on MONTH_MIN/MAX and DOY_MIN/MAX),
the FBW contains one band per week (up to 52, depends on MONTH_MIN/MAX and DOY_MIN/MAX), 
the FBD product contains one band per DOY (up to 365, depends on MONTH_MIN/MAX and DOY_MIN/MAX),
the 26 LSP products contain one band per year (do not overdo YEAR_MIN/MAX, this will give many useless bands).

Basic Statistics
The Basic Statistics (STA) product provides a summary of all observations (or the requested subset). It is a multi-layer image with following bands:
+                + 1	µ   + Average of index values
+                + 2	σ   + Standard deviation of index values
+                + 3	min   + Minimum index value
+                + 4	max   + Maximum index value
+                + 5	# of obs.   + Number of good quality observations 

Trend Analysis
The Trend Analysis (TRD) product contains trend parameters. It is a multi-layer image with following bands:
+                + 1	µ   + Average
+                + 2	a   + Intercept
+                + 3	b   + Trend
+                + 4	R²   + R squared
+                + 5	sig.   + Significance (-1, 0, 1)
+                + 6	RMSE   + Root Mean Squared Error
+                + 7	MAE   + Mean Absolute Error
+                + 8	max |e|   + Maximum Absolute Residual
+                + 9	# of obs.   + Number of good quality observations 

Change, Aftereffect, Trend
The Change, Aftereffect, Trend (CAT) product (following Hird et al. 2016, DOI: 10.1109/jstars.2015.2419594) contains extended change and trend parameters. It detects one change per time series, splits the time series into three parts, and derives trend parameters: (1) complete time series (this is the same as the TRD product), (2) time series before change, and (3) time series after change. It is a multi-layer image with following bands:
+                + 1	Change   + Magnitude of change
+                + 2	Time of change	Timestamp of the change (depends on the input time series, i.e. year/month/week/day)
+                + 3–11	Trend parameters for complete time series (see TRD product)
+                + 12–20	Trend parameters for time series before change (see TRD product)
+                + 21–29	Trend parameters for time series after change (see TRD product)

File format
The data are provided in (i) ENVI Standard format (flat binary images), or (ii) as GeoTiff (LZW compression with horizontal differencing). Each dataset consists of an image dataset (.dat/,tif) and additional metadata (.hdr). The image data have signed 16bit datatype and band sequential (BSQ) interleaving. Scaling factor is 10000 for most products.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info, band names etc.

