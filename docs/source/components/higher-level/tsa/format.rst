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


**Table 1:** Naming convention

+----------------+---------+---------------------------------------------------------+
+ Digits         + Description                                                       +
+================+=========+=========================================================+
+ 1–9            + Temporal range for the years as YYYY–YYYY                         +
+----------------+---------+---------------------------------------------------------+
+ 11-17          + Temporal range for the DOY as DDD–DDD                             +
+----------------+---------+---------------------------------------------------------+
+ 19-20          + Product Level                                                     +
+----------------+---------+---------------------------------------------------------+
+ 22-24          + Submodule                                                         +
+----------------+---------+---------------------------------------------------------+
+ 26-30          + Sensor ID                                                         +
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
+ 32–34          + Index Short Name                                                  +
+                +---------+---------------------------------------------------------+
+                + BLU     + Blue band                                               +
+                +---------+---------------------------------------------------------+
+                + GRN     + Green band                                              +
+                +---------+---------------------------------------------------------+
+                + RED     + Red band                                                +
+                +---------+---------------------------------------------------------+
+                + NIR     + Near Infrared band                                      +
+                +---------+---------------------------------------------------------+
+                + SW1     + Shortwave Infrared band 1                               +
+                +---------+---------------------------------------------------------+
+                + SW2     + Shortwave Infrared band 2                               +
+                +---------+---------------------------------------------------------+
+                + RE1     + Red Edge band 1                                         +
+                +---------+---------------------------------------------------------+
+                + RE2     + Red Edge band 2                                         +
+                +---------+---------------------------------------------------------+
+                + RE3     + Red Edge band 3                                         +
+                +---------+---------------------------------------------------------+
+                + BNR     + Broad Near Infrared band                                +
+                +---------+---------------------------------------------------------+
+                + NDV     + Normalized Difference Vegetation Index                  +
+                +---------+---------------------------------------------------------+
+                + EVI     + Enhanced Vegetation Index                               +
+                +---------+---------------------------------------------------------+
+                + NBR     + Normalized Burn Ratio                                   +
+                +---------+---------------------------------------------------------+
+                + ARV     + Atmospherically Resistant Vegetation Index              +
+                +---------+---------------------------------------------------------+
+                + SAV     + Soil Adjusted Vegetation Index                          +
+                +---------+---------------------------------------------------------+
+                + SRV     + Soil and Atmospherically Resistant Vegetation Index     +
+                +---------+---------------------------------------------------------+
+                + TCB     + Tasseled Cap Brightness                                 +
+                +---------+---------------------------------------------------------+
+                + TCG     + Tasseled Cap Greenness                                  +
+                +---------+---------------------------------------------------------+
+                + TCW     + Tasseled Cap Wetness                                    +
+                +---------+---------------------------------------------------------+
+                + TCD     + Tasseled Cap Disturbance Index (without rescaling)      +
+                +---------+---------------------------------------------------------+
+                + NDB     + Normalized Difference Building Index                    +
+                +---------+---------------------------------------------------------+
+                + NDW     + Normalized Difference Water Index                       +
+                +---------+---------------------------------------------------------+
+                + MNW     + modified Normalized Difference Water Index              +
+                +---------+---------------------------------------------------------+
+                + NDS     + Normalized Difference Snow Index                        +
+                +---------+---------------------------------------------------------+
+                + SMA     + Spectral Mixture Analysis abundance                     +
+                +---------+---------------------------------------------------------+
+                + BVV     + VV Polarized band                                       +
+                +---------+---------------------------------------------------------+
+                + BVH     + VH Polarized band                                       +
+                +---------+---------------------------------------------------------+
+                + NDRE    + Normalized Difference Red Edge Index                    +
+----------------+---------+---------------------------------------------------------+
+ 36-38          + Product Type                                                      +
+                +---------+---------------------------------------------------------+
+                + TSS     + Time Series Stack                                       +
+                +---------+---------------------------------------------------------+
+                + TSI     + Time Series Interpolation                               +
+                +---------+---------------------------------------------------------+
+                + RMS     + RMSE Time Series of SMA                                 +
+                +---------+---------------------------------------------------------+
+                + STM     + Spectral Temporal Metrics                               +
+                +---------+---------------------------------------------------------+
+                + FBX     + Fold-by-X Time Series (replace X with Table 2)          +
+                +---------+---------------------------------------------------------+
+                + TRX     + Trend Analysis on Folds (replace X with Table 2)        +
+                +---------+---------------------------------------------------------+
+                + CAX     + Extended CAT Analysis on Folds (replace X with Table 2) +
+                +---------+---------------------------------------------------------+
+                + SPL     + Spline fitted Time Series                               +
+----------------+---------+---------------------------------------------------------+
+ 36-42          + Product Type: Phenometrics (replace XXX with Table 3)             +
+                +---------+---------------------------------------------------------+
+                + XXX-LSP + Phenometrics                                            +
+                +---------+---------------------------------------------------------+
+                + XXX-TRP + Trend Analysis on Phenometrics                          +
+                +---------+---------------------------------------------------------+
+                + XXX-CAP + Extended CAT Analysis on Phenometrics                   +
+----------------+---------+---------------------------------------------------------+
+ 36-42          + Product Type: Polarmetrics (replace XXX with Table 3)             +
+                +---------+---------------------------------------------------------+
+                + XXX-POL + Polarmetrics                                            +
+                +---------+---------------------------------------------------------+
+                + XXX-TRO + Trend Analysis on Polarmetrics                          +
+                +---------+---------------------------------------------------------+
+                + XXX-CAO + Extended CAT Analysis on Polarmetrics                   +
+----------------+---------+---------------------------------------------------------+
+ 40-42 / 44-46  + File extension                                                    +
+                +---------+---------------------------------------------------------+
+                + tif     + image data in compressed GeoTiff format                 +
+                +---------+---------------------------------------------------------+
+                + dat     + image data in flat binary ENVI format                   +
+                +---------+---------------------------------------------------------+
+                + hdr     + metadata for ENVI format                                +
+----------------+---------+---------------------------------------------------------+


Folding
"""""""

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


.. _tsa-lsp_products:

Phenology
"""""""""

**Table 3:** Phenology name tags

+-----+---------------------------------------------------+-------+--------+
+ Tag + Description                                       + Polar + SPLITS +
+=====+===================================================+=======+========+
+ DEM + Date of Early Minimum                             + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ DSS + Date of Start of Season                           + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ DRI + Date of Rising Inflection                         +       + X      +
+-----+---------------------------------------------------+-------+--------+
+ DPS + Date of Peak of Season                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ DMS + Date of Mid of Season                             + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ DFI + Date of Falling Inflection                        +       + X      +
+-----+---------------------------------------------------+-------+--------+
+ DES + Date of End of Season                             + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ DLM + Date of Late Minimum                              + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ DEV + Date of Early Average Vector                      + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ DAV + Date of Average Vector                            + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ DLV + Date of Late Average Vector                       + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ DPY + Date of Start of Phenological Year                + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ DPV + delta Date of adaptive Start of Phenological Year + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ LTS + Length of Total Season                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ LGS + Length of Green Season                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ LGV + Length of between early/late vectors              + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VEM + Value of Early Minimum                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VSS + Value of Start of Season                          + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VRI + Value of Rising Inflection                        +       + X      +
+-----+---------------------------------------------------+-------+--------+
+ VPS + Value of Peak of Season                           + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VMS + Value of Mid of Season                            + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VFI + Value of Falling Inflection                       +       + X      +
+-----+---------------------------------------------------+-------+--------+
+ VES + Value of End of Season                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VLM + Value of Late Minimum                             + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VEV + Value of Early Average Vector                     + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VAV + Value of Average Vector                           + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VLV + Value of Late Average Vector                      + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VBL + Value of Base Level                               + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VSA + Value of Seasonal Amplitude                       + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ VGA + Value of Green Amplitude                          + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VPA + Value of Peak Amplitude                           + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VGM + Value of Green Mean                               + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ VGV + Value of Green Variability                        + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ IST + Integral of Total Season                          + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ IBL + Integral of Base Level                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ IBT + Integral of Base+Total                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ IGS + Integral of Green Season                          + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ IRR + Integral of Rising Rate                           + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ IFR + Integral of Falling Rate                          + X     +        +
+-----+---------------------------------------------------+-------+--------+
+ RAR + Rate of Average Rising                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ RAF + Rate of Average Falling                           + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ RMR + Rate of Maximum Rising                            + X     + X      +
+-----+---------------------------------------------------+-------+--------+
+ RMF + Rate of Maximum Falling                           + X     + X      +
+-----+---------------------------------------------------+-------+--------+


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

**Trend Analysis**

The Trend Analysis products contains trend parameters:

+------+-----------------------------+
+ Band + Description                 +
+======+=============================+
+ 1    + Average                     +
+------+-----------------------------+
+ 2    + Intercept                   +
+------+-----------------------------+
+ 3    + Trend                       +
+------+-----------------------------+
+ 4    + relative change             +
+------+-----------------------------+
+ 4    + R-squared                   +
+------+-----------------------------+
+ 5    + Significance (-1, 0, 1)     +
+------+-----------------------------+
+ 6    + Root Mean Squared Error     +
+------+-----------------------------+
+ 7    + Mean Absolute Error         +
+------+-----------------------------+
+ 8    + Maximum Absolute Residual   +
+------+-----------------------------+
+ 9    + Number of used observations +
+------+-----------------------------+


**Change and Trend**

The Change, Aftereffect, Trend (CAT) product (following [Hird et al. 2016](https://ieeexplore.ieee.org/document/7094220) contains extended change and trend parameters.
CAT detects one change per time series, splits the time series into three parts, and derives trend parameters for the three parts:

+----------+--------------------------------------------------------------------+
+ Band     + Description                                                        +
+==========+====================================================================+
+ 1        + Magnitude of change                                                +
+----------+--------------------------------------------------------------------+
+ 2        + Time of change                                                     +
+----------+--------------------------------------------------------------------+
+ 3 to 12  + Trend parameters for complete time series (see Trend product)      +
+----------+--------------------------------------------------------------------+
+ 13 to 22 + Trend parameters for time series before change (see Trend product) +
+----------+--------------------------------------------------------------------+
+ 23 to 32 + Trend parameters for time series after change (see Trend product)  +
+----------+--------------------------------------------------------------------+

