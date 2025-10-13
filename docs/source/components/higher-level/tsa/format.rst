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


File format
^^^^^^^^^^^

Refer to :ref:`hl-format` for details on the file format and metadata.


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
+                + ND1     + Normalized Difference Red Edge Index 1                  +
+                +---------+---------------------------------------------------------+
+                + ND2     + Normalized Difference Red Edge Index 2                  +
+                +---------+---------------------------------------------------------+
+                + CRE     + Chlorophyll Index Red Edge                              +
+                +---------+---------------------------------------------------------+
+                + NR1     + Normalized Difference Vegetation Index red edge 1       +
+                +---------+---------------------------------------------------------+
+                + NR2     + Normalized Difference Vegetation Index red edge 2       +
+                +---------+---------------------------------------------------------+
+                + NR3     + Normalized Difference Vegetation Index red edge 3       +
+                +---------+---------------------------------------------------------+
+                + N1N     + Normalized Difference Vegetation Index red edge 1 narrow+
+                +---------+---------------------------------------------------------+
+                + N2N     + Normalized Difference Vegetation Index red edge 2 narrow+
+                +---------+---------------------------------------------------------+
+                + N3N     + Normalized Difference Vegetation Index red edge 3 narrow+
+                +---------+---------------------------------------------------------+
+                + MRE     + Modified Simple Ratio red edge                          +
+                +---------+---------------------------------------------------------+
+                + MRN     + Modified Simple Ratio red edge narrow                   +
+                +---------+---------------------------------------------------------+
+                + CCI     + Chlorophyll Carotenoid Index                            +
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

+-----+---------------------------------------------------+
+ Tag + Description                                       +
+=====+===================================================+
+ DEM + Date of Early Minimum                             +
+-----+---------------------------------------------------+
+ DSS + Date of Start of Season                           +
+-----+---------------------------------------------------+
+ DPS + Date of Peak of Season                            +
+-----+---------------------------------------------------+
+ DMS + Date of Mid of Season                             +
+-----+---------------------------------------------------+
+ DES + Date of End of Season                             +
+-----+---------------------------------------------------+
+ DLM + Date of Late Minimum                              +
+-----+---------------------------------------------------+
+ DEV + Date of Early Average Vector                      +
+-----+---------------------------------------------------+
+ DAV + Date of Average Vector                            +
+-----+---------------------------------------------------+
+ DLV + Date of Late Average Vector                       +
+-----+---------------------------------------------------+
+ DPY + Date of Start of Phenological Year                +
+-----+---------------------------------------------------+
+ DPV + delta Date of adaptive Start of Phenological Year +
+-----+---------------------------------------------------+
+ LTS + Length of Total Season                            +
+-----+---------------------------------------------------+
+ LGS + Length of Green Season                            +
+-----+---------------------------------------------------+
+ LGV + Length of between early/late vectors              +
+-----+---------------------------------------------------+
+ VEM + Value of Early Minimum                            +
+-----+---------------------------------------------------+
+ VSS + Value of Start of Season                          +
+-----+---------------------------------------------------+
+ VPS + Value of Peak of Season                           +
+-----+---------------------------------------------------+
+ VMS + Value of Mid of Season                            +
+-----+---------------------------------------------------+
+ VES + Value of End of Season                            +
+-----+---------------------------------------------------+
+ VLM + Value of Late Minimum                             +
+-----+---------------------------------------------------+
+ VEV + Value of Early Average Vector                     +
+-----+---------------------------------------------------+
+ VAV + Value of Average Vector                           +
+-----+---------------------------------------------------+
+ VLV + Value of Late Average Vector                      +
+-----+---------------------------------------------------+
+ VBL + Value of Base Level                               +
+-----+---------------------------------------------------+
+ VSA + Value of Seasonal Amplitude                       +
+-----+---------------------------------------------------+
+ VGA + Value of Green Amplitude                          +
+-----+---------------------------------------------------+
+ VPA + Value of Peak Amplitude                           +
+-----+---------------------------------------------------+
+ VGM + Value of Green Mean                               +
+-----+---------------------------------------------------+
+ VGV + Value of Green Variability                        +
+-----+---------------------------------------------------+
+ IST + Integral of Total Season                          +
+-----+---------------------------------------------------+
+ IBL + Integral of Base Level                            +
+-----+---------------------------------------------------+
+ IBT + Integral of Base+Total                            +
+-----+---------------------------------------------------+
+ IGS + Integral of Green Season                          +
+-----+---------------------------------------------------+
+ IRR + Integral of Rising Rate                           +
+-----+---------------------------------------------------+
+ IFR + Integral of Falling Rate                          +
+-----+---------------------------------------------------+
+ RAR + Rate of Average Rising                            +
+-----+---------------------------------------------------+
+ RAF + Rate of Average Falling                           +
+-----+---------------------------------------------------+
+ RMR + Rate of Maximum Rising                            +
+-----+---------------------------------------------------+
+ RMF + Rate of Maximum Falling                           +
+-----+---------------------------------------------------+


Product type
^^^^^^^^^^^^

* Time Series
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
+ 5    + absolute change             +
+------+-----------------------------+
+ 6    + R-squared                   +
+------+-----------------------------+
+ 7    + Significance (-1, 0, 1)     +
+------+-----------------------------+
+ 8    + Root Mean Squared Error     +
+------+-----------------------------+
+ 9    + Mean Absolute Error         +
+------+-----------------------------+
+ 10   + Maximum Absolute Residual   +
+------+-----------------------------+
+ 11   + Number of used observations +
+------+-----------------------------+
+ 12   + Length of time series       +
+------+-----------------------------+


**Change and Trend**

The Change, Aftereffect, Trend (CAT) product (following [Hird et al. 2016](https://ieeexplore.ieee.org/document/7094220) contains extended change and trend parameters.
CAT detects one change per time series, splits the time series into three parts, and derives trend parameters for the three parts:

+----------+--------------------------------------------------------------------+
+ Band     + Description                                                        +
+==========+====================================================================+
+ 1        + Magnitude of change                                                +
+----------+--------------------------------------------------------------------+
+ 2        + Relative change                                                    +
+----------+--------------------------------------------------------------------+
+ 3        + Time of change                                                     +
+----------+--------------------------------------------------------------------+
+ 4 to 15  + Trend parameters for complete time series (see Trend product)      +
+----------+--------------------------------------------------------------------+
+ 16 to 27 + Trend parameters for time series before change (see Trend product) +
+----------+--------------------------------------------------------------------+
+ 28 to 39 + Trend parameters for time series after change (see Trend product)  +
+----------+--------------------------------------------------------------------+

