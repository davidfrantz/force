.. _tsa-param:

Parameterization
================

A parameter file is mandatory for the Time Series Analysis submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_TSA_START++`` and ``++PARAM_TSA_END++`` keywords enclose the parameter file.

The following parameter descriptions are a print-out of ``force-parameter``, which can generate an empty parameter file skeleton.


* **Input/output directories**

  * Lower Level datapool (parent directory of tiled input data)

    | *Type:* full directory path
    | ``DIR_LOWER = NULL``
    
  * Higher Level datapool (parent directory of tiled output data)

    | *Type:* full directory path
    | ``DIR_HIGHER = NULL``

  * This parameter controls whether the output is written as multi-band image, or if the stack will be exploded into single-band files.
  
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_EXPLODE = FALSE``

* **Masking**

  * Analysis Mask datapool (parent directory of tiled analysis masks)
    If no analsys mask should be applied, give NULL.

    | *Type:* full directory path
    | ``DIR_MASK = NULL``
    
  * Basename of analysis masks (e.g. WATER-MASK.tif).
    Masks need to be binary with 0 = off / 1 = on

    | *Type:* Basename of file
    | ``BASE_MASK = NULL``

* **Output options**

  * Output format, which is either uncompressed flat binary image format aka ENVI Standard or GeoTiff.
    GeoTiff images are compressed with LZW and horizontal differencing; BigTiff support is enabled; the Tiff is structured with striped blocks according to the TILE_SIZE (X) and BLOCK_SIZE (Y) specifications.
    Metadata are written to the ENVI header or directly into the Tiff to the FORCE domain.
    If the size of the metadata exceeds the Tiff's limit, an external .aux.xml file is additionally generated.

    | *Type:* Character. Valid values: {ENVI,GTiff}
    | ``OUTPUT_FORMAT = GTiff``

* **Parallel processing**

  * This module is using a streaming mechanism to speed up processing.
    There are three processing teams (3 Threads) that simultaneously handle Input, Processing, and Output.
    Example: when Block 2 is being processed, data from Block 3 are already being input and results from Block 1 are being output.
    Each team can have multiple sub-threads to speed up the work. 
    The number of threads to use for each team is given by following parameters.

    | *Type:* Integer. Valid range: [1,...
    | ``NTHREAD_READ = 8``
    | ``NTHREAD_COMPUTE = 22``
    | ``NTHREAD_WRITE = 4``

* **Processing extent and resolution**

  * Analysis extent, given in tile numbers (see tile naming)
    Each existing tile falling into this square extent will be processed
    A shapefile of the tiles can be generated with force-tabulate-grid

    | *Type:* Integer list. Valid range: [-999,9999]
    | ``X_TILE_RANGE = 0 0``
    | ``Y_TILE_RANGE = 0 0``

  * White list of tiles.
    Can be used to further limit the analysis extent to non-square extents.
    The white list is intersected with the analysis extent, i.e. only tiles included in both the analysis extent AND the white-list will be processed.
    Optional. If NULL, the complete analysis extent is processed

    | *Type:* full file path
    | ``FILE_TILE = NULL``
    
  * This parameter can be used to override the default blocksize of the input images (as specified in the datacube-definition.prj file).
    This can be necessary if the default blocksize is too large for your system and you cannot fit all necessary data into RAM.
    Note that processing of larger blocksizes is more efficient.
    The tilesize must be dividable by the blocksize without remainder.
    Set to 0, to use the default blocksize

    | *Type:* Character/Double. Valid range: 0 or [RESOLUTION,TILE_SIZE]
    | ``BLOCK_SIZE = 0``
    
  * Analysis resolution.
    The tile (and block) size must be dividable by this resolution without remainder, e.g. 30m resolution with 100km tiles is not possible

    | *Type:* Double. Valid range: ]0,BLOCK_SIZE]
    | ``RESOLUTION = 10``
    
  * How to reduce spatial resolution for cases when the image resolution is higher than the analysis resolution.
    If FALSE, the resolution is degraded using Nearest Neighbor resampling (fast).
    If TRUE, an approx. Point Spread Function (Gaussian lowpass with FWHM = analysis resolution) is used to approximate the acquisition of data at lower spatial resolution

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``REDUCE_PSF = FALSE``
    
  * If you have spatially enhanced some Level 2 ARD using the FORCE Level 2 ImproPhe module, this switch specifies whether the data are used at original (FALSE) or enhanced spatial resolution (TRUE).
  * If there are no improphe'd products, this switch doesn't have any effect

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``USE_L2_IMPROPHE = FALSE``


.. _tsa-sensor:    

* **Sensor white list**

  * Sensors to be used in the analysis.
    Multi-sensor analyses are restricted to the overlapping bands (see table).
    The resulting outputs are named according to their band designation, i.e. LNDLG, SEN2L, SEN2H, R-G-B or VVVHP.
    BAP Composites with such a band designation can be input again (e.g. SENSORS = LNDLG).
    Following sensors are available: 


    .. _table-tsa-sensor-bands:

    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SENSOR                         + BLUE + GREEN + RED + RE1 + RE2 + RE3 + BNIR + NIR + SWIR1 + SWIR2 + VV + VH +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND04  + Landsat 4 TM          + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND05  + Landsat 5 TM          + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND07  + Landsat 7 ETM+        + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LND08  + Landsat 8 OLI         + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2A  + Sentinel-2A           + 1    + 2     + 3   + 4   + 5   + 6   + 7    + 8   + 9     + 10    +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2B  + Sentinel-2B           + 1    + 2     + 3   + 4   + 5   + 6   + 7    + 8   + 9     + 10    +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + sen2a  + Sentinel-2A           + 1    + 2     + 3   +     +     +     + 7    +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + sen2b  + Sentinel-2B           + 1    + 2     + 3   +     +     +     + 7    +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1AIA  + Sentinel-1A IW asc.   +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1BIA  + Sentinel-1B IW asc.   +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1AID  + Sentinel-1A IW desc.  +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + S1BID  + Sentinel-1B IW desc.  +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + LNDLG  + Landsat legacy bands  + 1    + 2     + 3   +     +     +     +      + 4   + 5     + 6     +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2L  + Sentinel-2 land bands + 1    + 2     + 3   + 4   + 5   + 6   + 7    + 8   + 9     + 10    +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + SEN2H  + Sentinel-2 high-res   + 1    + 2     + 3   +     +     +     + 7    +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + R-G-B  + Visible bands         + 1    + 2     + 3   +     +     +     +      +     +       +       +    +    +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
    + VVVHP  + VV/VH Dual Polarized  +      +       +     +     +     +     +      +     +       +       + 1  + 2  +
    +--------+-----------------------+------+-------+-----+-----+-----+-----+------+-----+-------+-------+----+----+
 
    | *Type:* Character list. Valid values: {LND04,LND05,LND07,LND08,SEN2A,SEN2B,sen2a,sen2b,S1AIA,S1BIA,S1AID,S1BID,LNDLG,SEN2L,SEN2H,R-G-B,VVVHP}
    | ``SENSORS = LND08 SEN2A SEN2B``

* **QAI screening**

  * This list controls, which QAI flags are masked out before doing the analysis.

    | *Type:* Character list. Valid values: {NODATA,CLOUD_OPAQUE,CLOUD_BUFFER,CLOUD_CIRRUS,CLOUD_SHADOW,SNOW,WATER,AOD_FILL,AOD_HIGH,AOD_INT,SUBZERO,SATURATION,SUN_LOW,ILLUMIN_NONE,ILLUMIN_POOR,ILLUMIN_LOW,SLOPED,WVP_NONE}
    | ``SCREEN_QAI = NODATA CLOUD_OPAQUE CLOUD_BUFFER CLOUD_CIRRUS CLOUD_SHADOW SNOW SUBZERO SATURATION``
    
  * Threshold for removing outliers.
    Triplets of observations are used to determine the overall noise in the time series by computinglinearly interpolating between the bracketing observations.
    The RMSE of the residual between the middle value and the interpolation is the overall noise.
    Any observations, which have a residual larger than a multiple of the noise are iteratively filtered out (ABOVE_NOISE).
    Lower/Higher values filter more aggressively/conservatively. 
    Likewise, any masked out observation (as determined by the SCREEN_QAI filter) can be restored if its residual is lower than a multiple of the noise (BELOW_NOISE).
    Higher/Lower values will restore observations more aggressively/conservative.
    Give 0 to both parameters to disable the filtering.

    | *Type:* Float. Valid range: [0,...
    | ``ABOVE_NOISE = 3``
    | ``BELOW_NOISE = 1``

* **Processing timeframe**

  * Time extent for the analysis.
    All data between these dates will be used in the analysis.

    | *Type:* Date list. Format: YYYY-MM-DD
    | ``DATE_RANGE = 2010-01-01 2019-12-31``

  * DOY range for filtering the time extent.
    Day-of-Years that are outside of the given interval will be ignored.
    Example: DATE_RANGE = 2010-01-01 2019-12-31, DOY_RANGE = 91 273 will use all April-Sepember observations from 2010-2019.
    If you want to extend this window over years give DOY min > DOY max.
    Example: DATE_RANGE = 2010-01-01 2019-12-31, DOY_RANGE = 274 90 will use all October-March observations from 2010-2019.

    | *Type:* Integer list. Valid values: [1,365]
    | ``DOY_RANGE = 1 365``

* **Spectral index**

  * Perform the time series analysis using the specified band or index.
    Multiple indices can be processed ar once to avoid multiple reads of the same file.
    Only necessary bands will be input.
    You will be alerted if the index cannot be computed based on the requested SENSORS.
    The index SMA is a linear spectral mixture analysis and is dependent on the parameters specified in the SPECTRAL MIXTURE ANALYSIS section below.

    | *Type:* Character list. Valid values: {BLUE,GREEN,RED,NIR,SWIR1,SWIR2,RE1,RE2,RE3,BNIR,NDVI,EVI,NBR,NDTI,ARVI,SAVI,SARVI,TC-BRIGHT,TC-GREEN,TC-WET,TC-DI,NDBI,NDWI,MNDWI,NDMI,NDSI,SMA}
    | ``INDEX = NDVI EVI NBR``



    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + Index     + Name                                       + Formula                                                                                  + Reference                +
    +===========+============================================+==========================================================================================+==========================+
    + BLUE      + see :ref:`Sensor Bands <table-tsa-sensor-bands>`                                                                                                                 +
    +-----------+                                                                                                                                                                  +
    + GREEN     +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + RED       +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + RE1       +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + RE2       +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + RE3       +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + BNIR      +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + NIR       +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + SWIR1     +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + SWIR2     +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + VV        +                                                                                                                                                                  +
    +-----------+                                                                                                                                                                  +
    + VH        +                                                                                                                                                                  +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NDVI      + Normalized Difference Vegetation Index     + (NIR - RED) / (NIR + RED)                                                                + Tucker 1979              +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + EVI       + Enhanced Vegetation Index                  + | G * ((NIR - R) / (NIR + C1 * RED – C2 * BLUE + L))                                     + Huete et al. 2002        +
    +           +                                            + | with G = 2.5, L = 1, C1 = 6, C2 = 7.5                                                  +                          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NBR       + Normalized Burn Ratio                      + (NIR - SWIR2) / (NIR + SWIR2)                                                            + Key & Benson 2005        +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NDTI      + Normalized Difference Tillage Index        + (SWIR1 - SWIR2) / (SWIR1 + SWIR2)                                                        + Van Deventer et al. 1997 +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + ARVI      + Atmospherically Resistant Vegetation Index + | (NIR - RB) / (NIR + RB)                                                                + Kaufman & Tanré 1992     +
    +           +                                            + | with RB = RED - (BLUE - RED)                                                           +                          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + SAVI      + Soil Adjusted Vegetation Index             + | (NIR - RED) / (NIR + RED + L) * (1 + L)                                                + Huete 1988               +
    +           +                                            + | with L = 0.5                                                                           +                          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + SARVI     + Soil adj. and Atm.  Resistant Veg. Index   + | (NIR - RB) / (NIR + RB + L) * (1 + L)                                                  + Kaufman & Tanré 1992     +
    +           +                                            + | with RB = RED - (BLUE - RED)                                                           +                          +
    +           +                                            + | with L = 0.5                                                                           +                          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + TC-BRIGHT + Tasseled Cap Brightness                    +  0.2043 * BLUE  0.4158 * GREEN  0.5524 * RED 0.5741 * NIR  0.3124 * SWIR1  0.2303 *SWIR2 + Crist 1985               +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + TC-GREEN  + Tasseled Cap Greeness                      + -0.1603 * BLUE -0.2819 * GREEN -0.4934 * RED 0.7940 * NIR -0.0002 * SWIR1 -0.1446 *SWIR2 + Crist 1985               +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + TC-WET    + Tasseled Cap Wetness                       +  0.0315 * BLUE  0.2021 * GREEN  0.3102 * RED 0.1594 * NIR -0.6806 * SWIR1 -0.6109 *SWIR2 + Crist 1985               +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + TC-DI     +                                            + | TC-BRIGHT - (TC-GREEN + TC-WET)                                                        + Healey et al. 1995       +
    +           +                                            + | no rescaling applied (as opposed to Healey et al. 1995)                                +                          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NDBI      + Normalized Difference Built-Up Index       + (SWIR1 - NIR) / (SWIR1 + NIR)                                                            + Zha et al. 2003          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NDWI      + Normalized Difference Water Index          + (GREEN - NIR) / (GREEN + NIR)                                                            + McFeeters 1996           +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + MNDWI     + Modified Normalized Difference Water Index + (GREEN - SWIR1) / (GREEN + SWIR1)                                                        + Xu, H. 2006              +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NDMI      + Normalized Difference Moisture Index       + (NIR - SWIR1) / (NIR + SWIR1)                                                            + Gao 1996                 +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + NDSI      + Normalized Difference Snow Index           + (GREEN - SWIR1) / (GREEN + SWIR1)                                                        + Hall et al. 1995         +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+
    + SMA       + Spectral Mixture Analysis                  + | BOA = F * endmember + E                                                                + Smith et al. 1990        +
    +           +                                            + | Fraction F is retrieved using least-squares optimization                               +                          +
    +           +                                            + | from a couple of endmembers and BOA reflectance, E is model error                      +                          +
    +-----------+--------------------------------------------+------------------------------------------------------------------------------------------+--------------------------+

    
  * Standardize the TSS time series with pixel mean and/or standard deviation?

    | *Type:* Logical. Valid values: {NONE,NORMALIZE,CENTER}
    | ``STANDARDIZE_TSS = NONE``
    
  * Output the quality-screened Time Series Stack? This is a layer stack of index values for each date.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_TSS = FALSE``

* **Spectral mixture analysis**

  * This block only applies if INDEX includes SMA
   Endmember file holding the endmembers according to the SENSORS band subset

    | *Type:* full file path
    | ``FILE_ENDMEM  = NULL``

  * Sum-to-One constrained unmixing?

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``SMA_SUM_TO_ONE = TRUE``
    
  * Non-negativity constrained unmixing?

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``SMA_NON_NEG = TRUE``
    
  * Apply shade normalization? If TRUE, the last endmember FILE_ENDMEM needs to be the shade spectrum

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``SMA_SHD_NORM = TRUE``
    
  * Endmember to be used for the analysis.
    This number refers to the column, in which the desired endmember is stored (FILE_ENDMEM).

    | *Type:* Integer. Valid range: [1,NUMBER_OF_ENDMEMBERS]
    | ``SMA_ENDMEMBER = 1``
    
  * Output the SMA model Error? This is a layer stack of model RMSE for each date.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_RMS = FALSE``

* **Interpolation parameters**

  * Interpolation method.
    You can choose between no, linear, moving average or Radial Basis Function Interpolation.

    | *Type:* Character. Valid values: {NONE,LINEAR,MOVING,RBF}
    | ``INTERPOLATE = RBF``
    
  * Max temporal distance for the moving average filter in days.
    For each interpolation date, MOVING_MAX days before and after are considered.

    | *Type:* Integer. Valid range: [1,365]
    | ``MOVING_MAX = 16``
    
  * Sigma (width of the Gaussian bell) for the RBF filter in days.
    For each interpolation date, a Gaussian kernel is used to smooth the observations.
    The smoothing effect is stronger with larger kernels and the chance of having nodata values is lower.
    Smaller kernels will follow the time series more closely but the chance of having nodata values is larger.
    Multiple kernels can be combined to take advantage of both small and large kernel sizes.
    The kernels are weighted according to the data density within each kernel.

    | *Type:* Integer list. Valid range: [1,365]
    | ``RBF_SIGMA = 8 16 32``
    
  * Cutoff density for the RBF filter.
    The Gaussian kernels have infinite width, which is computationally slow, and doesn't make much sense as observations that are way too distant (in terms of time) are considered.
    Thus, the tails of the kernel are cut off.
    This parameter specifies, which percentage of the area under the Gaussian should be used.

    | *Type:* Float. Valid range: ]0,1]
    | ``RBF_CUTOFF = 0.95``

  * This parameter gives the interpolation step in days.

    | *Type:* Integer. Valid range: [1,...
    | ``INT_DAY = 16``
    
  * Standardize the TSI time series with pixel mean and/or standard deviation?

    | *Type:* Logical. Valid values: {NONE,NORMALIZE,CENTER}
    | ``STANDARDIZE_TSI = NONE``
    
  * Output the Time Series Interpolation? This is a layer stack of index values for each interpolated date.
    Note that interpolation will be performed even if OUTPUT_TSI = FALSE - unless you specify INTERPOLATE = NONE.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_TSI = FALSE``

* **Spectral temporal metrics**

  * Output Spectral Temporal Metrics? The remaining parameters in this block are only evaluated if TRUE

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_STM = FALSE``
    
  * Which Spectral Temporal Metrics should be computed? 
    The STM output files will have as many bands as you specify metrics (in the same order).
    Currently available statistics are the average, standard deviation, minimum, maximum, range, skewness, kurtosis, any quantile from 1-99%, and interquartile range.
    Note that median is Q50.

    | *Type:* Character list. Valid values: {MIN,Q01-Q99,MAX,AVG,STD,RNG,IQR,SKW,KRT,NUM}
    | ``STM = Q25 Q50 Q75 AVG STD``

* **Folding parameters**

  * Which statistic should be used for folding the time series? This parameter is only evaluated if one of the following outputs in this block is requested.
    Currently available statistics are the average, standard deviation, minimum, maximum, range, skewness, kurtosis, median, 10/25/75/90% quantiles, and interquartile range

    | *Type:* Character. Valid values: {MIN,Q10,Q25,Q50,Q75,Q90,MAX,AVG,STD,RNG,IQR,SKW,KRT,NUM
    | ``FOLD_TYPE = AVG``
    
  * Standardize the FB* time series with pixel mean and/or standard deviation?

    | *Type:* Logical. Valid values: {NONE,NORMALIZE,CENTER}
    | ``STANDARDIZE_FOLD = NONE``
    
  * Output the Fold-by-Year/Quarter/Month/Week/DOY time series? These are layer stacks of folded index values for each year, quarter, month, week or DOY.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_FBY = FALSE``
    | ``OUTPUT_FBQ = FALSE``
    | ``OUTPUT_FBM = FALSE``
    | ``OUTPUT_FBW = FALSE``
    | ``OUTPUT_FBD = FALSE``
    
  * Compute and output a linear trend analysis on any of the folded time series?
    Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.
    See also the TREND PARAMETERS block below.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_TRY = FALSE``
    | ``OUTPUT_TRQ = FALSE``
    | ``OUTPUT_TRM = FALSE``
    | ``OUTPUT_TRW = FALSE``
    | ``OUTPUT_TRD = FALSE``
    
  * Compute and output an extended Change, Aftereffect, Trend (CAT) analysis on any of the folded time series?
    Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.
    See also the TREND PARAMETERS block below.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_CAY = FALSE``
    | ``OUTPUT_CAQ = FALSE``
    | ``OUTPUT_CAM = FALSE``
    | ``OUTPUT_CAW = FALSE``
    | ``OUTPUT_CAD = FALSE``

* **Land surface phenology parameters**

  .. note::
     The Land Surface Phenology (LSP) options are only available if FORCE was compiled with SPLITS (see :ref:`install` section).

  * For estimating LSP for one year, some data from the previous/next year need to be considered to find the seasonal minima, which define a season.
    The parameters are given in DOY, i.e. LSP_DOY_PREV_YEAR = 273, and LSP_DOY_NEXT_YEAR = 91 will use all observations from October (Year-1) to March (Year+1)

    | *Type:* Integer. Valid range: [1,365]
    | ``LSP_DOY_PREV_YEAR = 273``
    | ``LSP_DOY_NEXT_YEAR = 91``
    
  * Seasonality is of Northern-, Southern-hemispheric or of mixed type?
    If mixed, the code will attempt to estimate the type on a per-pixel basis.

    | *Type:* Character. Valid values: {NORTH,SOUTH,MIXED}
    | ``LSP_HEMISPHERE = NORTH``
    
  * How many segments per year should be used for the spline fitting? 
    More segments follow the seasonality more closely, less segments smooth the time series stronger.

    | *Type:* Integer. Valid range: [1,...
    | ``LSP_N_SEGMENT = 4``
    
  * Amplitude threshold for detecing Start, and End of Season, i.e. the date, at which xx% of the amplitude is observed

    | *Type:* Float. Valid range: ]0,1[
    | ``LSP_AMP_THRESHOLD = 0.2``
    
  * LSP won't be derived if the seasonal index values do not exceed following value.
    This is useful to remove unvegetated surfaces.

    | *Type:* Integer. Valid range: [-10000,10000]
    | ``LSP_MIN_VALUE = 500``
    
  * LSP won't be derived if the seasonal amplitude is below following value
    This is useful to remove surfaces that do not have a seasonality.

    | *Type:* Integer. Valid range: [0,10000]
    | ``LSP_MIN_AMPLITUDE = 500``
    
  * Which Phenometrics should be computed? There will be a LSP output file for each metric (with years as bands).
    Currently available are the dates of the early minimum, start of season, rising inflection, peak of season, falling inflection, end of season, late minimum; 
    lengths of the total season, green season; values of the early minimum, start of season, rising inflection, peak of season, falling inflection, end of season, late minimum, base level, seasonal amplitude;
    integrals of the total season, base level, base+total, green season; 
    rates of averahe rising, average falling, maximum rising, maximum falling.

    | *Type:* Character list. Valid values: {DEM,DSS,DRI,DPS,DFI,DES,DLM,LTS,LGS,VEM,VSS,VRI,VPS,VFI,VES,VLM,VBL,VSA,IST,IBL,IBT,IGS,RAR,RAF,RMR,RMF}
    | ``LSP = VSS VPS VES VSA RMR IGS``
    
  * Standardize the LSP time series with pixel mean and/or standard deviation?

    | *Type:* Logical. Valid values: {NONE,NORMALIZE,CENTER}
    | ``STANDARDIZE_LSP = NONE``
    
  * Output the Spline fit? This is a layer stack of fitted index values for interpolated date.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_SPL = FALSE``
    
  * Output the Phenometrics? These are layer stacks per phenometric with as many bands as years (excluding one year at the beginning/end of the time series.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_LSP = FALSE``
    
  * Compute and output a linear trend analysis on the requested Phenometric time series? 
    Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.
    See also the TREND PARAMETERS block below.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_TRP = FALSE``
    
  * Compute and output an extended Change, Aftereffect, Trend (CAT) analysis on the requested Phenometric time series?
    Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.
    See also the TREND PARAMETERS block below.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_CAP = FALSE``

* **Trend parameters**

  * This parameter specifies the tail-type used for significance testing of the slope in the trend analysis.
    A left-, two-, or right-tailed t-test is performed.

    | *Type:* Character. Valid values: {LEFT,TWO,RIGHT}
    | ``TREND_TAIL = TWO``
    
  * Confidence level for significance testing of the slope in the trend analysis 
  
    | *Type:* Float. Valid range: [0,1]
    | ``TREND_CONF = 0.95``
