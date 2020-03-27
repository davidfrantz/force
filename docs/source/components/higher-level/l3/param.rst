.. _l3-param:

Level 3 Composite Parameterization
==================================

A parameter file is mandatory for the Level 3 Compositing submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_LEVEL3_START++`` and ``++PARAM_LEVEL3_END++`` keywords enclose the parameter file.

The following parameter descriptions are a print-out of ``force-parameter``, which can generate an empty parameter file skeleton.


* **Input/output directories**

  * Lower Level datapool (parent directory of tiled input data)
  
    | *Type:* full directory path
    | ``DIR_LOWER = NULL``
    
  * Higher Level datapool (parent directory of tiled output data)
  
    | *Type:* full directory path
    | ``DIR_HIGHER = NULL``

* **Masking**

  * Analysis Mask datapool (parent directory of tiled analysis masks).
    If no analsys mask should be applied, give NULL.
    
    | *Type:* full directory path
    | ``DIR_MASK = NULL``
    
  * Basename of analysis masks (e.g. WATER-MASK.tif).
    Masks need to be binary with 0 = off / 1 = on
    
    | *Type:* Basename of file
    | ``BASE_MASK = NULL``

* **Output options**

  * Output format, which is either uncompressed flat binary image format aka ENVI Standard or GeoTiff.
    GeoTiff images are compressed with LZW and horizontal differencing; BigTiff support is enabled; the Tiff is structured with striped blocks according to the TILE_SIZE (X) and BLOCK_SIZE (Y) speci    fications.
    Metadata are written to the ENVI header or directly into the Tiff to the FORCE domain.
    If the size of the metadata exceeds the Tiff's limit, an external .aux.xml file is additionally generated.

    | *Type:* Character. Valid values: {ENVI,GTiff}
    | ``OUTPUT_FORMAT = GTiff``

  * This parameter controls whether the output is written as multi-band image, or if the stack will be exploded into single-band files. The BAP product won't be exploded.
  
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_EXPLODE = FALSE``

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

  * Analysis extent, given in tile numbers (see tile naming).
    Each existing tile falling into this square extent will be processed.
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

    | *Type:* Double. Valid range: 0 or [RESOLUTION,TILE_SIZE]
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
    If there are no improphe'd products, this switch doesn't have any effect
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``USE_L2_IMPROPHE = FALSE``

* **Sensor white list**

  * Sensors to be used in the analysis.
    Multi-sensor analyses are restricted to the overlapping bands (see table).
    The resulting outputs are named according to their band designation, i.e. LNDLG, SEN2L, SEN2H, R-G-B or VVVHP.
    BAP Composites with such a band designation can be input again (e.g. SENSORS = LNDLG).
    Following sensors are available: 

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

* **Best Available Pixel (BAP) compositing**

  * This parameter specifies the target year for compositing.
  
    | *Type:* Integer. Valid values: [1900,2100]
    | ``YEAR_TARGET = 2018``
    
  * This parameter specifies thenumber of bracketing years (target year +- bracketing years), i.e. the compositing period.
    A value of 2 would result in a five-year compositing period.
    
    | *Type:* Integer. Valid values: [0,100]
    | ``YEAR_NUM = 2``
    
  * This parameter is a tradeoff parameter that balances the inter- and intra-annual selection.
    Lower values (e.g. 0.75) favor data from the target year.
    Higher values favor data that was acquired close to the target DOY (regardless of the year).

    | *Type:* Float. Valid values: [0,...
    | ``Y_FACTOR = 0.75``
    
  * These parameters specify the function values used for fitting the DOY scoring functions.
    The function type is automatically chosen from the given values, i.e.
    
    +--------------------+--------------+
    + Gaussian           + s0 < s1 > s2 +
    +--------------------+--------------+
    + Descending sigmoid + s0 > s1 > s2 +
    +--------------------+--------------+
    + Ascending sigmoid  + s0 < s1 < s2 +
    +--------------------+--------------+

    | *Type:* Float list, 3 values. Valid values: ]0,1[
    | ``DOY_SCORE = 0.01 0.99 0.01``
    
  * These parameters specify the DOYs used for fitting the DOY scoring functions in case of the static compositing.
    They are not used for the phenology-adaptive compositing. 
    However, in each case, the target date appearing in the file name is derived from these values.
    The target date is the value with highest score (see last parameter).
    Typically, the DOYs are in order, e.g. p0 = 60, p1 = 90, p2 = 120.
    However, the DOY scoring can also extend between the years (i.e. around the turn of the year).
    If p0 > p1: p0 is from previous year, e.g. p0 = 330, p1 = 30, p2 = 90.
    If p2 < p1: p2 is from next year, e.g. p0 = 300, p1 = 330, p2 = 30.
    
    | *Type:* Integer list, 3 values. Valid values: [1,365]
    | ``DOY_STATIC = 120 180 240``
    
  * This parameter specifies whether all available data within the requested time frame are used – or only from the season of interest.
    If FALSE, the composites only consider data for the period, in which the intra-annual score is higher than 0.01.
    If there is no clear-sky data within this period, data gaps are possible.
    If TRUE, all data from the requested years are used, thus the risk of having data gaps is lower.
    However, it is possible that data from unwanted parts of the year are selected.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OFF_SEASON = FALSE``
    
  * This parameter controls the strength of the DOY score.
    0 disables the use of this score.
    
    | *Type:* Float. Valid values: [0,1]
    | ``SCORE_DOY_WEIGHT = 1.0``
    
  * This parameter controls the strength of the Year score.
    0 disables the use of this score.
    
    | *Type:* Float. Valid values: [0,1]
    | ``SCORE_YEAR_WEIGHT = 1.0``
    
  * This parameter controls the strength of the cloud distance score.
    0 disables the use of this score.
    
    | *Type:* Float. Valid values: [0,1]
    | ``SCORE_CLOUD_WEIGHT = 0.4``
    
  * This parameter controls the strength of the haze score.
    0 disables the use of this score.
    
    | *Type:* Float. Valid values: [0,1]
    | ``SCORE_HAZE_WEIGHT = 0.3``
    
  * This parameter controls the strength of the correlation score.
    0 disables the use of this score.
    
    | *Type:* Float. Valid values: [0,1]
    | ``SCORE_CORREL_WEIGHT = 0.0``
    
  * This parameter controls the strength of the view zenith score.
    0 disables the use of this score.
    
    | *Type:* Float. Valid values: [0,1]
    | ``SCORE_VZEN_WEIGHT = 0.0``
    
  * This parameter indicates the distance (to the next cloud or cloud shadow) after which the sky is assumed to be clear (cloud score approaches 1.0).
    The distance needs to be given in meters.
    
    | *Type:* Float. Valid values: [1,...
    | ``DREQ = 3000``
    
  * This parameter indicates the view zenith angle at which the view zenith score approaches 0.0.
    The angle needs to be given in degree.

    | *Type:* Float. Valid values: [1,90]
    | ``VREQ = 7.5``
    
  * Output the composite?
  
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_BAP = TRUE``
    
  * Output the compositing information?
  
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_INF = TRUE``
    
  * Output the compositing scores?
  
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_SCR = FALSE``
    
  * Output quicklook of the composite?
  
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_OVV = FALSE``

* **Phenology Adaptive Compositing (PAC)**

  * This parameter defines whether the phenology-adpative compositing (TRUE) or the static compositing (FALSE) should be used.
    In case of the static version, the target DOYs are derived from DOY_STATIC.
    In case of the PAC, the target DOYs are retrived from the files given by LSP_FILE 

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``LSP_DO = FALSE``
    
  * Land Surface Phenology datapool (parent directory of tiled LSP)
  
    | *Type:* full directory path
    | ``DIR_LSP = NULL``
    
  * Basenames of the LSP datasets that are used as compositing targets (analogously to DOY_STATIC).
    Each file should be a multi-band image wherein the bands represent different years.
    The number of bands, and the corresponding years, need to be the same for all files.

    | *Type:* List with basenames of 3 files
    | ``BASE_LSP = NULL``
    
  * This parameter defines year, which corresponds to he 1st band of the LSP.
  
    | *Type:* Integer. Valid values: [1900,2100]
    | ``LSP_1ST_YEAR = 2000``
    
  * This parameter specifies the starting point of the LSP values.
    Internally, the data are represented as ‘Year x 365 + DOY’.
    Thus, LSP_START is an offset, which must be given as ‘Year x 365 + DOY’.
    If the values are provided in this format, use LSP_START = 1.
    If the LSP values would be provided relative to January 1 2000, use LSP_START = 730001, i.e. 2000*365+1.
    Leap years are not taken into account and each year consists of 365 days.
    
    | *Type:* Integer. Valid values: [1,2100*365]
    | ``LSP_START = 2000``
    
  * This parameter is a threshold in days.
    If the inter-annual variability of the LSP (of a given pixel) exceeds this value, the long-term average LSP is used instead of the yearly values.
    The value should be between 0 (long-term average is used for all pixels) and 365 (long-term average is never used).

    | *Type:* Integer list. Valid values: [0,365]
    | ``LSP_THRESHOLD = 182``
    
  * This parameter defines the nodata value for the LSP.
  
    | *Type:* Integer. Valid values: [-32767,32767]
    | ``LSP_NODATA = -32767``

