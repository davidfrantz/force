.. _l2i-param:

Parameterization
================

A parameter file is mandatory for the Level 2 ImproPhe submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_L2IMP_START++`` and ``++PARAM_L2IMP_END++`` keywords enclose the parameter file.

The following parameter descriptions are a print-out of ``force-parameter``, which can generate an empty parameter file skeleton.


* **Input/output directories**

  * Lower Level datapool (parent directory of tiled input data)

    | *Type:* full directory path
    | ``DIR_LOWER = NULL``

  * Higher Level datapool (parent directory of tiled output data)
    It is recommended to write the products to the Lower Level datapool

    | *Type:* full directory path
    | ``DIR_HIGHER = NULL``

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

  * Output format, which is either uncompressed flat binary image format aka ENVI Standard, GeoTiff, or COG. 
    GeoTiff images are compressed with LZW and horizontal differencing; BigTiff support is enabled; the Tiff is structured with striped blocks according to the TILE_SIZE (X) and BLOCK_SIZE (Y) specifications.
    Metadata are written to the ENVI header or directly into the Tiff to the FORCE domain.
    If the size of the metadata exceeds the Tiff's limit, an external .aux.xml file is additionally generated.

    | *Type:* Character. Valid values: {ENVI,GTiff,COG}
    | ``OUTPUT_FORMAT = GTiff``

  * File that contains custom GDAL output options. 
    This is only used if OUTPUT_FORMAT = CUSTOM. 
    If OUTPUT_FORMAT = CUSTOM, this file is mandatory.
    The file should be written in tag and value notation. 
    The first two lines are mandatory and specify GDAL driver and file extension, 
    e.g. DRIVER = GTiff and EXTENNSION = tif. 
    The driver name refers to the GDAL short driver names. 
    Lines 3ff can hold a variable number of GDAL options (up to 32 are allowed).
    Please note: with opening output options up to the user, it is now possible to
    give invalid or conflicting options that result in the failure of creating files.
    Type: full file path

    | *Type:* full file path
    | ``FILE_OUTPUT_OPTIONS = NULL``

  * This parameter controls whether the output is written as multi-band image, or if the stack will be exploded into single-band files.
  
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

  * Analysis extent, given in tile numbers (see tile naming)
    Each existing tile falling into this square extent will be processed
    A shapefile of the tiles can be generated with force-tabulate-grid

    | *Type:* Integer list. Valid range: [-999,9999]
    | ``X_TILE_RANGE = 0 0``
    | ``Y_TILE_RANGE = 0 0``

  * Allow-list of tiles.
    Can be used to further limit the analysis extent to non-square extents.
    The allow-list is intersected with the analysis extent, i.e. only tiles included in both the analysis extent AND the allow-list will be processed.
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
    If there are no improphe'd products, this switch doesn't have any effect

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``USE_L2_IMPROPHE = FALSE``

* **Sensor allow-list**

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

* **ImproPhe parameters**

  * This parameter defines the seasonal windows for which the Level 2 ARD should be aggregated.
    This parameter expects a list of DOYs that define the window breakpoints.
    If you specify 5 breakpoints, there will be four windows.
    The windows can extend to the previous/next year (e.g. 270 30 91 181 270 would extend into the previous year, 1 30 91 181 270 30 would extend into the next year.

    | *Type:* Integer list. Valid values: [1,365]
    | ``SEASONAL_WINDOW = 1 91 181 271 365``
    
  * This parameter defines the radius of the prediction kernel (in projection units, commonly in meters).
    A larger kernel increases the chance of finding a larger number of within-class pixels, but increases prediction time

    | *Type:* Double. Valid values: ]0,BLOCK_SIZE]
    | ``KERNEL_SIZE = 2500``
    
  * This parameter defines the radius of the kernel used for computing the heterogeneity proxies (in projection units, commonly in meters).
    The heterogeneity proxies are derived from a focal standard deviation filter.
    The width of the kernel should reflect the scale difference between the coarse and medium resolution data.

    | *Type:* Double. Valid values: ]0,BLOCK_SIZE]
    | ``KERNEL_TEXT = 330``

* **Level 2 ImproPhe parameters**

  * This parameter defines the sensors, whose spatial resolution should be improved.
    The SENSORS parameter above defines the sensors that serve as target images.
    For a list of available sensors, see the description for the SENSORS parameter.
    For improving the spatial resolution of Landsat to Sentinel-2, it is recommended to use ``SENSORS = sen2a sen2b``, and ``SENSORS_LOWRES = LND07 LND08``

    | *Type:* Character list. Valid values: {LND04,LND05,LND07,LND08,SEN2A,SEN2B,sen2a,sen2b,S1AIA,S1BIA,S1AID,S1BID}
    | ``SENSORS_LOWRES = LND07 LND08``

