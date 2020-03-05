.. _cso-param:

Parameterization
================

A parameter file is mandatory for the Clear Sky Observation submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_CSO_START++`` and ``++PARAM_CSO_END++`` keywords enclose the parameter file.

The following parameter descriptions are a print-out of ``force-parameter``, which can generate an empty parameter file skeleton.


* **Input/output directories**

  * Lower Level datapool (parent directory of tiled input data)

    | *Type:* full directory path
    | ``DIR_LOWER = NULL``
    
  * Higher Level datapool (parent directory of tiled output data)

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

* **CSO parameters**

  * This parameter specifies bin width for summarizing the CSOs.
    The width is given in months

    | *Type:* Integer. Valid values: [1,12]
    | ``MONTH_STEP = 3``

  * Which statistics should be computed? Currently available statistics are the number of observations, and aggregate statistics of the temporal difference between observations (available are average, standard deviation, minimum, maximum, range, skewness, kurtosis, any quantile from 1-99%, and interquartile range.
    Note that median is Q50.

    | *Type:* Character list. Valid values: {NUM,MIN,Q01-Q99,MAX,AVG,STD,RNG,IQR,SKW,KRT}
    | ``CSO = NUM AVG STD``

