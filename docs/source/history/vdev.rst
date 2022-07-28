.. _vdev:

Develop version
===============

- **General**

  - FORCE comes with some default output file formats, 
    most noteably the striped GTiff default format with LZW compression.

    It is now possible that users can define their own output file format with custom GDAL options.
    There is now a new parameter: ``FILE_OUTPUT_OPTIONS``. 
    This expects a file, and will only be activated when ``OUTPUT_FORMAT = CUSTOM``.

    The text file should be styled in tag and value notation like this:

    | DRIVER = GTiff
    | EXTENSION = tif
    | BIGTIFF = YES
    | COMPRESS = ZSTD
    | PREDICTOR = 2
    | ZLEVEL = 1
    | INTERLEAVE = BAND
    | TILED = YES
    | BLOCKXSIZE = 1024
    | BLOCKYSIZE = 1024

    **Important:** the file needs at least the ``DRIVER`` (GDAL short driver name) and ``EXTENSION``, 
    and then a variable number of GDAL options (up to 32 - this should be enough, right?).

    Some thoughts of caution: with opening this up to the user, 
    it is now possible to give invalid or conflicting options that result in the failure of creating files.

    Thanks to Max Freudenberg for suggesting and testing this feature.

- **FORCE L2PS**

  - scattered nodata pixels occured in Sentinel-2 imagery at random locations.
    This is because the cirrus band TOA reflectance can be 0 or negative over dark surfaces (e.g. water or shadow).
    As this is physically impossible (and we cannot recover from this during atmospheric correction), FORCE screens
    for *bad* pixels and masks them out.

    This condition was now relaxed for the cirrus band.
    Thanks to Max Helleis for bringing this up.

  - Added a small hiccup when parsing sensor ID. 
    Was only relevant when it failed by giving wrong input data.

- **FORCE HLPS**

  - deprecation note:

    The option to use the SPLITS algorithm to derive LSP metrics in ``force-higher-level``
    will be deprecated in with version 3.7.9.

    Please use the polar-based LSP retrieval method instead (recommended anyway).

  - in ``force-higher-level``: 
    since compiling against python for developing the UDF sub-module,
    ``force-higher-level`` did not honor ``ctrl+c`` anymore to abort the
    processing, and the program needed to be killed explicitly.
    Florian Katerndahl provided a fix for this! 
    ``ctrl+c`` works again.

  - in ``force-higher-level``, TSA sub-module:
    
    - New indices were added:
    
      - CCI (provided by J. Antonio Guzmán Q.)
      - EVI2 (suggested by Jonas Ardo)
      - Continuum Removal of SWIR1

    - A recipe for adding a new index was added to the repository in order to facilitate users the implementation of additional indices.
    - J. Antonio Guzmán Q. additionally changed the scaling factor for simple ratio indices like CIre (was 10000, is 1000).

  - in ``force-higher-level``, TSA sub-module:
    added the harmonic interpolation method from Zhu et al. 2015 (http://dx.doi.org/10.1016/j.rse.2015.02.009).
    This can be used with ``INTERPOLATE = HARMONIC``.
    ``HARMONIC_MODES = 3`` defines how many modes per season are used, 
    i.e. uni-modal (1), bi-modal (2), or tri-modal (3).

  - in ``force-higher-level``, TSA sub-module:
    added a simple near-real time monitoring component.
    When using the harmonic interpolation, the user can fit the harmonic to a subset of the time period with
    ``HARMONIC_FIT_RANGE``.

    For example, if the analysis timeframe is ``DATE_RANGE = 2015-01-01 2022-06-20``, 
    all data from 2015-2022 will be considered. If ``HARMONIC_FIT_RANGE = 2015-01-01 2017-12-31``, 
    the harmonic will only be fitted to the first 3 years of data.
    The new NRT product (``OUTPUT_NRT = TRUE``) will then contain the residual between the 
    extrapolated harmonic and the actual data from 2018 on.

    This can be used to identify changes in the present relative to the "usual" seasonality observed in the past.


  - in ``force-higher-level``, TSA sub-module (and probably others):
    Stefan Ernst noted that the TSA submodule did not produce any output when using Landsat 7, Landsat 9 and Sentinel-2 simultaneously.
    This gives us the unusual situation that we have >= 3 observations per day.

    This caused a divide-by-0 error in the linear interpolation that is used for detecting remaining noise in the time series.
    This is now fixed by simply computing the average when we have more then 2 obs/day.

  - in ``force-higher-level``, ML sub-module:
    a stack smashing bug occured when using more than 8 modelsets.
    This is now fixed. Thanks to Fabian Thiel for finding this.

- **FORCE AUX**

  - in ``force-qai-inflate``:
    changed output nodata from 1 (which is a valid value) to 255.
    Thanks to Fabian Thiel for bringing this up.

  - in ``force-lut-modis``: 
    Vincent Schut reportet that the program does not properly detect a 504 response from the server,
    and hangs infinitely.
    Thanks to Florian Katerndahl for adding a fix that catches HTTP responses >= 400.

  - in ``force-cube``:
    If a resulting image is completely nodata, it will automatically be removed.
    This works now for both raster-to-raster and vector-to-raster.
    Empty tiles will be removed as well.
    Thanks for László Henits for bringing this up. 
    Thanks to Stefan Ernst for suggesting a fix.

#-- No further changes yet.
