.. _txt-param:

Parameterization
================

A parameter file is mandatory for the Texture submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_TXT_START++`` and ``++PARAM_TXT_END++`` keywords enclose the parameter file.

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

* **Features**

  * This parameter specifies the feature(s) used for the analysis.
    The basename of a tiled dataset needs to be given, followed by an integer list that specifies the bands that are to be used.
    This parameter can be given multiple times if multiple features are to be used. 
    The features are used in the same order as given here, thus keep this in mind when training machine learning models with force-train.

    | *Type:* Basename of file, followed by Integer list
    | ``INPUT_FEATURE = 2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_STM.tif 1 2 3 4 5 6``
    | ``INPUT_FEATURE = 2018-2018_001-365_LEVEL4_TSA_SEN2L_NIR_STM.tif 7 8 9 10 11 12 13``
    | ``INPUT_FEATURE = 2018-2018_001-365_LEVEL4_TSA_SEN2L_RED_STM.tif 1 2 3 4 5 6 7 8 9 10 11 12 13``
    
  * Nodata value of the features.

    | *Type:* Integer. Valid values: [-32767,32767]
    | ``FEATURE_NODATA = -32767``
    
  * Should nodata values be excluded if any feature is nodata (TRUE).
  * Or just proceed (FALSE)?

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``FEATURE_EXCLUDE = FALSE``

* **Texture**

  * This parameter defines the radius of the kernel used for computing the texture metrics (in projection units, commonly in meters).

    | *Type:* Double. Valid values: ]0,BLOCK_SIZE]
    | ``TXT_RADIUS = 50``
    
  * This parameter defines the number of iterations for running the morphological operations.

    | *Type:* Integer. Valid values: [1,1000]
    | ``TXT_ITERATION = 1``
    
  * Which Texture Metrics should be computed? There will be one TXT output file for each metric with as many bands as there are features(in the same order).
    Currently available metrics are dilation, erosion, opening, closing, gradient, blackhat and tophat.

    | *Type:* Character list. Valid values: {DIL,ERO,OPN,CLS,GRD,BHT,THT}
    | ``TXT = DIL ERO BHT``
    
  * This parameter defines the basename for the output files.
    The basename will be appended by Module ID, product ID, and the file extension.

    | *Type:* Character.
    | ``TXT_BASE = TEXTURE``

    