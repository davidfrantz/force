.. _ml-param:

Parameterization
================

A parameter file is mandatory for the Machine Learning submodule of FORCE HLPS.

The file extension is ‘.prm’.
All parameters must be given, even if they are not used.
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file.

The ``++PARAM_ML_START++`` and ``++PARAM_ML_END++`` keywords enclose the parameter file.

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

    | *Type:* Integer. Valid values: [-32768,32767]
    | ``FEATURE_NODATA = -9999``
    
  * Should nodata values be excluded if any feature is nodata (TRUE).
    Or just proceed (FALSE)?

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``FEATURE_EXCLUDE = FALSE``

* **Machine learning**

  * Directory containing the machine learning model(s).
    Models can be trained with force-train

    | *Type:* full directory path
    | ``DIR_MODEL = NULL``
    
  * This parameter specifies the modelset used for prediction.
    The basename of the machine learning model(s) (.xml) must be given. 
    One or multiple models can be given.
    The predictions of the models are aggregated into the final prediction.
    The aggregation function is the average for regression problems, and the mode for classification problems.
    This parameter can be given multiple times, in which case multiple regressions/classifications can be computed.
    Then output files will have as many bands as modelsets are given.

    | *Type:* Basename of file, character list
    | ``FILE_MODEL = biomass-1.xml biomass-2.xml biomass-3.xml``
    | ``FILE_MODEL = canopy-cover.xml``
    | ``FILE_MODEL = tree-height.xml``

  * Machine learning method.
    Currently implemented are Random Forest and Support Vector Machines, both in regression and classification flavors.
    The method must match the models as given with FILE_MODEL.

    | *Type:* Character. Valid values: {SVR,SVC,RFR,RFC}
    | ``ML_METHOD = SVR``

  * This parameter only applies if multiple models are given for a modelset, and machine learning method is of regression flavor.
    The models are blended into the final prediction, and processing time scales linearly with the number of models given.
    However, the blended prediction will likely converge with increasing numbers of models, thus it may not be necessary to compute all models.
    This parameter sets the convergence threshold. 
    If the predictions differ less than this value (when adding another model), no more model will be added.
    This generally speeds up processing substantially.
    The convergence is tested for each pixel, i.e. each pixel is predicted with as many models as necessary to obtain a stable solution.

    | *Type:* Float. Valid range: [0,...
    | ``ML_CONVERGENCE = 0``
    
  * This parameter is a scaling factor to scale the prediction to fit into a 16bit signed integer.
    This parameter should be set in dependence on the scale used for training the model.

    | *Type:* Float. Valid range: ]0,...
    | ``ML_SCALE = 10000``
    
  * This parameter defines the basename for the output files.
    The basename will be appended by Module ID, product ID, and the file extension.

    | *Type:* Character.
    | ``ML_BASE = PREDICTION``
    
  * Output the Machine Learning Prediction?

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_MLP = FALSE``
    
  * Output the number of models used when blending the prediction? Makes most sense when ML_CONVERGENCE is used.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_MLI = FALSE``

  * Output the uncertainty of the blended prediction? This is the standard deviation of all predictions that are blended into the final prediction.
    Only makes sense when multiple models are given in a modelset.

    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_MLU = FALSE``

  * Output the Random Forest Class Probabilities? 
    This option is only available when ``ML_METHOD = RFC``. 
    If multiple models are given per modelset, the mean class probability is computed. 
    The output file will have as many bands as classes. 
    If multiple modelsets are given, the modelsets are appended after each other.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_RFP = FALSE``

  * Output the Random Forest Classification Margin? 
    This option is only available when ``ML_METHOD = RFC``. 
    If multiple models are given per modelset, the margin is based on the mean class probability. 
    If multiple modelsets are given, a margin is computed for each modelset.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_RFM = FALSE``
