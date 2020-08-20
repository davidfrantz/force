.. _l2-param:

Parameter file
==============

A parameter file is mandatory for FORCE L2PS. 

The file extension is ‘.prm’. 
All parameters must be given, even if they are not used. 
All parameters follow common tag-value notation.
Rudimentary checks are performed by the software components using this file. 

The ``++PARAM_LEVEL2_START++`` and ``++PARAM_LEVEL2_END++`` keywords enclose the parameter file. 

The following parameter descriptions are a print-out of ``force-parameter``, which can generate an empty parameter file skeleton.


* **Input/output directories**

  * The file queue specifies, which images are to be processed.
    The full path to the file needs to be given.
    Do  not  paste  the content of the file queue into the parameter file.
    The file queue is mandatory for force-level2, but may be NULL for force-l2ps.
  
    | *Type:* full file path
    | ``FILE_QUEUE = NULL``
    
  * This is the output directory where the Level 2 data will be stored.
    Note that data will be overwritten/mosaicked if you reprocess images.
    It is safe and recommended to use a single Level 2 data pool for different sensors (provided the same grid and projection is used).
    The higher-level programs of FORCE can handle different spatial resolutions (e.g.
    30m Landsat and 10m Sentinel-2).
    
    | *Type:* full directory path
    | ``DIR_LEVEL2 = NULL``
    
  * This is the directory where logfiles should be saved.
    
    | *Type:* full directory path
    | ``DIR_LOG = NULL``
    
  * This is a temporary directory that is used to extract compressed images for force-level2.
    Note that images already need to be extracted when using force-l2ps directly.
    The extracted data will be deleted once they were processed.
    If you cancel processing, you may want to delete any left-overs in this directory.
    A file 'cpu-$TIME' is temporarily created in DIR_TEMP.
    This file can be modified to re-adjust the number of CPUs while(!) force-level2 is running.
    Note that the effect is not immediate, as the load is only adjusted after one of the running jobs (images) is finished.
    
    | *Type:* full directory path
    | ``DIR_TEMP = NULL``
    
* **Digital Elevation Model**
    
  * This file specifies the DEM.
    It is highly recommended to use a DEM.
    It is used for cloud / cloud shadow detection, atmospheric correction and topographic correction.
    The DEM should be a mosaic that should completely cover the area you are preprocessing.
    If there are nodata values in the DEM, the Level 2 outputs will have holes, too.
    It is possible to process without a DEM (DEM = NULL).
    In this case, the surface is assumed flat @ z = 0m.
    Topographic correction cannot be used without a DEM.
    The quality of atmospheric correction and cloud /cloud shadow detection will suffer without a DEM.
    
    | *Type:* full file path
    | ``FILE_DEM = NULL``
    
  * Nodata value of the DEM.
    
    | *Type:* Integer. Valid range: [-32768,32767]
    | ``DEM_NODATA = -32767``
    
* **Data Cubes**
    
  * This indicates whether the images should be reprojected to the target coordinate system or if they should stay in their original UTM projection.
    If you want to work with force-higher-level routines, give TRUE.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``DO_REPROJ = TRUE``
    
  * This indicates whether the images should be gridded after processing.
    If TRUE, sub-directories for the tiles are generated in DIR_LEVEL2.
    If FALSE, sub-directories for the original spatial reference systems are generated in DIR_LEVEL2.
    If you want to work with force-higher-level routines, give TRUE.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``DO_TILE = TRUE``
    
  * This is the tile allow-list.
    It is an optional file that holds all tiles that should be output.
    Tiles, which are not specified in this file are not written to disc.
    This paremeter is ignored if DO_TILE = FALSE.
    If no tile allow-list should be used, give FILE_TILE = NULL, in which case all tiles are output.
    
    | *Type:* full file path
    | ``FILE_TILE = NULL``
    
  * This is the tile size (in target units, commonly in meters) of the gridded output.
    tiles are square; not used if DO_TILE = FALSE.
    
    | *Type:* Double. Valid range: ]0,...
    | ``TILE_SIZE = 30000``
    
  * This is the block size (in target units, commonly in meters) of the image chips.
    Blocks are stripes, i.e.
    they are as wide as the tile, and as high as specified here; not used if DO_TILE = FALSE or OUTPUT_FORMAT = ENVI.
    The blocks are the primary processing unit of the force-higher-level routines.
    
    | *Type:* Double. Valid range: ]0,TILE_SIZE]
    | ``BLOCK_SIZE = 3000``
    
  * This is the spatial resolution of Landsat output; not used if DO_REPROJ = FALSE.
    Note that the tile and block sizes must be a multiple of the pixel resolution.
    
    | *Type:* Double. Valid range: ]0,...
    | ``RESOLUTION_LANDSAT = 30``
    
  * This is the spatial resolution of Sentinel-2 output; not used if DO_REPROJ = FALSE.
    Note that the tile and block sizes must be a multiple of the pixel resolution.
    
    | *Type:* Double. Valid range: ]0,...
    | ``RESOLUTION_SENTINEL2 = 10``
    
  * These are the origin coordinates of the grid system in decimal degree (negative values for West/South).
    The upper left corner of tile X0000_Y0000 represents this point.
    It is a good choice to use a coordinate that is North-West of your study area – to avoid negative tile numbers.
    Not used if DO_TILE = FALSE.
    
    | *Type:* Double. Valid range: [-90,90]
    | *Type:* Double. Valid range: [-180,180]
    | ``ORIGIN_LON = -25``
    | ``ORIGIN_LAT = 60``
    
  * This defines the target coordinate system.
    If DO_REPROJ = FALSE, the projection string can be NULL.
    The coordinate system must either be given as WKT string - or can be a predefined coordinate/grid system.
    If one of the predefined systems are used, TILE_SIZE, BLOCK_SIZE, ORIGIN_LAT, and ORIGIN_LON are ignored and internally replaced with predefined values.
    Currently, EQUI7 and GLANCE7 are availble.
    Both are globally defined sets of projections with a corresponding grid system.
    EQUI7 consists of 7 Equi-Distant, continental projections, with a tile size of 100km.
    GLANCE7 consists of 7 Equal-Area, continental projections, with a tile size of 150km.
    One datacube will be generated for each continent.
    
    | *Type:* Character. Valid values: {<WKT>,EQUI7,GLANCE7}
    | ``PROJECTION = GLANCE7``
    
  * This is the resampling option for the reprojection; you can choose between Nearest Neighbor (NN), Bilinear (BL) and Cubic Convolution (CC); not used if DO_REPROJ = FALSE.
    
    | *Type:* Character. Valid values: {NN,BL,CC}
    | ``RESAMPLING = CC``
    
* **Radiometric correction options**
    
  * This indicates if topographic correction should be performed.
    If TRUE, a DEM need to be given.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``DO_ATMO = TRUE``
    
  * This indicates if atmospheric correction should be performed.
    If TRUE, Bottom-of-Atmosphere reflectance is computed.
    If FALSE, only Top-of-Atmosphere reflectance is computed.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``DO_TOPO = TRUE``
    
  * This indicates if BRDF correction should be performed.
    If TRUE, output is nadir BRDF adjusted reflectance instead of BOA reflectance (the output is named BOA nonetheless).
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``DO_BRDF = TRUE``
    
  * This indicates if adjacency effect correction should be performed.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``ADJACENCY_EFFECT = TRUE``
    
  * This indicates if multiple scattering (TRUE) or the single scattering approximation (FALSE) should be used in the radiative transfer calculations.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``MULTI_SCATTERING = TRUE``
    
* **Water vapor correction options**
    
  * This is the directory where the water vapor tables are located.
    Water vapor tables are not required for Sentinel-2, in this case DIR_WVPLUT may be NULL.
    For Landsat, it is recommended to use this functionality.
    As a minimum requirement, DIR_WVPLUT may be NULL and a global value for WATER_VAPOR needs to be specified.
    If a directory is given, WATER_VAPOR is ignored.
    DIR_WVPLUT must contain water vapor tables.
    The 12 climatology tables must exist at least.
    They are used if the daily tables do not exist or if there is no valid daily value.
    
    | *Type:* full directory path
    | ``DIR_WVPLUT = NULL``
    
  * This specifies a global value for atmospheric water vapor content in g cm-2.
    This parameter can be a dummy value to quickly process an image without needing to generate a water vapor database.
    Note that especially Landsat-8 is relatively insensitive to atmospheric water vapor (depending on wavelength), and external water vapor is not needed to process Sentinel-2.
    The error in using a dummy value is significant for the TM sensors.
    
    | *Type:* Float. Valid range: [0,15]
    | ``WATER_VAPOR = NULL``
    
* **Aerosol optical depth options**
    
  * This indicates whether the internal AOD estimation (TRUE) or externally generated AOD values should be used (FALSE).
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``DO_AOD  = TRUE``
    
  * This is the directory where the aerosol optical depth look-up-tables are located.
    They can be used to input external AOD values.
    It is recommended to use the internal algorithm only.
    If a path is given, and DO_ATMO = TRUE, internal AOD estimation is used and external AOD values are used as fallback option.
    
    | *Type:* full directory path
    | ``DIR_AOD  = NULL``
    
* **Cloud detection options**
    
  * This parameter cancels the processing of images that exceed the given threshold.
    The processing will be canceled after cloud detection.
    
    | *Type:* Integer. Valid range: ]0,100]
    | ``MAX_CLOUD_COVER_FRAME = 75``
    
  * This parameter works on a tile basis.
    It suppresses the output for chips (tiled image) that exceed the given threshold.
    
    | *Type:* Integer. Valid range: ]0,100]
    | ``MAX_CLOUD_COVER_TILE  = 75``
    
  * These are the main thresholds of the Fmask algorithm.
    
    | *Type:* Float. Valid range: [0,1]
    | ``CLOUD_THRESHOLD  = 0.225``
    | ``SHADOW_THRESHOLD = 0.02``
    
* **Resolution merging**
    
  * This parameter defines the method used for improving the spatial resolution of Sentinel-2’s 20 m bands to 10 m.
    Pixels flagged as cloud or shadow will be skipped.
    Following methods are available: IMPROPHE uses the ImproPhe code in a spectral-only setup; REGRESSION uses a multiparameter regression (results are expected to be best, but processing time is significant); STARFM uses a spectral-only setup of the Spatial and Temporal Adaptive Reflectance Fusion Model (prediction artifacts may occur between land cover boundaries); NONE disables resolution merge; in this case, 20m bands are quadrupled.
    
    | *Type:* Character. Valid values: {IMPROPHE,REGRESSION,STARFM,NONE}
    | ``RES_MERGE = IMPROPHE``
    
* **Co-Registration options**
    
  * This parameter only applies for Sentinel-2 data.
    This parameter defines the path to a directory that contains monthly Landsat NIR base images.
    If given, a co-registration is attempted.
    If it fails (no tie points), the image won't be processed.
    
    | *Type:* full directory path
    | ``DIR_COREG_BASE = NULL``
    
  * This parameter defines the nodata values of the coregistration base images.
    
    | *Type:* Integer. Valid values: [-32768,32767]
    | ``COREG_BASE_NODATA = -9999``
    
* **Miscellaneous options**
    
  * This parameter defines if impulse noise should be removed.
    Ony applies to 8bit input data.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``IMPULSE_NOISE = TRUE``
    
  * This parameter defines if nodata pixels should be buffered by 1 pixel.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``BUFFER_NODATA = FALSE``
    
* **TIER LEVEL**
    
  * This parameter specifies the acceptable tier level of Landsat Level 1 data.
    For pre-collection data, TIER = 1 will only accept L1T images, TIER = 2 will also accept L1Gt and L1G images.
    For collection data, TIER = 1 will only accept L1TP images, TIER = 2 will also accept T2 images, TIER = 3 will additionaly accept RT images.
    
    | *Type:* Integer. Valid range: [1,3]
    | ``TIER = 1``
    
* **Parallel processing**
    
  * Multiprocessing options (NPROC, DELAY) only apply when using the batch utility force-level2.
    They are not used by the core function force-l2ps.
    
  * This module is using hybrid parallelization, i.e.
    a mix of multiprocessing and multithreading.
    Each input image is one process, each process may use multiple threads.
    In general, it is most efficient to use as much multiprocessing as possible (i.e.
    NTHREAD = 1 or 2).
    However, if you only have a small number of images - or if your system does not have enough RAM, it is adviced to use less processes and more threads per process.
    The number of processes and threads is given by following parameters.
    
    | *Type:* Integer. Valid range: [1,...
    | ``NPROC = 32``
    | ``NTHREAD = 2``
    
  * This parameter controls whether the individual bands of the Level 1 input images are read sequentially or in parallel.
    Note that we observed two kinds of GDAL installation: (1) the JPEG driver reads each band parallely, but separated images in sequence - we recommend to disable PARALLEL_READS in this case (for Sentinel-2).
    (2) The GDAL JPEG drived does not do anything in parallel - use PARALLEL_READ to speed up the work (also use it for Landsat).
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``PARALLEL_READS = FALSE``
    
  * This parameter sets a delay before starting a new process.
    This can be helpful to prevent I/O jams when using a lot of processes.
    The delay is given in seconds.
    
    | *Type:* Integer. Valid range: [0,...
    | ``DELAY = 3``
    
  * This parameter sets a timeout for unzipping the Level 1 data (only applies when images are still in zip/tar.gz format.
    Only applies for force-level2).
    The timeout is given in seconds.
    
    | *Type:* Integer. Valid range: [0,...
    | ``TIMEOUT_ZIP = 30``
    
* **Output options**
    
  * Output format, which is either uncompressed flat binary image format aka ENVI Standard or GeoTiff.
    GeoTiff images are compressed with LZW and horizontal differencing; BigTiff support is enabled; the Tiff is structured with striped blocks according to the TILE_SIZE (X) and BLOCK_SIZE (Y) specifications.
    Metadata are written to the ENVI header or directly into the Tiff to the FORCE domain.
    If the size of the metadata exceeds the Tiff's limit, an external .aux.xml file is additionally generated.
    
    | *Type:* Character. Valid values: {ENVI,GTiff}
    | ``OUTPUT_FORMAT = GTiff``

  * Output the cloud/cloud shadow/snow distance output? Note that this is NOT the cloud mask (which is sitting in the mandatory QAI product).
    This product can be used in force-level3; no other higher-level FORCE module is using this.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_DST = FALSE``
    
  * Output Aerosol Optical Depth map for the green band? No higher-level FORCE module is using this.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_AOD = FALSE``
    
  * Output the Water Wapor map? No higher-level FORCE module is using this.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_WVP = FALSE``
    
  * Output the view zenith map? This product can be used in force-level3; no other higher-level FORCE module is using this.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_VZN = FALSE``
    
  * Output the  Haze Optimzed Transformation output? This product can be used in force-level3; no other higher-level FORCE module is using this.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_HOT = FALSE``
    
  * Output overview thumbnails? These are jpegs at reduced spatial resolution, which feature an RGB overview + quality information overlayed (pink: cloud, red: cirrus, cyan: cloud shadow, yellow: snow, orange: saturated, green: subzero reflectance).
    No higher-level FORCE module is using this.
    
    | *Type:* Logical. Valid values: {TRUE,FALSE}
    | ``OUTPUT_OVV = TRUE``
    
