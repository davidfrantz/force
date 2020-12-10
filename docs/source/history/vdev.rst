.. _vdev:

Develop version
===============

FORCE-dev
---------

* **General changes**

  * :ref:`depend` have changed.
    Instead of python, python3 is now specifically required.
    Also, added python3-dev as dependency (thanks to Willi Weber for the hint).

  * On some systems, FORCE failed to compile due to multiple definitions of _WVLUT_.
    Thanks to Ansgar Lößer for suggesting a fix for this.

  * Fixed the inconsistent use of the synonyms BLOCK_SIZE and CHUNK_SIZE. 
    It is BLOCK_SIZE from now on.
    Thanks to Jonas Ardo for the hint.

* **FORCE LEVEL 1 ARCHIVING SYSTEM**

  * Stefan Ernst fixed some issues in force-csd:

    * Reworked how AOIs in the form of vector files are handled.
      Geopackages are no longer used as database for intersecting user-defined AOIs with the tiles/footprints of Sentinel-2 and Landsat, 
      as the GDAL GPKG driver and SQLite caused issues with certains setups / installations. 
      The geopackage format is still supported as AOI input though.

    * Several small fixes
      Fixes target the reprojection of AOIs, handling of negative values for AOI bounding boxes, command line option parsing, etc.
      Only the Landsat product with the latest processing datestamp is downloaded if there are several products for one WRS-2 footprint on the same day.

* **FORCE L2PS**

  * Changed the BRDF correction strategy.
    Before, the reflectance was fixed to a sun zenith of 45° as suggested by Flood et al. 2013.
    Zhang et al. 2016 demonstrated that this strategy results in global mean absolute differences of the sun zenith greater than the maximum Landsat viewing zenith angle (7.5°).
    Now, the sun zenith is fixed to the latitude-varying local time, which only results in differences of 0.26°.
    We are using the mean local time of Landsat 8 and Sentinel-2 overpass.
    Note that the same change was implemented in the HLS dataset.
    Thanks for David Roy and Hankui Zhang for discussion and guidance on this topic.

  * Changed the behaviour of the cloud shadow flag.
    Before, the cloud shadow flag was not set if the cloud flag (any state) was set.
    Thus, when users wanted to use confident clouds only, i.e. not the buffered clouds, 
    there was a gap between the cloud and the shadow mask.
    This is mitigated now: cloud and cloud shadow flags can be set simultaneously. 
    This also enables "cloud shadow on top of cloud" scenarios.
    Thanks to Haili Hu for reporting this issue.

  * Changed the behaviour of over-saturated surface reflectance over cold cloud tops.
    Before, if surface reflectance > 200%, the pixel was set to nodata.
    Thus, holes in the clouds appeared.
    Now, the pixel is only flagged as saturated, and reflectance is capped at the maximum Int16 value.
    Note that this happens because the assumptions for estimating **surface** reflectance are not valid over clouds.
    Thanks to Philip Frost for reporting this issue.

  * Added a new parameter ``ERASE_CLOUDS``. 
    If this parameter is enabled, confident cloud detections will be erased in the reflectance product, 
    i.e. pixels are set to nodata. 
    The cloud flag in the QAI product will still mark these pixels as clouds. 
    This option may be usefule when disk space is of concern: 
    removing the clouds(when used with compressed output options) will result in smaller file sizes.

  * Some users noted that the cloud buffer is too large for their application.
    This release hands control of the buffer widths to the user.
    New parameters were added to the parameterfile: ``CLOUD_BUFFER``, ``SHADOW_BUFFER``, ``SNOW_BUFFER``.
    The size corresponds to the buffer radius in meters.

* **FORCE Import**

  * new program added: force-import-modis
    A new tool was added, which imports the MODIS Surface Reflectance Daily product MOD09GA/MYD09GA into FORCE.
    The tool generates a FORCE-compatible datacube in Sinusoidal projection; MODIS tile h18v03 is ingested into FORCE tile X0018_Y0003.
    The tool converts the MODIS hdf file into a pair of BOA/QAI images in compressed GeoTiff format according to FORCE data structure and naming convention.
    The BOA product holds the 7 surface reflectance bands (ordered by wavelength).
    The QAI product holds the Reflectance Data State QA, wherein the MODIS quality flags are translated into the usual FORCE quality flags.
    File naming is like this: 20150101_LEVEL2_MOD01_BOA.tif (MOD01 = MODIS Terra, MOD02 = MODIS Aqua).

* **FORCE HIGHER LEVEL**

  * force-higher-level is aware of the newly added MODIS sensors, i.e. MOD01 and MOD02 can be specified in the sensor list (``SENSORS``).
    A new spectral band is available as ``INDEX = SWIR0``, which represents MODIS band 5 (1230 - 1250).

  * in force-higher-level, TSA sub-module: 
    There is a new functionality, which enables users to plug-in their own python code in a very easy and user-friendly way.
    Thus, FORCE can now be complemented by custom user functionality without changing or recompling the C code.
    Two new parameters are now needed in the TSA parameterfile: 

    1) ``FILE_PYTHON`` points to a python file
    2) ``OUTPUT_PYP = TRUE/FALSE`` defines whether to use the script and output the corresponding data

    An example python script can be found in ``force/example/tsi-plugin.py``.
    Do not modify the function names and function arguments.

    A tutorial is planned to showcase the usage.

  * in force-higher-level, TSA sub-module, CAT analysis: 
    Fixed an arithmetic expression bug that occured when computing loss when the regression intercept was 0.

* **FORCE AUX**

  * force-pyramid uses multiprocessing to speed up computation (when multiple input images are given).

Master release: TBA

-- No changes yet, master is in sync with develop.
