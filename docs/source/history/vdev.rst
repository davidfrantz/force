.. _vdev:

Develop version
===============

- **General changes**

  - A lot of work has been done on improving the Docker builds and images.
    Many thanks to Peter Jonsson for his help with this! Most things won't be visible to the user, but should improve stability and security.
  - Due to some difficulties in installing the sf R-package, an older version of GDAL has been unknowingly resurfaced in the Docker container.
    Thanks to Benjamin Jakimow for reporting this issue.
    This has now been temporarily fixed by NOT installing the sf package in the Docker container.
    A more permanent fix will be implemented in the future.
    For now, note that ``force-map-accuracy`` will not work in the Docker container for the time being, 
    i.e., for dev versions after ``3.10.04-dev:::2026-01-16_16:23:49``.
    Please use an older Docker image (e.g., the main release ``3.10.04``) to use this program.
    This only affects the Docker container, not selfmade FORCE installations.
    This only affects the ``force-map-accuracy`` program, not any other program.

**FORCE L1AS**

  - All download capabilities were removed from FORCE. 
    Specifically, ``force-level1-landsat``, ``force-level1-csd`` and ``force-lut-modis`` were removed.
    Please see https://github.com/davidfrantz/force/issues/418 for more details.

**FORCE L2PS**

  - All download capabilities were removed from FORCE. 
    Specifically, ``force-level1-landsat``, ``force-level1-csd`` and ``force-lut-modis`` were removed.
    Please see https://github.com/davidfrantz/force/issues/418 for more details.
  - More resampling methods were added, including Cubic Spline, Lanczos, Average, Mode, Max, Min, Median, Q1, Q3, Sum, and RMS.
    Feature requested by Vu-Dong Pham.
  - ``force-level2-report`` has a new option ``-t`` to choose the theme of the generated charts.
    Available themes are inherited from ``echarts4r`` - look in their documentation for more details. 

- **FORCE HLPS**

  - in ``force-higher-level``, LEVEL3 module:
    It is now possible to use custom cutoff values for seasonal. cloud, haze, and view zenith scores when 
    selecting/weighting observations for compositing.
    Note that parameters ``USE_CLOUDY``, ``USE_HAZY``, ``USE_OFF_SEASON``, and ``USE_HIGH_VZEN`` have been refactored, 
    and are now replaced by ``CLOUDY_CUTOFF``, ``HAZY_CUTOFF``, ``OFF_SEASON_CUTOFF``, and ``VZEN_CUTOFF`` respectively.  
    These parameters now take float values instead of logical values.
    This allows users to set a threshold (between 0 and 1) below which observations are considered as cloudy, hazy, off-season, or with high view zenith angle, respectively.
    These are flat-out ignored for the compositing process.
    Setting these parameters to 0.0 will allow all observations to be considered.
    Thanks to Oleg Zheleznyy for proposing and contributing to this feature.
  - in ``force-higher-level``, there are up to three additional paramaters, which allow fine-tuning 
    the behaviour in case of reading errors in the input data. This is usually only relevant when
    the input data cube is not properly maintained, e.g., when some files are missing or corrupted.
    These parameters are ``READ_ERROR_MASK``, ``READ_ERROR_PRIMARY`` and ``READ_ERROR_SECONDARY``.
    Usable values (depending on context) are ``STOP``, ``SKIP``, and ``YOLO``. ``STOP`` will stop the program with an error message, ``SKIP`` will skip the problematic block of data and continue with the rest. ``YOLO`` will ignore any errors and tries to proceed. Obviously, use this with caution. All 
    option will print warning/error messages, so make sure to check your logs/stdout. ``YOLO`` is only
    available for ARD-type input data. Note that those parameters only act on reading errors, not on
    empty tiles. Feature requested by Vu-Dong Pham.

- **FORCE AUX**

  - force-tile-finder now allows to input either geographic (lon/lat) or projected (x/y) coordinates.
    The coordinate type can be specified via the new -t option (values: "geo" or "proj").
    The program still outputs both coordinate types for the given point, as well as the corresponding tile indices and pixel indices within that tile.
  - new program: ``force-runtime-data``, which provides information about the FORCE runtime data installation.
    This includes the current location of this data, 
    as well as available sensors and indices. 
    More features will be added in the future once the runtime data expands.

    .. code-block:: console 
    
      $ force-runtime-data -p # show the installation path of the runtime data
      $ force-runtime-data -s # show available sensors and their band names
      $ force-runtime-data -x # show available indices and their required bands

  - force-map-accuracy is now re-written in C. This avoids to rely on the sf package
    which caused some trouble in the Docker image as it caused a GDAL version conflict.
    The output format has been changed to markdown for nicer rendering.
    Apart from this, there is not really a change in usage. 
    Not that it really matters, but it is also faster now.


.. no changes to last release yet
