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
  - In ``force-level2`` / ``force-l2ps``, the setting of the grid origin has been refactored. 
    It is now necessary to specify the grid origin as a vector of two values (e.g., "``GRID_ORIGIN = -25 60``"), 
    instead of separate parameters for longitude and latitude.
    Additionally, it is now possible to specify the type of coordinates used for the grid origin (geographic or projected) 
    via the new parameter ``GRID_ORIGIN_TYPE``, which can take the values "GEO" or "MAP".
    This allows for more flexibility in defining the grid origin, especially when "clean" values in the map projection are desired.
    The first value in the ``GRID_ORIGIN`` vector is always interpreted as the x-coordinate (longitude or easting), and the second value as the y-coordinate (latitude or northing), regardless of the coordinate type.
    Thanks to Sebastian Hafner for highligting this 

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
  - in ``force-higher-level``, TSA and CSO modules:
    The new parameter ``ADAPTIVE_RANGE`` allows to use an adaptive date range for filtering images on a per-pixel basis. 
    It comes with a set of other new parameters to control the behaviour of this adaptive date range. 
    Technically, it works like the continuous fields already implemented in FORCE, e.g. in the CFIMP module or the phenology-adaptive compositing. 
    You require two input images (cubed, Int16), which hold start and end dates for each pixel. 
    Multiple filtering windows may be given (as multiband images). 
    The dates need to be given as "Year x 365 + DOY", i.e., the same format as used internally in FORCE, with an optional offset (see parameter ``ADAPTIVE_START``).
    Note that this option is just a filtering routine. 
    If you provide multiple windows, you will not receive different products for each window! 
    Also note that this option might badly interact with other functionality, e.g., deriving phenology metrics.
    Please also note, that this filtering option (contrary to filters like DATE_RANGE) does not reduce input as we need all the data in RAM to determine the adaptive date range for each pixel.
    Feature requested by Oleg Zheleznyy.


- **FORCE AUX**

  - force-tile-finder now allows to input either geographic (lon/lat) or projected map (x/y) coordinates.
    The coordinate type can be specified via the new -t option (values: "geo" or "map").
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
