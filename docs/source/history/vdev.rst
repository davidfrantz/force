.. _vdev:

Develop version
===============

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

- **FORCE AUX**

  - force-tile-finder now allows to input either geographic (lon/lat) or projected (x/y) coordinates.
    The coordinate type can be specified via the new -t option (values: "geo" or "proj").
    The program still outputs both coordinate types for the given point, as well as the corresponding tile indices and pixel indices within that tile.
  - new program: force-runtime-data
    This program provides information about the FORCE runtime data installation, including the current location of this data, 
    as well as available sensors and indices. More features will be added in the future once the runtime data expands.
    
    - Example to show the installation path of the runtime data:
      .. code-block:: console 
      
        $ force-runtime-data -p

    - Example to show all available sensors and their band names:
      .. code-block:: console

        $ force-runtime-data -s

    - Example to show all available indices and their required bands:
      .. code-block:: console

        $ force-runtime-data -x

.. no changes to last release yet
