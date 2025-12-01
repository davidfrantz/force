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

.. no changes to last release yet
