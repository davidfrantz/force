.. _level2-depend:

Data dependencies
=================


File queue
^^^^^^^^^^

A :ref:`queue` is required for :ref:`force-bulk`. 
The queue us not needed for :ref:`force-core` and :ref:`force-wrapper`.

The queue is automatically generated when using the :ref:`l1as`.


Digital Elevation Model
^^^^^^^^^^^^^^^^^^^^^^^

A digital elevation model (DEM) is highly recommended for :ref:`l2ps`.
It is used for cloud detection, topographic and atmospheric correction.
The user should provide a DEM that covers the complete(!) study area at adequate spatial resolution.
The DEM must be specified in the :ref:`l2-param` as ``FILE_DEM``.
There is not really a restriction on the source, projection or resolution; the file format must be supported by GDAL.
The DEM will be warped to the extent and resolution of the processed image using bilinear resampling.
The DEM needs to be provided in meters above sea level, and the nodata value needs to be given in the :ref:`l2-param` with ``FILE_DEM_NODATA``.

.. note::

  The user can choose to process without a DEM; in this case the surface is assumed flat @ z = 0m.
  Topographic correction cannot be used without a DEM.
  The quality of atmospheric correction and cloud /cloud shadow detection will suffer without DEM.

.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-dem/dem/>`_, which explains how to prepare a Digital Elevation Model (DEM).


Water Vapor Database
^^^^^^^^^^^^^^^^^^^^

A water vapor database is necessary for the atmospheric correction of Landsat data.
It is not needed for Sentinel-2.
The directory containing the database needs to be given in the :ref:`l2-param` with ``DIR_WVPLUT``.
For more information, see the :ref:`wvdb` module.

.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-wvdb/wvdb/>`_, which explains how to prepare the Water Vapor Database (WVDB).


Aerosol Optical Depth look-up-tables

These Look-up-Tables are optional, and may be used to override the internal AOD estimation or to provide backup values if the internal AOD estimation failed for any reason, as e.g. used in this `paper <http://doi.org/10.1080/2150704x.2015.1070314>`_
Potential usages are to employ an AOD climatology or a fixed AOD.
The directory containing the LUTs needs to be given in the :ref:`l2-param` with ``DIR_AOD``.

.. note::

  Generally, I advice to not use AOD LUTs.


For each DOY, one file needs to be prepared (you should prepare 366 files).
The file naming is ``AOD_DOY.txt``, e.g. AOD_076.txt.
The files are five column tables with no header, separated by white-space.
One line per coordinate; ended with an empty line.
The coordinate closest to the scene center will be selected, and the corresponding AOD value will be retrieved.
Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by three Ångström coefficients (3rd – 5th column; logarithmic formulation, see below).
The first coefficient is the turbidity coefficient (a0), the second coefficient is the Ångström exponent (a1), and the third coefficient can be used to describe spectral curvature in AOD (a2).
The spectral curvature can be disabled with a2 = 0; in this case the formulation simplifies to the classic Ångström equation.
AOD for any given wavelength is retrieved using following equation: 

ln⁡〖τ_a 〗=a_0+a_1∙ln⁡λ+a_2∙(ln⁡λ )^2


Master images for co-registration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Tile allow-list
^^^^^^^^^^^^^^^

A :ref:`tilelist` is optional, and may be used to suppress the output of tiles that are not in your study area.
This option was implemented to decrease the volume of the processed data.
The file is specified in the :ref:`l2-param` with ``FILE_TILE``.

