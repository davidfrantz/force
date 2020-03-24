.. _tsa-endmember:

Endmember file
==============

This file is needed for Spectral Mixture Analysis in the :ref:`tsa` submodule if ``INDEX`` includes ``SMA``.
The file is specified with the ``FILE_ENDMEM`` parameter.

The file defines the endmember spectra that should be used for the SMA.
The values need to be given in scaled reflectance (scale factor 10,000), i.e. the need to match the Level 2 ARD (see :ref:`level2-format`).

The files should be without header, ended with an empty line, columns separated by white-space.
There should be one column for each endmember.

If you want to apply shade normalization (``SMA_SHD_NORM = TRUE``), the shade spectrum (photogrammetric zero or measured shade) needs to be in the last column.

Note that you do not need to add a row of ones to apply the sum-to-one constraint.
This is handled internally, if ``SMA_SUM_TO_ONE = TRUE``.

There should be as many lines as there are overlapping bands for the chosen set of sensors (see :ref:`tsa-sensor`).

Examples:

* Generating a fraction time series based on ``SENSOR = LND04 LND05 LND07 LND08`` requires 6-band endmembers (Landsat legacy bands).

* Generating a fraction time series based on ``SENSOR = LND08 SEN2A SEN2B`` requires 6-band endmembers (Landsat legacy bands).

* Generating a fraction time series based on ``SENSOR = SEN2A SEN2B`` requires 10-band endmembers (Sentinel-2 land surface bands).

* Generating a fraction time series based on ``SENSOR = sen2a sen2b`` requires 4-band endmembers (Sentinel-2 high-res bands).


Example file (Landsat legacy bands using vegetation, soil, rock and shade endmembers):

.. code-block:: bash

  320 730 2620 0
  560 1450 3100 0
  450 2240 3340 0
  3670 2750 4700 0
  1700 4020 7240 0
  710 3220 5490 0

