.. _wvdb-format:

WVDB format
===========

The Water Vapor Database consists of two sets of tables, which both must be located in the directory given in ``DIR_WVPLUT`` in the Level 2 :ref:`l2-param`.

Daily tables are optional, but highly recommended.
They contain a water vapor value for each coordinate, and there is one table for each day.
Climatology tables are mandatory (unless ``DIR_WVPLUT = NULL``).
:ref:`l2ps` uses the climatology tables if a daily table is unavailable or if there is a fill value in the daily table.
Therefore, the minimum requirement is to prepare the 12 climatology tables, one for each month.


Daily tables
^^^^^^^^^^^^

For each date, one file can be prepared.
The file naming is ``WVP_YYYY-MM-DD.txt``, e.g. WVP_2003-08-24.txt.
The files are four column tables with no header, separated by white-space.
One line per coordinate; ended with an empty line.
The coordinate closest to the scene center will be selected, and the corresponding water vapor value will be retrieved.
The fill value is 9999 and TBD for source.

Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by water vapor (3rd column), and three-digit source (4th column).

Example:

.. code-block:: bash

  17.2642002 -14.4588003 2.448023 MYD
  16.9421997 -15.9028997 2.189836 MYD
  20.6735001 -13.0142002 9999.000 TBD
  20.3544006 -14.4588003 2.427723 MOD
  20.0323009 -15.9028997 2.499933 MOD

  
Climatology tables
^^^^^^^^^^^^^^^^^^

12 climatology tables must be prepared, one per month.
The file naming is ``WVP_0000-MM-00.txt``, e.g. WVP_0000-06-00.txt.
The files are five column tables with no header, separated by white-space.
One line per coordinate; ended with an empty line.
The coordinate closest to the scene center will be selected, and the corresponding water vapor value will be retrieved.

Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by long-term water vapor average (3rd column), long-term standard deviation of water vapor (4th column) and number of valid observations used for averaging (5th column).

The generation of climatology tables is mandatory (unless ``DIR_WVPLUT = NULL``).

Example:

.. code-block:: bash

  96.4300 34.6138 1.205356 0.398807 446
  96.0306 33.1801 1.360043 0.399460 447
  95.6409 31.7452 1.442830 0.350363 425
  95.2598 30.3093 1.642989 0.276430 311
  94.8869 28.8723 4.018294 0.812506 149
  94.5214 27.4344 6.426344 0.724956 123

