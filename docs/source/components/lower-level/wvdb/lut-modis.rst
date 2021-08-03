.. _lut-modis:

force-lut-modis
===============

The water vapor database may be generated/updated with force-lut-modis, which automatically downloads MODIS water vapor products (MOD05 and MYD05, collection 6.1) from the Level1 and Atmosphere Archive and Distribution System (LAADS) at NASA’s Goddard Space Flight Center.

.. note::

  Any permanent or temporary change/shutdown/decommissioning on LAADS’ or MODIS’ end may result in the nonfunctioning of force-lut-modis (as it happened between the release of FORCE v. 1.1 and v. 2.0).

.. note::

  The tables can also be generated with other data and/or software.
  If you want to create the tables on your own, refer to the :ref:`wvdb-format`.
  
.. seealso::
  
  An application-ready global water vapor database (daily values for 2000–2020 + monthly climatology) can be downloaded from `Zenodo <https://doi.org/10.5281/zenodo.4468700>`_.

.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-wvdb/wvdb/>`_, which explains how to prepare the Water Vapor Database (WVDB).
  
  
Usage
^^^^^

.. code-block:: bash

  force-lut-modis

  Usage: force-lut-modis [-h] [-v] [-i] [-d] [-t] [-c] coords-file wvp-dir geometa-dir download-dir

optional arguments
""""""""""""""""""

* ``-h`` show help page

* ``-v`` show version of FORCE

* ``-i`` show program's purpose

* ``-d`` date range as YYYYMMDD,YYYYMMDD. The default is *20000224,today*

* ``-t`` build daily tables? The default is *true*

* ``-c`` build climatology? The default is *true*

mandatory arguments
"""""""""""""""""""

* coords-file

  | A text file with coordinates, i.e. central coordinates of the WRS-2 frames.
  | Water vapor averages are estimated for each given coordinate.
  | The file should contain two columns separated by white space, and no header.
  | The first column should give the longitude, the second column the latitude with coordinates in decimal degree (negative values for West/South).

* wvp-dir

  | The final water vapor tables are saved in wvp-dir (this directory must exist).
  | Tables that are already in wvp-dir are not processed again (i.e. no download of geometa tables and hdf files; the user may delete the MODIS \*.hdf files after the water vapor tables are successfully generated).
  | This directory is the directory, which should be given in ``DIR_WVPLUT`` in the Level 2 :ref:`l2-param`.

* geometa-dir

  | MOD05/MYD05 data are swath products, and MOD03/MYD03 geometa tables are necessary to relate coordinates to MODIS granules.
  | The geometa tables are downloaded to geometa-dir (this directory must exist).
  | Tables that are already in geometa-dir are not downloaded again.
  | If the tool crashes because a table is invalid, it is necessary to manually delete this file and run the tool again.
  | Unfortunately, this happens from time to time due to incomplete downloads or if LAADS is unresponsive.
  | The program attempts to re-download a corrupt file up to 10 times, but this error can occur nonetheless.
  | Note that the geometa tables are a global product, and as such, the same geometa-dir can be used for different study areas.

* download-dir

  | The MODIS data are downloaded to download-dir (this directory must exist).
  | MODIS data that are already in download-dir are not downloaded again.
  | If the tool crashes because a dataset is corrupt, it is necessary to manually delete this file and run the tool again.
  | Unfortunately, this happens from time to time due to incomplete downloads or if LAADS is unresponsive.
  | The program attempts to re-download a corrupt file up to 10 times, but this error can occur nonetheless.
