.. _tut-wvdb:

Water Vapor Database
====================

**How to prepare the Water Vapor Database for Level 2 Processing**

This tutorial shows how to prepare the Water Vapor Database (WVDB) for the FORCE Level 2 Processing System (FORCE L2PS).

.. admonition:: Info

   *This tutorial uses FORCE v. 3.0*


Background
----------

During atmospheric correction, the effect of water vapor absorption can only be corrected if we know the amount of water vapor in the atmosphere.


If you are using Sentinel-2 data only, you can stop reading.
Sentinel-2 is equipped with a water vapor channel, and thus, water wapor amount can be estimated from the images.

Landsat, however, doesn't have such a band.
Therefore, we need to rely on external data, which needs to be precompiled into a water vapor database.

Water Vapor Database
--------------------

The database holds water vapor values for the central coordinates of each WRS-2 frame.
If available, day-specific values are used.


The database consists of one table for each day (``WVP_YYYY-MM-DD.txt``) 


.. code-block:: bash

   ls /data/Earth/global/wvp/wvdb/WVP_2010-07-*


    /data/Earth/global/wvp/wvdb/WVP_2010-07-01.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-02.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-03.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-04.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-05.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-06.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-07.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-08.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-09.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-10.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-11.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-12.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-13.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-14.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-15.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-16.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-17.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-18.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-19.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-20.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-21.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-22.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-23.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-24.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-25.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-26.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-27.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-28.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-29.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-30.txt
    /data/Earth/global/wvp/wvdb/WVP_2010-07-31.txt


Each file includes one value per coordinate.
In the example below, there are 13281 coordinates in each file (global land coverage).
The coordinate, which is closest to the center of the Landsat image is selected, and the atmospheric correction uses this value to account for gaseous absorption.


.. code-block:: bash

   wc -l /data/Earth/global/wvp/wvdb/WVP_2010-07-26.txt 


    13281 /data/Earth/global/wvp/wvdb/WVP_2010-07-26.txt



.. code-block:: bash

   head /data/Earth/global/wvp/wvdb/WVP_2010-07-26.txt


    -15.3934 80.7603 1.170018 MOD
    -22.8654 80.0056 9999.000000 TBD
    -29.2236 79.1137 9999.000000 TBD
    -34.5930 78.1151 0.614454 MOD
    -39.1269 77.0343 0.448552 MOD
    -42.9718 75.8898 0.260607 MOD
    -46.2552 74.6958 0.282855 MYD
    -49.0816 73.4629 0.337015 MOD
    -51.5357 72.1989 9999.000000 TBD
    -53.6847 70.9100 9999.000000 TBD


Climatology
-----------

If day-specific values are not available (no table is existing, or there is a fill value), a monthly long-term climatology is used instead.
The climatology consists of one table for each month (``WVP_0000-MM-00.txt``).


.. code-block:: bash

   ls /data/Earth/global/wvp/wvdb/WVP_0000*


    /data/Earth/global/wvp/wvdb/WVP_0000-01-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-02-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-03-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-04-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-05-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-06-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-07-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-08-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-09-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-10-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-11-00.txt
    /data/Earth/global/wvp/wvdb/WVP_0000-12-00.txt


Again, each file includes one value per coordinate.
The file holds the long-term average, long-term standard deviation, and the number of measurements used to compute these statistics.


.. code-block:: bash

   wc -l /data/Earth/global/wvp/wvdb/WVP_0000-07-00.txt 


    13281 /data/Earth/global/wvp/wvdb/WVP_0000-07-00.txt



.. code-block:: bash

   head /data/Earth/global/wvp/wvdb/WVP_0000-07-00.txt


    -15.3934 80.7603 1.177704 0.364894 300
    -22.8654 80.0056 1.079682 0.328948 311
    -29.2236 79.1137 0.695211 0.234917 383
    -34.5930 78.1151 0.549352 0.256754 445
    -39.1269 77.0343 0.472883 0.224957 480
    -42.9718 75.8898 0.410826 0.211346 476
    -46.2552 74.6958 0.384219 0.145523 457
    -49.0816 73.4629 0.415261 0.170940 456
    -51.5357 72.1989 0.515858 0.223122 422
    -53.6847 70.9100 0.546611 0.273735 276

.. figure:: img/wvdb.gif" width="750

   *Global animation of the climatology (monthly average)*

Uncertainty of the climatology
""""""""""""""""""""""""""""""

The uncertainty of using the climatology was assessed in this paper:
Frantz, D., Stellmes, M., & Hostert, P.
(2019).
A Global MODIS Water Vapor Database for the Operational Atmospheric Correction of Historic and Recent Landsat Imagery.
Remote Sensing, 11, 257.
https://doi.org/10.3390/rs11030257

Prepare the WVDB
----------------

We generally use a WVDB generated from MODIS water vapor products ([MOD05 and MYD05](https://modis.gsfc.nasa.gov/data/dataprod/mod05.php)).

Download the ready-to-go global WVDB
""""""""""""""""""""""""""""""""""""

You should start by downloading the pre-compiled WVDB with global coverage from [here](doi.pangaea.de/10.1594/PANGAEA.893109).
This saves you a lot of processing.
This freely available dataset was generated with the **FORCE WVDB** component, and is comprised of daily global water vapor data for February 2000 to July 2018 for each land-intersecting WRS-2 scene (13281 coordinates), as well as a monthly climatology that can be used if no daily value is available.

Generate the WVDB on your own
"""""""""""""""""""""""""""""

We try to update this dataset in regular intervals.
However, if you are in need of more up-to-date data, you can use the **FORCE WVDB** component to generate/update these tables on your own.


*Please note that you need access to the LAADS DAAC before using this tool (see last section on this page).*

**FORCE WVDB** needs a table with input coordinates (center coordinates of WRS-2 frames).
The [pre-compiled dataset](doi.pangaea.de/10.1594/PANGAEA.893109) includes such a table.
If you are not interested in global coverage, you can subset this file.
The file should contain two columns separated by white space, and no header.
The first column should give the longitude (X), the second column the latitude (Y) with coordinates in decimal degree (negative values for West/South).
Any other column is ignored (in the example below, the WRS-2 Path/Row is in the third column).


.. code-block:: bash

   wc -l /data/Earth/global/wvp/wvdb/wrs-2-land.coo


    13281 /data/Earth/global/wvp/wvdb/wrs-2-land.coo



.. code-block:: bash

   head /data/Earth/global/wvp/wvdb/wrs-2-land.coo


    -15.39340494140 80.76026666750 013001
    -22.86543244600 80.00558606640 013002
    -29.22356065160 79.11366800820 013003
    -34.59295680040 78.11513723200 013004
    -39.12687451150 77.03430642440 013005
    -42.97184515330 75.88984431700 013006
    -46.25519224080 74.69581438230 013007
    -49.08160498390 73.46286239410 013008
    -51.53569902300 72.19888348300 013009
    -53.68466715610 70.91003752470 013010


**FORCE WVDB** downloads each Terra/Aqua granule (collection 6.1) that intersects with any of these coordinates.
The files are downloaded from the Level1 and Atmosphere Archive and Distribution System ([LAADS](ladsweb.modaps.eosdis.nasa.gov)) at NASA’s Goddard Space Flight Center.
Note that any permanent or temporary change/shutdown/decommissioning on LAADS’ or MODIS’ end may result in the nonfunctioning of **FORCE WVDB**... Also note, that they perform a weekly maintenance, during which their servers are not accessable.

As with any other FORCE program, you can display short usage instructions by executing the program without any parameters.


.. code-block:: bash

   force-lut-modis


    usage: force-lut-modis coords dir-wvp dir-geometa dir-eoshdf
               [start-year start-month start-day
                end-year   end-month   end-day]




A coordinate file needs to be given as 1st argument.


The MODIS data are downloaded to dir-eoshdf (this directory must exist).
MODIS data that are already in dir-eoshdf are not downloaded again.
*If the tool crashes because a dataset is corrupt, it is necessary to manually delete this file and run the tool again.
Unfortunately, this happens from time to time due to incomplete downloads or if LAADS is unresponsive.
The program attempts to re-download a corrupt file up to 10 times, but this error can occur nonetheless.*

MOD05/MYD05 data are swath products, and MOD03/MYD03 geometa tables are necessary to relate coordinates to MODIS granules.
The geometa tables are downloaded to dir-geometa (this directory must exist).
Tables that are already in dir-geometa are not downloaded again.
*If the tool crashes because a table is invalid, it is necessary to manually delete this file and run the tool again.
Unfortunately, this happens from time to time due to incomplete downloads or if LAADS is unresponsive.
The program attempts to re-download a corrupt file up to 10 times, but this error can occur nonetheless.*

The final water vapor tables are saved in dir-wvp (this directory must exist).
Tables that are already in dir-wvp are not processed again (i.e. no download of geometa tables and hdf files).

The start and end arguments are optional and may be used for parallelization.
If they are not given, **FORCE WVDB** will download the entire time series of all coordinates provided (this can be a lot!).

This directory is the directory, to which DIR_WVPLUT in the FORCE L2PS parameter file should refer.
``DIR_WVPLUT = /data/Earth/global/wvp/wvdb``

If you have finished compiling the WVDB, you may delete the MODIS *.hdf files.

Download the entire data record (in one process - this is slow):


.. code-block:: bash

   force-lut-modis /data/Earth/global/wvp/wvdb/wrs-2-land.coo /data/Earth/global/wvp/wvdb /data/Earth/global/wvp/geo /data/Earth/global/wvp/hdf


Download one week:


.. code-block:: bash

   force-lut-modis /data/Earth/global/wvp/wvdb/wrs-2-land.coo /data/Earth/global/wvp/wvdb /data/Earth/global/wvp/geo /data/Earth/global/wvp/hdf 2010 07 01 2010 07 07


Use GNU parallel to download an entire month in 31 parallel processes.
This works by creating a list 1..31, which is distributed to 31 jobs.
Each job calls **FORCE WVDB** for one specific day in July 2010.
The curly braces are replaced with the list value given to each process.


.. code-block:: bash

   seq -w 1 31 | parallel -j31 force-lut-modis /data/Earth/global/wvp/wvdb/wrs-2-land.coo /data/Earth/global/wvp/wvdb /data/Earth/global/wvp/geo /data/Earth/global/wvp/hdf 2010 07 {} 2010 07 {}



Get access to the LAADS DAAC
----------------------------

.. note::

   *(edit 13.02.2020)*


You need authentification to download data from the LAADS DAAC.
This works by requesting an App Key from [NASA Earthdata](https://ladsweb.modaps.eosdis.nasa.gov/tools-and-services/data-download-scripts/#requesting).
You can make this key available to FORCE by putting the character string in a file ``.laads`` in your home directory.
With this, you should be able to download data.

