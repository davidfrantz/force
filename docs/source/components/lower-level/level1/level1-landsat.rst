.. _level1-landsat:

force-level1-landsat
====================

``force-level1-landsat`` offers a simple command line interface to communicate with the USGS/EROS machine-to-machine API. The tool can be used to search for Landsat Collection 2 Level 1 product bundles, create download links for the results and/or directly download the data.

Requirements
^^^^^^^^^^^^
User credentials to log in to the USGS EarthExplorer interface are required. Your user account needs to have access to the machine-to-machine API, which can be requested through the user profile page `here <https://ers.cr.usgs.gov/profile/access>`_.

aria2 is required to download product bundles. You can still create links and download them manually if aria2 is not available. Also see :ref:`depend`

Usage
^^^^^
| There are two subprogams, ``search`` and ``download``:
| ``search`` can be used to search the Landsat archive, generate download links, and also download the product bundles right away. \
| ``download`` can be used to download product bundles for which a list of download links was generated using ``search`` before.

force-level1-landsat search
+++++++++++++++++++++++++++

.. code-block::

  force-level1-landsat search
  usage: force-level1-landsat search [-h] [-s SENSOR] [-d DATERANGE] [-c CLOUDCOVER]
    [-m MONTHS] [-t {T1,T2}] [-l {L1TP,L1GT,L1GS}] [--download | -n] [-f FORCELOGS]
    [-q QUEUE_FILE] [--secret SECRET] aoi output_dir

* aoi
    | The area of interest. Valid input:
    | (1) .txt - text file containing one tile per line in the format ``PPPRRR``
    |     (``P`` = path, ``R`` = row)
    |     Keep padding zeroes. Correct: ``194023``, incorrect: ``19423``
    | (2) .shp, .gpkg, .geojson - vector file containing point, line, or polygon geometries.

* output-dir
    | The directory where the file containing the download URLs or downloaded products will be stored.
    | The ``--download`` option will deactivate saving of URLs.

* -s | \--sensor
    | Restrict results to specific sensor(s).
    | choices = ``TM``, ``ETM``, ``OLI`` (Landsat 4/5, Landsat 7, Landsat 8/9)
    | Default: All sensors

* -d | \--daterange
    | Start date and end date = date range to be considered.
    | Format: ``YYYYMMDD,YYYYMMDD``
    | Default: full archive until today.

* -c | \--cloudcover
    | Percent (land) cloud cover range to be considered.
    | Default: ``0,100``

* -m | \--months
    | Seasonal filter: define the months to be considered.
    | Default: ``1,2,3,4,5,6,7,8,9,10,11,12``

* -t | \--tier
    | Landsat collection tier level.
    | Valid tiers: ``T1,T2,RT``
    | Default: ``T1``

* -l | \--level
    | Landsat level of processing.
    | Valid levels: ``L1TP,L1GT,L1GS``
    | Default: ``L1TP``

* \--download
    | Download the product bundles directly after creating the download links.

* -n | \--no-action
    | Only search for product bundles and print info about search results without generating links or downloading.

* -f | \--forcelogs
    | Path to FORCE log file directory (Level-2 processing logs, directory will be searched recursively)
    | Links will only be generated for products that haven't been processed by FORCE yet.

* -q | \--queue-file
    | Path to FORCE queue file.
    | Downloaded product bundle file paths will be appended to the queue.

* \--secret
    | Path to the file containing the username and application token for M2MApi access.
    | Application tokens can be generated at https://ers.cr.usgs.gov/.
    | 1st line: 'app-token', 2nd line: ``user``, 3rd line: ``token``

.. code-block:: None

    force-level1-landsat search ~/berlin.shp ~/level1 -s OLI -d 20180101,20201231 -m 10,11 -c 0,70 --secret ~/.m2m.txt --no-action

    Sensor(s): OLI
    Tile(s): 192023,192024,193023,193024
    Date range: 2018-01-01 to 2020-12-31
    Included months: 10,11
    Cloud cover: 0% to 70%

    20 Landsat Level 1 scenes matching criteria found
    22.13 GB data volume found

.. note::

    The M2M API is rate limited to 15,000 requests/15min. If you exceed this limit, force-level1-landsat will wait for 15 minutes and continue afterwards. Checking for existing product bundles in the output directory happens before generating download URLs to reduce unnecessary requests.

force-level1-landsat download
+++++++++++++++++++++++++++++

.. code-block:: None

    force-level1-landsat download
    usage: force-level1-landsat download [-h] [-q QUEUE_FILE] url_file output_dir

* url-file
    | Path to the file containing the download links.

* output-dir
    | The directory where the product bundles will be stored.

* -q | \--queue-file
    | Path to FORCE queue file. Downloaded product bundle file paths will be appended to the queue.

.. code-block:: None

    force-level1-landsat download ~/urls_landsat_TM_ETM_OLI_20221001T174038.txt ~/level-1

    Loading urls from ~/urls_landsat_TM_ETM_OLI_20221001T174038.txt

    6 of 116 product bundles found in filesystem, 110 left to download.

    Downloading: 5%|===>                                    | 6/110 [08:36<2:29:13, 100.97s/product bundle/s]


.. note::

    The output directory will be checked recursively (i.e. including all subfolders) for existing product bundles and download URLs are only created for product bundles that were not found in the filesystem. All directories, .tar files, and .tar.gz files that match the Landsat Collections Level-1 naming convention are considered. Partial downloads (product bundles that are accompanied by .aria2 files) will be continued.