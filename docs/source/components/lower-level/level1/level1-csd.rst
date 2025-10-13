.. _level1-csd:

force-level1-csd
================

.. note:: 
    
    For downloading of `Landsat Collection 2 <https://www.usgs.gov/landsat-missions/landsat-collection-2>`_ data use :ref:`force-level1-landsat <level1-landsat>`.

.. note::
   | :ref:`level1-csd` is still functional, but due to changes on Google's end, the tool won't download data newer than August 28, 2024 (see `this issue <https://github.com/davidfrantz/force/issues/334>`_). For newer data, users are referred to the official APIs of CDSE.


FORCE can download Sentinel-2 data from cloud storage providers (currently Google Cloud Storage).

``force-level1-csd`` allows to search for image acquisitions that precisely match the user's requirements, manages data pool, and prepares/updates the file queue required for level 2 procesing.

.. note::

    If you need to bypass ``force-level1-csd``, it is recommended to store the images in a consistent data pool without duplicates.
    You will need to prepare a :ref:`queue`.


Usage
"""""

.. note::

    Before you start:
    The metadata catalogues need to be downloaded prior to the first run and downloading Sentinel-2 data requires authentication.
    See the :ref:`tut-l1csd` tutorial for more information on properly setting up ``force-level1-csd``.


``force-level1-csd`` requires four mandatory arguments to be specified.
Several optional arguments can be used to further refine the search.

.. code-block:: none

    Usage: force-level1-csd [optional arguments] metadata-dir level-1-datapool queue aoi


* **metadata-dir**

    | The directory where the metadata catalogues (csv files) are stored.
    | They need to be downloaded before the first use and should be updated regularly.

* **level-1-datapool**

    | All downloaded files are stored in the data pool, which should be an existing directory.
    | Scenes will not be downloaded if they already exist in this folder.

* **queue**

    | Downloaded files are appended to a file queue, which is needed for the Level 2 processing.
    | The file doesn't need to exist.
    | If it does exist, new lines will be appended on successful ingestion.
    | This queue is needed for Level 2 processing.
    | All images with ``QUEUED`` status will be processed, then set to ``DONE``.

* **area of interest**

    | The area that data should be downloaded for. It can be defined in three different ways
    | (1) user-supplied coordinates of your study area:
    |     The polygon must be closed (first ``X/Y`` = last ``X/Y``). ``X/Y`` must be given as
    |     decimal degrees with negative values for West and South coordinates.
    |     Either specify the path to a file, or the coordinates on the command line.
    |     If in a file, provide one coordinate per line.
    |     If on the command line, provide a comma separated list.
    | (2) a shapefile (point/polygon/line). On-the-fly reprojection is provided,
    |     but using EPSG4326 is recommended.
    | (3) scene identifier:
    |     Sentinel-2: MGRS tile as ``TXXXXX``. Make sure to keep the leading ``T``
    |       before the MGRS tile number.
    |     You can either give the path to a file, or give the IDs on the command line.
    |     If in a file, provide one ID per line.
    |     If on the command line, provide a comma separated list.



Running the search without any optional parameters will return all Sentinel-2 scenes for the specified ``aoi``.
To narrow down the search results, use the following parameters.

* **-c | \--cloudcover**
    | ``minimum,maximum``
    | The cloud cover range must be specified in %
    | Default: ``0,100``

* -d | \--daterange
    | ``starttime,endtime``
    | Dates must be given in the following format: YYYYMMDD,YYYYMMDD
    | Default: ``19700101,today``

* **-s | \--sensor**
    | Sensors to include in the query, comma-separated.
    | Valid sensors:
    | Sentinel
    | ``S2A`` - Sentinel-2A MSI
    | ``S2B`` - Sentinel-2B MSI
    | ``S2C`` - Sentinel-2C MSI
    | ``S2D`` - Sentinel-2D MSI (future-ready)
    | Default: ``S2A,S2B,S2C,S2D``

* **-t | \--tier**
    | Landsat collection tier level. Valid tiers: ``T1,T2,RT``
    | Default: ``T1``
    | deprecated option, has no effect anymore


The remaining optional arguments are used to perform a search without actually downloading data, store the metadata of search results, and download / update the metadata catalogues.

* **-n | \--no-act**
    | Will trigger a dry run that will only return the number of images
    | and their total data volume

* **-k | \--keep-meta**
    | Will write the results of the query to the level 1 datapool directory.
    | Two files will be created if Sentinel-2 data is queried
    | at the same time. Filename: ``csd_metadata_[satellite]_YYYY-MM-DDTHH-MM-SS``
    | ``[satellite]`` refers to Sentinel-2.

* **-l | \--logs**
    | Check for FORCE Level-2 log files and remove any products from the search
    | that have been processed previously. Note that this only checks for the
    | presence of log files, not for actual Level-2 products.

* **-u | \--update**
    | Will create or the metadata catalogue (download and extract from GCS)
    | If this option is used, only one mandatory argument is expected (metadata-dir).
    | Use the -s option to only update Sentinel-2 metadata.

.. note::

    The mandatory arguments are positional!
    They need to be provided in this exact order.
    The optional arguments can be placed anywhere and may also be combined.
    For example, ``-n -k -c 0,70`` could also be written as ``-nkc 0,70``.
    When values are passed to the optional arguments (cloud cover, date range, sensor, or tier), these must be separated by commas ``,`` and must not contain whitespace.


.. seealso::

    To learn more about FORCE Level 1 CSD, check the :ref:`tut-l1csd` tutorial.
    It covers the set up, usage, and provides some more general information.

