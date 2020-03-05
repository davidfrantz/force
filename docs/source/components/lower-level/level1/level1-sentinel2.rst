.. _level1-sentinel2:

Sentinel-2
==========

FORCE can process Level 1C Sentinel-2A and Sentinel-2B MSI data as provided by ESA through their data hub. 

The full resolution Sentinel-2 images must be acquired from the ESA archive (or from elsewhere).

It is recommended to store the unzipped images in a consistent data pool (without duplicates). 

A file queue needs to be prepared (see section VII.B). 

Downloading data, unzipping, managing the data pool, and preparing/updating the file queue can be handled by ``force-level1-sentinel2``.

.. code-block:: bash

  force-level1-sentinel2

  Usage: force-level1-sentinel2.sh Level-1-Datapool queue Boundingbox
                                starttime endtime min-cc max-cc [dry]

* Level-1-Datapool

  | The Sentinel-2 images are downloaded to the Level 1 data pool, which should be an existing directory. 
  | Files are not downloaded/imported if they are duplicates of already existing files.

* queue

  | A file queue (e.g. a file named ``level1-sentinel2-germany.txt``) needs to be given. 
  | If it does not exist, it will be created. 
  | If it exists, new imports are appended to this file. 
  | Outdated files (older production number) are removed from this queue, and the new imports are appended to the end. 
  | This queue is needed for Level 2 processing. All images with ``QUEUED`` status will be processed, then set to ``DONE``.


* Boundingbox

  | Each acquisition covered by the bounding box is downloaded/imported. 
  | The bounding box encloses your study area and must be given as ``"X1/Y1,X2/Y2,X3/Y3,...,X1/Y1"``. 
  | The box must be closed (first X/Y = last X/Y). 
  | X/Y must be given in decimal degree with negative values for West and South coordinates.
  | Note that the box doesn't have to be square, you can specify a polygon

* starttime endtime

  | Starttime and endtime specify a temporal subset and refer to the acquisition time.
  | Dates must be given as ``YYYY-MM-DD``

* min-cc max-cc

  The cloud cover range must be given in % ranging from 0 to 100

* [dry]

  This argument is optional, and if ``dry`` is specified, it will trigger a dry run that will only return the number of images and their total data volume

| Data will be downloaded from ESA’s API Hub, which requires an account at ESA’s end. 
| On your end, your login credentials must be placed in a hidden file ``.scihub`` in your home directory (you can chmod to 400), with user name in the first line and password in the second line.
| Note that special characters might be problematic. The End-of-Line character needs to be UNIX-style. 

It is possible to call ``force-level1-sentinel2`` (and subsequently ``force-level2``) from a cronjob, in which case a near-real time processing can be realized.


**A note on the Long Term Archive**

| In September 2019, ESA has activated the Long Term Archive (LTA) to roll out old (and potentially infrequently used) data products from the online storage system to offline storage. For details, see `here <https://scihub.copernicus.eu/userguide/LongTermArchive>`_. 
| As of now, the last year of data shall stay online, and is immediately ready for download. 
| Offline data may be pulled from offline to online storage upon request. The data retrieval shall happen within 24h and the products shall stay online for 3 days. If they were not downloaded within this time period, they need to be pulled again. A user quota is implemented to prevent users from pulling the entire archive - unfortunately this quota is currently 1 request per hour per user… 
| FORCE versions < 3.0 crash when trying to download offline products. FORCE L1AS v. >= 3.0 determines whether a product is online or offline. If online, the image is downloaded as described above. If offline, a pull request from offline to online storage is sent. ESA hasn't implemented any callback for this retrieval, thus FORCE L1AS will simply send pull requests for each requested offline image, probably download some available online images, and then exit. FORCE L1AS needs to be run again to retrieve the restored data. 


.. seealso:: Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-level1-s2/sentinel-2-l1c/>`_, which shows how to use ``force-level1-sentinel2``, how to set up a scheduled download, and more.
