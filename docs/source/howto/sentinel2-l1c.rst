.. _tut-s2l1c:

Sentinel-2 Level 1C
===================

**How to download and manage Sentinel-2 Level 1C data**

This tutorial explains how to use the FORCE Level 1 Archiving Suite (FORCE L1AS) to download, organize, and maintain a clean and consistent Sentinel-2 Level 1 data pool, as well as corresponding data queues needed for the Level 2 processing.

.. admonition:: Info

   *This tutorial uses FORCE v. 3.0*


----------
Overview**
ESA provides an application programming interface (API) for data query and automatic download (see `here <https://scihub.copernicus.eu/twiki/do/view/SciHubUserGuide/BatchScripting?redirectedfrom=SciHubUserGuide.8BatchScripting>`_).
Based on some user-defined parameters (coordinates etc.) FORCE L1AS pulls a metadata report from the Copernicus API Hub.
Each hit is compared
with the local data holdings you already downloaded.
If a new file is sitting on ESA's end, the missing image is downloaded.
A file queue is generated and updated accordingly - which is the main input to the FORCE Level 2 Processing System.


.. figure:: img/tutorial-l1sen2.png" width="750

   *Sentinel-2 downloader in the **FORCE Level 1 Archiving Suite***


----------
Setup**
Now, the first step is to get access to ESA's data and services.
For that, you need an account.
If you don't have one, register `here <https://scihub.copernicus.eu/dhus/#/self-registration>`_.
It's free.


Then, your credentials must be made available to FORCE L1AS to be able to request and receive ESA data.
On your machine, your login credentials must be placed in a hidden file ``.scihub`` in your home directory.
I advise to only give user reading rights to this file.
The user name goes in the first line, the password in the second line.
Please note that special characters might be problematic.
Also note, if you generate this file from a Windows machine, the Windows EOL character will cause problems.

----------
Download some data**
""""""""""
Instructions**
After setting up your account, you should be able to download Sentinel-2 data via FORCE L1AS.
As with any other FORCE program, you can display short usage instructions by executing the program without any parameters.


.. code-block:: bash

   force-level1-sentinel2


    
    Usage: force-level1-sentinel2 Level-1-Datapool queue Boundingbox
                       starttime endtime min-cc max-cc [dry]
    
      Level-1-Datapool
      An existing directory, your files will be stored here
    
      queue
      Downloaded files are appended to a file queue, which is needed for
      the Level 2 processing.
The file doesn't need to exist.
If it exists,
      new lines will be appended on successful ingestion
    
      Boundingbox
      The coordinates of your study area: "X1/Y1,X2/Y2,X3/Y3,...,X1/Y1"
      The box must be closed (first X/Y = last X/Y).
X/Y must be given as
      decimal degrees with negative values for West and South coordinates.
      Note that the box doesn't have to be square, you can specify a polygon
    
      starttime endtime
      Dates must be given as YYYY-MM-DD
    
      min-cc max-cc
      The cloud cover range must be given in %
    
      dry will trigger a dry run that will only return the number of images
      and their total data volume
    
      Your ESA credentials must be placed in /home/frantzda/.scihub
        First line: User name
        Second line: Password, special characters might be problematic
    


""""""""""
Dry run**
Please note that FORCE won't check that there is enough space on your hard disc.
If you don't dare to download all data straight away, there is a dry run option implemented that only checks how much data would be downloaded with the parameters you provided.
This is given by the optional ``dry`` keyword at the end of the command line.
The following query asks for all data in July 2019 with a maximum cloud coverage of 50%.
The region of interest here is a rather small area in Zambia's Northwestern Province, depicting a large copper mine in the Miombo forest.

The single most frequent error here is the specification of the coordinates.
If you don't receive any data for your study area, or end up somewhere else entirely, double check that the coordinates are given in the correct order.

- X = Longitude
- Y = Latitude


.. code-block:: bash

   force-level1-sentinel2 /data/Dagobah/S2L1C /data/Dagobah/S2L1C/zambia.txt "25.43/-12.46,25.94/-12.46,25.94/-11.98,25.39/-11.99,25.43/-12.46" 2019-07-01 2019-07-31 0 50 dry


    2020-02-15_15:36:36 - Found 13 S2A/B files.
    13 Sentinel-2 A/B L1C files available
    5.19094 GB data volume available


""""""""""
Download**
The actual download is triggered by omitting the ``dry`` option.
FORCE L1AS downloads all data that match the parameters provided - and which weren't downloaded before.
Note that the program checks against the files on the disc (not the file queue).
Each downloaded image is unzipped after the download.
If both steps were successful, the image is appended to the file queue.


Do not wonder if FORCE tells you that it has found exactly 100 S2A/B files.
The ESA API Hub only allows to retrieve metadata for 100 products.
Thus, FORCE iterates through the pages until no more image can be retrieved.

Please note that download speed varies considerably.. 


.. code-block:: bash

   force-level1-sentinel2 /data/Dagobah/S2L1C /data/Dagobah/S2L1C/zambia.txt "25.43/-12.46,25.94/-12.46,25.94/-11.98,25.39/-11.99,25.43/-12.46" 2019-07-01 2019-07-31 0 50


    2020-02-15_15:37:03 - Found 13 S2A/B files.
    2020-02-15_15:37:03 - Found 13 S2A/B files on this page.
    /data/Dagobah 100%[===================>] 729.50M  32.4MB/s    in 23s     
    /data/Dagobah 100%[===================>] 271.14M  30.5MB/s    in 7.0s    
    /data/Dagobah 100%[===================>] 742.98M  29.8MB/s    in 24s     
    /data/Dagobah 100%[===================>] 266.53M  28.1MB/s    in 11s     
    /data/Dagobah 100%[===================>] 732.80M  30.7MB/s    in 20s     
    /data/Dagobah 100%[===================>] 224.77M  69.2MB/s    in 3.2s    
    /data/Dagobah 100%[===================>] 730.90M  81.6MB/s    in 9.6s    
    /data/Dagobah 100%[===================>] 268.45M  42.5MB/s    in 7.9s    
    /data/Dagobah 100%[===================>] 704.98M  45.8MB/s    in 21s     
    /data/Dagobah 100%[===================>] 258.02M  47.5MB/s    in 5.9s    
    /data/Dagobah 100%[===================>] 754.09M  61.7MB/s    in 13s     
    /data/Dagobah 100%[===================>] 259.80M  66.5MB/s    in 4.3s    


In the Level 1 Datapool, there is now the file queue, as well as a directory for each MGRS tile that was retrieved.



.. code-block:: bash

   ls /data/Dagobah/S2L1C


    T35LLG  zambia.txt


In the directories, there are the unzipped images.



.. code-block:: bash

   ls /data/Dagobah/S2L1C/T*


    S2A_MSIL1C_20190707T080611_N0207_R078_T35LLG_20190707T100942.SAFE
    S2A_MSIL1C_20190710T081611_N0208_R121_T35LLG_20190710T103430.SAFE
    S2A_MSIL1C_20190717T080611_N0208_R078_T35LLG_20190717T110132.SAFE
    S2A_MSIL1C_20190720T081611_N0208_R121_T35LLG_20190720T134157.SAFE
    S2A_MSIL1C_20190727T080611_N0208_R078_T35LLG_20190727T115444.SAFE
    S2A_MSIL1C_20190730T081611_N0208_R121_T35LLG_20190730T103748.SAFE
    S2B_MSIL1C_20190702T080619_N0207_R078_T35LLG_20190702T115117.SAFE
    S2B_MSIL1C_20190705T081609_N0207_R121_T35LLG_20190705T110801.SAFE
    S2B_MSIL1C_20190712T080619_N0208_R078_T35LLG_20190712T110120.SAFE
    S2B_MSIL1C_20190715T081609_N0208_R121_T35LLG_20190715T121541.SAFE
    S2B_MSIL1C_20190722T080619_N0208_R078_T35LLG_20190722T110547.SAFE
    S2B_MSIL1C_20190725T081609_N0208_R121_T35LLG_20190725T120407.SAFE


Note that for Sentinel-2, the compression is realized in the image data, not in the container, thus unzipping the data does not inflate the file size (much).


.. code-block:: bash

   du -h -d 1 /data/Dagobah/S2L1C


    5.9G	/data/Dagobah/S2L1C/T35LLG
    5.9G	/data/Dagobah/S2L1C


The file queue is holding the full filepaths to all ingested images.
This is the main input to **force-level2**.
Each image is in a separate line.
A processing-state flag determines if the image is enqueued for Level 2 processing - or was already processed and will be ignored next time.
This flag is either ``QUEUED`` or ``DONE``.



.. code-block:: bash

   cat /data/Dagobah/S2L1C/zambia.txt


    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190730T081611_N0208_R121_T35LLG_20190730T103748.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190727T080611_N0208_R078_T35LLG_20190727T115444.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190725T081609_N0208_R121_T35LLG_20190725T120407.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190722T080619_N0208_R078_T35LLG_20190722T110547.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190720T081611_N0208_R121_T35LLG_20190720T134157.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190717T080611_N0208_R078_T35LLG_20190717T110132.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190715T081609_N0208_R121_T35LLG_20190715T121541.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190712T080619_N0208_R078_T35LLG_20190712T110120.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190710T081611_N0208_R121_T35LLG_20190710T103430.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190707T080611_N0207_R078_T35LLG_20190707T100942.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190705T081609_N0207_R121_T35LLG_20190705T110801.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190702T080619_N0207_R078_T35LLG_20190702T115117.SAFE QUEUED


----------
Download some more data**
Downloading more data is easy.
You can use the same datapool, and the same file queue for this.
Images are only downloaded if they weren't downloaded yet.
Thus, you can e.g. change the boundingbox, time frame or cloud coverage, and only download data that was not covered by the previous run.
In the following example, the Eastern X-Coordinates were increased by 1 degree.


.. code-block:: bash

   force-level1-sentinel2 /data/Dagobah/S2L1C /data/Dagobah/S2L1C/zambia.txt "25.43/-12.46,26.94/-12.46,26.94/-11.98,25.39/-11.99,25.43/-12.46" 2019-07-01 2019-07-31 0 50


    2020-02-15_15:43:57 - Found 26 S2A/B files.
    2020-02-15_15:43:57 - Found 26 S2A/B files on this page.
    /data/Dagobah 100%[===================>]  54.08M  43.4MB/s    in 1.2s    
    /data/Dagobah 100%[===================>] 794.27M  73.7MB/s    in 11s     
    /data/Dagobah 100%[===================>]  57.66M  61.0MB/s    in 0.9s    
    /data/Dagobah 100%[===================>] 782.06M  80.1MB/s    in 10s     
    /data/Dagobah 100%[===================>]  49.95M  52.0MB/s    in 1.0s    
    /data/Dagobah 100%[===================>] 555.54M  85.9MB/s    in 6.8s    
    /data/Dagobah 100%[===================>]  52.83M  57.7MB/s    in 0.9s    
    /data/Dagobah 100%[===================>] 788.67M  79.2MB/s    in 12s     
    /data/Dagobah 100%[===================>]  48.47M  52.4MB/s    in 0.9s    
    /data/Dagobah 100%[===================>] 779.62M  81.5MB/s    in 9.4s    
    /data/Dagobah 100%[===================>]  58.56M  50.9MB/s    in 1.1s    
    /data/Dagobah 100%[===================>] 781.21M  54.6MB/s    in 17s     


From the 26 available images, only 13 were retrieved, 13 were already downloaded before.
There are now several directories with different MGRS tiles in your Level 1 Datapool.


.. code-block:: bash

   ls /data/Dagobah/S2L1C/T*


    /data/Dagobah/S2L1C/T35LLG:
    S2A_MSIL1C_20190707T080611_N0207_R078_T35LLG_20190707T100942.SAFE
    S2A_MSIL1C_20190710T081611_N0208_R121_T35LLG_20190710T103430.SAFE
    S2A_MSIL1C_20190717T080611_N0208_R078_T35LLG_20190717T110132.SAFE
    S2A_MSIL1C_20190720T081611_N0208_R121_T35LLG_20190720T134157.SAFE
    S2A_MSIL1C_20190727T080611_N0208_R078_T35LLG_20190727T115444.SAFE
    S2A_MSIL1C_20190730T081611_N0208_R121_T35LLG_20190730T103748.SAFE
    S2B_MSIL1C_20190702T080619_N0207_R078_T35LLG_20190702T115117.SAFE
    S2B_MSIL1C_20190705T081609_N0207_R121_T35LLG_20190705T110801.SAFE
    S2B_MSIL1C_20190712T080619_N0208_R078_T35LLG_20190712T110120.SAFE
    S2B_MSIL1C_20190715T081609_N0208_R121_T35LLG_20190715T121541.SAFE
    S2B_MSIL1C_20190722T080619_N0208_R078_T35LLG_20190722T110547.SAFE
    S2B_MSIL1C_20190725T081609_N0208_R121_T35LLG_20190725T120407.SAFE
    
    /data/Dagobah/S2L1C/T35LMG:
    S2A_MSIL1C_20190707T080611_N0207_R078_T35LMG_20190707T100942.SAFE
    S2A_MSIL1C_20190710T081611_N0208_R121_T35LMG_20190710T103430.SAFE
    S2A_MSIL1C_20190717T080611_N0208_R078_T35LMG_20190717T110132.SAFE
    S2A_MSIL1C_20190720T081611_N0208_R121_T35LMG_20190720T134157.SAFE
    S2A_MSIL1C_20190727T080611_N0208_R078_T35LMG_20190727T115444.SAFE
    S2A_MSIL1C_20190730T081611_N0208_R121_T35LMG_20190730T103748.SAFE
    S2B_MSIL1C_20190702T080619_N0207_R078_T35LMG_20190702T115117.SAFE
    S2B_MSIL1C_20190705T081609_N0207_R121_T35LMG_20190705T110801.SAFE
    S2B_MSIL1C_20190712T080619_N0208_R078_T35LMG_20190712T110120.SAFE
    S2B_MSIL1C_20190715T081609_N0208_R121_T35LMG_20190715T121541.SAFE
    S2B_MSIL1C_20190722T080619_N0208_R078_T35LMG_20190722T110547.SAFE
    S2B_MSIL1C_20190725T081609_N0208_R121_T35LMG_20190725T120407.SAFE


The new files were appended to the file queue, too.


.. code-block:: bash

   cat /data/Dagobah/S2L1C/zambia.txt


    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190730T081611_N0208_R121_T35LLG_20190730T103748.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190727T080611_N0208_R078_T35LLG_20190727T115444.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190725T081609_N0208_R121_T35LLG_20190725T120407.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190722T080619_N0208_R078_T35LLG_20190722T110547.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190720T081611_N0208_R121_T35LLG_20190720T134157.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190717T080611_N0208_R078_T35LLG_20190717T110132.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190715T081609_N0208_R121_T35LLG_20190715T121541.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190712T080619_N0208_R078_T35LLG_20190712T110120.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190710T081611_N0208_R121_T35LLG_20190710T103430.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2A_MSIL1C_20190707T080611_N0207_R078_T35LLG_20190707T100942.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190705T081609_N0207_R121_T35LLG_20190705T110801.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LLG/S2B_MSIL1C_20190702T080619_N0207_R078_T35LLG_20190702T115117.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2A_MSIL1C_20190730T081611_N0208_R121_T35LMG_20190730T103748.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2A_MSIL1C_20190727T080611_N0208_R078_T35LMG_20190727T115444.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2B_MSIL1C_20190725T081609_N0208_R121_T35LMG_20190725T120407.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2B_MSIL1C_20190722T080619_N0208_R078_T35LMG_20190722T110547.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2A_MSIL1C_20190720T081611_N0208_R121_T35LMG_20190720T134157.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2A_MSIL1C_20190717T080611_N0208_R078_T35LMG_20190717T110132.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2B_MSIL1C_20190715T081609_N0208_R121_T35LMG_20190715T121541.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2B_MSIL1C_20190712T080619_N0208_R078_T35LMG_20190712T110120.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2A_MSIL1C_20190710T081611_N0208_R121_T35LMG_20190710T103430.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2A_MSIL1C_20190707T080611_N0207_R078_T35LMG_20190707T100942.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2B_MSIL1C_20190705T081609_N0207_R121_T35LMG_20190705T110801.SAFE QUEUED
    /data/Dagobah/S2L1C/T35LMG/S2B_MSIL1C_20190702T080619_N0207_R078_T35LMG_20190702T115117.SAFE QUEUED


----------
Setting up a scheduled download**
The same logic can be used to set up a scheduler for downloading your data at regular intervals.
For example, a daily cronjob can be installed to retrieve all data covering your study area.
A cronjob is installed by adding lines to the cronjob file.


.. code-block:: bash

   crontab -e


If your FORCE installation is not in a standard search path, you need to define the PATH variable, and include the path where FORCE is installed.
Then, use the command line from above and schedule it with cron notation.
Following line will start the download at 3:00 AM each day.
*Replace YOURNAME with your user name.*


.. code-block:: bash

   PATH=/home/YOURNAME/bin:/usr/bin:/bin:/usr/local/bin
0 3 * * * force-level1-sentinel2 /data/Dagobah/S2L1C /data/Dagobah/S2L1C/zambia.txt "25.43/-12.46,26.94/-12.46,26.94/-11.98,25.39/-11.99,25.43/-12.46" 2018-07-01 2018-07-31 0 50


----------
Long Term Archive**
In September 2019, ESA has activated the Long Term Archive (LTA) to roll out old (and potentially infrequently used) data products from the online storage system to offline storage.
For details, see `here <https://scihub.copernicus.eu/userguide/LongTermArchive>`_.
As of now (the following numbers might change in the future), the last year of data shall stay online, and is immediately ready for download.
Offline data may be pulled from offline to online storage upon request.
The data retrieval shall happen within 24h and the products shall stay online for 3 days.
If they were not downloaded within this time period, they need to be pulled again.
A user quota is implemented to prevent users from pulling the entire archive - unfortunately this quota is ridicously low, 1 request per hour per user... Let's all hope it doesn't stay this way :/

**FORCE >= 3.0** is able to handle LTA data.
Previous FORCE versions crash when trying to download offline products.

FORCE L1AS determines whether a product is online or offline.

- If online, the image is downloaded as described above.
- If offline, a pull request from offline to online storage is sent.
ESA hasn't implemented any callback for this retrieval, thus FORCE L1AS will simply send pull requests for each requested offline image, probably download some available online images, and then exit.
FORCE L1AS needs to be run again to retrieve the restored data.
To this end, it comes in handy to set up a download scheduler as desribed above.


.. code-block:: bash

   force-level1-sentinel2 /data/Dagobah/S2L1C /data/Dagobah/S2L1C/zambia.txt "25.43/-12.46,25.94/-12.46,25.94/-11.98,25.39/-11.99,25.43/-12.46" 2018-07-01 2018-07-31 0 50


    2020-02-15_15:49:18 - Found 12 S2A/B files.
    2020-02-15_15:49:18 - Found 12 S2A/B files on this page.
    S2B_MSIL1C_20180730T081559_N0206_R121_T35LLG_20180730T141111.SAFE: Pulling from Long Term Archive.
Success.
Rerun this program after a while
    S2B_MSIL1C_20180727T080609_N0206_R078_T35LLG_20180727T121446.SAFE: Pulling from Long Term Archive.
Failed.
You have exhausted your user quota.
Rerun this program after a while
    S2A_MSIL1C_20180725T081601_N0206_R121_T35LLG_20180725T121615.SAFE: Pulling from Long Term Archive.
Failed.
You have exhausted your user quota.
Rerun this program after a while
    S2A_MSIL1C_20180722T080611_N0206_R078_T35LLG_20180722T115605.SAFE: Pulling from Long Term Archive.
Failed.
You have exhausted your user quota.
Rerun this program after a while
    S2B_MSIL1C_20180720T081559_N0206_R121_T35LLG_20180720T121127.SAFE: Pulling from Long Term Archive.
Failed.
You have exhausted your user quota.
Rerun this program after a while
    S2B_MSIL1C_20180717T080609_N0206_R078_T35LLG_20180717T120239.SAFE: Pulling from Long Term Archive.
Failed.
You have exhausted your user quota.
Rerun this program after a while
    S2A_MSIL1C_20180715T081601_N0206_R121_T35LLG_20180715T103432.SAFE: Pulling from Long Term Archive.
Failed.
You have exhausted your user quota.
Rerun this program after a while
    S2A_MSIL1C_20180712T080611_N0206_R078_T35LLG_20180712T102334.SAFE: Pulling from Long Term Archive.
Failed.
Too Many Requests
    S2B_MSIL1C_20180710T081559_N0206_R121_T35LLG_20180710T120813.SAFE: Pulling from Long Term Archive.
Failed.
Too Many Requests
    S2B_MSIL1C_20180707T080609_N0206_R078_T35LLG_20180707T115209.SAFE: Pulling from Long Term Archive.
Failed.
Too Many Requests
    S2A_MSIL1C_20180705T081601_N0206_R121_T35LLG_20180705T103349.SAFE: Pulling from Long Term Archive.
Failed.
Too Many Requests
    S2A_MSIL1C_20180702T080611_N0206_R078_T35LLG_20180702T115230.SAFE: Pulling from Long Term Archive.
Failed.
Too Many Requests


------------

.. |author-pic| image:: profile/dfrantz.jpg

+--------------+--------------------------------------------------------------------------------+
+ |author-pic| + This tutorial was written by                                                   +
+              + `David Frantz <https://davidfrantz.github.io>`_,                               +
+              + main developer of **FORCE**,                                                   +
+              + postdoc at `EOL <https://www.geographie.hu-berlin.de/en/professorships/eol>`_. +
+              + *Views are his own.*                                                           +
+--------------+--------------------------------------------------------------------------------+
+ **EO**, **ARD**, **Data Science**, **Open Science**                                           +
+--------------+--------------------------------------------------------------------------------+
