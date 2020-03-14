.. _level2-bulk:

force-level2
============

This is the main program of FORCE L2PS.
It handles the mass processing of the Level 1 input.

In principle, the same algorithm is applied to all sensors – although specific processing options are triggered or are only available for some sensors (e.g. Sentinel-2 resolution merge).

Each image enqueued in the :ref:`queue` is processed to Level 2 according to the specifications in the :ref:`l2-param`.
After processing, the image is dequeued, and as such it is possible to schedule processing in near-real time, e.g. if called from a cronjob.
If reprocessing is required, the image needs to be enqueued again.

A logfile is written, which is recommended to inspect after running the program.


Usage
^^^^^

.. code-block:: bash

  force-level2

  Usage: force-level2 parameter-file

* parameter-file

  | The :ref:`l2-param` needs to be given as sole argument


The main parallelization strategy is multiprocessing, i.e. individual images are processed simultaneously. 
Each process can additionally use multithreading.
We recommend to use as many processes, and as few threads as possible.
However, a mild mix may be beneficial, e.g. 2 threads per process.
If processing only a few (or one) image, or if RAM is too small, increase the multithreading ratio accordingly.
This can speed up the work significantly.

Note that GNU parallel is used for multiprocessing.
It spawns jobs by using multiple ssh logins; the number of allowed ssh logins per user must be sufficiently large (ask your system administrator to increase this number if necessary).
With a small change in the script and passwordless access to other machines, GNU parallel also allows for parallelization across a cluster. 

It is highly advised to monitor the workload.
Swapping should be prevented – in this case, decrease the number of parallel jobs.

The number of CPUs can be re-adjusted while(!) force-level2 is running.
A file ‘cpu-$TIME’ is created in the temporary directory.
This file can be modified.
Note that the effect is not immediate, as the load is only adjusted after one of the running jobs (images) is finished.

To prevent an I/O jam at startup (by reading / extracting a lot of data simultaneously), a delay (in seconds) might be necessary.
GNU parallel will wait that long before starting a new job.
The necessary delay (or none) is dependent on your system’s architecture (I/O speed etc), on sensor to be processed, and whether packed archives or uncompressed images are given as input.

