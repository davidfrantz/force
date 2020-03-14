.. _level2-wrapper:

force-l2ps_
===========

This program is a wrapper script that acts as a bridge between :ref:`level2-bulk` and :ref:`level2-core`.
It is called from within :ref:`level2-bulk` for each image.

For the majority of users, it is recommended to use force-level2 instead of directly calling force-l2ps_.

If necessary, this tool unpacks the Level 1 zip/tar.gz containers. 
It calls :ref:`level2-core` to process the umpacked images.
If input data were unpacked by this tool, it removes the temporary data after processing.
The tool also redirects the messages printed by :ref:`level2-core` to logfiles.


Usage
^^^^^

.. code-block:: bash

  force-l2ps_

  Usage: force-l2ps_ image parfile bindir logdir tempdir timeout_zip
  
* image

  | The 1st argument is the input image.
  | In case of Landsat, it can be the ``.tar.gz`` archive, or the unpacked image.
  | In case of Sentinel-2, it can be the ``.zip`` archive, the unpacked ``.SAFE`` directory, or one tile (directory) within the ``GRANULE`` directory must be given as input.

* parfile

  | The :ref:`l2-param` needs to be given as second argument

* bindir

  | This is the directory, where :ref:`level2-core` is installed

* logdir

  | This is the directory, where logfiles should be stored. 

* tempdir

  | This is the directory for temporarily unpacking the ``.zip`` or ``.tar.gz`` containers

* timeout_zip

  | This is a timeout in seconds after which the unpacking of the ``.zip`` or ``.tar.gz`` containers will be cancelled

