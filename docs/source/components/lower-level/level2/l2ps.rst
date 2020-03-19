.. _level2-core:

force-l2ps
==========

The hidden workhorse of FORCE L2PS is force-l2ps, which is a lower-level routine called from within :ref:`level2-bulk` and :ref:`level2-wrapper`.

It processes one single image.
For the majority of users, it is recommended to use force-level2 instead of directly calling force-l2ps.
However, for specific purposes (e.g. testing/debugging or or if you need/want to implement your own job scheduler), the expert user may want to use this program directly.

Usage
^^^^^

.. code-block:: bash

  force-l2ps

  Usage: force-l2ps image-dir parameter-file

* image-dir

  | The 1st argument is the directory that contains the image data.
  | In case of Landsat, the *.tar.gz archive needs to be extracted before processing.
  | In case of Sentinel-2, the *.zip archive needs to extracted before processing and either the ``.SAFE`` directory or one tile (directory) within the ``GRANULE`` directory must be given as input.

* parameter-file

  | The :ref:`l2-param` needs to be given as second argument


The direct usage of force-l2ps is recommended for debugging or for detailed output.
The debugging mode also features extensive output where images for many processing steps are saved.
Note that these images are intended for software development and do not necessarily have intuitive file names; metadata or projections are also not appended.
If debug output is required, the software needs to be re-compiled, which should only be done by expert users.
If DEBUG is activated, :ref:`level2-bulk` does not allow you to process multiple images or to use parallel processing (your system will be unresponsive because too much data is simultaneously written to the disc, and parallel calls to force-l2ps would overwrite the debugging images).

For debugging, follow these steps either with:

1) Modify src/higher-level/const-cl.h

2) In the ‘compiler options’ section, uncomment ``FORCE_DEBUG``.

3) Re-compile the software

Or:

1) Run ./debug.sh

2) Re-compile the software
