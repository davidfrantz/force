.. _aux-stack:

force-stack
===========

force-stack creates a `GDAL virtual format <https://gdal.org/drivers/raster/vrt.html>`_ layer-stack of several bands from different physical files.

Usage
^^^^^

.. code-block:: bash
    
  force-stack [-h] [-v] [-i] {-o output-file} src-files

  -h  = show this help
  -v  = show version
  -i  = show program's purpose

  -o output-file  = stacked output file (.vrt)

  Positional arguments:
  - 'src-files': source files that will be stacked
  