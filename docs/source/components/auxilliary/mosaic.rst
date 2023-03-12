.. _aux-mosaic:

force-mosaic
============

force-mosaic creates a `GDAL virtual format <https://gdal.org/drivers/raster/vrt.html>`_-file for each FORCE product found in ``datacube-dir``. This function is agnostic to the processing level of the data found and identifies corresponding bands in different tiles by their base names.

Using VRTs eases the visualization of large study areas by virtually linking the respective files in multiple tiles (i. e. directories) together. 
Additionally, the output can be :ref:`stacked <aux-stack>` and :ref:`overviews generated <aux-pyramid>`.

Usage
^^^^^

.. note::
    In FORCE <= 3.6.5 only the ``datacube-dir`` could be specified as parameter.

.. code-block:: none

    force-mosaic [-h] [-v] [-i] [-j] [-m] datacube-dir

    -h  = show this help
    -v  = show version
    -i  = show program's purpose

    -j  = number of parallel processes (default: all)

    -m  = mosaic directory (default: mosaic)
        This should be a directory relative to the tiles

    Positional arguments:
        - 'datacube-dir': directory of existing datacube

