.. _tabulate-grid:

force-tabulate-grid
===================

In case of the gridded data structure, force-tabulate-grid can be used to extract the processing grid as ESRI shapefile, e.g. for visualization purposes or to generate a tile allow-list. Any gridded data cube (containing a data cube definition file, see VII.M) can be given as 1st argument. The approximate bounding box of your study area needs to be given with coordinates in decimal degree (negative values for West/South). The shapefile ‘datacube-grid.shp’ is stored in the same directory as the data cube.

Usage
^^^^^

.. code-block:: bash
    
   force-tabulate-grid [-h] [-v] [-i] [-b bottom,top,left,right] [-f format] datacube-dir

   -h  = show this help
   -v  = show version
   -i  = show program's purpose

   -b bottom,top,left,right  = bounding box
      use geographic coordinates! 4 comma-separated numbers

   -f format  = output format: shp or kml (default)

   Positional arguments:
   - 'datacube-dir': directory of existing datacube
