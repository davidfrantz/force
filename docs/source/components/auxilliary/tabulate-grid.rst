.. _tabulate-grid:

force-tabulate-grid
===================

In case of the gridded data structure, force-tabulate-grid can be used to extract the processing grid as ESRI shapefile, e.g. for visualization purposes or to generate a tile allow-list. Any gridded data cube (containing a data cube definition file, see VII.M) can be given as 1st argument. The approximate bounding box of your study area needs to be given with coordinates in decimal degree (negative values for West/South). The shapefile ‘datacube-grid.shp’ is stored in the same directory as the data cube.
Module	|	force-tabulate-grid

Usage	|	force-tabulate-grid     datacube     bottom     top     left     right

