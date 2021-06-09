.. _tile-finder:

Coordinate locator
==================

In case of the gridded data structure, a geographic coordinate can be converted to tile and pixel coordinates using force-tile-finder. This functionality is intended to spatially locate data. Any gridded data cube (containing a data cube definition file, see VII.M) can be given as 1st argument. Longitude and latitude must be given as 2nd and 3rd arguments with coordinates in decimal degree (negative values for West/South). The resolution needs to be given (4th argument) in order to relate coordinates to pixel positions.
Module	|	force-tile-finder

Usage	|	force-tile-finder     datacube     lon     lat     res

