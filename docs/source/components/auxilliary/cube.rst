.. _aux-cube:


force-cube
==========

Usage
^^^^^

.. code-block:: bash

    Usage: force-cube [-hvirsantobj] input-file(s)

    optional:
    -h = show this help
    -v = show version
    -i = show program's purpose
    -r = resampling method
         any GDAL resampling method for raster data, e.g. cubic (default)
         is ignored for vector data
    -s = pixel resolution of cubed data, defaults to 10
    -a = optional attribute name for vector data. force-cube will burn these values into the
         output raster. default: no attribute is used;
         a binary mask with geometry presence (1) or absence (0) is generated
    -l = layer name for vector data (default: basename of input, without extension)
    -n = output nodate value (defaults to 255)
    -t = output data type (defaults to Byte; see GDAL for datatypes; but note that FORCE HLPS
         only understands Int16 and Byte types correctly)
    -o = output directory: the directory where you want to store the cubes defaults to current
         directory 'datacube-definition.prj' needs to exist in there
    -b = basename of output file (without extension) defaults to the basename of the input-file
         cannot be used when multiple input files are given
    -j = number of jobs, defaults to 'as many as possible'

    mandatory:
    input-file(s) = the file(s) you want to cube


