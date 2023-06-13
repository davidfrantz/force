.. _aux-cube-init:

force-cube-init
===============

force-cube-init creates a datacube-definition of a certain projection.

Usage
^^^^^

.. code-block:: bash

    force-cube-init [-h] [-v] [-i] [-d datacube-dir] [-o lon/lat]
          [-t tile-size] [-c chunk-size] projection

    -h  = show this help
    -v  = show version
    -i  = show program's purpose

    -d datacube-dir = output directory for datacube definition
       default: current working directory

    -o lon,lat = origin coordinates of the grid
       use geographic coordinates!
       longitude is X!
       latitude  is Y!
       default: -25,60, is ignored for pre-defined projections!

    -t tile-size
       default: 30km, is ignored for pre-defined projections!

    -c chunk-size
       default: 3km, is ignored for pre-defined projections!

    Positional arguments:
    - Projection (custom WKT string or built-in projection