.. _aux-tile-extent:

force-tile-extent
=================

force-tile-extent creates a tile allow-list from a given area of interest (vector file). 

Usage
^^^^^

.. code-block:: bash
    
    force-tile-extent input-vector datacube-dir allow-list

       input-file:   a polygon vector file
       datacube-dir: the directory of a datacube;
                     datacube-definition.prj needs to exist in there
       allow-list:   a tile allow-list to restrict the processing extent.
                     This file will be written