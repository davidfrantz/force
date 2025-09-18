.. _cfi-format:

Output Format
=============

Data organization
^^^^^^^^^^^^^^^^^

The data are organized in a gridded data structure, i.e. data cubes.
The tiles manifest as directories in the file system, and the images are stored within. For mosaicking, use force-mosaic.

.. seealso:: 

  Check out this `tutorial <https://davidfrantz.github.io/tutorials/force-datacube/datacube/>`_, which explains what a datacube is, how it is parameterized, how you can find a POI, how to visualize the tiling grid, and how to conveniently display cubed data.


Data Cube definition
^^^^^^^^^^^^^^^^^^^^

The spatial data cube definition is appended to each data cube, i.e. to each directory containing tiled datasets, see :ref:`datacube-def`.


File format
^^^^^^^^^^^

Refer to :ref:`hl-format` for details on the file format and metadata.


Naming convention
^^^^^^^^^^^^^^^^^

Following 21-digit naming convention is applied to all output files:

Example filename: 2017_IMPROPHE_IGS.tif

**Table 1:** Naming convention

+----------------+---------+---------------------------------------------------------+
+ Digits         + Description                                                       +
+================+=========+=========================================================+
+ 1–4            + Year                                                              +
+----------------+---------+---------------------------------------------------------+
+ 6–13           + Processing Type                                                   +
+                +---------+---------------------------------------------------------+
+                + IMPROPHE                                                          +
+----------------+---------+---------------------------------------------------------+
+ 15–17          + Product Tag                                                       +
+                +---------+---------------------------------------------------------+
+                + XXX     + These 3-digit tags are specified in the parameter file  +
+----------------+---------+---------------------------------------------------------+
+ 19–21          + File extension                                                    +
+                +---------+---------------------------------------------------------+
+                + tif     + image data in compressed GeoTiff format                 +
+                +---------+---------------------------------------------------------+
+                + dat     + image data in flat binary ENVI format                   +
+                +---------+---------------------------------------------------------+
+                + hdr     + metadata                                                +
+----------------+---------+---------------------------------------------------------+


Product type
^^^^^^^^^^^^

* High resolution continuous fields
