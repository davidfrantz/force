.. _lsm-format:

Output Format
=============

Data organization
^^^^^^^^^^^^^^^^^

The data are organized in a gridded data structure, i.e. data cubes.
The tiles manifest as directories in the file system, and the images are stored within.

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

The basename of the output files can be defined in the parameter-file. The basename will be appended by Module ID, product ID, and the file extension.


Product type
^^^^^^^^^^^^

* Landscape Metrics
  
  Currently available metrics are unique patch ID, weighted mean patch area, weighted mean fractal dimension index, edge density, number of patches and
  effective mesh size. Additionally, arithmetic mean, geometric mean, standard deviation, maximum value and the area of the foreground class within the kernel are available.
