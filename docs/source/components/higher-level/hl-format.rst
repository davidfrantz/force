.. _hl-format:

Output format
=============

File format
^^^^^^^^^^^

Higher level images are generated with signed 16bit datatype in one of the following preconfigured formats:

.. include:: ../../../_static/files/format_gtiff.txt
  :code: shell

.. include:: ../../../_static/files/format_cog.txt
  :code: shell

.. include:: ../../../_static/files/format_envi.txt
  :code: shell

GTiff is the default and recommended format.

Note that due to GDAL constraints, it is not possible to use COGs with sub-tile processing chunks (i.e. ``CHUNK_SIZE`` smaller than ``TILE_SIZE``).
So, just don't do that. If you want to use COGs, set the chunk size to the tile size! 
You need enough memory to process a full tile at once, though. 
Alternatively, use GTiff as output format, which works with sub-tile chunks, and the new configuration is actually pretty comparable to COGs.

Users may also use a custom format by providing GDAL configuration options in the same format as shown above.

FORCE versions older than ``3.8.01-dev:::2025-09-17_14:04:58`` used a different GeoTiff configuration.
If you want to create GeoTiff files with the old configuration, you may use following settings in a custom format.
Note that you need to adapt the block sizes to your tile size, such that you have ``BLOCKXSIZE`` being the tile size in x direction and ``BLOCKYSIZE`` being a divisor of the tile size in y direction, usually 10 blocks.

.. include:: ../../../_static/files/format_gtiff_old.txt
  :code: shell


Metadata
^^^^^^^^

Metadata are written to all output products.
For GeoTiff format, the metadata are written into the GeoTiff file.
If the metadata is larger than allowed by the GeoTiff driver, the excess metadata will be written to an "auxiliary metadata" file with ``.aux.xml`` extension.
FORCE-specific metadata will be written to the FORCE domain, and thus are probably not visible unless the FORCE domain (or all domains) are specifically printed:

.. code-block:: bash

  gdalinfo -mdd all some-higher-level-product.tif

