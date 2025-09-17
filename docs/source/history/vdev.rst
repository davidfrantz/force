.. _vdev:

Develop version
===============

**Read through the changelog closely. In this version, there are some important changes that may affect your workflows!**

- **FORCE L2PS**

  - When merging chips of the same day/sensor to avoid data redundancy, the merging 
    strategy for the QAI files has been revised. Before, it was just updating the images,
    i.e., merely overlaying them on top of each other. Now, the QAI files are merged feature-based
    on a custom logic such that the most restrictive QAI value is kept. This ensures that reproducibility
    is guaranteed, independent of the order in which the chips are merged. 
  - It is now possible to specify the resampling method for the DEM.
    The parameter ``DEM_RESAMPLING`` can be set to ``NN`` (Nearest Neighbor), ``BL`` (Bilinear), or ``CC`` (Cubic Convolution).
    The default is ``BL``. This allows users to choose the resampling method that best fits their needs.
    Thanks to Oleg Zheleznyy for the suggestion.
  - When co-registering L7 SLC-Off imagery, a big portion of the data failed. 
    This issue was addressed by altering the no-data filtering logic in the pyramid building process of LSReg.
    Thanks to Oleg Zheleznyy for reporting this issue.
  - Updated COG default parameters. COG has been set as the default output format for L2PS 
    (note: HLPS still uses GTiff as default)
  - The default format for L2PS has been changed from ``GTiff`` to ``COG``.
    This affects the output format of the L2PS products. COGs are more efficient for cloud storage and access,
    making them a better choice for modern geospatial workflows. Users can still specify ``GTiff`` if needed.
  - The default creation options for ``GTiff`` files  have been updated to include ``COMPRESS=ZSTD`` and a gridded tile layout.
    Refer to https://github.com/davidfrantz/force/discussions/376 for more details.
    These options improve the compression and efficiency of the GTiff files. 
    Users can still use the previous parameterization through custom GDAL options if needed.
  - This choice has resulted in a decoupling of internal file layout and higher level processing (more below and in the issue above).
    As a result, the ``BLOCK_SIZE`` parameter has been deprecated and is no longer used in L2PS.
    Also, the block size is no longer part of the data cube definition.
    This comes with a new file format for the data cube definition in tag and value notification.
    **IMPORTANT: FORCE has backward compatibility for existing data cubes, but the new format is used
    when creating a new data cube. Existing data cubes do not need to be converted. 
    However, older software versions will not be able to read data cubes created with the new format!**
  - The ``TILE_SIZE`` has been changed into a two-dimensional parameter.
    This allows users to specify different tile sizes in x and y direction, which can be useful for certain applications.
    The sizes should be given in the unit of the data cube's coordinate reference system (CRS), usually meters.
    The sizes should be a multiple of the processing resolution. 
    Note that it is not generally recommended to make use of non-square tiles, but hey, you can now if you want to.

- **FORCE HLPS**

  - The internal file layout (blocks) has been decoupled from the sub-tile processing chunks. This means that the
    penalty for using a non-optimal block size became less significant.
    See the details described in https://github.com/davidfrantz/force/discussions/376.
    This has been implemented throughout FORCE. To make this clearer, the term "block" is now only used
    in the context of the internal file layout, whereas "chunk" is used for the processing units.
    These chunks can now be specified by the user via the ``CHUNK_SIZE`` parameter. 
    The old ``BLOCK_SIZE`` parameter has been deprecated and is no longer used in HLPS.
    Note that ``CHUNK_SIZE`` requires two values, i.e., the chunk size in x and y direction.
    The sizes should be given in the unit of the data cube's coordinate reference system (CRS), usually meters.
    The sizes should be a multiple of the processing resolution and the tile size should be divisible by the chunk size.
    This is to ensure that the chunks fit neatly into the tiles. This is the same behaviour as with the previous ``BLOCK_SIZE`` parameter.
    The only usage difference is that the chunk size can now follow a two-dimensional approach, i.e., different sizes in x and y direction,
    which e.g. can make processing way more efficient when using masks.
    Further note, that the chunk size must be greater than 0 and less than or equal to the tile size (in the specific direction).

    **IMPORTANT: When working with an old data cube, you should ensure that the chunk X-size is equal to the tile X-size to avoid a huge penalty on efficiency.**

  - It became apparent that using COGs is not possible when using sub-tile processing chunks. 
    So, just don't do that. If you want to use COGs, set the chunk size to the tile size!
    You need enough memory to process a full tile at once, though.
    https://github.com/davidfrantz/force/issues/374
    Alternatively, use GTiff as output format, which works with sub-tile chunks, and the new configuration is actually
    pretty comparable to COGs.

  - in ``force-higher-level``, UDF sub-module:
    a new feature was added to the UDF module, which allows users to add auxiliary products
    to the data array that is passed to the UDF. 
    The user can specify which auxiliary products to use in the configuration file via the new 
    ``REQUIRE_AUX_PRODUCTS`` parameter. The auxiliary products are specified as a white-space separated list,
    e.g. ``REQUIRE_AUX_PRODUCTS = DST VZN AOD``. Custom products may also be specified (*Int16!*), thus you can invent 
    and use new tags. An auxiliary product is a product should always accompany the main product (usually ``BOA``).
    In the UDF, the auxiliary products are appended to the data array, thus increasing the number of bands.
    The bandnames of the auxiliary products are set to the product name, e.g. ``DST`` for the DST product.
    If no auxiliary products are wanted, the user can set ``REQUIRE_AUX_PRODUCTS = NULL``.

  - in ``force-higher-level``, UDF sub-module:
    the ``REQUIRE_AUX_PRODUCTS`` mechanism has been implemented here as well. 
    You may use the ``DST``, ``HOT``, and ``VZN`` products.
    Before, the usage of a specific product was solely managed by using a corresponding score > 0. 
    To prevent accidental use of a product, the user must now explicitly specify the product in addition to the score.

  - in ``force-higher-level``, all ARD-based modules:
    a new parameter ``DATE_IGNORE_LANDSAT_7`` was added. During the last years of Landsat 7's life,
    the satellite was slowly de-orbited, which caused the acquisition times to slowly shift away from
    the nominal time. Operational production continued. This parameter allows the user to ignore
    the Landsat 7 data after a specific point in time. The default is ``2099-12-31``, which just means 
    that all Landsat 7 data will be used.

  - in ``force-higher-level``, all modules:
    a new parameter ``FAIL_IF_EMPTY`` was added by Florian Katerndahl.
    If set to ``TRUE``, the module will fail if no input data is found for the specified configuration, 
    e.g. time range, or if no output was written. This is meant to be used in complex workflows, where
    one want the program to signal an error condition. 
    The default is ``FALSE``, which is better suited for production runs, 
    as e.g. an NRT application may not have data for a specific day or region.
    In this case, only a warning is displayed with some hints on how to resolve potential issues.

--  **FORCE AUX**

  - ``force-mosaic`` was overhauled:
    The tool should now be much more efficient and faster as it splits the working load into smaller chunks
    instead of relying a singular find command. As suggested by Max Freudenberg, the tool can now be used
    if your data cube does not have files in *.tif or *.dat extentsion. The user can now specify the
    file extension with the ``-e`` option. The default is ``tif``, whereas ``dat`` is no longer part of the default.
    The tool can now also be used on write-protected data cubes by specifiying an output directory that no longer
    needs to be a subdirectoy of the data cube. The default is still a `mosaic` folder within the datacube.
    
  - added a new tool ``force-virtual-datacube``:
    This tool allows users to create a virtual datacube from a physical datacube. It is useful for creating
    a virtual representation of the data without duplicating the actual files, thus saving disk space.
    It can be used to combine multiple datacubes into a single virtual dataset, which can be useful for analysis.
    The tool can be used with various options to customize the output, such as specifying the pattern of files
    to include and whether to overwrite existing files.

  - Minor adjustments have been made in a couple of AUX tools to reflect the changes in handling chunks.
  