.. _vdev:

Develop version
===============

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

- **FORCE HLPS**

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
