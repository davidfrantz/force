.. _vdev:

Develop version
===============

- **FORCE HLPS**

  - Analogous to Python UDFs, R UDFs can now be used through FORCE. This applies both to
    the TSA and UDF submodules. Three new parameters were added: ``FILE_RSTATS``, ``RSTATS_TYPE``,
    ``OUTPUT_RSP``. At least two functions must be present in the UDF: ``force_rstats_init``, as well as
    ``force_rstats_pixel`` or ``force_rstats_block`` (depending whether ``RSTATS_TYPE`` is set to PIXEL
    or BLOCK). To run PIXEL-type functions, you need to install the ``snow`` and ``snowfall`` packages. 
    To run the FORCE components that are now compiled against the R API (mostly force-higher-level, 
    but also force-qai-inflate), you need to provide two environment variables, i.e.
    ``R_HOME`` and ``LD_LIBRARY_PATH``. You can do it like this:

    - export R_HOME=$(R RHOME)
    - export LD_LIBRARY_PATH=$R_HOME/lib
    - force-higher-level parameters.prm

    In the pre-built Docker container, everything is pre-configured already.

  - Added a new parameter: ``STREAMING = FALSE`` can disable the streaming strategy of FORCE HLPS.
    When TRUE (previous behaviour, and still the default), FORCE will perform reading, computing and 
    writing at the same time. If FALSE, these operations are done after one another in sequential mode.
    Disabling streaming might be necessary for some UDFs that otherwise produce threading conflicts 
    with the internally used OpenMP functionality. When using R UDFs, FORCE will issue a warning and
    gracefully disable streaming.

  - Added a new parameter: ``PRETTY_PROGRESS``. HLPS  will display progress information on screen. 
    When TRUE (previous behaviour, and still the default), the progress information overwrites itself 
    to produce a pretty displayal. However, this can cause error messages (or printing in UDFs) to be 
    overwritten. If disabled (FALSE), the progress information will be simply be appended  on screen (stdout).
  
  - Removed warning message that no output is produced when using the sampling submodule.
    It is now checked whether the files are actually written and will only warn if 
    no input was detected or if no files were written. 
    The behaviour for all other submodules stays the same.

  - It is now possible to specify the product types that should be used. This is to give the user more freedom
    with respect to data sources to be used. 
    There are two new parameters:
    ``PRODUCT_TYPE_MAIN`` and ``PRODUCT_TYPE_QUALITY``. The main product is usually a reflectance product like ``BOA``.
    When using composites, you may use BAP. This can be anything, but make sure that the string can uniquely 
    identify your product. As an example, do not use ``LEVEL2`` as this will not filter products apropriately.
    Note that the product should contain the bands that are to be expected with the sensor used, e.g. 10 bands 
    when sensor is SEN2A. The quality product type should be a bit flag product like ``QAI``. When using composites, 
    you may use INF. This can be anything, but make sure that the product contains quality bit flags as outputted 
    by FORCE L2PS. As an exception, it is also possible to give ``NULL`` if you don't have any quality masks.
    In this case, FORCE will only be able to filter nodata values, but no other quality flags as defined with ``SCREEN_QAI``.
    Feature requested by Marcel Schwieder and Felix Lobert.

  - endmember, sampling, feature and response tables can be a csv now, i.e. "," has become an allowed column separator. 
    Tab and whitespace are still allowed, too.
    @German users, take care to use English csv settings!


- **FORCE AUX**

  - new auxilliary program `force-init`.
    This program will create a new project with reasonably named folders that
    can be used as a starting point for a new project with some suggestions 
    on organizing things. 
    Unused folders can be deleted and the naming and structure is a mere suggestion and by no 
    means prescriptive or mandatory.
    This tool is especially meant for beginners.

  - new auxilliary program `force-datacube-size`.
    This program prints the size of your datacube, per sensor and in total.

  - ``force-tabulate-grid`` has been updated to produce a properly named output file.

  - The CLI help of `force-tile-finder` has been corrected concerning the separator for the coordinates.

  .. -- No further changes yet.
