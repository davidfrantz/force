.. _vdev:

Develop version
===============

- **FORCE HLPS**

  - Analogous to Python UDFs, R UDFs can now be used through FORCE. This applies both to
    the TSA and UDF submodules. Three new parameters were added: ``FILE_RSTATS``, ``RSTATS_TYPE``,
    ``OUTPUT_RSP``. At least two functions must be present in the UDF: ``force_rstats_init``, as well as
    ``force_rstats_pixel`` or ``force_rstats_block`` (depending whether ``RSTATS_TYPE`` is set to PIXEL
    or BLOCK).

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
  

- **FORCE AUX**

  - new auxilliary program `force-init`.
    This program will create a new project with reasonably named folders that
    can be used as a starting point for a new project with some suggestions 
    on organizing things. 
    Unused folders can be deleted and the naming and structure is a mere suggestion and by no 
    means prescriptive or mandatory.
    This tool is especially meant for beginners.

  - ``force-tabulate-grid`` has been updated to produce a properly named output file.

  - The CLI help of `force-tile-finder` has been corrected concerning the separator for the coordinates.

  .. -- No further changes yet.
