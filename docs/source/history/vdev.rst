.. _vdev:

Develop version
===============

- **FORCE HLPS**

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
