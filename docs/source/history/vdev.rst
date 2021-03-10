.. _vdev:

Develop version
===============

FORCE-dev
---------

Master release: TBA

* **FORCE HIGHER LEVEL**

  * in force-higher-level, LSM sub-module: 
    Franz Schug added a new metric: ``ARE``, which yields the pixel area of the foreground class.
    Note: this value is capped at 32767.

  * in force-higher-level, TSA sub-module:
    A segmentation fault occured when using the SAVI Index.
    This is now fixed.
    Thanks for Janos Steiner for reporting this issue.

  * in force-higher-level, TSA sub-module:
    Implemented the new kNDVI following Camps-Valls et al. 2021.
    Use with ``INDEX = kNDVI``.
    Sigma is fixed to 0.5*(NIR+RED).

* **FORCE WVDB**

  * We updated the ready-to-use, global water vapor database. 
    The dataset is comprised of daily global water vapor data for February 2000 to December 2020 for each land-intersecting Worldwide Reference System 2 (WRS-2) scene, as well as a monthly climatology that can be used if no daily value is available. 
    The dataset is freely available at `<https://doi.org/10.5281/zenodo.4468700>`_. 
    This dataset may relieve you of the burden to generate the water vapor database on your own.

#-- No changes yet, master is in sync with develop.
