.. _vdev:

Develop version
===============

FORCE-dev
---------

Master release: TBA

* **General changes**

  * FORCE no longer uses the terms white-list, master, and slave.
    These were replaced (in code and docs) with allow-list, base, and target.

  * Bandnames were added to all output products.

* **Changes for Docker**

  * In Docker, retrieving the user credentials was problematic, i.e. the user/password for ``force-level2-sentinel2`` and ``force-lut-modis``.
    We have now solved it by adding an environment variable.
    These two programs will look for an environment variable ``FORCE_CREDENTIALS``, which the Docker user can specify with s.th. like this: 

    .. code-block:: bash

       docker run --env FORCE_CREDENTIALS=/app/credentials fegyi001/force env
    
    In this directory, you should place the ``.scihub`` and ``.laads`` files.
    
    If the environment variable is not set, FORCE will look in the user's home directory (as before).
    Thus, for non-Docker users, nothing changes (although you can choose the environment variable, too).

    Thanks to Haili Hu and Gergely Padányi-Gulyás for developing this solution.
    
* **FORCE L2PS**

  * Due to the ban of the term "master", the ``DIR_MASTER`` and ``MASTER_NODATA`` tags have changed to ``DIR_COREG_BASE`` and ``COREG_BASE_NODATA``.

* **FORCE HIGHER LEVEL**

  * added new sub-module to force-higher-level:
  
    library-completeness LIB. 
    This submodule takes a feature table (e.g. spectral library used for training a machine learning classifier), and tests each feature vector against the image features.
    The output is a minimum MAE map, which indicates if your library is complete - or if there are e.g. landcovers that you do not have any samples for (likely your classification/regression will be worse there).
    It is suggested to not use this sub-module on the native spatial resolution, but on 100m or similar.
    force-parameter has a new option to generate a LIB parameter file.
    Thanks to Franz Schug for prototyping this method.

  * in force-higher-level, most sub-modules: 
  
    Added a new parameter ``OUTPUT_EXPLODE``.
    If FALSE, multi-band images are written (as before).
    If TRUE, the output is exploded into single-band images.
    Note that this can result in an extremely large number of files.

  * in force-higher-level, various sub-modules: 
  
    Explicitly added the nodata value for output products, which formerly caused strange behaviour when there only was nodata within the processing mask of one block.
    Thanks to Stefan Ernst for reporting this issue.

  * in force-higher-level, sampling sub-module: 
  
    The limitation of only having one response variable was lifted.
    Accordingly, the input table can have more than 3 columns, i.e. 1) X-, 2) Y-coordinates, and 3+) response variables.
    The output response file will hold all response variables.
    Some improvements were made w.r.t. performance, i.e. the input table is only read once, and a "we-already-have-sampled-this-coordinate" is used to skip finished samples.

  * in force-higher-level, CSO sub-module: 
  
    Fixed a critical memory error related to the CSO nodata value.

  * in force-higher-level, machine learning sub-module, random forest classification:
  
    Random Forest class probabilities can now be output. 
    The Random Forest classification margin can now be output.
    Two new parameters were added: ``OUTPUT_RFP`` & ``OUTPUT_RFM``.
    Thanks to Benjamin Jakimow for suggesting this improvement.
    
  * in force-higher-level, TSA sub-module: 
  
    Added additional spectral indices: Normalized Difference Tillage Index, and Normalized Difference Moisture Index
    Thanks to Benjamin Jakimow for suggesting this improvement.

* **FORCE AUX**

  * new program force-synthmix:
  
    Andreas Rabe has provided a SynthMix program!
    SynthMix can be used to generate training data for machine learning regression to map sub-pixel fractions of land cover, tree cover etc.
    SynthMix is a very elegant method to create a proper training dataset, makes it much easier to generate training data for fractional cover, and needs very few input data (as opposed to traditional methods).
    For details, see. `Okujeni et al. "Support vector regression and synthetically mixed training data for quantifying urban land cover." Remote Sensing of Environment 137 (2013): 184-197. <https://www.sciencedirect.com/science/article/pii/S0034425713002009>`_. 
    For a ecent example, see `Schug et al. "Mapping urban-rural gradients of settlements and vegetation at national scale using Sentinel-2 spectral-temporal metrics and regression-based unmixing with synthetic training data." Remote Sensing of Environment 246 (2020): 111810 <https://www.sciencedirect.com/science/article/pii/S0034425720301802>`_
    force-parameter has a new option to generate a SynthMix parameter file.

  * new program force-procmask:
  
    This program can generate processing masks from cubed, continuous input images, e.g. to generate a mask with all pixels that have NDVI > 0.8
  
  * new program force-tile-extent:
  
    This program takes a polygon vector file (e.g. shapefile of a country), and suggests a processing extent for higher-level processing (``X_TILE_RANGE`` & ``Y_TILE_RANGE``)
    It further gives a recommendation whether you should use a tile allow-list.
    This list is also generated.

  * in force-train:

    The response file can now have multiple columns, i.e. different variables.
    A new tag ``RESPONSE_VARIABLE`` is used to select the variable, which should be used for training the model.

  * in force-cube:
  
    If a resulting image is completely nodata, it will automatically be removed.
    
    