.. _lsm:

Landscape Metrics
=================

The Landscape Metrics submodule of HLPS computes area depending metrics or patches from several input features. This metrics characterize the structure of the landscape and can be used as training data for machine learning.
Landscape metrics are computed for pixels covererd by the foreground class, no metrics are computed for the pixels covered by the background class.
The foreground class is defined by the type of the threshold for each feature given in the parameter-file. All pixels that are greater than, lower than or equal to this threshold are interpreted as foreground class (in dependence of threshold type). 
The minimum size (in pixels) of an area to be considered as a patch will be defined in the parameter-file. Patches with fewer pixels will be omitted.

# Workflow
# A glimpse of what you get:


.. toctree::
   :maxdepth: 2

   param.rst
   format.rst
   