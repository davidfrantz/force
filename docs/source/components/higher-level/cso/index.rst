.. _cso:

Clear Sky Observations
======================

FORCE Clear Sky Observations (FORCE CSO) is intended for data availability mining. For given time steps (e.g. 3 months), per-pixel statistics about data availability are calculated, i.e. number of CSOs, and average (standard deviation, min, max, etc.) days between consecutive CSOs. A per-tile summary is written to the image header and a per-project summary is printed to screen. This section summarizes the usage of FORCE CSO, its helper programs and the output format.
FORCE CSO can only be used with Level 2 ARD.



A glimpse of what you get:

 
Fig. 6. Quarterly Clear Sky Observation statistics.
The CSO statistics were computed for the 2015 Landsat 7–8 acquisitions over Turkey.
[CSO statistics were generated with force-cso, then mosaicked with force-mosaic]


.. toctree::
   :maxdepth: 2

   param.rst
   process.rst
   format.rst

   