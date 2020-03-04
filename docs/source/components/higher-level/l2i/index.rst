FORCE L2IMP – Level 2 ImproPhe
==============================

FORCE Level 2 ImproPhe (L2IMP) is intended to increase the spatial resolution of lower resolution Level 2 ARD using higher resolution Level 2 ARD, e.g. to improve the spatial resolution of 30m Landsat imagery to 10m using Sentinel-2 data as targets. This only works for years where both data sources exist. The data fusion is performed with the ImproPhe algorithm. Note that this module is heavy on processing time.
This section summarizes the usage of FORCE L2IMP, its helper programs and the output format.
FORCE L2IMP can only be used with Level 2 ARD.



A glimpse of what you get:

 
Fig. 9. 30m Landsat ARD, and ImproPhed 10m Landsat ARD.
The figure shows image subsets from North Rhine-Westphalia, Germany. Using the ImproPhe algorithm, the spatial resolution was improved to 10m using multi-temporal Sentinel-2 A/B high-res bands as prediction targets.
[Data were fused using force-l2imp]


.. toctree::
   :maxdepth: 2

   param.rst
   process.rst
   format.rst

   