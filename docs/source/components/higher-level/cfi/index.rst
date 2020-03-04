Improving the Spatial Resolution of Land Surface Phenology
==========================================================

FORCE ImproPhe is intended to increase the spatial resolution of coarse continuous fields. It was originally developed to refine Land Surface Phenology metrics derived from MODIS, using sparse Landsat data as spectral and multi-temporal targets for data fusion. Regarding phenology, it can be used to obtain a Landsat-like phenology even in areas / during times when Landsat data alone is insufficient (in terms of temporal density). FORCE permits the use of Landsat and/or Sentinel-2 data as target datasets for the improPhement. ImproPhe can also be applied to other coarse resolution data (for best results, some link to spectral-temporal land surface processes should exist – e.g. increasing the spatial resolution of rainfall data won’t work).
FORCE also features a tool to increase the spatial resolution of lower resolution ARD using higher resolution ARD using the ImproPhe algorithm (spectral + multi-temporal parameterization, see VI.G). The ImproPhe code is also implemented as resolution merge option in FORCE L2PS (VI.B) to increase the spatial resolution of Sentinel-2’s 20m bands to 10m (spectral + mono-temporal implementation).
This section summarizes the usage of FORCE ImproPhe, its helper programs and the output format.
FORCE ImproPhe can only be used with Level 2 ARD.



A glimpse of what you get:

 
Fig. 8. Coarse resolution (500m) and ImproPhed (30m) LSP metrics.
Rate of Maximum Rise (R), Integral of Green Season (G), and Value of Early Minimum (B) phenometrics for a study site in Brandenburg, Germany. Using the ImproPhe algorithm, the LSP metrics were improved to 30m spatial resolution using Landsat and (degraded) Sentinel-2 targets.
[Data were fused using force-improphe]

.. toctree::
   :maxdepth: 2

   param.rst
   process.rst
   format.rst

   