Level 3 Compositing
===================

The FORCE Level 3 Processing System (FORCE L3PS) is intended to generate temporal aggregations of Level 2 data to provide seamless, gap free, and highly Analysis Ready Data (hARD) over very large areas. This includes pixel-based composites as well as spectral temporal statistics (e.g. average reflectance within compositing period). hARD are the optimal input for many machine learning algorithms, e.g. for land cover /change classification purposes. Pixel-based composites can either be static or phenology-adaptive. In the latter case, a Land Surface Phenology (LSP) dataset is required (see subsection). This section summarizes the usage of FORCE L3PS, its helper programs and the output format.
FORCE L3PS can only be used with Level 2 ARD.

submodules

A glimpse of what you get:

 
Fig. 3. Phenology Adaptive Composite (PAC) using Landsat 5–7.
The Best Available Pixel (BAP) composite (phenology-adaptive code: End of Season 2018) was computed for Angola, Zambia, Zimbabwe, Botswana and Namibia. 
[The composite was generated with force-level3, then mosaicked using force-mosaic] 


.. toctree::
   :maxdepth: 2

   process.rst
   param.rst
   lsp.rst
   format.rst

   