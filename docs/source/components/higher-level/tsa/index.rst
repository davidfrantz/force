.. _tsa:

Time Series Analysis
====================

The Time Series Analysis submodule provides out-of-the-box time series preparation and analysis functionality. 
The user can select from a number of spectral indices or unmix spectra using custom endmembers. 
The submodule is capable of extracting quality-controlled time series with a number of aggregation and interpolation techniques. 
Annual Land Surface Phenology metrics can be derived, and change and trend analyses can be performed on any of the generated time series. 
Many outputs of FORCE TSA are either considered as highly Analysis Ready Data (hARD), or even as highly Analysis Ready Data plus (hARD+).
hARD products are excellent input for many machine learning algorithms, e.g. for land cover / change classification purposes. 
hARD+ products can be directly used to fuel your research questions without any further processing. 


**A glimpse of what you get:**
 
.. image:: tsa.jpg

**Figure** Phenology-based change and trend analysis. 
Change, Aftereffect, Trend transformation (CAT) showing both long-term (30+ years) gradual and abrupt changes over Crete, Greece. 
The CAT transform was applied to the Value of Base Level (VBL) annual time series, which was itself derived by inferring Land Surface Phenology (LSP) metrics from dense time series of green vegetation abundance derived from linear spectral mixture analysis (SMA). 
[All this was done in one step using this submodule]


.. toctree::
   :maxdepth: 2

   process.rst
   param.rst
   endmember.rst
   format.rst

   