.. _tsa:

Time Series Analysis
====================

FORCE Time Series Analysis (FORCE TSA) is intended to provide out-of-the-box time series preparation and analysis functionality. The user can select from a number of spectral indices or unmix spectra using custom endmembers. FORCE TSA is capable of extracting quality-controlled time series with a number of aggregation and interpolation techniques. Annual Land Surface Phenology metrics can be derived, and change and trend analyses can be performed on any of the generated time series. Many outputs of FORCE TSA are considered as highly Analysis Ready Data plus (hARD+), meaning that generated products can be directly used to fuel your research questions without any further processing. This section summarizes the usage of FORCE TSA, its helper programs and the output format.
FORCE TSA can only be used with Level 2 ARD.



A glimpse of what you get:

 
Fig. 4. Phenology-based change and trend analysis. 
Change, Aftereffect, Trend transformation (CAT) showing both long-term (30+ years) gradual and abrupt changes over Crete, Greece. The CAT transform was applied to the Value of Base Level (VBL) annual time series, which was itself derived by inferring Land Surface Phenology (LSP) metrics from dense time series of green vegetation abundance derived from linear spectral mixture analysis (SMA). 
[All this was done in one step using force-tsa; then mosaicked using force-mosaic]


.. toctree::
   :maxdepth: 2

   process.rst
   param.rst
   endmember.rst
   format.rst

   