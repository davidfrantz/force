.. _about:

About
=====

FORCE is an all-in-one processing engine for medium-resolution Earth Observation image archives. FORCE uses the data cube concept to mass-generate Analysis Ready Data, and enables large area + time series applications. With FORCE, you can perform all essential tasks in a typical Earth Observation Analysis workflow, i.e. going from data to information.

FORCE natively supports the integrated processing and analysis of 

  * Landsat 4/5 TM, 
  * Landsat 7 ETM+, 
  * Landsat 8 OLI and 
  * Sentinel-2 A/B MSI.

Non-native data sources can also be processed, e.g. Sentinel-1 SAR data or environmental variables.

* The main features are

  * Integration of Landsat 4–8, and Sentinel-2 A/B as Virtual Constellation.

  * Data management of Landsat and Sentinel-2 Level 1 data + Download of Sentinel-2 data.

  * Near-realtime (NRT) processing capability.

  * Generation of Analysis Ready Data (ARD): Data Cubes
  
    * Advanced cloud and cloud shadow detection. 
    * Quality screening. 
    * Integrated atmospheric and topographic correction: one algorithm for all sensors. 
    * Adjacency effect correction. 
    * BRDF correction. 
    * Resolution merge of Sentinel-2 bands: 20m –> 10m. 
    * Co-registration of Sentinel-2 images
    * Data cubing: reprojection / gridding.

  * Generation of highly Analysis Ready Data (hARD): Large area. Gap free. Easy to use. Ideal input for Machine Learners!
  
    * Best Available Pixel (BAP) composites. 
    * Phenology Adaptive Composites (PAC). 
    * Spectral Temporal Metrics (STM)
    * Time Series generation
    * Time series folding
    * Time series interpolation
    * Texture metrics
    * Landscape metrics

  * Generation of highly Analysis Ready Data plus (hARD+). Open in a GIS and analyze!
  
    * Land Surface Phenology (LSP)
    * Trend analysis
    * Change, Aftereffect, Trend (CAT) analysis
    * Machine Learning regression
    * Machine Learning classification

  * Detailed data mining of the Clear Sky Observation (CSO) availability.

  * Data Fusion. 
  
    * Improving spatial resolution of coarse continuous fields: MODIS LSP –> medium resolution LSP. 
    * Improving spatial resolution of lower resolution ARD using higher resolution ARD: 30m Landsat –> 10m using Sentinel-2 targets

