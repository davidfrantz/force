Processing
==========

Processing
The core module of FORCE CSO is force-cso, which generates per-pixel statistics about data availability. The processed images are written to the output directory given in the CSO parameter file (see VII.D for all available options).
Module	|	force-cso

Usage	|	force-cso     par-file

The parameter file needs to be given as 1st argument. All options need to be specified in the parameter file (see VII.D for all available options).

Quick guide
FORCE CSO is not covered by a scientific publication. Therefore, this section briefly summarizes the key functionality; see also section VII.G for all available options and possible keywords.
The analysis can be constrained to a certain area. The general extent in tile coordinates should be specified, and a tile white-list may be used.
The analysis can be performed with observations from all available sensors, or a subset of these. The spatial resolution on which the analysis is performed, needs to be given too. Image decimation/replication is taken care of (using nearest neighbor resampling).
Quality control is in full user control. All provided quality flags (see VI.B.5) can be used individually. Use this option rigorously!
The most important setting are the temporal properties. A temporal range needs to be specified in terms of years, e.g. 2017–2018. Clear Sky Observation statistics are generated for each time step as defined by the MONTH_STEP parameter (e.g. 6 months). Within each interval, the number of Clear Sky Observations are counted. In addition, several statistics based on the temporal difference between the observations are calculated, e.g. the maximum time between two observations in each interval. Note that the beginning and end of the intervals act as boundaries for this assessment and are also considered, for an example, see Fig. 7.
 
Fig. 7. Processing scheme of FORCE CSO.

This processing scheme reflects the fact, that a single measure of data availability might not yield representative results – depending on the application. A combined look at different statistics, or at a more uncommon metric, may provide more insight into the applicability of a specific method – or might explain uncertainties associated with a specific method. As an example, the data availability for the first and second half of 2018 as depicted in Fig. 7 is equal in terms of the number of observations, and the average time between observations. However, there are large differences in the maximum time between observations as the data are clumped in the first half. This may have important implications, e.g. for the detectability of harvesting events.
Currently available statistics are number of Clear Sky Observations, as well as the average, standard deviation, minimum, maximum, range, skewness, kurtosis, median, 25/75% quantiles, and interquartile range of the difference between CSOs. Happy data mining.

