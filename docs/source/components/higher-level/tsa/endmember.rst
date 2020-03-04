Endmember file
==============

This file is needed for Spectral Mixture Analysis in FORCE TSA, i.e. if INDEX = SMA. Suggested file extension is ‘.emb’. The file defines the endmember spectra that should be used for the SMA. The values need to be given in scaled reflectance (scale factor 10,000). The files should be without header, ended with an empty line, columns separated by white-space.
There should be one column for each endmember. If you want to apply shade normalization, the shade spectrum (photogrammetric zero or measured shade) needs to be in the last column. There should be as many lines as there are overlapping bands for the chosen set of sensors. See Table 8 on p. 65 for the matching table.
As an example, generating a fraction time series based on LND04, LND05, LND07, and LND08 requires 6-band endmembers (Landsat legacy bands). Generating a fraction time series based on LND08, SEN2A and SEN2B requires 6-band endmembers (Landsat legacy bands). Generating a fraction time series based on SEN2A and SEN2B requires 10-band endmembers (Sentinel-2 land surface bands). Generating a fraction time series based on sen2a and sen2b requires 4-band endmembers (Sentinel-2 high-res bands).
Example (Landsat legacy bands using vegetation, soil, rock and shade endmembers):
320 730 2620 0
560 1450 3100 0
450 2240 3340 0
3670 2750 4700 0
1700 4020 7240 0
710 3220 5490 0

