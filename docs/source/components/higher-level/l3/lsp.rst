Land Surface Phenology
======================

A Land Surface Phenology (LSP) dataset may be input to generate phenology-adaptive composites – 
or sth. similar that is dependent on spatial variation of the target dates. 
This is optional and may be omitted if static composites or temporal statistics are used. 
The LSP dataset needs to be prepared in the same grid as the Level 2 data (i.e. in a mirrored data structure). 
Three (or more) images need to be prepared for each tile, i.e. 
seasonal parameters describing points in time (e.g. the timing of start of season, peak of season, end of season, ...). 
For compositing, a sequence of three images needs to be selected as temporal target, 
and the filenames must contain a unique ID (e.g. SOS, POS, EOS). 
The data are expected to be in ENVI standard or GeoTiff format. 
Each file is a multi-layer image with years as bands (the first year is specified in the Level 3 parameter file. 
Note that missing years are not allowed (use a fill band instead). 
The values should be in days relative to a custom starting point. 
Leap years are not taken into account and each year consists of 365 days.

