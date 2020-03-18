.. _level2-depend:

Data dependencies
=================


File queue
^^^^^^^^^^

A :ref:`queue` is required for :ref:`force-bulk`. 
The queue us not needed for :ref:`force-core` and :ref:`force-wrapper`.

The queue is automatically generated when using the :ref:`l1as`.


Tile white-list

*better put in higher-level?*

This file is optional, and may be used to suppress the output of tiles that are not in your study area.
This option was implemented to decrease the volume of the processed data.
The file extension is ‘.til’.
The file is specified with FILE_TILE in the various parameter files.
The file must be prepared as follows: the 1st line must give the number of tiles for which output should be created.
The corresponding tile IDs must be given in the following lines, one ID per line; end with an empty line.
The sorting does not matter.
The tool force-tabulate-grid can be used to create a shapefile of the grid, which in turn can be used to determine the desired tiles (see section VI.I.4).
Example:
4524
X0044_Y0014
X0044_Y0015
X0045_Y0013
X0045_Y0014
X0045_Y0015
X0045_Y0016
X0045_Y0017
X0045_Y0018
… [file truncated]



Digital Elevation Model
A digital elevation model (DEM) is recommended for FORCE L2PS (it is used for cloud detection, topographic and atmospheric correction).
The user should provide a DEM that covers the complete study area at adequate spatial resolution.
The DEM must be specified in the parameter file (see section VII.A).
There is not really a restriction on the source, projection or resolution; the file format must be supported by GDAL.
The DEM will be warped to the extent and resolution of the processed image using bilinear resampling.
The DEM needs to be provided in meters above sea level with nodata  32767.

The user can choose to process without a DEM; in this case the surface is assumed flat @ z = 0m.
Topographic correction cannot be used without a DEM.
The quality of atmospheric correction and cloud /cloud shadow detection will suffer without DEM.


WVDB

see :ref:`wvdb`.

AOD

AOD look-up-tables
These Look-up-Tables are optional, and may be used to override FORCE L2PS’ internal AOD estimation or to provide backup values if the internal AOD estimation failed for any reason.
Potential usages are to employ an AOD climatology or a fixed AOD.
The directory (containing the LUTs) is specified with DIR_AOD in the Level 2 parameter file (see section VII.A.1).
For each DOY, one file needs to be prepared (you should prepare 366 files).
The file naming is AOD_DOY.txt; e.g.
AOD_076.txt.
The files are five column tables with no header, separated by white-space.
One line per coordinate; ended with an empty line.
The coordinate closest to the scene center will be selected, and the corresponding AOD will be retrieved.
Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by three Ångström coefficients (3rd – 5th column; logarithmic formulation, see below).
The first coefficient is the turbidity coefficient (a0), the second coefficient is the Ångström exponent (a1), and the third coefficient can be used to describe spectral curvature in AOD (a2).
The spectral curvature can be disabled with a2 = 0; in this case the formulation simplifies to the classic Ångström equation.
AOD for any given wavelength is retrieved using following equation: 
ln⁡〖τ_a 〗=a_0+a_1∙ln⁡λ+a_2∙(ln⁡λ )^2	(1)


Coordinate file
This file is needed for some tools, e.g.
FORCE WVDB.
Suggested file extension is ‘.coo’.
The file defines coordinates that should be processed with some functionality.
The files are two column tables with no header, separated by white-space.
One line per coordinate; ended with an empty line.
Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West).
Example:
17.2642 -14.4588
16.9422 -15.9029
20.6735 -13.0142
20.3544 -14.4588
20.0323 -15.9029
… [file truncated]



COREG


