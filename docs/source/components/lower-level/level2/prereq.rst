Prerequisites
=============


Tile white-list
This file is optional, and may be used to suppress the output of tiles that are not in your study area. This option was implemented to decrease the volume of the processed data. The file extension is ‘.til’. The file is specified with FILE_TILE in the various parameter files. The file must be prepared as follows: the 1st line must give the number of tiles for which output should be created. The corresponding tile IDs must be given in the following lines, one ID per line; end with an empty line. The sorting does not matter.
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
A digital elevation model (DEM) is recommended for FORCE L2PS (it is used for cloud detection, topographic and atmospheric correction). The user should provide a DEM that covers the complete study area at adequate spatial resolution. The DEM must be specified in the parameter file (see section VII.A). There is not really a restriction on the source, projection or resolution; the file format must be supported by GDAL. The DEM will be warped to the extent and resolution of the processed image using bilinear resampling. The DEM needs to be provided in meters above sea level with nodata  32767. 
The user can choose to process without a DEM; in this case the surface is assumed flat @ z = 0m. Topographic correction cannot be used without a DEM. The quality of atmospheric correction and cloud /cloud shadow detection will suffer without DEM.

WVDB
Water Vapor look-up-tables
The usage of a spatially and temporally explicit water vapor database for atmospheric correction in FORCE L2PS is optional for Landsat data. Water vapor LUTs are ignored for Sentinel-2 data, but should be considered for Landsat (especially for the TM-type of sensors). The directory containing the water vapor look-up-tables is specified with the DIR_WVPLUT in the Level 2 parameter file (see section VII.A.1). However, the algorithm can also be parameterized with a global water vapor value, see WATER_VAPOR.
FORCE WVDB provides a software component to generate such a database on the basis of MODIS MOD05/MYD05 data; see section VI.H. Note that the initial build of the water vapor database may need some time. However, the user may also use other data sources and/or tools to generate the required tables. Or you can download an application-ready database from https://doi.org/10.1594/PANGAEA.893109 (global coverage, 2000–July 2018).
There are two sets of tables that can/need to be generated:
	Daily tables are optional, but recommended. They contain a water vapor value for each coordinate, and there is one table for each day.
	Climatology tables are mandatory (unless DIR_WVPLUT = NULL). L2PS uses the climatology tables if a daily table is unavailable or if there is a fill value in the daily table. Therefore, the minimum requirement is to prepare the 12 climatology tables, one for each month.
Daily tables
For each date, one file can be prepared. The file naming is WVP_YYYY-MM-DD.txt; e.g. WVP_2003-08-24.txt. The files are four column tables with no header, separated by white-space. One line per coordinate; ended with an empty line. The coordinate closest to the scene center will be selected, and the corresponding value will be retrieved. 
Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by water vapor (3rd column), and three-digit source (4th column). The fill value is 9999 and TBD for source. The generation of daily tables is not mandatory, but highly recommended. If there is no table for a specific day, or if there is a fill value in the table, the corresponding climatology table is used instead.
Example:
17.2642002 -14.4588003 2.448023 MYD
16.9421997 -15.9028997 2.189836 MYD
20.6735001 -13.0142002 9999.000 TBD
20.3544006 -14.4588003 2.427723 MOD
20.0323009 -15.9028997 2.499933 MOD
… [file truncated]

Climatology tables
12 climatology tables must be prepared, one per month. The file naming is WVP_0000-MM-00.txt; e.g. WVP_0000-06-00.txt. The files are five column tables with no header, separated by white-space. One line per coordinate; ended with an empty line. The coordinate closest to the scene center will be selected, and the corresponding value will be retrieved. 
Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by long-term water vapor average (3rd column), long-term standard deviation of water vapor (4th column) and number of valid observations used for averaging (5th column). The generation of climatology tables is mandatory (unless DIR_WVPLUT = NULL).
Example:
96.4300 34.6138 1.205356 0.398807 446
96.0306 33.1801 1.360043 0.399460 447
95.6409 31.7452 1.442830 0.350363 425
95.2598 30.3093 1.642989 0.276430 311
94.8869 28.8723 4.018294 0.812506 149
94.5214 27.4344 6.426344 0.724956 123
… [file truncated]


AOD

AOD look-up-tables
These Look-up-Tables are optional, and may be used to override FORCE L2PS’ internal AOD estimation or to provide backup values if the internal AOD estimation failed for any reason. Potential usages are to employ an AOD climatology or a fixed AOD. The directory (containing the LUTs) is specified with DIR_AOD in the Level 2 parameter file (see section VII.A.1).
For each DOY, one file needs to be prepared (you should prepare 366 files). The file naming is AOD_DOY.txt; e.g. AOD_076.txt. The files are five column tables with no header, separated by white-space. One line per coordinate; ended with an empty line. The coordinate closest to the scene center will be selected, and the corresponding AOD will be retrieved.
Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West), followed by three Ångström coefficients (3rd – 5th column; logarithmic formulation, see below). The first coefficient is the turbidity coefficient (a0), the second coefficient is the Ångström exponent (a1), and the third coefficient can be used to describe spectral curvature in AOD (a2). The spectral curvature can be disabled with a2 = 0; in this case the formulation simplifies to the classic Ångström equation. AOD for any given wavelength is retrieved using following equation: 
ln⁡〖τ_a 〗=a_0+a_1∙ln⁡λ+a_2∙(ln⁡λ )^2	(1)


Coordinate file
This file is needed for some tools, e.g. FORCE WVDB. Suggested file extension is ‘.coo’. The file defines coordinates that should be processed with some functionality. The files are two column tables with no header, separated by white-space. One line per coordinate; ended with an empty line. Longitude (1st column) and latitude (2nd column) need to be given as geographic coordinates in decimal degree (negative values for South/West).
Example:
17.2642 -14.4588
16.9422 -15.9029
20.6735 -13.0142
20.3544 -14.4588
20.0323 -15.9029
… [file truncated]



COREG


