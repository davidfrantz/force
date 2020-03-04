Output Format
=============

Quicklooks
Quicklooks of the processed Level 2 data can be generated using force-quicklook-level2. It searches files that match the pattern ‘*BOA*.dat/tif’ in the directory given as 1st argument (typically the output directory of FORCE L2PS), and creates a quicklook in the directory given as 2nd argument. The data organization of the Level 2 archive will be mirrored in the quicklook archive (tiles or original spatial reference system). Existing quicklooks are skipped (to enable efficient and scheduled processing, e.g. if called from a cronjob). The true color quicklooks have a fixed stretch from 0–maxval (3rd argument). The upper limit of the stretch needs to be given in scaled reflectance (scaling factor 10,000), e.g. 1500. This value can be decreased/increased for very dark/bright landscapes. The quicklooks are generated with gdal_translate in JPEG format and have a fixed size of 256 x 256 px.
Module	|	force-quicklook-level2

Usage	|	force-quicklook-level2     L2-archive     QUICKLOOK-archive     maxval

Data organization
Depending on parameterization, the output data are organized according to their original spatial reference system (WRS-2 frames / MGRS zones) or are provided in a gridded data structure as ARD (strongly recommended!). The tiles (or original reference system) manifest as directories in the file system, and the images are stored within. The user can choose to keep the original projection (UTM) or to reproject all data to one consistent projection (strictly recommended for ARD!). See section VII.H for more details.


datacube-definition.prj
Data Cube definition
The spatial data cube definition is appended to each data cube, i.e. to each directory containing tiled datasets. The file ‘datacube-definition.prj’ is a 6-line text file that contains the (1) projection as WKT string, (2) origin of the tile system as geographic Longitude, (3) origin of the tile system as geographic Latitude, (4) origin of the tile system as projected X-coordinate, (5) origin of the tile system as projected Y-coordinate, and (6) width of the tiles in projection units. Do not modify or delete any of these files!
Example (WKT string is one line): 
PROJCS["ETRS89 / LAEA Europe",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0
,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",52],PARAMETER["longitude_of_center",10],PARAMETER["false_easting",4321000],PARAMETER["false_northing",3210000],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AUTHORITY["EPSG","3035"]]
-25.000000
60.000000
2456026.250000
4574919.500000
30000.000000


Naming convention
Following 29-digit naming convention is applied to all output files:

20160823_LEVEL2_SEN2A_BOA.tif
20160823_LEVEL2_SEN2A_BOA.hdr

Digits 1–8	Acquisition date as YYYYMMDD
Digits 10–15	Product Level
Digits 17–21	Sensor ID
LND04		Landsat 4 Thematic Mapper
LND05		Landsat 5 Thematic Mapper
LND07		Landsat 7 Enhanced Thematic Mapper +
LND08		Landsat 8 Operational Land Imager
SEN2A		Sentinel-2A MultiSpectral Instrument
SEN2B		Sentinel-2B MultiSpectral Instrument
Digits 23–25	Product Type
		BOA		Bottom-of-Atmosphere Reflectance (standard output, scale: 10000, nodata: -9999)
		TOA		Top-of-Atmosphere Reflectance (secondary standard output, scale: 10000, nodata: -9999)
		QAI		Quality Assurance Information (standard output, bit coding, nodata: 1)
		AOD		Aerosol Optical Depth (550 nm, optional output, scale: 1000, nodata: -9999)
		CLD		Cloud / Cloud shadow distance (optional output, scale: 10000, nodata: -9999)
		WVP		Water vapor (optional output, scale: 1000, nodata: -9999)
		VZN		View zenith (optional output, scale: 100, nodata: -9999)
		HOT		Haze Optimized Transformation (optional output, scale: 10000, nodata: -9999)
Digits 27–29	File extension
		tif		image data in compressed GeoTiff format
		dat		image data in flat binary ENVI format
		hdr		metadata

Product type
Reflectance data (BOA / TOA) and Quality Assurance Information (QAI) are standard output and cannot be disabled.
AOD / CLD / WVP / VZN / HOT output are optional, images are single-band at the same resolution as BOA / TOA.

BOA / TOA reflectance
Bottom-of-Atmosphere (BOA) reflectance is standard output if atmospheric correction is used. Top-of-Atmosphere (TOA) reflectance is standard output if atmospheric correction is not used. BOA / TOA data contain multiple bands (≙ wavelengths, see metadata and following tables). All bands are provided at the same spatial resolution (set by RESOLUTION parameter). Bands intended for atmospheric characterization are not output (e.g. ultra-blue, water vapor or cirrus bands). Following tables summarize the output bands for each sensor.

Table 2. Landsat 4–5 Thematic Mapper (TM) bands.
Summary of USGS band definitions, and corresponding output bands of FORCE L2PS.
USGS Level 1 band name	Wavelength name	Wavelength range 
in µ	Resolution in m	FORCE Level 2 output band 
1	Blue	0.45–0.52	30	1
2	Green	0.52–0.60	30	2
3	Red	0.63–0.69	30	3
4	Near Infrared	0.76–0.90	30	4
5	Shortwave Infrared 1	1.55–1.75	30	5
6	Thermal Infrared	10.40–12.50	30 (120*)	-**
7	Shortwave Infrared 2	2.08–2.35	30	6
*	Band is acquired at 120m resolution, but USGS products are resampled and provided at 30m.
**	Thermal band is used internally for cloud / cloud shadow detection, but not output.

Table 3. Landsat 7 Enhanced Thematic Mapper Plus (ETM+) bands.
Summary of USGS band definitions, and corresponding output bands of FORCE L2PS.
USGS Level 1 band name	Wavelength name	Wavelength range 
in µ	Resolution in m	FORCE Level 2 output band 
1	Blue	0.45–0.52	30	1
2	Green	0.52–0.60	30	2
3	Red	0.63–0.69	30	3
4	Near Infrared	0.77–0.90	30	4
5	Shortwave Infrared 1	1.55–1.75	30	5
6	Thermal Infrared	10.40–12.50	30 (60*)	-**
7	Shortwave Infrared 2	2.09–2.35	30	6
8	Panchromatic	0.52–0.90	15	-
*	Band is acquired at 60m resolution, but USGS products are resampled and provided at 30m.
**	Thermal band is used internally for cloud / cloud shadow detection, but not output.

Table 4. Landsat 8 Operational Land Imager (OLI) / Thermal Infrared Sensor (TIRS) bands.
Summary of USGS band definitions, and corresponding output bands of FORCE L2PS.
USGS Level 1 band name	Wavelength name	Wavelength range 
in µ	Resolution in m	FORCE Level 2 output band 
1	Ultra-Blue	0.435–0.451	30	-**
2	Blue	0.452–0.512	30	1
3	Green	0.533–0.590	30	2
4	Red	0.636–0.673	30	3
5	Near Infrared	0.851–0.879	30	4
6	Shortwave Infrared 1	1.566–1.651	30	5
7	Shortwave Infrared 2	2.107–2.294	30	6
8	Panchromatic	0.503–0.676	15	-
9	Cirrus	1.363–1.384	30	-***
10	Thermal Infrared 1	10.60–11.19	30 (100*)	-****
11	Thermal Infrared 2	11.50–12.51	30 (100*)	-
*	Bands are acquired at 100m resolution, but USGS products are resampled and provided at 30m.
**	Ultra-Blue band is used internally for aerosol retrieval, but not output.
***	Cirrus band is used internally for cirrus cloud detection, but not output.
****	Thermal band is used internally for cloud / cloud shadow detection, but not output.

Table 5. Sentinel-2 A/B MultiSpectral Instrument (MSI) bands.
Summary of ESA band definitions, and corresponding output bands of FORCE L2PS.
ESA Level 1 band name	Wavelength name	Wavelength range 
in µ	Resolution in m	FORCE Level 2 output band 
1	Ultra-Blue	0.430–0.457	60	-*
2	Blue	0.440–0.538	10	1
3	Green	0.537–0.582	10	2
4	Red	0.646–0.684	10	3
5	Red Edge 1	0.694–0.713	20	4
6	Red Edge 2	0.731–0.749	20	5
7	Red Edge 3	0.769–0.797	20	6
8	Broad Near Infrared	0.760–0.908	10	7
8A	Near Infrared	0.848–0.881	20	8
9	Water Vapor	0.932–0.958	60	-**
10	Cirrus	1.337–1.412	60	-***
11	Shortwave Infrared 1	1.539–1.682	20	9
12	Shortwave Infrared 2	2.078–2.320	20	10
*	Ultra-Blue band is used internally for aerosol retrieval, but not output.
**	Water vapor band is used internally for water vapor retrieval, but not output.
***	Cirrus band is used internally for cirrus cloud detection, but not output.

Quality Assurance Information
Quality Assurance Information (QAI product) are key for any higher-level analysis of ARD. This product contain the cloud masks etc. USE QAI RIGOROUSLY!!! QAI are provided bit-wise for each pixel, thus the integers have to be parsed using following convention (see also force-qai-inflate in section VI.I.2). As an example, integer 28672 would be a poorly illuminated, sloped pixel where water vapor could not have been estimated.

Bit:	15	14	13	12	11	10	9	8	7	6	5	4	3	2	1	0	
Flag:	0	1	1	1	0	0	0	0	0	0	0	0	0	0	0	0	∑ = 28672

Nodata values are values where no data was observed, where no auxiliary information was given (nodata values in DEM), or where data is substantially corrupt. The latter case includes impulse noise (for Landsat 5–7), or pixels that would end up with reflectance > 2.0 or reflectance < -1.0.
Clouds are automatically detected, 100% accuracy for any given image cannot be given. In general, we tried to balance cloud masking, but we believe that it is most important to reduce omission errors for automated processing and analysis of large image archives. Therefore, commission error is probably higher than omission error for any given scene (over time, we expect this to level out on the pixel scale). Opaque clouds (confident cloud) are buffered by 300m (less confident cloud). Cirri are only detected for Landsat 8 and Sentinel-2, using thresholds of the cirrus and blue band.
Cloud shadows are detected on the basis of the cloud layer. If a cloud is missed, the cloud shadow is missed, too. If a false positive cloud is detected, false positive cloud shadows follow. As cloud shadow detection is of less accuracy, all high shadow matches are included in the cloud shadow mask, therefore commission error is larger than in the cloud mask.
Cloud, cloud shadow, snow and water flags are exclusive. A pixel cannot have multiple flags. Water takes precedence over snow. Snow takes precedence over cloud. Note that snow and cold clouds can be easily confused. No shadow is matched for snow pixels. Opaque clouds, and buffered clouds take precedence over cirrus clouds. Clouds take precedence over cloud shadows. 
It is advised to always filter for snow, clouds, and cloud shadows (unless you are specifically interested in one of them). 
Aerosol Optical Depth is estimated for fairly coarse grid cells. If there is no valid AOD estimation in any cell, values are interpolated. If there is no valid AOD estimation for the complete image, a fill value is assigned (AOD is guessed). If AOD @550nm is higher than 0.6, it is flagged as high aerosol; this is not necessarily critical, but should be used with caution (see subzero flag).
If reflectance in any band is < 0, the subzero flag is set. This can point to overestimation of AOD. Depending on application, you should use this data with caution.
If DNs were saturated, or if reflectance in any band is > 1, the saturation flag is set. Depending on application, you should use this data with caution.
If sun elevation is smaller than 15°, the high sun zenith flag is set. Use this data with caution, radiative transfer computations might be out of specification.
The illumination state is related to the quality of the topographic correction. If the incidence angle is smaller than 55°, quality is best. If the incidence angle is larger than 80°, the quality of the topographic correction is low, and data artefacts are possible. If the area is not illuminated at all, no topographic correction is done (values are the same as without topographic correction).
The slope flag indicates whether a simple cosine correction (slope ≤ 2°) was used for topographic correction, or if the enhanced C-correction was used (slope > 2°).
The water vapor flag indicates whether water vapor was estimated, or if the scene average was used to fill. Water vapor is not estimated over water and cloud shadow pixels. This flag only applies to Sentinel-2 images.

Table 6. Quality Assurance Information (QAI) description.
[continued on next pages]
Bit No.	Parameter Name	Bit comb.	Integer	State
0	Valid data	0	0	valid
		1	1	no data
1-2	Cloud state	00	0	clear
		01	1	less confident cloud 
(i.e. buffered cloud)
		10	2	confident, opaque cloud
		11	3	cirrus
3	Cloud shadow flag	0	0	no
		1	1	yes
4	Snow flag	0	0	no
		1	1	yes
5	Water flag	0	0	no
		1	1	yes
6-7	Aerosol state	00	0	Estimated (best quality)
		01	1	interpolated (mid quality)
		10	2	high (might or might not 
be problematic, watch out)
		11	3	fill (use with caution, 
AOD estimate is just a guess)
8	Subzero flag	0	0	no
		1	1	yes (use with caution)
9	Saturation flag	0	0	no
		1	1	yes (use with caution)
10	High sun zenith flag	0	0	no
		1	1	yes (use with caution)
11-12	Illumination state	00	0	good (best quality for topographic 
correction)
		01	1	low (good quality for topographic 
correction)
		10	2	poor (low quality for topographic 
correction, artefacts are possible)
		11	3	shadow (no topographic 
correction applied)
13	Slope flag	0	0	no (< 2° slope, topogr. correction: 
cosine correction applied)
		1	1	yes (> 2° slope, topogr. correction: 
enhanced C-correction applied)
14	Water vapor flag	0	0	measured (best quality, only 
used for Sentinel-2)
		1	1	fill (scene average, e.g. over 
water, only used for Sentinel-2)
15	Empty	0	0	TBD

Aerosol Optical Depth
The Aerosol Optical Depth (AOD) product is optional output. It contains the AOD of the green band (~550 nm). This product is not used by any of the higher-level FORCE modules.

Cloud / cloud shadow / snow distance
The Cloud / cloud shadow / snow distance (CLD) product is optional output. The cloud distance gives the distance to the next opaque cloud, buffered cloud, cirrus cloud, cloud shadow or snow. This product can be used in FORCE L3PS to generate Best Available Pixel (BAP) composites.
Note that this is not the actual cloud mask! For cloud masks and quality screening, rather use the QAI product.

Water vapor
The Water vapor (WVP) product is optional output. It contains the atmospheric water vapor (as derived from the Sentinel-2 data on pixel level, or as ingested with the water vapor database for Landsat). This product is not used by any of the higher-level FORCE modules.

View zenith
The View zenith (VZN) product is optional output. It contains the view zenith (the average view zenith for Sentinel-2, and an approximated view zenith for Landsat). This product can be used in FORCE L3PS to generate Best Available Pixel (BAP) composites.

Haze Optimized Transformation
The 	Haze Optimized Transformation (HOT) product is optional output. It contains the HOT index, which is computed on TOA reflectance (and therefore cannot be computed on Level 2 ARD). The HOT is useful to avoid hazy and residual cloud contamination. This product can be used in FORCE L3PS to generate Best Available Pixel (BAP) composites.

File format
The data are provided in (i) ENVI Standard format (flat binary images), or (ii) as GeoTiff (LZW compression with horizontal differencing). Each dataset consists of an image dataset (.dat/,tif) and additional metadata (.hdr). The image data have signed 16bit datatype and band sequential (BSQ) interleaving.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info etc. Additional information like acquisition time (including date and time), cloud cover per image and tile, sun position and processing information are also provided.

Logfile
A logfile is created by force-level2 in the output directory. Following 29-digit naming convention is applied:
FORCE-L2PS_20170712040001.log
Digits 1–10	Processing module
Digits 12–25	Processing time (start time) as YYYYMMDDHHMMSS
Digits 27–29	File extension

Typical entries look like this:
LC08_L1TP_195023_20180110_20180119_01_T1: sc:   0.10%. cc:  89.59%. AOD: 0.2863. # of targets: 0/327.  4 product(s) written. Success! Processing time: 32 mins 37 secs
LC08_L1TP_195023_20170328_20170414_01_T1: sc:   0.00%. cc:   2.56%. AOD: 0.0984. # of targets: 394/6097.  6 product(s) written. Success! Processing time: 19 mins 03 secs
LC08_L1TP_195023_20170312_20170317_01_T1: sc:   0.29%. cc:  91.85%. Skip. Processing time: 13 mins 22 secs 
The first entry indicates the image ID, followed by overall snow and cloud cover, aerosol optical depth @ 550 nm (scene average), the number of dark targets for retrieving aerosol optical depth (over water/vegetation), the number of products written (number of tiles, this is dependent on tile cloud cover, and FILE_TILE), and a supportive success indication. In the case the overall cloud coverage is higher than allowed, the image is skipped. The processing time (real time) is appended at the end.
