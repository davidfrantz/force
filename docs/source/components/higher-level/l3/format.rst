Output Format
=============

Quicklooks
Quicklooks of the processed Level 3 data can be generated using force-quicklook-level3. It searches files that match the pattern ‘*BAP*.dat/tif/vrt’ in the directory given as 1st argument (typically the output directory of FORCE L3PS), and creates a quicklook in the directory given as 2nd argument. The true color quicklooks have a fixed stretch from 0–maxval (3rd argument). The upper limit of the stretch needs to be given in scaled reflectance (scaling factor 10,000), e.g. 1500. This value can be decreased/increased for very dark/bright landscapes. The quicklooks are generated with gdal_translate in JPEG format. The 4th argument needs to be given if the Level 3 products are not tiled ( physical mosaic or virtual mosaic). If the data are tiled, the quicklooks have a fixed size of 256 x 256 px, otherwise it is 20 % of the mosaic size. The data organization of the Level 3 archive will be mirrored in the quicklook archive. Existing quicklooks are skipped (to enable efficient and scheduled processing, e.g. if called from a cronjob). 
Module	|	force-quicklook-level3

Usage	|	force-quicklook-level3     L3-archive     QUICKLOOK-archive     maxval     [--no-tile]

Output format
Data organization
Depending on parameterization, the output data are organized in the gridded data structure used for Level 2 processing, or mosaics are generated (covering the complete requested area). The tiles (or a directory named ‘mosaic’) manifest as directories in the file system, and the images are stored within.

Naming convention
Following 29-digit naming convention is applied to all output files:

20150415_LEVEL3_MULTI_BAP.tif
20150415_LEVEL3_MULTI_BAP.hdr

Digits 1–8	Temporal target for compositing as YYYYMMDD
Digits 10–15	Product Level
Digits 17–21	Set of spectral bands
		LNDLG		Landsat legacy bands
		SEN2L		Sentinel-2 land surface bands
		SEN2H		Sentinel-2 high-res bands
		R-G-B		Visual bands
Digits 23–25	Product Type
		BAP		Best Available Pixel composite (optional output, scale: 10000, nodata: -9999)
		INF		Compositing Information (optional output, nodata: 1/-9999)
		SCR		Compositing Scores (optional output, scale: 10000, nodata: -9999)
		AVG		Temporal Average (optional output, scale: 10000, nodata: -9999)
		STD		Temporal Standard Deviation (optional output, scale: 10000, nodata: -9999)
		MIN		Temporal Minimum (optional output, scale: 10000, nodata: -9999)
		MAX		Temporal Maximum (optional output, scale: 10000, nodata: -9999)
		RNG		Temporal Range (optional output, scale: 10000, nodata: -9999)
		SKW		Temporal Skewness (optional output, scale: 10000, nodata: -9999)
		KRT		Temporal Kurtosis (optional output, scale: 10, nodata: -9999)
		Q25		Temporal 0.25 Quantile (optional output, scale: 10000, nodata: -9999)
		Q50		Temporal 0.50 Quantile (optional output, scale: 10000, nodata: -9999)
		Q75		Temporal 0.75 Quantile (optional output, scale: 10000, nodata: -9999)
		IQR		Temporal Interquartile Range (optional output, scale: 10000, nodata: -9999)
Digits 27–29	File extension
		tif		image data in compressed GeoTiff format
		dat		image data in flat binary ENVI format
		hdr		metadata

Product type
Reflectance data
Reflectance data (BAP and temporal statistics) contain multiple bands (≙ wavelengths). All bands are provided at the same spatial resolution. Single-sensor composites have the same bands as the corresponding Level 2 data. Multi-sensor composites only contain overlapping bands. Exclusive bands are discarded. Note that no spectral adjustment is made.

Compositing information
The compositing information (INF) product contains information about the selected observation in the BAP product. It is a multi-layer image with following bands:
		1	QAI		Quality Assurance Information of BAP (bit coding, nodata: 1)
		2	# of obs.		Number of cloud-free observations within compositing period
		3	D		Acquisition DOY of BAP
		4	Y		Acquisition Year of BAP
		5	ΔD		Difference between D and Target DOY
		6	Sensor		Sensor ID of BAP (in the order, which was given in the parameter file)

Compositing score
The compositing score (SCR) product contains the scores of the BAP (minimum: 0, maximum: 1, scale: 10000). It is a multi-layer image with following bands:
		1	ST		Total score
		2	SD		DOY score (intra-annual score)
		3	SY		Year score (inter-annual score)
		4	SC		Cloud distance score
		5	SH		Haze score
		6	SR		Correlation score
		7	SV		View angle score

File format
The data are provided in (i) ENVI Standard format (flat binary images), or (ii) as GeoTiff (LZW compression with horizontal differencing). Each dataset consists of an image dataset (.dat/,tif) and additional metadata (.hdr). The image data have signed 16bit datatype and band sequential (BSQ) interleaving.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info, band names etc.

