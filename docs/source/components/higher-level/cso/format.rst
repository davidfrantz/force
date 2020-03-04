Output Format
=============

Output format
Data organization
The output data are organized in the gridded data structure used for Level 2 processing. The tiles manifest as directories in the file system, and the images are stored within. For mosaicking, use force-mosaic.

Naming convention
Following 37-digit naming convention is applied to all output files:

2000-2010_03M_CSO-STATS_LNDLG_NUM.tif
2000-2010_03M_CSO-STATS_LNDLG_NUM.hdr

Digits 1–9	Temporal range for the years as YYYY–YYYY
Digits 11–13	Temporal binning in months
Digits 15–23	Processing Type
		CSO-STATS
Digits 25–29	Set of spectral bands
		LNDLG		Landsat legacy bands
		SEN2L		Sentinel-2 land surface bands
		SEN2H		Sentinel-2 high-res bands
		R-G-B		Visual bands
Digits 31–33	Product Type
		NUM		Number of Observations (optional output, scale 1, nodata: 0)
		AVG		Average of dt (optional output, scale: 1, nodata: 0)
		STD		Standard Deviation of dt (optional output, scale: 1, nodata: 0)
		MIN		Minimum of dt (optional output, scale: 1, nodata: 0)
		MAX		Maximum of dt (optional output, scale: 1, nodata: 0)
		RNG		Range of dt (optional output, scale: 1, nodata: 0)
		SKW		Skewness of dt (optional output, scale: 10000, nodata: 0)
		KRT		Kurtosis of dt (optional output, scale: 1000, nodata: 0)
		Q25		0.25 Quantile of dt (optional output, scale: 1, nodata: 0)
		Q50		0.50 Quantile of dt (optional output, scale: 1, nodata: 0)
		Q75		0.75 Quantile of dt (optional output, scale: 1, nodata: 0)
		IQR		Interquartile Range of dt (optional output, scale: 1, nodata: 0)
Digits 35–37	File extension
		tif		image data in compressed GeoTiff format
		dat		image data in flat binary ENVI format
		hdr		metadata

Product type
There are several product types available, all are optional. The Number of Observations (NUM) product contains the number of clear sky observations for each pixel and bin. The other products are statistical summaries of the temporal difference between consecutive observations within these bins.

File format
The data are provided in compressed GeoTiff or flat binary ENVI Standard format. Each dataset consists of an image dataset (.tif/.dat) and metadata (.hdr). The image data have signed 16bit datatype and band sequential (BSQ) interleave format. The products have as many bands as there are intervals within the defined year range.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info, band names etc.

