Output Format
=============


Output format
Data organization
The output data are organized in the gridded data structure used for Level 2 processing. The tiles manifest as directories in the file system, and the images are stored within. For mosaicking, use force-mosaic (will likely give an error for the TSS and RMS products because there are different numbers of observations in different tiles).

Naming convention
Following 65-digit naming convention is applied to all output files:

1984-2017_182-274_LEVEL4_TSA_MULTI_TCG_C0_S0_FAVG_TY_C95T_TRD.tif
1984-2017_182-274_LEVEL4_TSA_MULTI_TCG_C0_S0_FAVG_TY_C95T_TRD.hdr

Digits 1–9	Temporal range for the years as YYYY–YYYY
Digits 11–17	Temporal range for the DOY as DDD–DDD
Digits 19–24	Product Level
		LEVEL4
Digits 26–28	Processing Type
		TSA
Digits 30–34	Set of spectral bands
		LNDLG		Landsat legacy bands
		SEN2L		Sentinel-2 land surface bands
		SEN2H		Sentinel-2 high-res bands
		R-G-B		Visual bands
Digits 36–38	3-digit index compact name
		BLU		Blue band
		GRN		Green band
		RED		Red band
		NIR		Near Infrared band
		SW1		Shortwave Infrared band 1
		SW2		Shortwave Infrared band 2
		RE1		Red Edge band 1
		RE2		Red Edge band 2
		RE3		Red Edge band 3
		BNR		Broad Near Infrared band
		NDV		Normalized Difference Vegetation Index
		EVI		Enhanced Vegetation Index
		NBR		Normalized Burn Ratio
		ARV		Atmospherically Resistant Vegetation Index
		SAV		Soil Adjusted Vegetation Index
		SRV		Soil and Atmospherically Resistant Vegetation Index
		TCB		Tasseled Cap Brightness
		TCG		Tasseled Cap Greenness
		TCW		Tasseled Cap Wetness
		TCD		Tasseled Cap-based Disturbance Index
		SMA		Spectral Mixture Analysis abundance
Digits 40–41	Center flag
		C0		no
		C1		yes
Digits 43–44	Standardize flag
		S0		no
		S1		yes
Digits 46–49	Folding method
		FAVG		fold with average
		FMIN		fold with minimum
		FMAX		fold with maximum
		FLSP		fold with Land Surface Phenology
		F***		fold with Land Surface Phenology, in combination with TRD/CAT product type.
				Refer to next subsection.
Digits 51–52	Trend on …
		TY		folded years
		TM		folded months
		TW		folded weeks
		TD		folded days
Digits 54–57	Significance parameters
		C95T		95% (or other) confidence level, two-tailed t-test
		C95L		95% (or other) confidence level, left-tailed t-test
		C95R		95% (or other) confidence level, right-tailed t-test
Digits 59–61	Product Type
		TSS		Time Series Stack
		RMS		RMSE Time Series of SMA
		STA		Basic Statistics
		TSI		Time Series Interpolation
		TRD		Trend Analysis
		CAT		Extended CAT Analysis
		FBY		Fold-by-Year Stack
		FBM		Fold-by-Month Stack
		FBW		Fold-by-Week Stack
		FBD		Fold-by-Day Stack
		***		26 Land Surface Phenology metrics (refer to next subsection)
Digits 63–65	File extension
		tif		image data in compressed GeoTiff format
		dat		image data in flat binary ENVI format
		hdr		metadata

Special naming convention for Land Surface Phenology
Due to the one-to-26 nature of the LSP metrics, the naming convention is a bit more complex. First there are 26 LSP metrics, which are defined with following LSP name tags.
The outputted LSP metrics will have fold-by-LSP tag (digits 46–49: FLSP), and product type (digits 59–61) according to LSP name tag.
If Trend Analysis (TRD) or Extended CAT Analysis (CAT) product type based on LSP metrics is requested, the change / trend will be computed on each LSP metric. Therefore, naming convention needs to be modified. The outputted products will have folding method set to the LSP name tag (digits 46–49: e.g. FDEM, FVEM, …), and TRD / CAT product type (digits 59–61).

LSP name tags:
		DEM		Date of Early Minimum
		DSS		Date of Start of Season
		DRI		Date of Rising Inflection
		DPS		Date of Peak of Season
		DFI		Date of Falling Inflection
		DES		Date of End of Season
		DLM		Date of Late Minimum
		LTS		Length of Total Season
		LGS		Length of Green Season
		VEM		Value of Early Minimum
		VSS		Value of Start of Season
		VRI		Value of Rising Inflection
		VPS		Value of Peak of Season
		VFI		Value of Falling Inflection
		VES		Value of End of Season
		VLM		Value of Late Minimum
		VBL		Value of Base Level
		VSA		Value of Seasonal Amplitude
		IST		Integral of Total Season
		IBL		Integral of Base Level
		IBT		Integral of Base+Total
		IGS		Integral of Green Season
		RAR		Rate of Average Rising
		RAF		Rate of Average Falling
		RMR		Rate of Maximum Rising
		RMF		Rate of Maximum Falling

Product type
Time Series
Time Series products have as many bands as there are available or requested time steps. If no temporal subset was specified:
the TSS product contains one band per available acquisition (this may vary between the tiles), 
the RMS product contains one band per available acquisition (this may vary between the tiles), 
the TSI product contains one band per interpolation step,
the FBY product contains one band per year (do not overdo YEAR_MIN/MAX, this will give many useless bands), 
the FBM product contains one band per month (up to 12, depends on MONTH_MIN/MAX and DOY_MIN/MAX),
the FBW contains one band per week (up to 52, depends on MONTH_MIN/MAX and DOY_MIN/MAX), 
the FBD product contains one band per DOY (up to 365, depends on MONTH_MIN/MAX and DOY_MIN/MAX),
the 26 LSP products contain one band per year (do not overdo YEAR_MIN/MAX, this will give many useless bands).

Basic Statistics
The Basic Statistics (STA) product provides a summary of all observations (or the requested subset). It is a multi-layer image with following bands:
		1	µ		Average of index values
		2	σ		Standard deviation of index values
		3	min		Minimum index value
		4	max		Maximum index value
		5	# of obs.		Number of good quality observations 

Trend Analysis
The Trend Analysis (TRD) product contains trend parameters. It is a multi-layer image with following bands:
		1	µ		Average
		2	a		Intercept
		3	b		Trend
		4	R²		R squared
		5	sig.		Significance (-1, 0, 1)
		6	RMSE		Root Mean Squared Error
		7	MAE		Mean Absolute Error
		8	max |e|		Maximum Absolute Residual
		9	# of obs.		Number of good quality observations 

Change, Aftereffect, Trend
The Change, Aftereffect, Trend (CAT) product (following Hird et al. 2016, DOI: 10.1109/jstars.2015.2419594) contains extended change and trend parameters. It detects one change per time series, splits the time series into three parts, and derives trend parameters: (1) complete time series (this is the same as the TRD product), (2) time series before change, and (3) time series after change. It is a multi-layer image with following bands:
		1	Change		Magnitude of change
		2	Time of change	Timestamp of the change (depends on the input time series, i.e. year/month/week/day)
		3–11	Trend parameters for complete time series (see TRD product)
		12–20	Trend parameters for time series before change (see TRD product)
		21–29	Trend parameters for time series after change (see TRD product)

File format
The data are provided in (i) ENVI Standard format (flat binary images), or (ii) as GeoTiff (LZW compression with horizontal differencing). Each dataset consists of an image dataset (.dat/,tif) and additional metadata (.hdr). The image data have signed 16bit datatype and band sequential (BSQ) interleaving. Scaling factor is 10000 for most products.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info, band names etc.

