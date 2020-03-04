Output Format
=============


Output format
Data organization
The output data are organized in the gridded data structure used for Level 2 processing. The tiles manifest as directories in the file system, and the images are stored within. For mosaicking, use force-mosaic.

Naming convention
Following 21-digit naming convention is applied to all output files:

2017_IMPROPHE_IGS.tif
2017_IMPROPHE_IGS.tif

Digits 1–4	Year
Digits 6–13	Processing Type
		IMPROPHE
Digits 15–17	Product Tag
XXX		These custom3-digit tags are specified in the parameter file
Digits 19–21	File extension
		tif		image data in compressed GeoTiff format
		dat		image data in flat binary ENVI format
		hdr		metadata

File format
The data are provided in compressed GeoTiff or flat binary ENVI Standard format. Each dataset consists of an image dataset (.tif/.dat) and metadata (.hdr). The image data have signed 16bit datatype. Each predicted image is stored as separate file.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info, band names etc.
