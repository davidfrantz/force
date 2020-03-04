Output Format
=============

Output format
Data organization
The output data are organized as the Level 2 data. The output data are appended to the input Level 2 data as new product, i.e. two additional files (image + metadata) appear next to the existing data. Note that the new product will have higher spatial resolution (more pixels) than the other products (e.g. QAI). Higher-level FORCE routines can handle this. For any higher-level FORCE module, you can choose to use the improved product or the original one.

Naming convention
Following 29-digit naming convention is applied to all output files:

20180823_LEVEL2_LND08_IMP.tif
20180823_LEVEL2_LND08_IMP.hdr

The naming convention is the same as for the Level 2 data (see VI.B.5). The only difference is the Product Type, which is set to IMP.
Digits 23–25	Product Type
		IMP		ImproPhed Bottom-of-Atmosphere Reflectance (standard output, scale: 10000, nodata: -9999)

Product type
There is only one product type, i.e. the ImproPhed Bottom-of-Atmosphere Reflectance (IMP). The IMP product has the same specification as the BOA product, but spatial resolution was enhanced.

File format
The data are provided in compressed GeoTiff or flat binary ENVI Standard format. Each dataset consists of an image dataset (.tif/.dat) and metadata (.hdr). The image data have signed 16bit datatype. Each predicted image is stored as separate file.
The metadata (.hdr) are provided in ENVI Standard format as human-readable text using tag and value notation. Metadata include image characteristics like dimensions, data type, band interleave, coordinate reference system, map info, band names etc.

