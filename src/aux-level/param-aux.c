/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2020 David Frantz

FORCE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORCE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FORCE.  If not, see <http://www.gnu.org/licenses/>.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This program holds functionality for writing parameter skeletons
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "param-aux.h"


/** This function writes parameters into a parameter skeleton file: lower
+++ level directories
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_dirs(FILE *fp, bool verbose){


  fprintf(fp, "\n# INPUT/OUTPUT DIRECTORIES\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# The file queue specifies, which images are to be processed. The full path\n");
    fprintf(fp, "# to the file needs to be given. Do  not  paste  the content of the file queue\n");
    fprintf(fp, "# into the parameter file. The file queue is mandatory for force-level2, but\n");
    fprintf(fp, "# may be NULL for force-l2ps.\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_QUEUE = NULL\n");

  if (verbose){
    fprintf(fp, "# This is the output directory where the Level 2 data will be stored. Note\n");
    fprintf(fp, "# that data will be overwritten/mosaicked if you reprocess images. It is\n");
    fprintf(fp, "# safe and recommended to use a single Level 2 data pool for different\n");
    fprintf(fp, "# sensors (provided the same grid and projection is used). The higher-level\n");
    fprintf(fp, "# programs of FORCE can handle different spatial resolutions (e.g. 30m\n");
    fprintf(fp, "# Landsat and 10m Sentinel-2).\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_LEVEL2 = NULL\n");

  if (verbose){
    fprintf(fp, "# This is the directory where logfiles should be saved.\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_LOG = NULL\n");

  if (verbose){
    fprintf(fp, "# This is a temporary directory that is used to extract compressed images\n");
    fprintf(fp, "# for force-level2. Note that images already need to be extracted when using\n");
    fprintf(fp, "# force-l2ps directly. The extracted data will be deleted once they were\n");
    fprintf(fp, "# processed. If you cancel processing, you may want to delete any left-overs\n");
    fprintf(fp, "# in this directory. A file 'cpu-$TIME' is temporarily created in DIR_TEMP.\n");
    fprintf(fp, "# This file can be modified to re-adjust the number of CPUs while(!) force-\n");
    fprintf(fp, "# level2 is running. Note that the effect is not immediate, as the load is\n");
    fprintf(fp, "# only adjusted after one of the running jobs (images) is finished. \n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_TEMP = NULL\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level DEM pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_dem(FILE *fp, bool verbose){


  fprintf(fp, "\n# DIGITAL ELEVATION MODEL\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n"); 

  if (verbose){  
    fprintf(fp, "# This file specifies the DEM. It is highly recommended to use a DEM. It is\n");
    fprintf(fp, "# used for cloud / cloud shadow detection, atmospheric correction and topo-\n");
    fprintf(fp, "# graphic correction. The DEM should be a mosaic that should completely cover\n");
    fprintf(fp, "# the area you are preprocessing. If there are nodata values in the DEM, the\n");
    fprintf(fp, "# Level 2 outputs will have holes, too. It is possible to process without a\n");
    fprintf(fp, "# DEM (DEM = NULL). In this case, the surface is assumed flat @ z = 0m.\n");
    fprintf(fp, "# Topographic correction cannot be used without a DEM. The quality of atmo-\n");
    fprintf(fp, "# spheric correction and cloud /cloud shadow detection will suffer without\n");
    fprintf(fp, "# a DEM.\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_DEM = NULL\n");
  
  if (verbose){
    fprintf(fp, "# Nodata value of the DEM.\n");
    fprintf(fp, "# Type: Integer. Valid range: [-32767,32767]\n");
  }
  fprintf(fp, "DEM_NODATA = -32767\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level cube pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_cube(FILE *fp, bool verbose){


  fprintf(fp, "\n# DATA CUBES\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This indicates whether the images should be reprojected to the target\n");
    fprintf(fp, "# coordinate system or if they should stay in their original UTM projection.\n");
    fprintf(fp, "# If you want to work with force-higher-level routines, give TRUE.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "DO_REPROJ = TRUE\n");
  
  if (verbose){
    fprintf(fp, "# This indicates whether the images should be gridded after processing.\n");
    fprintf(fp, "# If TRUE, sub-directories for the tiles are generated in DIR_LEVEL2. \n");
    fprintf(fp, "# If FALSE, sub-directories for the original spatial reference systems\n");
    fprintf(fp, "# are generated in DIR_LEVEL2. If you want to work with force-higher-level\n");
    fprintf(fp, "# routines, give TRUE.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "DO_TILE = TRUE\n");
  
  if (verbose){
    fprintf(fp, "# This is the tile white list. It is an optional file that holds all tiles\n");
    fprintf(fp, "# that should be output. Tiles, which are not specified in this file are\n");
    fprintf(fp, "# not written to disc. This paremeter is ignored if DO_TILE = FALSE.\n");
    fprintf(fp, "# If no tile white list should be used, give FILE_TILE = NULL, in which\n");
    fprintf(fp, "# case all tiles are output.\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_TILE = NULL\n");

  if (verbose){
    fprintf(fp, "# This is the tile size (in target units, commonly in meters) of the \n");
    fprintf(fp, "# gridded output. tiles are square; not used if DO_TILE = FALSE. \n");
    fprintf(fp, "# Type: Double. Valid range: ]0,...\n");
  }
  fprintf(fp, "TILE_SIZE = 30000\n");

  if (verbose){
    fprintf(fp, "# This is the block size (in target units, commonly in meters) of the \n");
    fprintf(fp, "# image chips. Blocks are stripes, i.e. they are as wide as the tile,\n");
    fprintf(fp, "# and as high as specified here; not used if DO_TILE = FALSE or \n");
    fprintf(fp, "# OUTPUT_FORMAT = ENVI. The blocks are the primary processing unit of\n");
    fprintf(fp, "# the force-higher-level routines.\n");
    fprintf(fp, "# Type: Double. Valid range: ]0,TILE_SIZE]\n");
  }
  fprintf(fp, "BLOCK_SIZE = 3000\n");

  if (verbose){
    fprintf(fp, "# This is the spatial resolution of Landsat output; not used if DO_REPROJ \n");
    fprintf(fp, "# = FALSE. Note that the tile and block sizes must be a multiple of the\n");
    fprintf(fp, "# pixel resolution.\n");
    fprintf(fp, "# Type: Double. Valid range: ]0,...\n");
  }
  fprintf(fp, "RESOLUTION_LANDSAT = 30\n");

  if (verbose){
    fprintf(fp, "# This is the spatial resolution of Sentinel-2 output; not used if DO_REPROJ \n");
    fprintf(fp, "# = FALSE. Note that the tile and block sizes must be a multiple of the\n");
    fprintf(fp, "# pixel resolution.\n");
    fprintf(fp, "# Type: Double. Valid range: ]0,...\n");
  }
  fprintf(fp, "RESOLUTION_SENTINEL2 = 10\n");

  if (verbose){
    fprintf(fp, "# These are the origin coordinates of the grid system in decimal degree\n");
    fprintf(fp, "# (negative values for West/South). The upper left corner of tile \n");
    fprintf(fp, "# X0000_Y0000 represents this point. It is a good choice to use a coord-\n");
    fprintf(fp, "# inate that is North-West of your study area – to avoid negative tile\n");
    fprintf(fp, "# numbers. Not used if DO_TILE = FALSE.\n");
    fprintf(fp, "# Type: Double. Valid range: [-90,90]\n");
    fprintf(fp, "# Type: Double. Valid range: [-180,180]\n");
  }
  fprintf(fp, "ORIGIN_LON = -25\n");
  fprintf(fp, "ORIGIN_LAT = 60\n");

  if (verbose){
    fprintf(fp, "# This defines the target coordinate system. If DO_REPROJ = FALSE, the\n");
    fprintf(fp, "# projection string can be NULL. The coordinate system must either be\n");
    fprintf(fp, "# given as WKT string - or can be a predefined coordinate/grid system.\n");
    fprintf(fp, "# If one of the predefined systems are used, TILE_SIZE, BLOCK_SIZE,\n");
    fprintf(fp, "# ORIGIN_LAT, and ORIGIN_LON are ignored and internally replaced with\n");
    fprintf(fp, "# predefined values. Currently, EQUI7 and GLANCE7 are availble. Both\n");
    fprintf(fp, "# are globally defined sets of projections with a corresponding grid \n");
    fprintf(fp, "# system. EQUI7 consists of 7 Equi-Distant, continental projections,\n");
    fprintf(fp, "# with a tile size of 100km. GLANCE7 consists of 7 Equal-Area, conti-\n");
    fprintf(fp, "# nental projections, with a tile size of 150km. One datacube will be\n");
    fprintf(fp, "# generated for each continent.\n");
    fprintf(fp, "# Type: Character. Valid values: {<WKT>,EQUI7,GLANCE7}\n");
  }
  fprintf(fp, "PROJECTION = GLANCE7\n");

  if (verbose){
    fprintf(fp, "# This is the resampling option for the reprojection; you can choose\n");
    fprintf(fp, "# between Nearest Neighbor (NN), Bilinear (BL) and Cubic Convolution\n");
    fprintf(fp, "# (CC); not used if DO_REPROJ = FALSE.\n");
    fprintf(fp, "# Type: Character. Valid values: {NN,BL,CC}\n");
  }
  fprintf(fp, "RESAMPLING = CC\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level atmospheric correction pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_atcor(FILE *fp, bool verbose){


  fprintf(fp, "\n# RADIOMETRIC CORRECTION OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# This indicates if topographic correction should be performed. If TRUE,\n");
    fprintf(fp, "# a DEM need to be given.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "DO_ATMO = TRUE\n");

  if (verbose){
    fprintf(fp, "# This indicates if atmospheric correction should be performed. If TRUE,\n");
    fprintf(fp, "# Bottom-of-Atmosphere reflectance is computed. If FALSE, only Top-of-Atmo-\n");
    fprintf(fp, "# sphere reflectance is computed.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "DO_TOPO = TRUE\n");
  
  if (verbose){
    fprintf(fp, "# This indicates if BRDF correction should be performed. If TRUE, output is\n");
    fprintf(fp, "# nadir BRDF adjusted reflectance instead of BOA reflectance (the output is\n");
    fprintf(fp, "# named BOA nonetheless).\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "DO_BRDF = TRUE\n");

  if (verbose){
    fprintf(fp, "# This indicates if adjacency effect correction should be performed.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "ADJACENCY_EFFECT = TRUE\n");

  if (verbose){
    fprintf(fp, "# This indicates if multiple scattering (TRUE) or the single scattering\n");
    fprintf(fp, "# approximation (FALSE) should be used in the radiative transfer calculations.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "MULTI_SCATTERING = TRUE\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level water vapor correction pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_wvp(FILE *fp, bool verbose){


  fprintf(fp, "\n# WATER VAPOR CORRECTION OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# This is the directory where the water vapor tables are located. Water\n");
    fprintf(fp, "# vapor tables are not required for Sentinel-2, in this case DIR_WVPLUT\n");
    fprintf(fp, "# may be NULL. For Landsat, it is recommended to use this functionality.\n");
    fprintf(fp, "# As a minimum requirement, DIR_WVPLUT may be NULL and a global value\n");
    fprintf(fp, "# for WATER_VAPOR needs to be specified. If a directory is given, \n");
    fprintf(fp, "# WATER_VAPOR is ignored. DIR_WVPLUT must contain water vapor tables.\n");
    fprintf(fp, "# The 12 climatology tables must exist at least. They are used if the\n");
    fprintf(fp, "# daily tables do not exist or if there is no valid daily value.\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_WVPLUT = NULL\n");

  if (verbose){
    fprintf(fp, "# This specifies a global value for atmospheric water vapor content in\n");
    fprintf(fp, "# g cm-2. This parameter can be a dummy value to quickly process an image\n");
    fprintf(fp, "# without needing to generate a water vapor database. Note that especially\n");
    fprintf(fp, "# Landsat-8 is relatively insensitive to atmospheric water vapor (depending\n");
    fprintf(fp, "# on wavelength), and external water vapor is not needed to process\n");
    fprintf(fp, "# Sentinel-2. The error in using a dummy value is significant for the TM\n");
    fprintf(fp, "# sensors.\n");
    fprintf(fp, "# Type: Float. Valid range: [0,15]\n");
  }
  fprintf(fp, "WATER_VAPOR = NULL\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level aod estimation pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_aod(FILE *fp, bool verbose){


  fprintf(fp, "\n# AEROSOL OPTICAL DEPTH OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# This indicates whether the internal AOD estimation (TRUE) or externally\n");
    fprintf(fp, "# generated AOD values should be used (FALSE).\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "DO_AOD  = TRUE\n");

  if (verbose){
    fprintf(fp, "# This is the directory where the aerosol optical depth look-up-tables are\n");
    fprintf(fp, "# located. They can be used to input external AOD values. It is recom-\n");
    fprintf(fp, "# mended to use the internal algorithm only. If a path is given, and \n");
    fprintf(fp, "# DO_ATMO = TRUE, internal AOD estimation is used and external AOD values\n");
    fprintf(fp, "# are used as fallback option.\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_AOD  = NULL\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level cloud detection pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_cloud(FILE *fp, bool verbose){
  
  
  fprintf(fp, "\n# CLOUD DETECTION OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter cancels the processing of images that exceed the given\n");
    fprintf(fp, "# threshold. The processing will be canceled after cloud detection.\n");
    fprintf(fp, "# Type: Integer. Valid range: ]0,100]\n");
  }
  fprintf(fp, "MAX_CLOUD_COVER_FRAME = 75\n");
  
  if (verbose){  
    fprintf(fp, "# This parameter works on a tile basis. It suppresses the output for chips\n");
    fprintf(fp, "# (tiled image) that exceed the given threshold.\n");
    fprintf(fp, "# Type: Integer. Valid range: ]0,100]\n");
  }
  fprintf(fp, "MAX_CLOUD_COVER_TILE  = 75\n");

  if (verbose){
    fprintf(fp, "# These are the main thresholds of the Fmask algorithm.\n");
    fprintf(fp, "# Type: Float. Valid range: [0,1]\n");
  }
  fprintf(fp, "CLOUD_THRESHOLD  = 0.225\n");
  fprintf(fp, "SHADOW_THRESHOLD = 0.02\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level resolution merging pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_resmerge(FILE *fp, bool verbose){


  fprintf(fp, "\n# RESOLUTION MERGING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the method used for improving the spatial reso-\n");
    fprintf(fp, "# lution of Sentinel-2’s 20 m bands to 10 m. Pixels flagged as cloud or\n");
    fprintf(fp, "# shadow will be skipped. Following methods are available: IMPROPHE uses\n");
    fprintf(fp, "# the ImproPhe code in a spectral-only setup; REGRESSION uses a multi-\n");
    fprintf(fp, "# parameter regression (results are expected to be best, but processing\n");
    fprintf(fp, "# time is significant); STARFM uses a spectral-only setup of the Spatial\n");
    fprintf(fp, "# and Temporal Adaptive Reflectance Fusion Model (prediction artifacts\n");
    fprintf(fp, "# may occur between land cover boundaries); NONE disables resolution merge;\n");
    fprintf(fp, "# in this case, 20m bands are quadrupled.\n");
    fprintf(fp, "# Type: Character. Valid values: {IMPROPHE,REGRESSION,STARFM,NONE}\n");
  }
  fprintf(fp, "RES_MERGE = IMPROPHE\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level coregistration pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_coreg(FILE *fp, bool verbose){


  fprintf(fp, "\n# CO-REGISTRATION OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter only applies for Sentinel-2 data. This parameter defines\n");
    fprintf(fp, "# the path to a directory that contains monthly Landsat NIR target images.\n");
    fprintf(fp, "# If given, a co-registration is attempted. If it fails (no tie points),\n");
    fprintf(fp, "# the image won't be processed.\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_MASTER = NULL\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the nodata values of the master images.\n");
    fprintf(fp, "# Type: Integer. Valid values: [-32767,32767]\n");
  }
  fprintf(fp, "MASTER_NODATA = -32767\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level miscellaneous pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_misc(FILE *fp, bool verbose){


  fprintf(fp, "\n# MISCELLANEOUS OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines if impulse noise should be removed. Ony applies\n");
    fprintf(fp, "# to 8bit input data.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "IMPULSE_NOISE = TRUE\n");

  if (verbose){
    fprintf(fp, "# This parameter defines if nodata pixels should be buffered by 1 pixel.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "BUFFER_NODATA = FALSE\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level tier pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_tier(FILE *fp, bool verbose){


  fprintf(fp, "\n# TIER LEVEL\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");  
  
  if (verbose){
    fprintf(fp, "# This parameter specifies the acceptable tier level of Landsat Level 1 data.\n");
    fprintf(fp, "# For pre-collection data, TIER = 1 will only accept L1T images, TIER = 2\n");
    fprintf(fp, "# will also accept L1Gt and L1G images. For collection data, TIER = 1 will\n");
    fprintf(fp, "# only accept L1TP images, TIER = 2 will also accept T2 images, TIER = 3\n");
    fprintf(fp, "# will additionaly accept RT images.\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,3]\n");
  }
  fprintf(fp, "TIER = 1\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level threading pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_thread(FILE *fp, bool verbose){


  fprintf(fp, "\n# PARALLEL PROCESSING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  fprintf(fp, "# Multiprocessing options (NPROC, DELAY) only apply when using the batch\n");
  fprintf(fp, "# utility force-level2. They are not used by the core function force-l2ps.\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This module is using hybrid parallelization, i.e. a mix of multiprocessing\n");
    fprintf(fp, "# and multithreading. Each input image is one process, each process may use\n");
    fprintf(fp, "# multiple threads. In general, it is most efficient to use as much multi-\n");
    fprintf(fp, "# processing as possible (i.e. NTHREAD = 1 or 2). However, if you only have\n");
    fprintf(fp, "# a small number of images - or if your system does not have enough RAM,\n");
    fprintf(fp, "# it is adviced to use less processes and more threads per process. The\n");
    fprintf(fp, "# number of processes and threads is given by following parameters.\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,...\n");
  }
  fprintf(fp, "NPROC = 32\n");
  fprintf(fp, "NTHREAD = 2\n");
  
  if (verbose){
    fprintf(fp, "# This parameter controls whether the individual bands of the Level 1 input\n");
    fprintf(fp, "# images are read sequentially or in parallel. Note that we observed two kinds\n");
    fprintf(fp, "# of GDAL installation: (1) the JPEG driver reads each band parallely, but \n");
    fprintf(fp, "# separated images in sequence - we recommend to disable PARALLEL_READS in this\n");
    fprintf(fp, "# case (for Sentinel-2). (2) The GDAL JPEG drived does not do anything in \n");
    fprintf(fp, "# parallel - use PARALLEL_READ to speed up the work (also use it for Landsat).\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "PARALLEL_READS = FALSE\n");

  if (verbose){
    fprintf(fp, "# This parameter sets a delay before starting a new process. This can be help-\n");
    fprintf(fp, "# ful to prevent I/O jams when using a lot of processes. The delay is given\n");
    fprintf(fp, "# in seconds.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "DELAY = 3\n");

  if (verbose){
    fprintf(fp, "# This parameter sets a timeout for unzipping the Level 1 data (only applies when\n");
    fprintf(fp, "# images are still in zip/tar.gz format. Only applies for force-level2).\n");
    fprintf(fp, "# The timeout is given in seconds.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "TIMEOUT_ZIP = 30\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: lower
+++ level output pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_ll_output(FILE *fp, bool verbose){


  fprintf(fp, "\n# OUTPUT OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# Output format, which is either uncompressed flat binary image format aka\n");
    fprintf(fp, "# ENVI Standard or GeoTiff. GeoTiff images are compressed with LZW and hori-\n");
    fprintf(fp, "# zontal differencing; BigTiff support is enabled; the Tiff is structured \n");
    fprintf(fp, "# with striped blocks according to the TILE_SIZE (X) and BLOCK_SIZE (Y) speci-\n");
    fprintf(fp, "# fications. Metadata are written to the ENVI header or directly into the Tiff\n");
    fprintf(fp, "# to the FORCE domain. If the size of the metadata exceeds the Tiff's limit,\n");
    fprintf(fp, "# an external .aux.xml file is additionally generated.\n");
    fprintf(fp, "# Type: Character. Valid values: {ENVI,GTiff}\n");
  }
  fprintf(fp, "OUTPUT_FORMAT = GTiff\n");

  if (verbose){
    fprintf(fp, "# Output the cloud/cloud shadow/snow distance output? Note that this is NOT\n");
    fprintf(fp, "# the cloud mask (which is sitting in the mandatory QAI product). This pro-\n");
    fprintf(fp, "# duct can be used in force-level3; no other higher-level FORCE module is\n");
    fprintf(fp, "# using this.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_DST = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output Aerosol Optical Depth map for the green band? No higher-level FORCE\n");
    fprintf(fp, "# module is using this.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_AOD = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output the Water Wapor map? No higher-level FORCE module is using this.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_WVP = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output the view zenith map? This product can be used in force-level3; no\n");
    fprintf(fp, "# other higher-level FORCE module is using this.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_VZN = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output the  Haze Optimzed Transformation output? This product can be\n");
    fprintf(fp, "# used in force-level3; no other higher-level FORCE module is using this.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_HOT = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output overview thumbnails? These are jpegs at reduced spatial resolution,\n");
    fprintf(fp, "# which feature an RGB overview + quality information overlayed (pink: cloud,\n");
    fprintf(fp, "# red: cirrus, cyan: cloud shadow, yellow: snow, orange: saturated, green:\n");
    fprintf(fp, "# subzero reflectance). No higher-level FORCE module is using this.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_OVV = TRUE\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level directories
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_dirs(FILE *fp, bool verbose){


  fprintf(fp, "\n# INPUT/OUTPUT DIRECTORIES\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Lower Level datapool (parent directory of tiled input data)\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_LOWER = NULL\n");
  
  if (verbose){
    fprintf(fp, "# Higher Level datapool (parent directory of tiled output data)\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_HIGHER = NULL\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level masking pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_mask(FILE *fp, bool verbose){


  fprintf(fp, "\n# MASKING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Analysis Mask datapool (parent directory of tiled analysis masks)\n");
    fprintf(fp, "# If no analsys mask should be applied, give NULL.\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_MASK = NULL\n");

  if (verbose){
    fprintf(fp, "# Basename of analysis masks (e.g. WATER-MASK.tif).\n");
    fprintf(fp, "# Masks need to be binary with 0 = off / 1 = on\n");
    fprintf(fp, "# Type: Basename of file\n");
  }
  fprintf(fp, "BASE_MASK = NULL\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level extent pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_extent(FILE *fp, bool verbose){


  fprintf(fp, "\n# PROCESSING EXTENT AND RESOLUTION\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Analysis extent, given in tile numbers (see tile naming)\n");
    fprintf(fp, "# Each existing tile falling into this square extent will be processed\n");
    fprintf(fp, "# A shapefile of the tiles can be generated with force-tabulate-grid\n");
    fprintf(fp, "# Type: Integer list. Valid range: [-999,9999]\n");
  }
  fprintf(fp, "X_TILE_RANGE = 0 0\n");
  fprintf(fp, "Y_TILE_RANGE = 0 0\n");

  if (verbose){
    fprintf(fp, "# White list of tiles. Can be used to further limit the analysis extent to\n");
    fprintf(fp, "# non-square extents. The white list is intersected with the analysis extent,\n");
    fprintf(fp, "# i.e. only tiles included in both the analysis extent AND the white-list will\n");
    fprintf(fp, "# be processed.\n");
    fprintf(fp, "# Optional. If NULL, the complete analysis extent is processed\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_TILE = NULL\n");
  
  if (verbose){
    fprintf(fp, "# This parameter can be used to override the default blocksize of the input\n");
    fprintf(fp, "# images (as specified in the datacube-definition.prj file). This can be\n");
    fprintf(fp, "# necessary if the default blocksize is too large for your system and you\n");
    fprintf(fp, "# cannot fit all necessary data into RAM. Note that processing of larger\n");
    fprintf(fp, "# blocksizes is more efficient. The tilesize must be dividable by the blocksize\n");
    fprintf(fp, "# without remainder. Set to 0, to use the default blocksize\n");
    fprintf(fp, "# Type: Double. Valid range: 0 or [RESOLUTION,TILE_SIZE]\n");
  }
  fprintf(fp, "BLOCK_SIZE = 0\n");
  
  if (verbose){
    fprintf(fp, "# Analysis resolution. The tile (and block) size must be dividable by this\n");
    fprintf(fp, "# resolution without remainder, e.g. 30m resolution with 100km tiles is not possible\n");
    fprintf(fp, "# Type: Double. Valid range: ]0,BLOCK_SIZE]\n");
  }
  fprintf(fp, "RESOLUTION = 10\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level psf aggregation pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_psf(FILE *fp, bool verbose){


  if (verbose){
    fprintf(fp, "# How to reduce spatial resolution for cases when the image resolution is higher\n");
    fprintf(fp, "# than the analysis resolution. If FALSE, the resolution is degraded using Nearest\n");
    fprintf(fp, "# Neighbor resampling (fast). If TRUE, an approx. Point Spread Function (Gaussian\n");
    fprintf(fp, "# lowpass with FWHM = analysis resolution) is used to approximate the acquisition\n");
    fprintf(fp, "# of data at lower spatial resolution\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "REDUCE_PSF = FALSE\n");
   
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level improphe input pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_improphed(FILE *fp, bool verbose){


  if (verbose){
    fprintf(fp, "# If you have spatially enhanced some Level 2 ARD using the FORCE Level 2 ImproPhe\n");
    fprintf(fp, "# module, this switch specifies whether the data are used at original (FALSE) or\n");
    fprintf(fp, "# enhanced spatial resolution (TRUE). If there are no improphe'd products, this\n");
    fprintf(fp, "# switch doesn't have any effect\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "USE_L2_IMPROPHE = FALSE\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level sensor pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_sensor(FILE *fp, bool verbose){


  fprintf(fp, "\n# SENSOR WHITE LIST\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Sensors to be used in the analysis. Multi-sensor analyses are restricted\n");
    fprintf(fp, "# to the overlapping bands. Following sensors are available: LND04 (6-band\n");
    fprintf(fp, "# Landsat 4 TM), LND05 (6-band Landsat 5 TM), LND07 (6-band Landsat 7 ETM+),\n");
    fprintf(fp, "# LND08 (6-band Landsat 8 OLI), SEN2A (10-band Sentinel-2A), SEN2B (10-band\n");
    fprintf(fp, "# Sentinel-2B), sen2a (4-band Sentinel-2A), sen2b (4-band Sentinel-2B),\n");
    fprintf(fp, "# S1AIA (2-band VV-VH Sentinel-1A IW ascending), S1BIA (2-band VV-VH Senti-\n");
    fprintf(fp, "# nel-1B IW ascending), S1AID (2-band VV-VH Sentinel-1A IW descending), S1BID\n");
    fprintf(fp, "# (2-band VV-VH Sentinel-1B IW descending).\n");
    fprintf(fp, "# The resulting outputs are named according to their band designation, i.e. \n");
    fprintf(fp, "# LNDLG ((6-band Landsat legacy bands), SEN2L (10-band Sentinel-2 land surface\n");
    fprintf(fp, "# bands), SEN2H (4-band Sentinel-2 high-res bands), R-G-B (3-band visual) or\n");
    fprintf(fp, "# VVVHP (VV/VH polarized).\n");
    fprintf(fp, "# BAP Composites with such a band designation can be input again (e.g. \n");
    fprintf(fp, "# SENSORS = LNDLG).\n");
    fprintf(fp, "# Type: Character list. Valid values: {LND04,LND05,LND07,LND08,SEN2A,\n");
    fprintf(fp, "#   SEN2B,sen2a,sen2b,S1AIA,S1BIA,S1AID,S1BID,LNDLG,SEN2L,SEN2H,R-G-B,VVVHP}\n");
  }
  fprintf(fp, "SENSORS = LND08 SEN2A SEN2B\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level QAI pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_qai(FILE *fp, bool verbose){


  fprintf(fp, "\n# QAI SCREENING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This list controls, which QAI flags are masked out before doing the analysis.\n");
    fprintf(fp, "# Type: Character list. Valid values: {NODATA,CLOUD_OPAQUE,CLOUD_BUFFER,\n");
    fprintf(fp, "#   CLOUD_CIRRUS,CLOUD_SHADOW,SNOW,WATER,AOD_FILL,AOD_HIGH,AOD_INT,SUBZERO,\n");
    fprintf(fp, "#   SATURATION,SUN_LOW,ILLUMIN_NONE,ILLUMIN_POOR,ILLUMIN_LOW,SLOPED,WVP_NONE}\n");
  }
  fprintf(fp, "SCREEN_QAI = NODATA CLOUD_OPAQUE CLOUD_BUFFER CLOUD_CIRRUS CLOUD_SHADOW SNOW SUBZERO SATURATION\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level noise filtering pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_noise(FILE *fp, bool verbose){


  if (verbose){
    fprintf(fp, "# Threshold for removing outliers. Triplets of observations are used to determine\n");
    fprintf(fp, "# the overall noise in the time series by computinglinearly interpolating between\n");
    fprintf(fp, "# the bracketing observations. The RMSE of the residual between the middle value\n");
    fprintf(fp, "# and the interpolation is the overall noise. Any observations, which have a\n");
    fprintf(fp, "# residual larger than a multiple of the noise are iteratively filtered out\n");
    fprintf(fp, "# (ABOVE_NOISE). Lower/Higher values filter more aggressively/conservatively.\n");
    fprintf(fp, "# Likewise, any masked out observation (as determined by the SCREEN_QAI filter)\n");
    fprintf(fp, "# can be restored if its residual is lower than a multiple of the noise\n");
    fprintf(fp, "# (BELOW_NOISE). Higher/Lower values will restore observations more aggres-\n");
    fprintf(fp, "# sively/conservative. Give 0 to both parameters to disable the filtering.\n");
    fprintf(fp, "# Type: Float. Valid range: [0,...\n");
  }
  fprintf(fp, "ABOVE_NOISE = 3\n");
  fprintf(fp, "BELOW_NOISE = 1\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level temporal extent pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_time(FILE *fp, bool verbose){


  fprintf(fp, "\n# PROCESSING TIMEFRAME\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# Time extent for the analysis. All data between these dates will be used in\n");
    fprintf(fp, "# the analysis.\n");
    fprintf(fp, "# Type: Date list. Format: YYYY-MM-DD\n");
  }
  fprintf(fp, "DATE_RANGE = 2010-01-01 2019-12-31\n");
  
  if (verbose){
    fprintf(fp, "# DOY range for filtering the time extent. Day-of-Years that are outside of\n");
    fprintf(fp, "# the given interval will be ignored. Example: DATE_RANGE = 2010-01-01 \n");
    fprintf(fp, "# 2019-12-31, DOY_RANGE = 91 273 will use all April-Sepember observations from\n");
    fprintf(fp, "# 2010-2019. If you want to extend this window over years give DOY min > \n");
    fprintf(fp, "# DOY max. Example: DATE_RANGE = 2010-01-01 2019-12-31, DOY_RANGE = 274 90 \n");
    fprintf(fp, "# will use all October-March observations from 2010-2019.\n");
    fprintf(fp, "# Type: Integer list. Valid values: [1,365]\n");
  }
  fprintf(fp, "DOY_RANGE = 1 365\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level output pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_output(FILE *fp, bool verbose){


  fprintf(fp, "\n# OUTPUT OPTIONS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Output format, which is either uncompressed flat binary image format aka\n");
    fprintf(fp, "# ENVI Standard or GeoTiff. GeoTiff images are compressed with LZW and hori-\n");
    fprintf(fp, "# zontal differencing; BigTiff support is enabled; the Tiff is structured \n");
    fprintf(fp, "# with striped blocks according to the TILE_SIZE (X) and BLOCK_SIZE (Y) speci-\n");
    fprintf(fp, "# fications. Metadata are written to the ENVI header or directly into the Tiff\n");
    fprintf(fp, "# to the FORCE domain. If the size of the metadata exceeds the Tiff's limit,\n");
    fprintf(fp, "# an external .aux.xml file is additionally generated.\n");
    fprintf(fp, "# Type: Character. Valid values: {ENVI,GTiff}\n");
  }
  fprintf(fp, "OUTPUT_FORMAT = GTiff\n");

  if (verbose){
    fprintf(fp, "# This parameter controls whether the output is written as multi-band image, or\n");
    fprintf(fp, "# if the stack will be exploded into single-band files.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_EXPLODE = FALSE\n");

  //if (verbose){
  //  fprintf(fp, "# If an output file already exists.. Overwrite?\n");
  //  fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  //}
  //fprintf(fp, "OUTPUT_OVERWRITE = FALSE\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level threading pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_thread(FILE *fp, bool verbose){


  fprintf(fp, "\n# PARALLEL PROCESSING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This module is using a streaming mechanism to speed up processing. There\n");
    fprintf(fp, "# are three processing teams (3 Threads) that simultaneously handle Input,\n");
    fprintf(fp, "# Processing, and Output. Example: when Block 2 is being processed, data\n");
    fprintf(fp, "# from Block 3 are already being input and results from Block 1 are being\n");
    fprintf(fp, "# output. Each team can have multiple sub-threads to speed up the work. The\n");
    fprintf(fp, "# number of threads to use for each team is given by following parameters.\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,...\n");
  }
  fprintf(fp, "NTHREAD_READ = 8\n");
  fprintf(fp, "NTHREAD_COMPUTE = 22\n");
  fprintf(fp, "NTHREAD_WRITE = 4\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level BAP pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_bap(FILE *fp, bool verbose){


  fprintf(fp, "\n# Best Available Pixel (BAP) compositing\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# This parameter specifies the target year for compositing.\n");
    fprintf(fp, "# Type: Integer. Valid values: [1900,2100]\n");
  }
  fprintf(fp, "YEAR_TARGET = 2018\n");

  if (verbose){
    fprintf(fp, "# This parameter specifies thenumber of bracketing years (target year +- \n");
    fprintf(fp, "# bracketing years), i.e. the compositing period. A value of 2 would result\n");
    fprintf(fp, "# in a five-year compositing period.\n");
    fprintf(fp, "# Type: Integer. Valid values: [0,100]\n");
  }
  fprintf(fp, "YEAR_NUM = 2\n");

  if (verbose){
    fprintf(fp, "# This parameter is a tradeoff parameter that balances the inter- and intra-\n");
    fprintf(fp, "# annual selection. Lower values (e.g. 0.75) favor data from the target year.\n");
    fprintf(fp, "# Higher values favor data that was acquired close to the target DOY (regard-\n");
    fprintf(fp, "# less of the year).\n");
    fprintf(fp, "# Type: Float. Valid values: [0,...\n");
  }
  fprintf(fp, "Y_FACTOR = 0.75\n");

  if (verbose){
    fprintf(fp, "# These parameters specify the function values used for fitting the DOY\n");
    fprintf(fp, "# scoring functions. The function type is automatically chosen from the \n");
    fprintf(fp, "# given values, i.e.\n");
    fprintf(fp, "#   Gaussian           s0 < s1 > s2\n");
    fprintf(fp, "#   Descending sigmoid s0 > s1 > s2\n");
    fprintf(fp, "#   Ascending sigmoid  s0 < s1 < s2\n");
    fprintf(fp, "# Type: Float list, 3 values. Valid values: ]0,1[\n");
  }
  fprintf(fp, "DOY_SCORE = 0.01 0.99 0.01\n");

  if (verbose){
    fprintf(fp, "# These parameters specify the DOYs used for fitting the DOY scoring\n");
    fprintf(fp, "# functions in case of the static compositing. They are not used for the \n");
    fprintf(fp, "# phenology-adaptive compositing. However, in each case, the target date\n");
    fprintf(fp, "# appearing in the file name is derived from these values. The target date\n");
    fprintf(fp, "# is the value with highest score (see last parameter).\n");
    fprintf(fp, "# Typically, the DOYs are in order, e.g. p0 = 60, p1 = 90, p2 = 120.\n");
    fprintf(fp, "# However, the DOY scoring can also extend between the years (i.e. around \n");
    fprintf(fp, "# the turn of the year). If p0 > p1: p0 is from previous year, e.g. p0 = 330,\n");
    fprintf(fp, "# p1 = 30, p2 = 90. If p2 < p1: p2 is from next year, e.g. p0 = 300, p1 = 330,\n");
    fprintf(fp, "# p2 = 30.\n");
    fprintf(fp, "# Type: Integer list, 3 values. Valid values: [1,365]\n");
  }
  fprintf(fp, "DOY_STATIC = 120 180 240\n");

  if (verbose){
    fprintf(fp, "# This parameter specifies whether all available data within the requested time\n");
    fprintf(fp, "# frame are used – or only from the season of interest. If FALSE, the composites \n");
    fprintf(fp, "# only consider data for the period, in which the intra-annual score is higher \n");
    fprintf(fp, "# than 0.01. If there is no clear-sky data within this period, data gaps are \n");
    fprintf(fp, "# possible. If TRUE, all data from the requested years are used, thus the risk\n");
    fprintf(fp, "# of having data gaps is lower. However, it is possible that data from unwanted\n");
    fprintf(fp, "# parts of the year are selected.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OFF_SEASON = FALSE\n");

  if (verbose){
    fprintf(fp, "# This parameter controls the strength of the DOY score.\n");
    fprintf(fp, "# 0 disables the use of this score.\n");
    fprintf(fp, "# Type: Float. Valid values: [0,1]\n");
  }
  fprintf(fp, "SCORE_DOY_WEIGHT = 1.0\n");

  if (verbose){
    fprintf(fp, "# This parameter controls the strength of the Year score.\n");
    fprintf(fp, "# 0 disables the use of this score.\n");
    fprintf(fp, "# Type: Float. Valid values: [0,1]\n");
  }
  fprintf(fp, "SCORE_YEAR_WEIGHT = 1.0\n");

  if (verbose){
    fprintf(fp, "# This parameter controls the strength of the cloud distance score.\n");
    fprintf(fp, "# 0 disables the use of this score.\n");
    fprintf(fp, "# Type: Float. Valid values: [0,1]\n");
  }
  fprintf(fp, "SCORE_CLOUD_WEIGHT = 0.4\n");

  if (verbose){
    fprintf(fp, "# This parameter controls the strength of the haze score.\n");
    fprintf(fp, "# 0 disables the use of this score.\n");
    fprintf(fp, "# Type: Float. Valid values: [0,1]\n");
  }
  fprintf(fp, "SCORE_HAZE_WEIGHT = 0.3\n");

  if (verbose){
    fprintf(fp, "# This parameter controls the strength of the correlation score.\n");
    fprintf(fp, "# 0 disables the use of this score.\n");
    fprintf(fp, "# Type: Float. Valid values: [0,1]\n");
  }
  fprintf(fp, "SCORE_CORREL_WEIGHT = 0.0\n");

  if (verbose){
    fprintf(fp, "# This parameter controls the strength of the view zenith score.\n");
    fprintf(fp, "# 0 disables the use of this score.\n");
    fprintf(fp, "# Type: Float. Valid values: [0,1]\n");
  }
  fprintf(fp, "SCORE_VZEN_WEIGHT = 0.0\n");

  if (verbose){
    fprintf(fp, "# This parameter indicates the distance (to the next cloud or cloud shadow) after \n");
    fprintf(fp, "# which the sky is assumed to be clear (cloud score approaches 1.0). The distance\n");
    fprintf(fp, "# needs to be given in meters.\n");
    fprintf(fp, "# Type: Float. Valid values: [1,...\n");
  }
  fprintf(fp, "DREQ = 3000\n");

  if (verbose){
    fprintf(fp, "# This parameter indicates the view zenith angle at which the view zenith score\n");
    fprintf(fp, "# approaches 0.0. The angle needs to be given in degree. \n");
    fprintf(fp, "# Type: Float. Valid values: [1,90]\n");
  }
  fprintf(fp, "VREQ = 7.5\n");


  if (verbose){
    fprintf(fp, "# Output the composite?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_BAP = TRUE\n");  

  if (verbose){
    fprintf(fp, "# Output the compositing information?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_INF = TRUE\n");  

  if (verbose){
    fprintf(fp, "# Output the compositing scores?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_SCR = FALSE\n");  

  if (verbose){
    fprintf(fp, "# Output quicklook of the composite?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_OVV = FALSE\n");  

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level PAC pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_pac(FILE *fp, bool verbose){


  fprintf(fp, "\n# Phenology Adaptive Compositing (PAC)\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# This parameter defines whether the phenology-adpative compositing (TRUE)\n");
    fprintf(fp, "# or the static compositing (FALSE) should be used. In case of the static \n");
    fprintf(fp, "# version, the target DOYs are derived from DOY_STATIC. In case of the \n");
    fprintf(fp, "# PAC, the target DOYs are retrived from the files given by LSP_FILE\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "LSP_DO = FALSE\n");

  if (verbose){
    fprintf(fp, "# Land Surface Phenology datapool (parent directory of tiled LSP)\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_LSP = NULL\n");

  if (verbose){
    fprintf(fp, "# Basenames of the LSP datasets that are used as compositing targets \n");
    fprintf(fp, "# (analogously to DOY_STATIC). Each file should be a multi-band image\n");
    fprintf(fp, "# wherein the bands represent different years. The number of bands, and\n");
    fprintf(fp, "# the corresponding years, need to be the same for all files.\n");
    fprintf(fp, "# Type: List with basenames of 3 files\n");
  }
  fprintf(fp, "BASE_LSP = NULL\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines year, which corresponds to he 1st band of the\n");
    fprintf(fp, "# LSP.\n");
    fprintf(fp, "# Type: Integer. Valid values: [1900,2100]\n");
  }
  fprintf(fp, "LSP_1ST_YEAR = 2000\n");

  if (verbose){
    fprintf(fp, "# This parameter specifies the starting point of the LSP values. \n");
    fprintf(fp, "# Internally, the data are represented as ‘Year x 365 + DOY’. Thus, LSP_START\n");
    fprintf(fp, "# is an offset, which must be given as ‘Year x 365 + DOY’. If the values are \n");
    fprintf(fp, "# provided in this format, use LSP_START = 1. If the LSP values would be pro-\n");
    fprintf(fp, "# vided relative to January 1 2000, use LSP_START = 730001, i.e. 2000*365+1. \n");
    fprintf(fp, "# Leap years are not taken into account and each year consists of 365 days.\n");
    fprintf(fp, "# Type: Integer. Valid values: [1,2100*365]\n");
  }
  fprintf(fp, "LSP_START = 2000\n");
  
  if (verbose){
    fprintf(fp, "# This parameter is a threshold in days. If the inter-annual variability of the \n");
    fprintf(fp, "# LSP (of a given pixel) exceeds this value, the long-term average LSP is used \n");
    fprintf(fp, "# instead of the yearly values. The value should be between 0 (long-term average\n");
    fprintf(fp, "# is used for all pixels) and 365 (long-term average is never used).\n");
    fprintf(fp, "# Type: Integer list. Valid values: [0,365]\n");
  }
  fprintf(fp, "LSP_THRESHOLD = 182\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the nodata value for the LSP.\n");
    fprintf(fp, "# Type: Integer. Valid values: [-32767,32767]\n");
  }
  fprintf(fp, "LSP_NODATA = -32767\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level spectral indices pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_index(FILE *fp, bool verbose){


  fprintf(fp, "\n# SPECTRAL INDEX\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# Perform the time series analysis using the specified band or index.\n");
    fprintf(fp, "# Multiple indices can be processed ar once to avoid multiple reads of the\n");
    fprintf(fp, "# same file. Only necessary bands will be input. You will be alerted if the\n");
    fprintf(fp, "# index cannot be computed based on the requested SENSORS. The index SMA is\n");
    fprintf(fp, "# a linear spectral mixture analysis and is dependent on the parameters\n");
    fprintf(fp, "# specified in the SPECTRAL MIXTURE ANALYSIS section below.\n");
    fprintf(fp, "# Type: Character list. Valid values: {BLUE,GREEN,RED,NIR,SWIR1,SWIR2,RE1,\n");
    fprintf(fp, "#   RE2,RE3,BNIR,NDVI,EVI,NBR,NDTI,ARVI,SAVI,SARVI,TC-BRIGHT,TC-GREEN,TC-WET,\n");
    fprintf(fp, "#   TC-DI,NDBI,NDWI,MNDWI,NDMI,NDSI,SMA}\n");
  }
  fprintf(fp, "INDEX = NDVI EVI NBR\n");

  if (verbose){
    fprintf(fp, "# Standardize the TSS time series with pixel mean and/or standard deviation?\n");
    fprintf(fp, "# Type: Logical. Valid values: {NONE,NORMALIZE,CENTER}\n");
  }
  fprintf(fp, "STANDARDIZE_TSS = NONE\n");

  if (verbose){
    fprintf(fp, "# Output the quality-screened Time Series Stack? This is a layer stack of\n");
    fprintf(fp, "# index values for each date.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_TSS = FALSE\n");  
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level unmixing pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_sma(FILE *fp, bool verbose){


  fprintf(fp, "\n# SPECTRAL MIXTURE ANALYSIS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  fprintf(fp, "# This block only applies if INDEX includes SMA\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Endmember file holding the endmembers according to the SENSORS band subset\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_ENDMEM  = NULL\n");

  if (verbose){
    fprintf(fp, "# Sum-to-One constrained unmixing?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "SMA_SUM_TO_ONE = TRUE\n");

  if (verbose){
    fprintf(fp, "# Non-negativity constrained unmixing?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "SMA_NON_NEG = TRUE\n");

  if (verbose){
    fprintf(fp, "# Apply shade normalization? If TRUE, the last endmember FILE_ENDMEM needs\n");
    fprintf(fp, "# to be the shade spectrum\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "SMA_SHD_NORM = TRUE\n");

  if (verbose){
    fprintf(fp, "# Endmember to be used for the analysis. This number refers to the column,\n");
    fprintf(fp, "# in which the desired endmember is stored (FILE_ENDMEM).\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,NUMBER_OF_ENDMEMBERS]\n");
  }
  fprintf(fp, "SMA_ENDMEMBER = 1\n");

  if (verbose){
    fprintf(fp, "# Output the SMA model Error? This is a layer stack of model RMSE for\n");
    fprintf(fp, "# each date.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_RMS = FALSE\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level interpolation pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_tsi(FILE *fp, bool verbose){


  fprintf(fp, "\n# INTERPOLATION PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Interpolation method. You can choose between no, linear, moving average\n");
    fprintf(fp, "# or Radial Basis Function Interpolation.\n");
    fprintf(fp, "# Type: Character. Valid values: {NONE,LINEAR,MOVING,RBF}\n");
  }
  fprintf(fp, "INTERPOLATE = RBF\n");
  
  if (verbose){
    fprintf(fp, "# Max temporal distance for the moving average filter in days. For each\n");
    fprintf(fp, "# interpolation date, MOVING_MAX days before and after are considered.\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,365]\n");
  }
  fprintf(fp, "MOVING_MAX = 16\n");
  
  if (verbose){
    fprintf(fp, "# Sigma (width of the Gaussian bell) for the RBF filter in days. For each\n");
    fprintf(fp, "# interpolation date, a Gaussian kernel is used to smooth the observations.\n");
    fprintf(fp, "# The smoothing effect is stronger with larger kernels and the chance of\n");
    fprintf(fp, "# having nodata values is lower. Smaller kernels will follow the time series\n");
    fprintf(fp, "# more closely but the chance of having nodata values is larger. Multiple\n");
    fprintf(fp, "# kernels can be combined to take advantage of both small and large kernel\n");
    fprintf(fp, "# sizes. The kernels are weighted according to the data density within each\n");
    fprintf(fp, "# kernel.\n");
    fprintf(fp, "# Type: Integer list. Valid range: [1,365]\n");
  }
  fprintf(fp, "RBF_SIGMA = 8 16 32\n");
  
  if (verbose){
    fprintf(fp, "# Cutoff density for the RBF filter. The Gaussian kernels have infinite width,\n");
    fprintf(fp, "# which is computationally slow, and doesn't make much sense as observations\n");
    fprintf(fp, "# that are way too distant (in terms of time) are considered. Thus, the\n");
    fprintf(fp, "# tails of the kernel are cut off. This parameter specifies, which percen-\n");
    fprintf(fp, "# tage of the area under the Gaussian should be used.\n");
    fprintf(fp, "# Type: Float. Valid range: ]0,1]\n");
  }
  fprintf(fp, "RBF_CUTOFF = 0.95\n");
  
  if (verbose){
    fprintf(fp, "# This parameter gives the interpolation step in days.\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,...\n");
  }
  fprintf(fp, "INT_DAY = 16\n");

  if (verbose){
    fprintf(fp, "# Standardize the TSI time series with pixel mean and/or standard deviation?\n");
    fprintf(fp, "# Type: Logical. Valid values: {NONE,NORMALIZE,CENTER}\n");
  }
  fprintf(fp, "STANDARDIZE_TSI = NONE\n");

  if (verbose){
    fprintf(fp, "# Output the Time Series Interpolation? This is a layer stack of index\n");
    fprintf(fp, "# values for each interpolated date. Note that interpolation will be per-\n"); 
    fprintf(fp, "# formed even if OUTPUT_TSI = FALSE - unless you specify INTERPOLATE = NONE.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_TSI = FALSE\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level STM pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_stm(FILE *fp, bool verbose){


  fprintf(fp, "\n# SPECTRAL TEMPORAL METRICS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# Output Spectral Temporal Metrics? The remaining parameters in this block\n");
    fprintf(fp, "# are only evaluated if TRUE\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_STM = FALSE\n");


  if (verbose){
    fprintf(fp, "# Which Spectral Temporal Metrics should be computed? The STM output files\n");
    fprintf(fp, "# will have as many bands as you specify metrics (in the same order).\n");
    fprintf(fp, "# Currently available statistics are the average, standard deviation, mini-\n");
    fprintf(fp, "# mum, maximum, range, skewness, kurtosis, any quantile from 1-99%%, and\n");
    fprintf(fp, "# interquartile range. Note that median is Q50.\n");
    fprintf(fp, "# Type: Character list. Valid values: {MIN,Q01-Q99,MAX,AVG,STD,RNG,IQR,SKW,KRT,NUM}\n");
  }
  fprintf(fp, "STM = Q25 Q50 Q75 AVG STD\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level folding pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_fold(FILE *fp, bool verbose){


  fprintf(fp, "\n# FOLDING PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Which statistic should be used for folding the time series? This parameter\n");
    fprintf(fp, "# is only evaluated if one of the following outputs in this block is requested.\n");
    fprintf(fp, "# Currently available statistics are the average, standard deviation, mini-\n");
    fprintf(fp, "# mum, maximum, range, skewness, kurtosis, median, 10/25/75/90%% quantiles,\n");
    fprintf(fp, "# and interquartile range\n");
    fprintf(fp, "# Type: Character. Valid values: {MIN,Q10,Q25,Q50,Q75,Q90,MAX,AVG,STD,\n");
    fprintf(fp, "#   RNG,IQR,SKW,KRT,NUM\n");
  }
  fprintf(fp, "FOLD_TYPE = AVG\n");

  if (verbose){
    fprintf(fp, "# Standardize the FB* time series with pixel mean and/or standard deviation?\n");
    fprintf(fp, "# Type: Logical. Valid values: {NONE,NORMALIZE,CENTER}\n");
  }
  fprintf(fp, "STANDARDIZE_FOLD = NONE\n");

  if (verbose){
    fprintf(fp, "# Output the Fold-by-Year/Quarter/Month/Week/DOY time series? These are layer\n");
    fprintf(fp, "# stacks of folded index values for each year, quarter, month, week or DOY.\n"); 
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_FBY = FALSE\n");
  fprintf(fp, "OUTPUT_FBQ = FALSE\n");
  fprintf(fp, "OUTPUT_FBM = FALSE\n");
  fprintf(fp, "OUTPUT_FBW = FALSE\n");
  fprintf(fp, "OUTPUT_FBD = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Compute and output a linear trend analysis on any of the folded time series?\n");
    fprintf(fp, "# Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.\n");
    fprintf(fp, "# See also the TREND PARAMETERS block below.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_TRY = FALSE\n");
  fprintf(fp, "OUTPUT_TRQ = FALSE\n");
  fprintf(fp, "OUTPUT_TRM = FALSE\n");
  fprintf(fp, "OUTPUT_TRW = FALSE\n");
  fprintf(fp, "OUTPUT_TRD = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Compute and output an extended Change, Aftereffect, Trend (CAT) analysis on\n");
    fprintf(fp, "# any of the folded time series?\n");
    fprintf(fp, "# Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.\n");
    fprintf(fp, "# See also the TREND PARAMETERS block below.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_CAY = FALSE\n");
  fprintf(fp, "OUTPUT_CAQ = FALSE\n");
  fprintf(fp, "OUTPUT_CAM = FALSE\n");
  fprintf(fp, "OUTPUT_CAW = FALSE\n");
  fprintf(fp, "OUTPUT_CAD = FALSE\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level phenometric pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_lsp(FILE *fp, bool verbose){


  fprintf(fp, "\n# LAND SURFACE PHENOLOGY PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  fprintf(fp, "# The Land Surface Phenology (LSP) options are only available if FORCE was\n");
  fprintf(fp, "# compiled with SPLITS (see installation section in the FORCE user guide).\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# For estimating LSP for one year, some data from the previous/next year\n");
    fprintf(fp, "# need to be considered to find the seasonal minima, which define a season.\n");
    fprintf(fp, "# The parameters are given in DOY, i.e. LSP_DOY_PREV_YEAR = 273, and \n");
    fprintf(fp, "# LSP_DOY_NEXT_YEAR = 91 will use all observations from October (Year-1)\n");
    fprintf(fp, "# to March (Year+1)\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,365]\n");
  }
  fprintf(fp, "LSP_DOY_PREV_YEAR = 273\n");
  fprintf(fp, "LSP_DOY_NEXT_YEAR = 91\n");
  
  if (verbose){
    fprintf(fp, "# Seasonality is of Northern-, Southern-hemispheric or of mixed type? If\n");
    fprintf(fp, "# mixed, the code will attempt to estimate the type on a per-pixel basis.\n");
    fprintf(fp, "# Type: Character. Valid values: {NORTH,SOUTH,MIXED}\n");
  }
  fprintf(fp, "LSP_HEMISPHERE = NORTH\n");
  
  if (verbose){
    fprintf(fp, "# How many segments per year should be used for the spline fitting? More\n");
    fprintf(fp, "# segments follow the seasonality more closely, less segments smooth the\n");
    fprintf(fp, "# time series stronger.\n");
    fprintf(fp, "# Type: Integer. Valid range: [1,...\n");
  }
  fprintf(fp, "LSP_N_SEGMENT = 4\n");
  
  if (verbose){
    fprintf(fp, "# Amplitude threshold for detecing Start, and End of Season, i.e. the date,\n");
    fprintf(fp, "# at which xx%% of the amplitude is observed\n");
    fprintf(fp, "# Type: Float. Valid range: ]0,1[\n");
  }
  fprintf(fp, "LSP_AMP_THRESHOLD = 0.2\n");
  
  if (verbose){
    fprintf(fp, "# LSP won't be derived if the seasonal index values do not exceed following\n");
    fprintf(fp, "# value. This is useful to remove unvegetated surfaces.\n");
    fprintf(fp, "# Type: Integer. Valid range: [-10000,10000]\n");
  }
  fprintf(fp, "LSP_MIN_VALUE = 500\n");
  
  if (verbose){
    fprintf(fp, "# LSP won't be derived if the seasonal amplitude is below following value\n");
    fprintf(fp, "# This is useful to remove surfaces that do not have a seasonality.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,10000]\n");
  }
  fprintf(fp, "LSP_MIN_AMPLITUDE = 500\n");
  
  if (verbose){
    fprintf(fp, "# Which Phenometrics should be computed? There will be a LSP output file for\n");
    fprintf(fp, "# each metric (with years as bands).\n");
    fprintf(fp, "# Currently available are the dates of the early minimum, start of season,\n");
    fprintf(fp, "# rising inflection, peak of season, falling inflection, end of season, late\n");
    fprintf(fp, "# minimum; lengths of the total season, green season; values of the early\n");
    fprintf(fp, "# minimum, start of season, rising inflection, peak of season, falling \n");
    fprintf(fp, "# inflection, end of season, late minimum, base level, seasonal amplitude;\n");
    fprintf(fp, "# integrals of the total season, base level, base+total, green season; rates\n");
    fprintf(fp, "# of averahe rising, average falling, maximum rising, maximum falling.\n");
    fprintf(fp, "# Type: Character list. Valid values: {DEM,DSS,DRI,DPS,DFI,DES,DLM,LTS,LGS,\n");
    fprintf(fp, "#   VEM,VSS,VRI,VPS,VFI,VES,VLM,VBL,VSA,IST,IBL,IBT,IGS,RAR,RAF,RMR,RMF}\n");
  }
  fprintf(fp, "LSP = VSS VPS VES VSA RMR IGS\n");
  
  if (verbose){
    fprintf(fp, "# Standardize the LSP time series with pixel mean and/or standard deviation?\n");
    fprintf(fp, "# Type: Logical. Valid values: {NONE,NORMALIZE,CENTER}\n");
  }
  fprintf(fp, "STANDARDIZE_LSP = NONE\n");
  
  if (verbose){
    fprintf(fp, "# Output the Spline fit? This is a layer stack of fitted index values for\n");
    fprintf(fp, "# interpolated date.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_SPL = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output the Phenometrics? These are layer stacks per phenometric with as many\n");
    fprintf(fp, "# bands as years (excluding one year at the beginning/end of the time series.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_LSP = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Compute and output a linear trend analysis on the requested Phenometric time\n");
    fprintf(fp, "# series? Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.\n");
    fprintf(fp, "# See also the TREND PARAMETERS block below.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_TRP = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Compute and output an extended Change, Aftereffect, Trend (CAT) analysis on\n");
    fprintf(fp, "# the requested Phenometric time series?\n");
    fprintf(fp, "# Note that the OUTPUT_FBX parameters don't need to be TRUE to do this.\n");
    fprintf(fp, "# See also the TREND PARAMETERS block below.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_CAP = FALSE\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level trend pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_trend(FILE *fp, bool verbose){


  fprintf(fp, "\n# TREND PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter specifies the tail-type used for significance testing of\n");
    fprintf(fp, "# the slope in the trend analysis. A left-, two-, or right-tailed t-test\n");
    fprintf(fp, "# is performed.\n");
    fprintf(fp, "# Type: Character. Valid values: {LEFT,TWO,RIGHT}\n");
  }
  fprintf(fp, "TREND_TAIL = TWO\n");
  
  if (verbose){
    fprintf(fp, "# Confidence level for significance testing of the slope in the trend analysis\n");
    fprintf(fp, "# Type: Float. Valid range: [0,1]\n");
  }
  fprintf(fp, "TREND_CONF = 0.95\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level CSO pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_cso(FILE *fp, bool verbose){


  fprintf(fp, "\n# CSO PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter specifies bin width for summarizing the CSOs. The width\n");
    fprintf(fp, "# is given in months\n");
    fprintf(fp, "# Type: Integer. Valid values: [1,12]\n");
  }
  fprintf(fp, "MONTH_STEP = 3\n");
  
  if (verbose){
    fprintf(fp, "# Which statistics should be computed? Currently available statistics are\n");
    fprintf(fp, "# the number of observations, and aggregate statistics of the temporal\n");
    fprintf(fp, "# difference between observations (available are average, standard deviation,\n");
    fprintf(fp, "# minimum, maximum, range, skewness, kurtosis, any quantile from 1-99%%, and\n");
    fprintf(fp, "# interquartile range. Note that median is Q50.\n");
    fprintf(fp, "# Type: Character list. Valid values: {NUM,MIN,Q01-Q99,MAX,AVG,STD,RNG,IQR,SKW,KRT}\n");
  }
  fprintf(fp, "CSO = NUM AVG STD\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level ImproPhe pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_imp(FILE *fp, bool verbose){


  fprintf(fp, "\n# ImproPhe PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the seasonal windows for which the Level 2 ARD\n");
    fprintf(fp, "# should be aggregated. This parameter expects a list of DOYs that define\n");
    fprintf(fp, "# the window breakpoints. If you specify 5 breakpoints, there will be four\n");
    fprintf(fp, "# windows. The windows can extend to the previous/next year (e.g. 270 30\n");
    fprintf(fp, "# 91 181 270 would extend into the previous year, 1 30 91 181 270 30 would\n");
    fprintf(fp, "# extend into the next year.\n");
    fprintf(fp, "# Type: Integer list. Valid values: [1,365]\n");
  }
  fprintf(fp, "SEASONAL_WINDOW = 1 91 181 271 365\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the radius of the prediction kernel (in projection\n");
    fprintf(fp, "# units, commonly in meters). A larger kernel increases the chance of finding\n");
    fprintf(fp, "# a larger number of within-class pixels, but increases prediction time\n");
    fprintf(fp, "# Type: Double. Valid values: ]0,BLOCK_SIZE]\n");
  }
  fprintf(fp, "KERNEL_SIZE = 2500\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the radius of the kernel used for computing the\n");
    fprintf(fp, "# heterogeneity proxies (in projection units, commonly in meters). The\n");
    fprintf(fp, "# heterogeneity proxies are derived from a focal standard deviation filter.\n");
    fprintf(fp, "# The width of the kernel should reflect the scale difference between the\n");
    fprintf(fp, "# coarse and medium resolution data.\n");
    fprintf(fp, "# Type: Double. Valid values: ]0,BLOCK_SIZE]\n");
  }
  fprintf(fp, "KERNEL_TEXT = 330\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level CF ImproPhe pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_cfi(FILE *fp, bool verbose){


  fprintf(fp, "\n# Continuous Field ImproPhe PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Continuous Field datapool (parent directory of tiled continuous fields)\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_COARSE = NULL\n");

  if (verbose){
    fprintf(fp, "# Basenames of the continuous fields (e.g. LSP-POS.tif). The spatial reso-\n");
    fprintf(fp, "# lution of each file provided will be improved. Multiple files can be given\n");
    fprintf(fp, "# as character list; they should represent different continuous fields, e.g.\n");
    fprintf(fp, "# amplitude and peak-of-season phenometrics. Each file can be a multi-band\n");
    fprintf(fp, "# file wherein the bands represent different years. The number of bands, and\n");
    fprintf(fp, "# the corresponding years, need to be the same for all files.\n");
    fprintf(fp, "# Type: List with basename of files\n");
  }
  fprintf(fp, "BASE_COARSE = NULL\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines year, which corresponds to he 1st band of the\n");
    fprintf(fp, "# continuous fields.\n");
    fprintf(fp, "# Type: Integer. Valid values: [1900,2100]\n");
  }
  fprintf(fp, "COARSE_1ST_YEAR = 2000\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the years, for which the spatial resolution should\n");
    fprintf(fp, "# be improved. The corresponding bands of the continuous fields are computed\n");
    fprintf(fp, "# based on this list and the COARSE_1ST_YEAR parameter; please note that the\n");
    fprintf(fp, "# CF ImproPhe module cannot handle skipped or non-ordered years present in the\n");
    fprintf(fp, "# continuous field files.\n");
    fprintf(fp, "# Type: Integer list. Valid values: [1900,2100]\n");
  }
  fprintf(fp, "COARSE_PREDICT_YEARS = 2000 2005 2010 2011 2012 2013 2014 2015\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the nodata value for the continuous fields.\n");
    fprintf(fp, "# Type: Integer. Valid values: [-32767,32767]\n");
  }
  fprintf(fp, "COARSE_NODATA = -32767\n");
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level L2 ImproPhe pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_l2i(FILE *fp, bool verbose){


  fprintf(fp, "\n# Level 2 ImproPhe PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the sensors, whose spatial resolution should be\n");
    fprintf(fp, "# improved. The SENSORS parameter above defines the sensors that serve as\n");
    fprintf(fp, "# target images. For a list of available sensors, see the description for\n");
    fprintf(fp, "# the SENSORS parameter. For improving the spatial resolution of Landsat to\n");
    fprintf(fp, "# Sentinel-2, it is recommended to use \"SENSORS = sen2a sen2b\", and\n");
    fprintf(fp, "# \"SENSORS_LOWRES = LND07 LND08\"\n");
    fprintf(fp, "# Type: Character list. Valid values: {LND04,LND05,LND07,LND08,SEN2A,\n");
    fprintf(fp, "#   SEN2B,sen2a,sen2b,S1AIA,S1BIA,S1AID,S1BID}\n");
  }
  fprintf(fp, "SENSORS_LOWRES = LND07 LND08\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level feature pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_feature(FILE *fp, bool verbose){


  fprintf(fp, "\n# FEATURES\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter specifies the feature(s) used for the analysis. The base-\n");
    fprintf(fp, "# name of a tiled dataset needs to be given, followed by an integer list that\n");
    fprintf(fp, "# specifies the bands that are to be used. This parameter can be given\n");
    fprintf(fp, "# multiple times if multiple features are to be used. The features are used\n");
    fprintf(fp, "# in the same order as given here, thus keep this in mind when training\n");
    fprintf(fp, "# machine learning models with force-train.\n");
    fprintf(fp, "# Type: Basename of file, followed by Integer list\n");
  }
  fprintf(fp, "INPUT_FEATURE = 2018-2018_001-365_LEVEL4_TSA_SEN2L_NDV_STM.tif 1 2 3 4 5 6\n");
  fprintf(fp, "INPUT_FEATURE = 2018-2018_001-365_LEVEL4_TSA_SEN2L_NIR_STM.tif 7 8 9 10 11 12 13\n");
  fprintf(fp, "INPUT_FEATURE = 2018-2018_001-365_LEVEL4_TSA_SEN2L_RED_STM.tif 1 2 3 4 5 6 7 8 9 10 11 12 13\n");
  
  if (verbose){
    fprintf(fp, "# Nodata value of the features.\n");
    fprintf(fp, "# Type: Integer. Valid values: [-32767,32767]\n");
  }
  fprintf(fp, "FEATURE_NODATA = -32767\n");
  
  if (verbose){
    fprintf(fp, "# Should nodata values be excluded if any feature is nodata (TRUE). Or just\n");
    fprintf(fp, "# proceed (FALSE)?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "FEATURE_EXCLUDE = FALSE\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level texture pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_txt(FILE *fp, bool verbose){


  fprintf(fp, "\n# TEXTURE\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the radius of the kernel used for computing the\n");
    fprintf(fp, "# texture metrics (in projection units, commonly in meters).\n");
    fprintf(fp, "# Type: Double. Valid values: ]0,BLOCK_SIZE]\n");
  }
  fprintf(fp, "TXT_RADIUS = 50\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the number of iterations for running the morpho-.\n");
    fprintf(fp, "# logical operations.\n");
    fprintf(fp, "# Type: Integer. Valid values: [1,1000]\n");
  }
  fprintf(fp, "TXT_ITERATION = 1\n");
  
  if (verbose){
    fprintf(fp, "# Which Texture Metrics should be computed? There will be one TXT output file\n");
    fprintf(fp, "# for each metric with as many bands as there are features(in the same order).\n");
    fprintf(fp, "# Currently available metrics are dilation, erosion, opening, closing, gradient,\n");
    fprintf(fp, "# blackhat and tophat.\n");
    fprintf(fp, "# Type: Character list. Valid values: {DIL,ERO,OPN,CLS,GRD,BHT,THT}\n");
  }
  fprintf(fp, "TXT = DIL ERO BHT\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the basename for the output files. The basename will\n");
    fprintf(fp, "# be appended by Module ID, product ID, and the file extension.\n");
    fprintf(fp, "# Type: Character.\n");
  }
  fprintf(fp, "TXT_BASE = TEXTURE\n");
  

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level landscape metrics pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_lsm(FILE *fp, bool verbose){


  fprintf(fp, "\n# Landscape Metrics\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the radius of the kernel used for computing the\n");
    fprintf(fp, "# landscape metrics (in projection units, commonly in meters).\n");
    fprintf(fp, "# Type: Double. Valid values: ]0,BLOCK_SIZE]\n");
  }
  fprintf(fp, "LSM_RADIUS = 50\n");
  
  if (verbose){
    fprintf(fp, "# This parameter determines if the kernel for landscape metrics calculation\n");
    fprintf(fp, "# is circular or squared.\n");
    fprintf(fp, "# # Type: Character. Valid values: {CIRCLE,SQUARE}\n");
  }
  fprintf(fp, "LSM_KERNEL_SHAPE = CIRCLE\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the type of the threshold that is used to define\n");
    fprintf(fp, "# the foreground class (greater then, less than, equal). This parameter is\n");
    fprintf(fp, "# a character list, which defines the threshold type for each feature given.\n");
    fprintf(fp, "# The list needs to be as long as there are features (including bands).\n");
    fprintf(fp, "# Type: Character list. Valid values: {GT,LT,EQ}\n");
  }
  fprintf(fp, "LSM_THRESHOLD_TYPE = EQ LT EQ EQ GT LT EQ LT GT EQ GT EQ GT GT GT LT LT EQ GT GT "
              "GT EQ GT LT LT LT\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the threshold. All pixels that are greater than, lower\n");
    fprintf(fp, "# than or equal to this threshold are defined as foreground class (in dependence\n");
    fprintf(fp, "# of LSM_THRESHOLD_TYPE). Landscape metrics are computed for pixels covererd by\n"); 
    fprintf(fp, "# the foreground class, no metrics are computed for the pixels covered by the\n");
    fprintf(fp, "# background class. This parameter is an integer list, which defines the threshold\n");
    fprintf(fp, "# for each feature given. The list needs to be as long as there are features (in-\n");
    fprintf(fp, "# cluding bands).\n");
    fprintf(fp, "# Type: Integer list. Valid values [-32767,32767]\n");
  }
  fprintf(fp, "LSM_THRESHOLD = 2000 2000 3500 2000 -2000 5000 7500 -3500 500 750 890 999 0 0 0 0 0 "
              "50 5500 1500 300 78 250 500 500 500\n");

  if (verbose){
    fprintf(fp, "# This parameter determines if the landscape metrics are also calculated for \n");
    fprintf(fp, "# pixels covered by the background class.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "LSM_ALL_PIXELS = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Which Landscape Metrics should be computed? There will be one LSM output file\n");
    fprintf(fp, "# for each metric with as many bands as there are features (in the same order).\n");
    fprintf(fp, "# Currently available metrics are unique patch ID, weighted mean patch area, \n");
    fprintf(fp, "# weighted mean fractal dimension index, edge density, number of patches and \n");
    fprintf(fp, "# effective mesh size.\n");
    fprintf(fp, "# Additionally, arithmetic mean, geometric mean, standard deviation and maximum\n");
    fprintf(fp, "# value within the kernel are available.\n");
    fprintf(fp, "# Type: Character list. Valid values: {UCI,MPA,FDI,EDD,NBR,EMS,AVG,GEO,STD,MAX}\n");
  }
  fprintf(fp, "LSM = UCI MPA FDI EDD NBR EMS AVG GEO STD MAX\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the basename for the output files. The basename will\n");
    fprintf(fp, "# be appended by Module ID, product ID, and the file extension.\n");
    fprintf(fp, "# Type: Character.\n");
  }
  fprintf(fp, "LSM_BASE = LSM\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level library completeness pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_lib(FILE *fp, bool verbose){


  fprintf(fp, "\n# Library Completeness\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Directory containing the libraries.\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_LIBRARY = NULL\n");
   
  if (verbose){
    fprintf(fp, "# This parameter specifies the libraries which should be tested against the \n");
    fprintf(fp, "# features. The basename(s) must be given. One or multiple libraries can be\n");
    fprintf(fp, "# given. The output files will have as many bands (+1 overall band) as there\n");
    fprintf(fp, "# are libraries. The libraries should be text files with samples in rows, and\n");
    fprintf(fp, "# features in columns (no header). The column separator is white-space. The\n");
    fprintf(fp, "# features in the library must correspond to the given features.\n");
    fprintf(fp, "# Type: Basename of file, character list\n");
  }
  fprintf(fp, "FILE_LIBRARY = biomass.txt builtup.txt land-cover.txt\n");

  if (verbose){
    fprintf(fp, "# This parameter defines whether the features should be rescaled before\n");
    fprintf(fp, "# testing for library completeness.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "LIB_RESCALE = TRUE\n");

  if (verbose){
    fprintf(fp, "# This parameter defines the basename for the output files. The basename will\n");
    fprintf(fp, "# be appended by Module ID, product ID, and the file extension.\n");
    fprintf(fp, "# Type: Character.\n");
  }
  fprintf(fp, "LIB_BASE = LIB\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level sampling pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_smp(FILE *fp, bool verbose){


  fprintf(fp, "\n# SAMPLING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# File with coordinates, at which the features should be sampled.\n");
    fprintf(fp, "# 1st column: x-coordinate, 2nd column: y-coordinate, 3rd column: response\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_POINTS = NULL\n");

  if (verbose){
    fprintf(fp, "# File with sampled features. This file should not exist.\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_SAMPLE = NULL\n");

  if (verbose){
    fprintf(fp, "# File with the response variable corresponding to the sampled features.\n");
    fprintf(fp, "# This file should not exist.\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_RESPONSE = NULL\n");

  if (verbose){
    fprintf(fp, "# File with the coordinates corresponding to the sampled features.\n");
    fprintf(fp, "# This file should not exist.\n");
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_COORDINATES = NULL\n");

  if (verbose){
    fprintf(fp, "# Are the coordinates in FILE_POINTS in the projection of the datacube\n");
    fprintf(fp, "# (X/Y: TRUE)? Or are they geographic coordinates (Lon/Lat: FALSE)\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "PROJECTED = FALSE\n");

  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level machine learning pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_ml(FILE *fp, bool verbose){


  fprintf(fp, "\n# MACHINE LEARNING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Directory containing the machine learning model(s). Models can be trained\n");
    fprintf(fp, "# with force-train\n");
    fprintf(fp, "# Type: full directory path\n");
  }
  fprintf(fp, "DIR_MODEL = NULL\n");
  
  
  if (verbose){
    fprintf(fp, "# This parameter specifies the modelset used for prediction. The basename\n");
    fprintf(fp, "# of the machine learning model(s) (.xml) must be given. One or multiple\n");
    fprintf(fp, "# models can be given. The predictions of the models are aggregated into\n");
    fprintf(fp, "# the final prediction. The aggregation function is the average for regres-\n");
    fprintf(fp, "# sion problems, and the mode for classification problems. This parameter\n");
    fprintf(fp, "# can be given multiple times, in which case multiple regressions/classifi-\n");
    fprintf(fp, "# cations can be computed. Then output files will have as many bands as\n");
    fprintf(fp, "# modelsets are given.\n");
    fprintf(fp, "# Type: Basename of file, character list\n");
  }
  fprintf(fp, "FILE_MODEL = biomass-1.xml biomass-2.xml biomass-3.xml\n");
  fprintf(fp, "FILE_MODEL = canopy-cover.xml\n");
  fprintf(fp, "FILE_MODEL = tree-height.xml\n");
  
  if (verbose){
    fprintf(fp, "# Machine learning method. Currently implemented are Random Forest and\n");
    fprintf(fp, "# Support Vector Machines, both in regression and classification flavors.\n");
    fprintf(fp, "# The method must match the models as given with FILE_MODEL.\n");
    fprintf(fp, "# Type: Character. Valid values: {SVR,SVC,RFR,RFC}\n");
  }
  fprintf(fp, "ML_METHOD = SVR\n");
  
  if (verbose){
    fprintf(fp, "# This parameter only applies if multiple models are given for a modelset,\n");
    fprintf(fp, "# and machine learning method is of regression flavor. The models are blended\n");
    fprintf(fp, "# into the final prediction, and processing time scales linearly with the\n");
    fprintf(fp, "# number of models given. However, the blended prediction will likely converge\n");
    fprintf(fp, "# with increasing numbers of models, thus it may not be necessary to compute\n");
    fprintf(fp, "# all models. This parameter sets the convergence threshold. If the predic-\n");
    fprintf(fp, "# tions differ less than this value (when adding another model), no more model\n");
    fprintf(fp, "# will be added. This generally speeds up processing substantially. The con-\n");
    fprintf(fp, "# vergence is tested for each pixel, i.e. each pixel is predicted with as many\n");
    fprintf(fp, "# models as necessary to obtain a stable solution.\n");
    fprintf(fp, "# Type: Float. Valid range: [0,...\n");
  }
  fprintf(fp, "ML_CONVERGENCE = 0\n");
  
  if (verbose){
    fprintf(fp, "# This parameter is a scaling factor to scale the prediction to fit into a\n");
    fprintf(fp, "# 16bit signed integer. This parameter should be set in dependence on the\n");
    fprintf(fp, "# scale used for training the model.\n");
    fprintf(fp, "# Type: Float. Valid range: ]0,...\n");
  }
  fprintf(fp, "ML_SCALE = 10000\n");
  
  if (verbose){
    fprintf(fp, "# This parameter defines the basename for the output files. The basename will\n");
    fprintf(fp, "# be appended by Module ID, product ID, and the file extension.\n");
    fprintf(fp, "# Type: Character.\n");
  }
  fprintf(fp, "ML_BASE = PREDICTION\n");

  if (verbose){
    fprintf(fp, "# Output the Machine Learning Prediction?\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_MLP = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output the number of models used when blending the prediction? Makes most\n");
    fprintf(fp, "# sense when ML_CONVERGENCE is used.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_MLI = FALSE\n");
  
  if (verbose){
    fprintf(fp, "# Output the uncertainty of the blended prediction? THis is the standard\n");
    fprintf(fp, "# deviation of all predictions that are blended into the final prediction.\n");
    fprintf(fp, "# Only makes sense when multiple models are given in a modelset.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "OUTPUT_MLU = FALSE\n");
  
  
  return;
}


/** This function writes parameters into a parameter skeleton file: higher
+++ level machine learning training pars
--- fp:      parameter skeleton file
--- verbose: add description, or use more compact format for experts?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void write_par_hl_train(FILE *fp, bool verbose){
  
  
  fprintf(fp, "\n# INPUT\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# File that is holding the features for training (and probably validation).\n");
    fprintf(fp, "# The file needs to be a table with features in columns, and samples in rows.\n");
    fprintf(fp, "# Column delimiter is whitespace. The same number of features must be given\n");
    fprintf(fp, "# for each sample. Do not include a header. The samples need to match the\n");
    fprintf(fp, "# response file.\n");  
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_FEATURES = NULL\n");
  
  if (verbose){
    fprintf(fp, "# File that is holding the response for training (class labels or numeric\n");
    fprintf(fp, "# values). The file needs to be a table with one column, and samples in rows.\n");
    fprintf(fp, "# Do not include a header. The samples need to match the feature file.\n");  
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_RESPONSE = NULL\n");

  fprintf(fp, "\n# OUTPUT\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# File for storing the Machine Learning model in xml format. This file\n");
    fprintf(fp, "# will be overwritten if it exists.\n");  
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_MODEL = NULL\n");

  if (verbose){
    fprintf(fp, "# File for storing the logfile. This file will be overwritten if it exists.\n");  
    fprintf(fp, "# Type: full file path\n");
  }
  fprintf(fp, "FILE_LOG = NULL\n");

  fprintf(fp, "\n# TRAINING\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# This parameter specifies how many samples (in %%) should be used for\n");
    fprintf(fp, "# training the model. The other samples are left out, and used to vali-\n");
    fprintf(fp, "# date the model.\n");
    fprintf(fp, "# Type: Float. Valid range: ]0,100]\n");
  }
  fprintf(fp, "PERCENT_TRAIN = 70\n");
  
  if (verbose){
    fprintf(fp, "# This parameter specifies whether the samples should be randomly drawn (TRUE)\n");
    fprintf(fp, "# or if the first n samples (FALSE) should be used for training.\n");  
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "RANDOM_SPLIT = TRUE\n");
  
  if (verbose){
    fprintf(fp, "# Machine learning method. Currently implemented are Random Forest and\n");
    fprintf(fp, "# Support Vector Machines, both in regression and classification flavors.\n");
    fprintf(fp, "# Type: Character. Valid values: {SVR,SVC,RFR,RFC}\n");
  }
  fprintf(fp, "ML_METHOD = RFC\n");

  fprintf(fp, "\n# RANDOM FOREST PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  fprintf(fp, "# This block only applies if method is Random Forest\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  
  if (verbose){
    fprintf(fp, "# Maximum number of trees in the forest. If RF_OOB_ACCURACY is 0, all trees\n");
    fprintf(fp, "# will be grown. If RF_OOB_ACCURACY is set, the algorithm won't grow addi-\n");
    fprintf(fp, "# tional trees if the accuracy is already met. If set to 0, additional trees\n");
    fprintf(fp, "# are grown until RF_OOB_ACCURACY is met; note that this might never happen.\n");
    fprintf(fp, "# RF_NTREE and RF_OOB_ACCURACY cannot both be 0.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "RF_NTREE = 500\n");
  
  if (verbose){
    fprintf(fp, "# Required accuracy of the ensemble, measured as OOB error. See also RF_NTREE.\n");
    fprintf(fp, "# Type: Float. Valid range: [0,...\n");
  }
  fprintf(fp, "RF_OOB_ACCURACY = 0\n");
  
  if (verbose){
    fprintf(fp, "# The number of randomly selected features at each tree node, which are used\n");
    fprintf(fp, "# to find the best split. If set to 0, the square root of the number of all\n");
    fprintf(fp, "# feature is used for the classification flavor; a third of the number of\n");
    fprintf(fp, "# all feature is used for the regression flavor.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "RF_NFEATURE = 0\n");
  
  if (verbose){
    fprintf(fp, "# This parameter indicates whether the variable importance should be computed.\n");
    fprintf(fp, "# Type: Logical. Valid values: {TRUE,FALSE}\n");
  }
  fprintf(fp, "RF_FEATURE_IMPORTANCE = TRUE\n");
  
  if (verbose){
    fprintf(fp, "# If the number of samples in a node is less than this parameter then the\n");
    fprintf(fp, "# node will not be split. If set to 0, it defaults to 1 for the classi-\n");
    fprintf(fp, "# fication flavor; 5 for the regression flavor.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "RF_DT_MINSAMPLE = 0\n");
  
  if (verbose){
    fprintf(fp, "# The maximum possible depth of the tree. That is the training algorithms\n");
    fprintf(fp, "# attempts to split a node while its depth is less than RF_DT_MAXDEPTH.\n");
    fprintf(fp, "# The root node has zero depth. If set to 0, the maximum possible depth\n");
    fprintf(fp, "# is used.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "RF_DT_MAXDEPTH = 0\n");
  
  if (verbose){
    fprintf(fp, "# Termination criteria for regression trees. If all absolute differences\n");
    fprintf(fp, "# between an estimated value in a node and values of train samples in\n");
    fprintf(fp, "# this node are less than this parameter then the node will not be\n");
    fprintf(fp, "# split further.\n");
    fprintf(fp, "# Type: Float. Valid range: [0.01,...\n");
  }
  fprintf(fp, "RF_DT_REG_ACCURACY = 0.01\n");

  fprintf(fp, "\n# SUPPORT VECTOR MACHINE PARAMETERS\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");
  fprintf(fp, "# This block only applies if method is Support Vector Machine\n");
  fprintf(fp, "# ------------------------------------------------------------------------\n");

  if (verbose){
    fprintf(fp, "# Maximum number of iterations for the iterative SVM training procedure\n");
    fprintf(fp, "# which solves a partial case of constrained quadratic optimization problem.\n");
    fprintf(fp, "# If SVM_ACCURACY is 0, all iterations will be used. If SVM_ACCURACY is set,\n");
    fprintf(fp, "# the algorithm will stop if the accuracy is already met. If set to 0, addi-\n");
    fprintf(fp, "# tional iterations are computed until SVM_ACCURACY is met; note that this\n");
    fprintf(fp, "# might never happen. SVM_MAXITER and SVM_ACCURACY cannot both be 0.\n");
    fprintf(fp, "# Type: Integer. Valid range: [0,...\n");
  }
  fprintf(fp, "SVM_MAXITER = 1000000\n");
  
  if (verbose){
    fprintf(fp, "# Required accuracy of the optimization. See also SVM_MAXITER.\n");
    fprintf(fp, "# Type: Float. Valid range: [0,...\n");
  }
  fprintf(fp, "SVM_ACCURACY = 0.001\n");
  
  if (verbose){
    fprintf(fp, "# Cross-validation parameter. The training set is divided into kFold sub-\n");
    fprintf(fp, "# sets. One subset is used to test the model, the others form the train set.\n");
    fprintf(fp, "# So, the SVM algorithm is executed kFold times.\n");
    fprintf(fp, "# Type: Float. Valid range: [1,...\n");
  }
  fprintf(fp, "SVM_KFOLD = 10\n");
  
  if (verbose){
    fprintf(fp, "# Parameter ϵ of a SVM optimization problem. \n");
    fprintf(fp, "# Type: Float. Valid range: [0,...\n");
  }
  fprintf(fp, "SVM_P = 0\n");
  
  if (verbose){
    fprintf(fp, "# Parameter C of a SVM optimization problem. This parameter expects three\n");
    fprintf(fp, "# values which are used to perform a grid search, i.e. minimum value, \n");
    fprintf(fp, "# maximum value, logarithmic step.\n");
    fprintf(fp, "# Type: Float list. Valid range: [0,...\n");
  }
  fprintf(fp, "SVM_C_GRID = 0.001 10000 1\n");
  
  if (verbose){
    fprintf(fp, "# Parameter γ of a kernel function. This parameter expects three values\n");
    fprintf(fp, "# which are used to perform a grid search, i.e. minimum value, maximum\n");
    fprintf(fp, "# value, logarithmic step.\n");
    fprintf(fp, "# Type: Float list. Valid range: [0,...\n");
  }
  fprintf(fp, "SVM_GAMMA_GRID = 0.000010 10000 10\n");

  return;
}

