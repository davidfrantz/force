/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2022 David Frantz

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
Landsat Level 1 metadata header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/



#ifndef META_LND_LL_H
#define META_LND_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <math.h>    // common mathematical functions
#include <float.h>   // macro constants of the floating-point library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/utils-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/runtime_data-cl.h"

#include "../lower-level/param-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char spacecraft_name[NPOW_10];
  char processing_level[NPOW_10];
  int collection_number;
  char collection_category[NPOW_10];
  char landsat_product_id[NPOW_10];
  int wrs_type;
  int wrs_path;
  int wrs_row;
  char wrs_path_row[NPOW_10];
  char date_char[NPOW_10];
  char time_char[NPOW_10];
  date_t date; 
  char **image_files;
  int nodata;
  int saturation;
  int nband;
  double ulx, uly;
  int nrow, ncol, ncell;
  double res;
  int epsg;
  int tier;
  float *reflectance_scale;
  float *reflectance_offset;
  float *radiance_min;
  float *radiance_max;
  float *quantize_min;
  float *quantize_max;
  float *k1;
  float *k2;
} lnd_mtd_t;

void free_metadata_landsat(lnd_mtd_t *mtd);
int parse_metadata_landsat_platform(char *d_level1, lnd_mtd_t *mtd);
int parse_metadata_landsat_mtl(char *d_level1, rtd_t *rtd, lnd_mtd_t *mtd);

#ifdef __cplusplus
}
#endif

#endif

