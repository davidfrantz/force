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
Sentinel-2 Level 1 metadata header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/



#ifndef META_S2_LL_H
#define META_S2_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <math.h>    // common mathematical functions
#include <float.h>   // macro constants of the floating-point library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/utils-cl.h"
#include "../cross-level/xml-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/runtime_data-cl.h"

#include "../lower-level/param-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double res;
  int nrow, ncol, ncell;
  //float *sun_z, *sun_a;
  float ***zen, ***azi;
  float nodata;
} full_view_grid_t;

typedef struct {
  double res;
  int nrow, ncol, ncell;
  float *zen, *azi;
  float nodata;
} simple_view_grid_t;

typedef struct {
  char spacecraft_name[NPOW_10];
  char processing_level[NPOW_10];
  char **image_files;
  char granule_path[NPOW_10];
  float processing_baseline;
  date_t date; 
  int nband;
  int ndetector;
  double ulx, uly;
  int nrow, ncol, ncell;
  int col_offset, row_offset;
  double res;
  int epsg;
  float scale;
  float *offset;
  int nodata;
  int saturation;
  seq_t *rsr;
  full_view_grid_t view_grid;
  simple_view_grid_t subset_view_grid;
} s2_mtd_t;

void free_metadata_sentinel2(s2_mtd_t *mtd);

int parse_mgrs_tile(char *b_level1, char *mgrs, size_t size);
int parse_metadata_sentinel2_platform(char *d_level1, s2_mtd_t *mtd);
int parse_metadata_sentinel2_safe(char *d_level1, rtd_t *rtd, s2_mtd_t *mtd);
int parse_metadata_sentinel2_granule(char *d_level1, int detector_number, rtd_t *rtd, s2_mtd_t *mtd);

int construct_sentinel2_view_grid(par_ll_t *pl2, s2_mtd_t *mtd);
void interpolate_sentinel2_view_grid(s2_mtd_t *mtd);
void collapse_sentinel2_view_grid(s2_mtd_t *mtd);
void subset_sentinel2_view_grid(par_ll_t *pl2, s2_mtd_t *mtd);

#ifdef __cplusplus
}
#endif

#endif

