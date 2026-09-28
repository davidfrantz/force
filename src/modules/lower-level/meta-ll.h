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
Level 1 metadata header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef META_LL_H
#define META_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/runtime_data-cl.h"
#include "../cross-level/utils-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/meta_lnd-ll.h"
#include "../lower-level/meta_s2-ll.h"


#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
  float lmax, lmin;     // radiance min/max
  float qmax, qmin;     // quantized DN min/msx
} radiance_cal_t;

typedef struct {
  float scale, offset;     // reflectance scaling factor
} reflectance_cal_t;

typedef struct {
  float k1, k2;         // conversion factors brightness temperature
} temperature_cal_t;

typedef struct {
  radiance_cal_t    radiance;     // radiance calibration
  reflectance_cal_t reflectance;  // reflectance calibration
  temperature_cal_t temperature;  // brightness temperature calibration
  int type;        // _CAL_RAD_, _CAL_BT_, _CAL_REF_
} cal_t;

typedef struct {
  int mission; // mission name
  int band_number;     // number of bands
  string_vector_t image_path; // path to image files
  int saturation;      // saturation value
  int nodata;          // nodata value
  double res;          // spatial resolution
  double ulx, uly;     // upper left corner coordinates
  int nrow, ncol, ncell;      // number of rows and columns
  int col_offset, row_offset; // offset of subset in original image
  date_t date;        // acquisition date
  int epsg;
  int tier;            // processing tier
  cal_t *cal;          // calibration DN->TOA reflectance / BT
  float *wavelength;   // band center wavelength
  char refsys_type[NPOW_10]; // reference system type
  char refsys_id[NPOW_10];  // reference system ID
  simple_view_grid_t view_grid; // view grid
} meta_t;

meta_t *allocate_metadata();
void free_metadata(meta_t *meta);
int init_metadata(meta_t *meta);
int test_metadata(meta_t *meta);
void print_metadata(meta_t *meta);
int parse_metadata_landsat(par_ll_t *pl2, rtd_t *rtd, meta_t *meta);
int parse_metadata_sentinel2(par_ll_t *pl2, rtd_t *rtd, meta_t *meta);
int parse_metadata_mission(par_ll_t *pl2, meta_t *meta);
int parse_metadata(par_ll_t *pl2, rtd_t *rtd, meta_t **metadata);

#ifdef __cplusplus
}
#endif

#endif

