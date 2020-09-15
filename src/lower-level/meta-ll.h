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
Level 1 metadata header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef META_LL_H
#define META_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/stack-cl.h"
#include "../lower-level/table-ll.h"
#include "../lower-level/param-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char  orig_band[NPOW_03];   // Band ID in original file name
  char  fname[NPOW_10]; // file name
  int   fill;           // fill value
  int   rsr_band;       // ID in RSR table
  float lmax, lmin;     // radiance min/max
  float qmax, qmin;     // quantized DN min/msx
  float rmul, radd;     // reflectance scaling factor
  float k1, k2;         // conversion factors brightness temperature
} cal_t;

typedef struct {
  float nodata;
  float **szen, **sazi; // sun  zenith / azimuth from metadata
  float **vzen, **vazi; // view zenith / azimuth from metadata
  int nx, ny;           // number of cells in angle grid
  int left, right;
  int top, bottom;
} s2_meta;

typedef struct {
  int fill;               // fill value
  int dtype;           // data type (bytes)
  int sat;             // saturation value
  cal_t *cal;          // calibration DN->TOA reflectance / BT
  char refsys[NPOW_04];  // worldwide reference system ID
  int tier;            // tier level
  s2_meta s2;          // Sentinel-2 calibration specific
} meta_t;

meta_t *allocate_metadata();
void free_metadata(meta_t *meta);
int init_metadata(meta_t *meta);
cal_t *allocate_calibration(int nb);
void free_calibration(cal_t *cal);
int init_calibration(cal_t *cal);
int check_metadata(par_ll_t *pl2, meta_t *meta, stack_t *DN);
int print_metadata(meta_t *meta, int nb);
int parse_metadata_landsat(par_ll_t *pl2, meta_t *meta, stack_t **dn);
int parse_metadata_sentinel2(par_ll_t *pl2, meta_t *meta, stack_t **dn);
void parse_metadata_band(char *d_level1, char *tag, char *value, cal_t *cal, int lid, int type);
int parse_metadata_mission(par_ll_t *pl2);
int parse_metadata(par_ll_t *pl2, meta_t **metadata, stack_t **DN);

#ifdef __cplusplus
}
#endif

#endif

