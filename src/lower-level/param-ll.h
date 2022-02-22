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
Level 2 Processing paramater header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef PARAM_LL_H
#define PARAM_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/param-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  params_t *params;
  /** input/output filenames/directories **/
  char f_par[NPOW_10];    // parameter file
  char d_level1[NPOW_10]; // input Level-1 directory
  char b_level1[NPOW_10]; // basename of input Level-1 directory
  char *d_level2;         // output Level-2 directory
  char *d_log;            // logfile directory
  char *d_temp;           // temporary directory
  char *d_out;            // exact output directory
  char *fdem;             // Digital Elevation Model
  char *d_coreg;         // Master image for coregistration
  char *f_tile;           // tile allow-list
  char *f_queue;          // file queue
  char *d_aod;            // directory of AOD LUT
  char *d_wvp;            // directory of water vapor LUT
  char *f_gdalopt;        // file for GDAL options

  /** output parameters **/
  int format; // output format
  int odst;   // flag: output cloud/shadow distance
  int ovzn;   // flag: output view zenith angle
  int ohot;   // flag: output haze optimized transformation
  int oaod;   // flag: output AOD @ 550nm
  int owvp;   // flag: output water vapor
  int oovv;   // flag: output product overview

  /** projection/tiling parameters **/
  int dotile;         // flag: tile
  int doreproj;       // flag: reproject
  char proj[NPOW_10]; // output projection
  char **proj_;       // helper for parsing proj
  int nproj_;         // helper for parsing proj
  
  double orig_lat;      // origin latitude
  double orig_lon;      // origin longitude
  double tilesize;      // tilesize
  double chunksize;     // chunksize
  double res_landsat;   // output resolution Landsat
  double res_sentinel2; // output resolution S2
  double res;           // output resolution
  int resample;         // resampling option

  /** resolution parameters **/
  int psf;       // flag: point spread function
  int vimp;      // flag: use ImproPhe'd data
  int  resmerge; // resolution merge
  
  /** L1 post-processing parameters **/
  int impulse;   // scan for impulse noise?
  int bufnodata; // buffer nodata pixels by 1?

  /** L2 processing parameters **/
  int dotopo;  // flag: topographic correction
  int doatmo;  // flag: atmospheric correction
  int dobrdf;  // flag: BRDF reduction
  int doaod;   // flag: estimate AOD
  int domulti; // flag: multiscattering
  int doenv;   // flag: environment reflectance

  int dem_nodata;    // DEM nodata
  int coreg_nodata;  // Master nodata
  
  float maxcc, maxtc;     // max. allowable cloud cover per scene/tile
  float wvp;              // water vapor dummy value
  float cldprob, shdprob; // Fmask thresholds
  float cldbuf, shdbuf, snwbuf; // buffer sizes
  int erase_cloud;        // erase the clouds?
  int tier;               // tier level

  /** parallel processing **/
  int nproc;   // number of CPUs for multi-processing
  int nthread; // number of CPUs for multi-threading
  int ithread; // use threads for reading bands in parallel?
  int delay;   // delay for starting a new process
  int timeout;   // delay for starting a new process
  
} par_ll_t;

par_ll_t *allocate_param_lower();
void free_param_lower(par_ll_t *pl2);
int parse_param_lower(par_ll_t *pl2);

#ifdef __cplusplus
}
#endif

#endif

