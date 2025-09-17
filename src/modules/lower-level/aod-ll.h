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
Aerosol Optical Depth header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef AOD_LL_H
#define AOD_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/imagefuns-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/meta-ll.h"
#include "../lower-level/atc-ll.h"
#include "../lower-level/topo-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

// spectral library
typedef struct {
  int n;     // number of spectra
  float **s; // spectra
} speclib_t;

// single dark target
typedef struct {
  int k;                      // number of valid pixels
  bool valid;                 // valid object?
  int imin, imax, jmin, jmax; // bounding box
  int i, j;                   // location of centroid
  float ms, mv, razi;         // cos of sun & view, relative azimuth
  int e, f, g;              // coarse grid cell
  float  *ttoa, *etoa;        // target and environment reflectance
  
  int r;                      // target radius
  float cosi, sky, z;         // target illumination angle, sky view and elevation
  float Ha, Hr;               // elevation correction factors for AOD/MOD
  float  *aod, *est;       // fitted and estimated AOD
  float rsq;                  // R-squared of best fit, and ID of spectrum
  int lib_id;                 // ID of best spectrum
  int ang_fit;                // type of AOD fit; 0: curved, 1: linear
  float coef[3];              // Angstrom coefficients of best fit
} darkobj_t;

// dark target container
typedef struct {
  darkobj_t *wat;       // water targets
  darkobj_t *shd;       // shadow targets
  darkobj_t *veg;       // vegetation targets
  int kwat, kshd, kveg; // number of candidate targets
  int nwat, nshd, nveg; // number of valid targets
} dark_t;

float *aodfileread(par_ll_t *pl2, atc_t *atc);
int extract_dark_target(atc_t *atc, brick_t *TOA, brick_t *QAI, top_t *TOP, int type, darkobj_t **DOBJ);
int aod_from_target(par_ll_t *pl2, meta_t *meta, atc_t *atc, double res, darkobj_t *dobj, int num, int type);
speclib_t *water_lib(int nb, meta_t *meta);
speclib_t *land_lib(int nb, meta_t *meta);
speclib_t *veg_lib(int nb, int blue, int green, int red);
int aod_lib_to_target(atc_t *atc, double res, bool multi, darkobj_t dobj, speclib_t *lib, int type, float **aodest);
int aod_linear_fit(atc_t *atc, float *logaod, float *angb, float *angn);
int aod_polynomial_fit(atc_t *atc, int naod, float *logaod, float *a0, float *a1, float *a2);
int aod_map(atc_t *atc, dark_t *dark);
int interpolate_aod_map(atc_t *atc, float **map_aod);
int compile_aod(par_ll_t *pl2, meta_t *meta, atc_t *atc, brick_t *TOA, brick_t *QAI, top_t *TOP);

#ifdef __cplusplus
}
#endif

#endif

