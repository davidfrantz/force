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
BAP selection header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef BAP_HL_H
#define BAP_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../higher-level/param-hl.h"
#include "../higher-level/level3-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float ce[3];     // target DOY, i.e. DOY triplet p0, p1, p2
  float a, b; // fitted parameters for logistic S-curve or Gaussian
} target_t;

target_t *compile_target_static(par_bap_t *bap);
target_t *compile_target_adaptive(par_bap_t *bap, ard_t *lsp, int p, short nodata);
bool pixel_is_water(ard_t *ard, int nt, int p);
int corr_matrix(ard_t *ard, int nt, int nb, int p, float **cor);
int haze_stats(ard_t *ard, int nt, int p, par_scr_t *score, par_bap_t *bap, float *mean, float *sd);
int water_score(ard_t *ard, int nt, int p, par_scr_t *score);
int parametric_score(ard_t *ard, int nt, int p, target_t *target, float **cor, par_scr_t *score, int *tdist, par_bap_t *bap);
int bap_compositing(ard_t *ard, level3_t *level3, int nt, int nb, short nodata, int p, par_scr_t *score, int *tdist, float hmean, float hsd, bool water, par_bap_t *bap);
int bap_overview(level3_t *l3, int nx, int ny, int nb, double res, short nodata);

#ifdef __cplusplus
}
#endif

#endif

