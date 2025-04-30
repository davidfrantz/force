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
BRDF header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef BRDF_LL_H
#define BRDF_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <math.h>    // common mathematical functions

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../lower-level/sunview-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

int brdf_factor(brick_t *sun, brick_t *view, brick_t *cor, int g);
float brdf_forward(float ti, float tv, float phi, float iso, float vol, float geo);
void LiKernel(float hbratio, float brratio, float tantv, float tanti, float sinphi, float cosphi, float *result);
void GetPhaang(float cos1, float cos2, float sin1, float sin2, float cos3, float *cosres, float *res,float *sinres);
void GetpAngles(float brratio, float tan1, float *sinp, float *cosp, float *tanp);
void GetDistance(float tan1, float tan2, float cos3,float *res);
void GetOverlap(float hbratio, float distance, float cos1, float cos2, float tan1, float tan2, float sin3, float *overlap, float *temp);

#ifdef __cplusplus
}
#endif

#endif

