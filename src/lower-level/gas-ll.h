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
Atmospheric Gas header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef WVP_LL_H
#define WVP_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h>  // boolean data type

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/brick-cl.h"
#include "../cross-level/quality-cl.h"
#include "../lower-level/meta-ll.h"
#include "../lower-level/atc-ll.h"
#include "../lower-level/radtran-ll.h"
#include "../cross-level/cite-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

// water vapor LUT
typedef struct {
  float ***val;
  int nb;
  int nw;
  int nm;
} wvp_lut_t;

// global instance
wvp_lut_t _WVLUT_;

int wvp_transmitt_lut(meta_t *meta, atc_t *atc);
brick_t *water_vapor(meta_t *meta, atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *DEM);
short *gas_transmittance(atc_t *atc, int b, brick_t *WVP, brick_t *QAI);
float ozone_amount(float lon, float lat, int doy);
float water_vapor_from_lut();

#ifdef __cplusplus
}
#endif

#endif

