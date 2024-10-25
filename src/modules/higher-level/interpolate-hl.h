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
Time series interpolation header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TS_INTERPOL_HL_H
#define TS_INTERPOL_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/cite-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/tsa-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int     nk;
  int     nbin;
  float **kernel;
  int     *max_ce;
  int      hbin;
  float   *max_w;
} rbf_t;

int tsa_interpolation(tsa_t *ts, small *mask_, int nc, int nt, int nr, int ni, short nodata, par_tsi_t *tsi);

#ifdef __cplusplus
}
#endif

#endif

