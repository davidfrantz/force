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
Topographic effects header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TOPO_LL_H
#define TOPO_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/brick_io-cl.h"
#include "../cross-level/quality-cl.h"
#include "../cross-level/stats-cl.h"
#include "../cross-level/cite-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/atc-ll.h"
#include "../lower-level/radtran-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  brick_t *dem; // binned digital elevation model
  brick_t *exp; // exposition (slope / aspect)
  brick_t *ill; // illumination angle
  brick_t *sky; // sky view factor
  brick_t *c;   // C-factor SWIR2
} top_t;

void free_topography(top_t *top);
int compile_topography(par_ll_t *pl2, atc_t *atc, top_t **topography, brick_t *QAI);
brick_t *cfactor_topography(atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *DEM, brick_t *EXP, brick_t *ILL);
int average_elevation_cell(int g, brick_t *CDEM, brick_t *FDEM, brick_t *QAI);

#ifdef __cplusplus
}
#endif

#endif

