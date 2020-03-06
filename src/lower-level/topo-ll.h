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
Topographic effects header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TOPO_LL_H
#define TOPO_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/stack-cl.h"
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
  stack_t *dem; // binned digital elevation model
  stack_t *exp; // exposition (slope / aspect)
  stack_t *ill; // illumination angle
  stack_t *sky; // sky view factor
  stack_t *c;   // C-factor SWIR2
} top_t;

void free_topography(top_t *top);
int compile_topography(par_ll_t *pl2, atc_t *atc, top_t **topography, stack_t *QAI);
stack_t *cfactor_topography(atc_t *atc, stack_t *TOA, stack_t *QAI, stack_t *DEM, stack_t *EXP, stack_t *ILL);
int average_elevation_cell(int g, stack_t *CDEM, stack_t *FDEM, stack_t *QAI);

#ifdef __cplusplus
}
#endif

#endif

