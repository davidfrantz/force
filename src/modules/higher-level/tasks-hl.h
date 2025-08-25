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
Higher level tasks header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TASKS_HL_H
#define TASKS_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/brick_io-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/lock-cl.h"
#include "../higher-level/progress-hl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/quality-hl.h"
#include "../higher-level/read-ard-hl.h"
#include "../higher-level/read-aux-hl.h"
#include "../higher-level/level3-hl.h"
#include "../higher-level/tsa-hl.h"
#include "../higher-level/cso-hl.h"
#include "../higher-level/ml-hl.h"
#include "../higher-level/texture-hl.h"
#include "../higher-level/lsm-hl.h"
#include "../higher-level/lib-hl.h"
#include "../higher-level/sample-hl.h"
#include "../higher-level/cf-improphe-hl.h"
#include "../higher-level/l2-improphe-hl.h"
#include "../higher-level/spec-adjust-hl.h"
#include "../higher-level/udf-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

void read_higher_level (progress_t *pro, off_t *ibytes, brick_t **MASK, ard_t **ARD1, ard_t **ARD2, int *nt1, int *nt2, cube_t *cube, par_hl_t *phl);
void compute_higher_level (progress_t *pro, brick_t **MASK, ard_t **ARD1, ard_t **ARD2, int *nt1, int *nt2, cube_t *cube, par_hl_t *phl, aux_t *aux, brick_t ***OUTPUT, int *nprod);
void output_higher_level (progress_t *pro, off_t *obytes, brick_t ***OUTPUT, int *nprod, par_hl_t *phl);
int handle_no_io(off_t ibytes, off_t obytes, par_hl_t *phl);

#ifdef __cplusplus
}
#endif

#endif

