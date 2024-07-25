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
CSO Processing header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef CSO_HL_H
#define CSO_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/cite-cl.h"
#include "../cross-level/brick-cl.h"
#include "../cross-level/stats-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/read-ard-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  short **cso_[NPOW_08];
  date_t *d_cso;
} cso_t;

brick_t **clear_sky_observations(ard_t *ard, brick_t *mask, int nt, par_hl_t *phl, cube_t *cube, int *nproduct);

#ifdef __cplusplus
}
#endif

#endif

