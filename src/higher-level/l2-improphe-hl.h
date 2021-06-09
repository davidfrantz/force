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
Level 2 ImproPhe header header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef L2IMP_HL_H
#define L2IMP_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/pca-cl.h"
#include "../cross-level/cite-cl.h"
#include "../higher-level/read-ard-hl.h"
#include "../higher-level/improphe-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  short ***imp_;
} l2i_t;

brick_t **level2_improphe(ard_t *ard_hr, ard_t *ard_mr, brick_t *mask, int nt_hr, int nt_mr, par_hl_t *phl, cube_t *cube, int *nproduct);

#ifdef __cplusplus
}
#endif

#endif

