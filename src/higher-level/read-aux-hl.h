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
Reading aux files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef READAUX_HL_H
#define READAUX_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/stats-cl.h"
#include "../cross-level/read-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/tsa-hl.h"
#include "../higher-level/lib-hl.h"
#include "../higher-level/ml-hl.h"
#include "../higher-level/sample-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  aux_emb_t endmember;
  aux_lib_t library;
  aux_ml_t  ml;
  aux_smp_t sample;
} aux_t;

aux_t *read_aux(par_hl_t *phl);
void free_aux(par_hl_t *phl, aux_t *aux);

#ifdef __cplusplus
}
#endif

#endif

