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
Atmospheric correction header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef ATMO_LL_H
#define ATMO_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/brick-cl.h"
#include "../cross-level/cube-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/atc-ll.h"
#include "../lower-level/topo-ll.h"
#include "../lower-level/radtran-ll.h"
#include "../lower-level/brdf-ll.h"
#include "../lower-level/gas-ll.h"
#include "../lower-level/aod-ll.h"
#include "../lower-level/cloud-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

brick_t **radiometric_correction(par_ll_t *pl2, meta_t *meta, int mission, atc_t *atc, cube_t *cube, brick_t *TOA, brick_t *QAI, brick_t *AOI, top_t *TOP, int *nprod);

#ifdef __cplusplus
}
#endif

#endif

