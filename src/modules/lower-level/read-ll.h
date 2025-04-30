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
Read Level 1 header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef READ_LL_H
#define READ_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdal.h"           // public (C callable) GDAL entry points

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/brick_io-cl.h"
#include "../cross-level/quality-cl.h"
#include "../cross-level/sun-cl.h"
#include "../cross-level/imagefuns-cl.h"
#include "../cross-level/stats-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/meta-ll.h"
#include "../lower-level/atc-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

int read_level1(meta_t *meta, int mission, brick_t *DN, par_ll_t *pl2);
int bounds_level1(meta_t *meta, brick_t *DN, brick_t **QAI, par_ll_t *pl2);
int impulse_noise_level1(meta_t *meta, brick_t *DN, brick_t *QAI, par_ll_t *pl2);
int convert_level1(meta_t *meta, int mission, atc_t *atc, brick_t *DN, brick_t **toa, brick_t *QAI);

#ifdef __cplusplus
}
#endif

#endif

