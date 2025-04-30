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
Sun/view geometry header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef SUNVIEW_LL_H
#define SUNVIEW_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/brick_io-cl.h"
#include "../cross-level/quality-cl.h"
#include "../cross-level/sun-cl.h"
#include "../lower-level/meta-ll.h"
#include "../lower-level/atc-ll.h"
#include "../lower-level/acix-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

double standard_sunzenith(date_t *dmeta, double lat, double lon);
int sun_target_view(par_ll_t *pl2, meta_t *meta, int mission, atc_t *atc, brick_t *QAI);
int viewgeo(par_ll_t *pl2, brick_t *QAI, atc_t *atc);
int view_angle(meta_t *meta, int mission, atc_t *atc, brick_t *QAI, int f, int e, int g);

#ifdef __cplusplus
}
#endif

#endif

