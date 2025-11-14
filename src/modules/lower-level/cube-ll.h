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
Datacube header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef CUBE_LL_H
#define CUBE_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/string-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/tile-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/brick_io-cl.h"
#include "../cross-level/quality-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/meta-ll.h"
#include "../lower-level/equi7-ll.h"
#include "../lower-level/glance7-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

int cube_level2(par_ll_t *pl2, meta_t *meta, cube_t *cube, brick_t **LEVEL2, int nprod);
multicube_t *start_multicube(par_ll_t *pl2, brick_t *brick);

#ifdef __cplusplus
}
#endif

#endif

