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
Brick I/O header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef BRICK_IO_CL_H
#define BRICK_IO_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library'

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/datesys-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/quality-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

int      warp_from_brick_to_unknown_brick(bool tile, int rsm, int threads, brick_t *src, cube_t *cube);
int      warp_from_disc_to_known_brick(int rsm, int threads, const char *fname, brick_t *dst, int src_b, int dst_b, int src_nodata);
int      write_brick(brick_t *brick);

#ifdef __cplusplus
}
#endif

#endif

