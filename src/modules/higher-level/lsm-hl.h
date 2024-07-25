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
Landscape metrics header
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (C) 2013-2022 Franz Schug, David Frantz
Contact: fschug@wisc.edu
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef LANDSCAPE_HL_H
#define LANDSCAPE_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <limits.h>  // macro constants of the integer types


// check if all needed
#include "../cross-level/brick-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/quality-cl.h"
#include "../cross-level/stats-cl.h"
#include "../cross-level/cite-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/read-ard-hl.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  short **mpa_;
  short **uci_;
  short **fdi_;
  short **edd_;
  short **nbr_;
  short **ems_;
  short **avg_;
  short **std_;
  short **geo_;
  short **max_;
  short **are_;
} lsm_t;

brick_t **landscape_metrics(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct);

#ifdef __cplusplus
}
#endif

#endif

