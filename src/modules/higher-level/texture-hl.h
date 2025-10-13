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
Texture header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TEXTURE_HL_H
#define TEXTURE_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../higher-level/read-ard-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  short **ero_;
  short **dil_;
  short **opn_;
  short **cls_;
  short **grd_;
  short **tht_;
  short **bht_;
} txt_t;

brick_t **texture(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct);

#ifdef __cplusplus
}
#endif

#endif

