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
Reading ARD header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef READARD_HL_H
#define READARD_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/stack-cl.h"
#include "../cross-level/imagefuns-cl.h"
#include "../cross-level/quality-cl.h"
#include "../higher-level/param-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  stack_t *DAT;
  stack_t *QAI;
  stack_t *DST;
  stack_t *AOD;
  stack_t *HOT;
  stack_t *VZN;
  stack_t *WVP;
  stack_t *MSK;
  short  **dat;  // quantitative data (reflectance, or index)
  short   *qai;  // quality assurance information (bit-coding)
  short   *dst;  // cloud / cloud shadow distance
  short   *aod;  // aerosol optical depth @ 550 nm
  short   *hot;  // haze optimized transformation
  short   *vzn;  // view zenith angle
  short   *wvp;  // water vapor
  small   *msk;  // encoded mask
  //int     sid;  // sensor id
  //date_t  date; // acquisition date
} ard_t;

stack_t *read_mask(int *success, int tx, int ty, int chunk, cube_t *cube, par_hl_t *phl);
ard_t *read_features(int *nt, int tx, int ty, int chunk, cube_t *cube, par_hl_t *phl);
ard_t *read_confield(int *nt, int tx, int ty, int chunk, cube_t *cube, par_hl_t *phl);
ard_t *read_ard(int *nt, int tx, int ty, int chunk, cube_t *cube, par_sen_t *sen, par_hl_t *phl);
stack_t *read_block(char *file, int ard_type, par_sen_t *sen, int read_b, int read_nb, short nodata, int datatype, int chunk, int tx, int ty, cube_t *cube, bool psf, double partial_x, double partial_y);
stack_t *add_blocks(char *file, int ard_type, par_sen_t *sen, int read_b, int read_nb, short nodata, int datatype, int chunk, int tx, int ty, cube_t *cube, bool psf, double radius, stack_t *ARD);
int free_ard(ard_t *ard, int nt);

#ifdef __cplusplus
}
#endif

#endif

