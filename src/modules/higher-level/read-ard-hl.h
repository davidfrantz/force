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
Reading ARD header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef READARD_HL_H
#define READARD_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/imagefuns-cl.h"
#include "../cross-level/quality-cl.h"
#include "../higher-level/param-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  brick_t *DAT;
  brick_t *QAI;
  brick_t *DST;
  brick_t *AOD;
  brick_t *HOT;
  brick_t *VZN;
  brick_t *WVP;
  brick_t *AUX;
  brick_t *MSK;
  short  **dat;  // quantitative data (reflectance, or index)
  short   *qai;  // quality assurance information (bit-coding)
  short   *dst;  // cloud / cloud shadow distance
  short   *aod;  // aerosol optical depth @ 550 nm
  short   *hot;  // haze optimized transformation
  short   *vzn;  // view zenith angle
  short   *wvp;  // water vapor
  short   *aux;  // free-folating auxiliary product
  small   *msk;  // encoded mask
  //int     sid;  // sensor id
  //date_t  date; // acquisition date
} ard_t;

brick_t *read_mask(int *success, off_t *ibytes, int tx, int ty, int chunk, cube_t *cube, par_hl_t *phl);
ard_t *read_features(off_t *ibytes, int *nt, int tx, int ty, int chunk, cube_t *cube, par_hl_t *phl);
ard_t *read_confield(off_t *ibytes, int *nt, int tx, int ty, int chunk, cube_t *cube, par_hl_t *phl);
ard_t *read_ard(off_t *ibytes, int *nt, int tx, int ty, int chunk, cube_t *cube, par_sen_t *sen, par_hl_t *phl);
brick_t *read_block(char *file, int ard_type, par_sen_t *sen, int read_b, int read_nb, short nodata, int datatype, int chunk, int tx, int ty, cube_t *cube, bool psf, double partial_x, double partial_y);
brick_t *add_blocks(char *file, int ard_type, par_sen_t *sen, int read_b, int read_nb, short nodata, int datatype, int chunk, int tx, int ty, cube_t *cube, bool psf, double radius, brick_t *ARD);
int free_ard(ard_t *ard, int nt);

#ifdef __cplusplus
}
#endif

#endif

