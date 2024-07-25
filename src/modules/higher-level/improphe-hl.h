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
ImproPhe header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef IMP_HL_H
#define IMP_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/stats-cl.h"
#include "../higher-level/read-ard-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

int improphe(float **hr_, float *hr_tex_, float **mr_, float **mr_tex_, short **pred_, float **KDIST, float nodata_hr, short nodata_mr, int i, int j, int p, int nx, int ny, int h, int nb_hr, int nb_mr, int nk, int mink);
double rescale_weight(double weight, double minweight, double maxweight);
short **average_season(ard_t *ard, small *mask_, int nb, int nc, int nt, short nodata, int nwin, int *dwin, int ywin, bool *is_empty);
int standardize_float(float *data, float nodata, int nc);
float *focal_sd(float **DAT, float nodata, int h, int nx, int ny, int nb, int bstart);

#ifdef __cplusplus
}
#endif

#endif

