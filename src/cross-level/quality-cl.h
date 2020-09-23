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
Quality assurance header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef QUALITY_CL_H
#define QUALITY_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

void set_qai(brick_t *qai, int index, int p, short val);
short get_qai(brick_t *qai, int index, int p, int bitfields);
bool get_off(brick_t *qai, int p);
char get_cloud(brick_t *qai, int p);
bool get_shadow(brick_t *qai, int p);
bool get_snow(brick_t *qai, int p);
bool get_water(brick_t *qai, int p);
char get_aerosol(brick_t *qai, int p);
bool get_subzero(brick_t *qai, int p);
bool get_saturation(brick_t *qai, int p);
bool get_lowsun(brick_t *qai, int p);
char get_illumination(brick_t *qai, int p);
bool get_slope(brick_t *qai, int p);
bool get_vaporfill(brick_t *qai, int p);
void set_off(brick_t *qai, int p, short val);
void set_cloud(brick_t *qai, int p, short val);
void set_shadow(brick_t *qai, int p, short val);
void set_snow(brick_t *qai, int p, short val);
void set_water(brick_t *qai, int p, short val);
void set_aerosol(brick_t *qai, int p, short val);
void set_subzero(brick_t *qai, int p, short val);
void set_saturation(brick_t *qai, int p, short val);
void set_lowsun(brick_t *qai, int p, short val);
void set_illumination(brick_t *qai, int p, short val);
void set_slope(brick_t *qai, int p, short val);
void set_vaporfill(brick_t *qai, int p, short val);

#ifdef __cplusplus
}
#endif

#endif

