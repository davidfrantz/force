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
#include "../cross-level/stack-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

void set_qai(stack_t *qai, int index, int p, short val);
short get_qai(stack_t *qai, int index, int p, int bitfields);
bool get_off(stack_t *qai, int p);
char get_cloud(stack_t *qai, int p);
bool get_shadow(stack_t *qai, int p);
bool get_snow(stack_t *qai, int p);
bool get_water(stack_t *qai, int p);
char get_aerosol(stack_t *qai, int p);
bool get_subzero(stack_t *qai, int p);
bool get_saturation(stack_t *qai, int p);
bool get_lowsun(stack_t *qai, int p);
char get_illumination(stack_t *qai, int p);
bool get_slope(stack_t *qai, int p);
bool get_vaporfill(stack_t *qai, int p);
void set_off(stack_t *qai, int p, short val);
void set_cloud(stack_t *qai, int p, short val);
void set_shadow(stack_t *qai, int p, short val);
void set_snow(stack_t *qai, int p, short val);
void set_water(stack_t *qai, int p, short val);
void set_aerosol(stack_t *qai, int p, short val);
void set_subzero(stack_t *qai, int p, short val);
void set_saturation(stack_t *qai, int p, short val);
void set_lowsun(stack_t *qai, int p, short val);
void set_illumination(stack_t *qai, int p, short val);
void set_slope(stack_t *qai, int p, short val);
void set_vaporfill(stack_t *qai, int p, short val);

#ifdef __cplusplus
}
#endif

#endif

