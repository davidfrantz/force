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
Quality assurance header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef QUALITY_CL_H
#define QUALITY_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick_base-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

short get_qai(brick_t *qai, int index, int p, int bitfields);
short get_qai_from_value(short value, int index, int bitfields);

void set_qai(brick_t *qai, int index, int p, short val);
void set_qai_to_value(short *value, int index, short val);

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

bool get_off_from_value(short qai_value);
char get_cloud_from_value(short qai_value);
bool get_shadow_from_value(short qai_value);
bool get_snow_from_value(short qai_value);
bool get_water_from_value(short qai_value);
char get_aerosol_from_value(short qai_value);
bool get_subzero_from_value(short qai_value);
bool get_saturation_from_value(short qai_value);
bool get_lowsun_from_value(short qai_value);
char get_illumination_from_value(short qai_value);
bool get_slope_from_value(short qai_value);
bool get_vaporfill_from_value(short qai_value);

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

void set_off_to_value(short *value, short val);
void set_cloud_to_value(short *value, short val);
void set_shadow_to_value(short *value, short val);
void set_snow_to_value(short *value, short val);
void set_water_to_value(short *value, short val);
void set_aerosol_to_value(short *value, short val);
void set_subzero_to_value(short *value, short val);
void set_saturation_to_value(short *value, short val);
void set_lowsun_to_value(short *value, short val);
void set_illumination_to_value(short *value, short val);
void set_slope_to_value(short *value, short val);
void set_vaporfill_to_value(short *value, short val);

void merge_qai_from_values(brick_t *qai, int p, short qai_1, short qai_2);

#ifdef __cplusplus
}
#endif

#endif

