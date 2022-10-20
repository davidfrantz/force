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
Utility functions header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef UTILS_CL_H
#define UTILS_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <math.h>    // common mathematical functions
#include <time.h>    // date and time handling functions
#include <stdbool.h> // boolean data type
#include <float.h>   // macro constants of the floating-point library


#ifdef __cplusplus
extern "C" {
#endif

void print_ivector(int    *v, const char *name, int n, int big);
void print_fvector(float  *v, const char *name, int n, int big, int small);
void print_dvector(double *v, const char *name, int n, int big, int small);
double proctime(time_t start);
void proctime_print(const char *string, time_t start);
void fproctime_print(FILE *fp, const char *string, time_t start);
bool fequal(float a, float b);

#ifdef __cplusplus
}
#endif

#endif

