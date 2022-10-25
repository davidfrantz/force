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
String handling header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef STRING_CL_H
#define STRING_CL_H

#include <stdio.h>    // core input and output functions
#include <stdlib.h>   // standard general utilities library
#include <string.h>   // string handling functions
#include <limits.h>   // macro constants of the integer types
#include <errno.h>    // error numbers

#include "../cross-level/const-cl.h"

#ifdef __cplusplus
extern "C" {
#endif

void copy_string(char *dst, size_t size, const char *src);
int char_to_int(const char *src, int *val);
int char_to_float(const char *src, float *val);

#ifdef __cplusplus
}
#endif

#endif

