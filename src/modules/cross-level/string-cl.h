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
#include <stdbool.h> // boolean data type

#include "../cross-level/const-cl.h"
#include "../cross-level/alloc-cl.h"

#ifdef __cplusplus
extern "C" {
#endif

void copy_string(char *dst, size_t size, const char *src);
void concat_string_2(char *dst, size_t size, const char *src1, const char *src2, const char *delim);
void concat_string_3(char *dst, size_t size, const char *src1, const char *src2, const char *src3, const char *delim);
void replace_string(char *src, const char *search, const char *replace, size_t src_len);
int char_to_int(const char *src, int *val);
int char_to_float(const char *src, float *val);
bool strings_equal(const char *str1, const char *str2);
bool vector_contains(const char **vector, size_t size, const char *target);
int vector_contains_pos(const char **vector, size_t size, const char *target);

#ifdef __cplusplus
}
#endif

#endif

