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
GDAL options
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef GDALOPT_CL_H
#define GDALOPT_CL_H

#include <stdio.h>   // core input and output functions
#include <stdbool.h>  // boolean data type
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/read-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  string_t driver;    // GDAL driver short name
  string_t extension; // file extension
  string_vector_t options[_TV_LENGTH_]; // GDAL output options as string vector (tag and value)
} gdalopt_t;

void default_gdaloptions(int format, gdalopt_t *gdalopt);
void parse_gdaloptions(char *fname, gdalopt_t *gdalopt);
void print_gdaloptions(gdalopt_t *gdalopt);
void copy_gdaloptions(gdalopt_t *dst, gdalopt_t *src);
void free_gdaloptions(gdalopt_t *gdalopt);

#ifdef __cplusplus
}
#endif

#endif

