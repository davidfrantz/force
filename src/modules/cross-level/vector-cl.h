/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2025 David Frantz

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
Vector header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef VECTOR_CL_H
#define VECTOR_CL_H

#include <stdio.h>   // core input and output functions
#include <string.h>  // string handling functions

#include "gdal.h"           // public (C callable) GDAL entry points

#include "../cross-level/const-cl.h"
#include "../cross-level/brick-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

GDALDatasetH warp_vector_from_disc(char *input_path, const char *proj);
brick_t *rasterize_vector_from_memory(GDALDatasetH vector_dataset, brick_t *destination_brick);
brick_t *rasterize_vector_from_disc(char *input_path, brick_t *destination_brick);

#ifdef __cplusplus
}
#endif

#endif

