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
Index parsing header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef INDEX_PARSE_HL_H
#define INDEX_PARSE_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h>  // boolean data type

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/sys-cl.h"
#include "../higher-level/sensor-hl.h"

#include <jansson.h> // JSON library


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int n;              // number of indices
  int i;              // current index
  int *type;          // index type
  char **names;       // index names
  char ***band_names; // band names required for each index
  int *n_bands;       // number of bands required for each index
} index_t;

void free_indices(index_t *index);
int retrieve_indices(index_t *index, sen_t *sen);

#ifdef __cplusplus
}
#endif

#endif

