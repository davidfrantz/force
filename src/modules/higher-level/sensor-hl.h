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
Sensor header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef SENSOR_HL_H
#define SENSOR_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h>  // boolean data type

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/sys-cl.h"

#include <jansson.h> // JSON library


#ifdef __cplusplus
extern "C" {
#endif

// Level 2 band dictionary
typedef struct {
  int    n;
  int    n_bands;
  int  **band_number;
  char **band_names; 
  char **sensor;
  char  *target;
  char  *main_product;
  char  *quality_product;
  char **aux_products;
  int    n_aux_products;
  int spec_adjust; // spectral band adjustment to S2A?
} sen_t;


int retrieve_sensor(sen_t *sen);

#ifdef __cplusplus
}
#endif

#endif

