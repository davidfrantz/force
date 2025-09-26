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
  int    *senid;
  int    nb;
  int  **band;
  char **domain; 
  char **sensor;
  char   target[NPOW_10];
  char  *main_product;
  char  *quality_product;
  char **aux_products;
  int    n_aux_products;

  int spec_adjust; // spectral band adjustment to S2A?

  int blue;
  int green;
  int red;
  int rededge1;
  int rededge2;
  int rededge3;
  int bnir;
  int nir;
  int swir0;
  int swir1;
  int swir2;
  int vv;
  int vh;

  float w_blue;
  float w_green;
  float w_red;
  float w_rededge1;
  float w_rededge2;
  float w_rededge3;
  float w_bnir;
  float w_nir;
  float w_swir0;
  float w_swir1;
  float w_swir2;
  float w_vv;
  float w_vh;
} par_sen_t;


int parse_sensor(par_sen_t *sen);

#ifdef __cplusplus
}
#endif

#endif

