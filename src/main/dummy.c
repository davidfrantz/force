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
This program is for testing small things. Needs to be compiled on demand
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include <float.h>   // macro constants of the floating-point library
#include <limits.h>  // macro constants of the integer types
#include <math.h>    // common mathematical functions
#include <stdbool.h> // boolean data type
#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions
#include <ctype.h>   // transform individual characters
#include <time.h>    // date and time handling functions

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing

#include "../modules/cross-level/alloc-cl.h"
#include "../modules/cross-level/const-cl.h"
#include "../modules/cross-level/quality-cl.h"
#include "../modules/cross-level/sys-cl.h"


//#include "higher-level/read-ard-hl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
//#include "cpl_conv.h"       // various convenience functions for CPL
//#include "gdalwarper.h"     // GDAL warper related entry points and defs
//#include "ogr_spatialref.h" // coordinate systems services

#include <jansson.h>

int main ( int argc, char *argv[] ){
char d_exe[NPOW_10];
char f_sensor[NPOW_10];

  get_install_directory(d_exe, NPOW_10);
  concat_string_3(f_sensor, NPOW_10, d_exe, "force-misc/sensors", "LND05.json", "/");


  json_error_t error;
  json_t *root = json_load_file(f_sensor, 0, &error);
  if (!root) {
    fprintf(stderr, "Error: %s\n", error.text);
    return 1;
  }
    // Get description
    json_t *desc = json_object_get(root, "description");
    if (json_is_string(desc)) {
        printf("Description: %s\n", json_string_value(desc));
    } else {
        printf("Description is not a string\n");
    }

  json_t *bands = json_object_get(root, "bands");
  if (json_is_integer(bands)) {
      printf("Bands: %lld\n", json_integer_value(bands));
  } else {
      printf("Bands is not an integer\n");
  }

  // Get band_names array
  json_t *band_names = json_object_get(root, "band_names");
  if (json_is_array(band_names)) {
      printf("Band names:\n");
      size_t i;
      for (i = 0; i < json_array_size(band_names); i++) {
          json_t *name = json_array_get(band_names, i);
          if (json_is_string(name)) {
              printf("  %02d: %s\n", i, json_string_value(name));
          }
      }
  } else {
      printf("band_names is not an array\n");
  }

  json_decref(root);
  
   return 0; 
}


