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


//#include "higher-level/read-ard-hl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
//#include "cpl_conv.h"       // various convenience functions for CPL
//#include "gdalwarper.h"     // GDAL warper related entry points and defs
//#include "ogr_spatialref.h" // coordinate systems services



int main ( int argc, char *argv[] ){
short qai[2] = { 12, 74 };
short merged;

  merged = qai[0];

  for ( int i = 0; i < 2; i++ ){
    printf("qai[%d] = %d\n", i, qai[i]);
    printf("off:          %d\n", get_off_from_value(qai[i]));
    printf("cloud:        %d\n", get_cloud_from_value(qai[i]));
    printf("shadow:       %d\n", get_shadow_from_value(qai[i]));
    printf("snow:         %d\n", get_snow_from_value(qai[i]));
    printf("water:        %d\n", get_water_from_value(qai[i]));
    printf("aerosol:      %d\n", get_aerosol_from_value(qai[i]));
    printf("subzero:      %d\n", get_subzero_from_value(qai[i]));
    printf("saturation:   %d\n", get_saturation_from_value(qai[i]));
    printf("lowsun:       %d\n", get_lowsun_from_value(qai[i]));
    printf("illumination: %d\n", get_illumination_from_value(qai[i]));
    printf("slope:        %d\n", get_slope_from_value(qai[i]));
    printf("vaporfill:    %d\n", get_vaporfill_from_value(qai[i]));
    printf("\n");

  }
  
  printf("max cloud:    %d\n", 
    (get_cloud_from_value(qai[0]) > get_cloud_from_value(qai[1])) ? 
     get_cloud_from_value(qai[0]) : get_cloud_from_value(qai[1]));
  printf("max aerosol   %d\n", 
    (get_aerosol_from_value(qai[0]) > get_aerosol_from_value(qai[1])) ? 
     get_aerosol_from_value(qai[0]) : get_aerosol_from_value(qai[1]));




  // Merge off/on flag
  set_off_to_value(&merged,get_off_from_value(qai[0]) && get_off_from_value(qai[1]));

  // Merge cloud flag
  set_cloud_to_value(&merged,
    (get_cloud_from_value(qai[0]) > get_cloud_from_value(qai[1])) ? 
     get_cloud_from_value(qai[0]) : get_cloud_from_value(qai[1]));

  // Merge cloud shadow flag
  set_shadow_to_value(&merged,
    get_shadow_from_value(qai[0]) || get_shadow_from_value(qai[1]));

  // Merge snow flag
  set_snow_to_value(&merged,
    get_snow_from_value(qai[0]) || get_snow_from_value(qai[1]));

  // Merge water flag
  set_water_to_value(&merged,
    get_water_from_value(qai[0]) || get_water_from_value(qai[1]));

  // Merge aerosol flag
  set_aerosol_to_value(&merged,
    (get_aerosol_from_value(qai[0]) > get_aerosol_from_value(qai[1])) ? 
     get_aerosol_from_value(qai[0]) : get_aerosol_from_value(qai[1]));

  // Merge subzero reflectance flag
  set_subzero_to_value(&merged,
    get_subzero_from_value(qai[0]) || get_subzero_from_value(qai[1]));

  // Merge saturated reflectance flag
  set_saturation_to_value(&merged,
    get_saturation_from_value(qai[0]) || get_saturation_from_value(qai[1]));

  // Merge low sun angle flag
  set_lowsun_to_value(&merged,
    get_lowsun_from_value(qai[0]) || get_lowsun_from_value(qai[1]));

  // Merge illumination flag
  set_illumination_to_value(&merged,
    (get_illumination_from_value(qai[0]) < get_illumination_from_value(qai[1])) ? 
     get_illumination_from_value(qai[0]) : get_illumination_from_value(qai[1]));

  // Merge slope flag
  set_slope_to_value(&merged,
    get_slope_from_value(qai[0]) || get_slope_from_value(qai[1]));

  // Merge water vapor fill flag
  set_vaporfill_to_value(&merged,
    get_vaporfill_from_value(qai[0]) || get_vaporfill_from_value(qai[1]));

  printf("merged qai = %d\n", merged);
  printf("off:          %d\n", get_off_from_value(merged));
  printf("cloud:        %d\n", get_cloud_from_value(merged));
  printf("shadow:       %d\n", get_shadow_from_value(merged));
  printf("snow:         %d\n", get_snow_from_value(merged));
  printf("water:        %d\n", get_water_from_value(merged));
  printf("aerosol:      %d\n", get_aerosol_from_value(merged));
  printf("subzero:      %d\n", get_subzero_from_value(merged));
  printf("saturation:   %d\n", get_saturation_from_value(merged));
  printf("lowsun:       %d\n", get_lowsun_from_value(merged));
  printf("illumination: %d\n", get_illumination_from_value(merged));
  printf("slope:        %d\n", get_slope_from_value(merged));
  printf("vaporfill:    %d\n", get_vaporfill_from_value(merged));
  printf("\n");

   return 0; 
}


