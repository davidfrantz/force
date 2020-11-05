/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2020 David Frantz

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

#include "cross-level/param-cl.h"
#include "cross-level/alloc-cl.h"
#include "cross-level/dir-cl.h"
#include "cross-level/const-cl.h"
#include "cross-level/stats-cl.h"
#include "cross-level/brick-cl.h"
#include "cross-level/warp-cl.h"

#include "higher-level/read-ard-hl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdalwarper.h"     // GDAL warper related entry points and defs
#include "ogr_spatialref.h" // coordinate systems services



int main ( int argc, char *argv[] ){

// EQUI7 Asia
char proj[NPOW_10] = "PROJCS[\"Azimuthal_Equidistant\",GEOGCS[\"GCS_WGS_1984\",DATUM[\"D_WGS_1984\",SPHEROID[\"WGS_1984\",6378137.0,298.257223563]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\",0.0174532925199433]],PROJECTION[\"Azimuthal_Equidistant\"],PARAMETER[\"false_easting\",4340913.84808],PARAMETER[\"false_northing\",4812712.92347],PARAMETER[\"longitude_of_center\",94.0],PARAMETER[\"latitude_of_center\",47.0],UNIT[\"Meter\",1.0]]";

double x0, y0;
double x1, y1;
double x2, y2;

  for (x0=0; x0<=180; x0+=45){
  for (y0=-20;  y0<=80;  y0+=20){

    printf("warp (%.0f/%.0f) to ", x0, y0);

    if ((warp_geo_to_any(x0, y0,&x1, &y1, proj)) == FAILURE) return 1;

    printf("(%.0f/%.0f) to ", x1, y1);

    if ((warp_any_to_geo(x1, y1,&x2, &y2, proj)) == FAILURE) return 1;

    printf("(%.0f/%.0f), delta = (%.0f/%.0f)\n", x2, y2, x2-x0, y2-y0);

  }
  }


  return 0; 
}


