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
#include "../modules/cross-level/string-cl.h"


//#include "higher-level/read-ard-hl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
//#include "cpl_conv.h"       // various convenience functions for CPL
//#include "gdalwarper.h"     // GDAL warper related entry points and defs
//#include "ogr_spatialref.h" // coordinate systems services


int main ( int argc, char *argv[] ){

  char pattern[100] = "MOD05_L2.A2021202.0020.061.2021202161526.hdf";
  char doy[4] = "ABC";

  overwrite_string_part(pattern, 0, "MCD", 3);
  overwrite_string_part(pattern, 3, "07_L1", 5);
  pattern[22] = '\0';
  
  overwrite_string_part(doy, 0, pattern+14, 3);
  doy[3] = '\0';

  puts(pattern);
  puts(doy);

  exit(0);


   return 0; 
}


