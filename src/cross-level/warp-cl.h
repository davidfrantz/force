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
Warp header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef WARP_CL_H
#define WARP_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

int warp_geo_to_any(double  srs_x, double  srs_y, double *dst_x, double *dst_y, char *dst_wkt);
int warp_any_to_geo(double  srs_x, double  srs_y, double *dst_x, double *dst_y, char *src_wkt);
int warp_any_to_any(double  srs_x, double  srs_y, double *dst_x, double *dst_y, char *src_wkt, char *dst_wkt);

#ifdef __cplusplus
}
#endif

#endif

