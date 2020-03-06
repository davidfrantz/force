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
MODIS Atmospheric Water Vapor header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef MODWVP_LL_H
#define MODWVP_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h>  // boolean data type

#include "../cross-level/const-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/download-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

void get_laads_key(char auth[], int size);
int parse_coord_list(char *fcoords, float ***COORD);
int failure_terra(date_t d);
int failure_aqua(date_t d);
int read_modis_geometa(char *fname, char ***fid, float ****gr, bool **v, int *nl, int *nv);
int modis_intersect(float lat, float lon, float ***gr, bool *v, int nl, int nv, int **ptr);
int modis_inside(float lat, float lon, float ***gr, int nintersect, int *ptr);
void compile_modis_wvp(char *dir_geo, char *dir_hdf, date_t d_now, char *sen, int nc, float **COO, double **avg, double **count, char *key);
void choose_modis_wvp(bool aqua, int nc, float *WVP, char **SEN, double *modavg, double *mydavg, double *modctr, double *mydctr);
void write_wvp_lut(char *fname, int nc, float **COO, float *WVP, char **SEN);
void write_avg_table(char *dir_wvp, int nc, float **COO, double ***AVG);
void read_wvp_lut(char *fname, int nc, float **COO, float *WVP);
void create_wvp_lut(char *dir_geo, char *dir_hdf, char *tablename, date_t d_now, int nc, float **COO, float *WVP, char **SEN, char *key);

#ifdef __cplusplus
}
#endif

#endif

