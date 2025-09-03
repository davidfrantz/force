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
Datacube header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef CUBE_CL_H
#define CUBE_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions
#include <math.h>    // common mathematical functions

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/lock-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char dir_path[NPOW_10];  // path of datacube
  char def_path[NPOW_10];  // path of datacube definition
  coord_t origin_geo; // origin of grid, geographic
  coord_t origin_map; // origin of grid, destination projection
  int tile_extent[2][2]; // tile extent [_X_, _Y_][_MIN_, _MAX_]
  double tile_size[2];   // tile size in destination unit
  double chunk_size[2];  // chunk size in destination unit
  double resolution[2];  // spatial resolution [_X_][_Y_]
  array_2d_int allowed_tiles; // allow-listed tiles: n x [_X_][_Y_]
  dim_t dim_tiles;     // number of tiles (square extent)
  dim_t dim_chunks;    // number of chunks per tile
  dim_t dim_pixels;    // number of pixels per tile
  dim_t dim_subpixels; // number of pixels per chunk
  char projection[NPOW_10]; // destination projection
} cube_t;

typedef struct{
  int n;
  cube_t **cube;
  bool *cover;
} multicube_t;

cube_t *allocate_datacube();
multicube_t *allocate_multicube(int n);
void free_datacube(cube_t *cube);
void free_multicube(multicube_t *multicube);
void init_datacube(cube_t *cube);
void init_multicube(multicube_t *multicube, int n);
void print_datacube(cube_t *cube);
void print_multicube(multicube_t *multicube);
void update_datacube_extent(cube_t *cube, int tminx, int tmaxx, int tminy, int tmaxy);
void update_datacube_res(cube_t *cube, double res);
int write_datacube_def(cube_t *cube);
cube_t *read_datacube_def(char *d_read);
cube_t *copy_datacube_def(char *d_read, char *d_write, double blockoverride);
int tile_find(double ulx, double uly, double *tilex, double *tiley, int *idx, int *idy, cube_t *cube);
int tile_align(cube_t *cube, double ulx, double uly, double *new_ulx, double *new_uly);

#ifdef __cplusplus
}
#endif

#endif

