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
This file contains functions that handle tiling operations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "tile-cl.h"


/** Read tile allow-list
+++ This function reads the tile allow-list. If none is given, it is gra-
+++ cefully ignored. The X/Y IDs and the number of entries is returned.
--- f_tile: path of tile allow-list
--- X:      X tile coordinate IDs (returned)
--- Y:      Y tile coordinate IDs (returned)
--- k:      number of tiles       (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_readlist(char *f_tile, int **X, int **Y, int *k){
FILE *fp = NULL;
char  buffer[NPOW_10] = "\0";
int i, *x = NULL, *y = NULL;
char cx[5], cy[5];


  // if no file is specified, make sure that processing can continue
  if ((strcmp(f_tile, "NULL") == 0)){
    *k = 0;
    alloc((void**)&x, 1, sizeof(int)); *X = x;
    alloc((void**)&y, 1, sizeof(int)); *Y = y;
    return SUCCESS;
  }


  if ((fp = fopen(f_tile, "r")) == NULL){
    printf("Unable to open tile file!\n"); return FAILURE;}

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Error while reading from tile file!\n"); return FAILURE;}

  // number of positive tiles should be in 1st line
  if ((*k = atoi(buffer)) <= 0){
    printf("Error retrieving number of lines in tile file!\n"); return FAILURE;}

  alloc((void**)&x, *k, sizeof(int)); 
  alloc((void**)&y, *k, sizeof(int));

  // read line by line, one Tile ID should be in each line
  for (i=0; i<*k; i++){

    if (fgets(buffer, NPOW_10, fp) == NULL){
      printf("Error reading line %d in tile file\n", i+2); return FAILURE;}

    strncpy(cx, buffer+1, 4); cx[4] = '\0'; 
    strncpy(cy, buffer+7, 4); cy[4] = '\0'; 
    x[i] = atoi(cx);
    y[i] = atoi(cy);

  }

  fclose(fp);
  
  #ifdef FORCE_DEBUG
  printf("positive tile list contains %d tiles\n", *k);
  #endif

  *X = x;
  *Y = y;
  return SUCCESS;
}


/** Test if tile is allow-listed
+++ This function tests whether a given tile is allow-listed or not.
--- allow_x: allow-listed X tile coordinate IDs
--- allow_y: allow-listed Y tile coordinate IDs
--- allow_k: number of allow-listed tiles
--- x:       x tile coordinate ID
--- y:       y tile coordinate ID
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_allowlisted(int *allow_x, int *allow_y, int allow_k, int x, int y){
int k;
bool allow = false;

  if (allow_k > 0){
    for (k=0; k<allow_k; k++){
      if (x == allow_x[k] && y == allow_y[k]) allow = true;
    }
  } else {
    allow = true;
  }
  
  if (allow) return SUCCESS; else return FAILURE;
}


/** Compile list of active tiles
+++ This function compiles a list of active tiles, i.e. tiles for which
+++ processing should be done. The tile allow-list is read, and intersec-
+++ ted with the processing extent. The X/Y IDs and the number of tiles
+++ are returned.
--- f_tile: path of tile allow-list
--- cube:   datacube struct
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_active(char *f_tile, cube_t *cube){
int tx, ty;
int *allow_x = NULL, *allow_y = NULL, allow_k;


  // read tile allow-list
  if (tile_readlist(f_tile, &allow_x, &allow_y, &allow_k) != SUCCESS){
    printf("Reading tile file failed! "); return FAILURE;}

  // allocate active tile list
  alloc((void**)&cube->tx, cube->tnc, sizeof(int));
  alloc((void**)&cube->ty, cube->tnc, sizeof(int));
  cube->tn = 0;

  // how many tiles really to do?
  for (ty=cube->tminy; ty<=cube->tmaxy; ty++){
  for (tx=cube->tminx; tx<=cube->tmaxx; tx++){

    // if tile is not allowlisted (if specified), skip
    if (tile_allowlisted(allow_x, allow_y, allow_k, tx, ty) == FAILURE) continue;

    cube->tx[cube->tn] = tx;
    cube->ty[cube->tn] = ty;
    cube->tn++;

  }
  }

  free((void*)allow_x);
  free((void*)allow_y);

  // is there any tile to do?
  if (cube->tn == 0){
    printf("No active tile to process. "); return FAILURE;}

  return SUCCESS;
}

