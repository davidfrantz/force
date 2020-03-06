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


/** Read tile white-list
+++ This function reads the tile white-list. If none is given, it is gra-
+++ cefully ignored. The X/Y IDs and the number of entries is returned.
--- f_tile: path of tile white list
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


/** Test if tile is white-listed
+++ This function tests whether a given tile is white-listed or not.
--- white_x: white-listed X tile coordinate IDs
--- white_y: white-listed Y tile coordinate IDs
--- white_k: number of white-listed tiles
--- x:       x tile coordinate ID
--- y:       y tile coordinate ID
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_whitelisted(int *white_x, int *white_y, int white_k, int x, int y){
int k;
bool white = false;

  if (white_k > 0){
    for (k=0; k<white_k; k++){
      if (x == white_x[k] && y == white_y[k]) white = true;
    }
  } else {
    white = true;
  }
  
  if (white) return SUCCESS; else return FAILURE;
}


/** Compile list of active tiles
+++ This function compiles a list of active tiles, i.e. tiles for which
+++ processing should be done. The tile white-list is read, and intersec-
+++ ted with the processing extent. The X/Y IDs and the number of tiles
+++ are returned.
--- f_tile: path of tile white list
--- cube:   datacube struct
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_active(char *f_tile, cube_t *cube){
int tx, ty;
int *white_x = NULL, *white_y = NULL, white_k;


  // read tile white-list
  if (tile_readlist(f_tile, &white_x, &white_y, &white_k) != SUCCESS){
    printf("Reading tile file failed! "); return FAILURE;}

  // allocate active tile list
  alloc((void**)&cube->tx, cube->tnc, sizeof(int));
  alloc((void**)&cube->ty, cube->tnc, sizeof(int));
  cube->tn = 0;

  // how many tiles really to do?
  for (ty=cube->tminy; ty<=cube->tmaxy; ty++){
  for (tx=cube->tminx; tx<=cube->tmaxx; tx++){

    // if tile is not whitelisted (if specified), skip
    if (tile_whitelisted(white_x, white_y, white_k, tx, ty) == FAILURE) continue;

    cube->tx[cube->tn] = tx;
    cube->ty[cube->tn] = ty;
    cube->tn++;

  }
  }

  free((void*)white_x);
  free((void*)white_y);

  // is there any tile to do?
  if (cube->tn == 0){
    printf("No active tile to process. "); return FAILURE;}

  return SUCCESS;
}

