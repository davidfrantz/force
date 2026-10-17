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
This file contains functions that handle tiling operations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "tile-cl.h"


/** Read tile allow-list
+++ This function reads the tile allow-list. If none is given, it is gra-
+++ cefully ignored. The X/Y IDs and the number of entries is returned.
--- f_tile: path of tile allow-list
--- allowed_tiles: allow-listed tiles: [_X_][_Y_] x n
--- number: number of allow-listed tiles
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_readlist(char *f_tile, int ***allowed_tiles, int *number){

  if (f_tile == NULL || allowed_tiles == NULL || number == NULL){
    printf("Invalid input.\n"); 
    return FAILURE;
  }

  // if no file is specified, make sure that processing can continue
  if (strings_equal(f_tile, "NULL")){
    *number = 0;
    *allowed_tiles = NULL;
    return SUCCESS;
  }


  FILE *fp = NULL;
  if ((fp = fopen(f_tile, "r")) == NULL){
    printf("Unable to open tile file!\n"); 
    *number = 0;
    *allowed_tiles = NULL;
    return FAILURE;
  }

  char  buffer[NPOW_10] = "\0";
  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Error while reading from tile file!\n"); 
    *number = 0;
    *allowed_tiles = NULL;
    return FAILURE;
  }

  buffer[strcspn(buffer, "\r\n#")] = 0;

  // number of positive tiles should be in 1st line
  int n;
  char_to_int(buffer, &n);
  if (n <= 0){
    printf("Number of lines in tile file must be positive!\n"); 
    *number = 0;
    *allowed_tiles = NULL;
    return FAILURE;
  }
  
  
  int **tiles = NULL;
  alloc_2D((void***)&tiles, 2, n, sizeof(int)); 
  
  // read line by line, one Tile ID should be in each line
  for (int i=0; i<n; i++){
    
    if (fgets(buffer, NPOW_10, fp) == NULL){
      printf("Error reading line %d in tile file\n", i+2); return FAILURE;}

    buffer[strcspn(buffer, "\r\n#")] = 0;

    char cx[5], cy[5];
    strncpy(cx, buffer+1, 4); cx[4] = '\0'; 
    strncpy(cy, buffer+7, 4); cy[4] = '\0'; 
    tiles[_X_][i] = atoi(cx);
    tiles[_Y_][i] = atoi(cy);

  }

  fclose(fp);
  
  #ifdef FORCE_DEBUG
  printf("Allowed tile list contains %d tiles\n", n);
  #endif

  *number = n;
  *allowed_tiles = tiles;
  return SUCCESS;
}


/** Test if tile is allow-listed
+++ This function tests whether a given tile is allow-listed or not.
+++ If no allow-list is given, all tiles are considered allow-listed.
--- allowed_tiles: allow-listed tiles: [_X_][_Y_] x n 
--- allowed_number: number of allow-listed tiles
--- x:       x tile coordinate ID
--- y:       y tile coordinate ID
+++ Return:  true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool tile_allowlisted(int **allowed_tiles, int allowed_number, int x, int y){

  bool allow = false;

  if (allowed_number > 0 && allowed_tiles != NULL){
    for (int i=0; i<allowed_number; i++){
      if (x == allowed_tiles[_X_][i] && y == allowed_tiles[_Y_][i]) allow = true;
    }
  } else {
    allow = true;
  }
  
  return allow;
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


  if (f_tile == NULL || cube == NULL){
    printf("Invalid input.\n"); 
    return FAILURE;
  }

  // read tile allow-list
  int **allowed_tiles = NULL;
  int allowed_number = 0;
  if (tile_readlist(f_tile, &allowed_tiles, &allowed_number) != SUCCESS){
    printf("Reading tile file failed! "); return FAILURE;}

  // allocate active tile list
  alloc_2D((void***)&cube->allowed_tiles, 2, cube->dim_tiles.cells, sizeof(int));
  cube->n_allowed_tiles = 0;

  // how many tiles really to do?
  for (int ty=cube->tile_extent[_Y_][_MIN_]; ty<=cube->tile_extent[_Y_][_MAX_]; ty++){
  for (int tx=cube->tile_extent[_X_][_MIN_]; tx<=cube->tile_extent[_X_][_MAX_]; tx++){

    // if tile is not allowlisted (if specified), skip
    if (!tile_allowlisted(allowed_tiles, allowed_number, tx, ty)) continue;

    cube->allowed_tiles[_X_][cube->n_allowed_tiles] = tx;
    cube->allowed_tiles[_Y_][cube->n_allowed_tiles] = ty;
    cube->n_allowed_tiles++;

  }
  }

  if (allowed_tiles != NULL) free_2D((void**)allowed_tiles, 2);

  // is there any tile to do?
  if (cube->n_allowed_tiles == 0){
    printf("No active tile to process. "); return FAILURE;}

  return SUCCESS;
}

