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
This file contains functions that define the datacubes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "cube-cl.h"


/** This function allocates a datacube
+++ Return: datacube (must be freed with free_datacube)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
cube_t *allocate_datacube(){
cube_t *cube = NULL;


  alloc((void**)&cube, 1, sizeof(cube_t));
  init_datacube(cube);    

  return cube;
}


/** This function allocates multiple datacubes
--- n:      number of datacubes
+++ Return: multiple datacubes (must be freed with free_multicube)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
multicube_t *allocate_multicube(int n){
multicube_t *multicube = NULL;


  alloc((void**)&multicube, 1, sizeof(multicube_t));
  init_multicube(multicube, n);    

  return multicube;
}


/** This function frees a datacube
--- cube:   datacube
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_datacube(cube_t *cube){

  if (cube == NULL) return;

  if (cube->allowed_tiles != NULL) free_2D((void**)cube->allowed_tiles, 2);
  cube->allowed_tiles = NULL;
  cube->n_allowed_tiles = 0;

  free((void*)cube); cube = NULL;

  return;
}


/** This function frees multiple datacubes
--- cube:   multiple datacubes
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_multicube(multicube_t *multicube){
int c;

  if (multicube == NULL) return;

  for (c=0; c<multicube->n; c++) free_datacube(multicube->cube[c]);
  if (multicube->cube  != NULL) free((void*)multicube->cube);
  if (multicube->cover != NULL) free((void*)multicube->cover);
  multicube->cube  = NULL;
  multicube->cover = NULL;
  multicube->n = 0;

  free((void*)multicube); multicube = NULL;

  return;
}


/** This function initializes a datacube
--- cube:   datacube
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_datacube(cube_t *cube){

  cube->dir_path[0] = '\0';
  cube->def_path[0] = '\0';
  cube->projection[0] = '\0';
  memset(&cube->origin_geo, 0, sizeof(coord_t));
  memset(&cube->origin_map, 0, sizeof(coord_t));
  memset(cube->tile_extent, 0, 2*2*sizeof(int));
  memset(cube->tile_size, 0, 2*sizeof(double));
  cube->resolution = 0;
  memset(&cube->dim_tiles, 0, sizeof(dim_t));
  memset(&cube->dim_tile_pixels, 0, sizeof(dim_t));
  cube->allowed_tiles = NULL;
  cube->n_allowed_tiles = 0;

  return;
}


/** This function initializes multiple datacubes
--- multicube: multiple datacubes
--- n:         number of datacubes
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_multicube(multicube_t *multicube, int n){
int c;

  multicube->n = n;
  alloc((void**)&multicube->cube, n, sizeof(cube_t*));
  for (c=0; c<n; c++) multicube->cube[c] = allocate_datacube();
  alloc((void**)&multicube->cover, n, sizeof(bool));

  return;
}


/** This function prints a datacube
--- cube:   datacube
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_datacube(cube_t *cube){

  printf("Data cube definition:\n");
  printf("Path: %s\n", cube->dir_path);
  printf("Projection: %s\n", cube->projection);
  printf("Active resolution: %.2f\n", cube->resolution);
  printf("Origin (geo): %.2f (X) %.2f (Y)\n",  cube->origin_geo.x, cube->origin_geo.y);
  printf("Origin (map): %.2f (X) %.2f (Y)\n",  cube->origin_map.x, cube->origin_map.y);
  printf("Tile extent: X%04d to X%04d\n", cube->tile_extent[_X_][_MIN_], cube->tile_extent[_X_][_MAX_]);
  printf("Tile extent: Y%04d to Y%04d\n", cube->tile_extent[_Y_][_MIN_], cube->tile_extent[_Y_][_MAX_]);
  printf("Number of tiles: %d (X) x %d (Y) = %d\n", cube->dim_tiles.cols, cube->dim_tiles.rows, cube->dim_tiles.cells);
  printf("Number of allow-listed tiles: %d\n", cube->n_allowed_tiles);
  printf("Tile size: %.2f (X) %.2f (Y)\n", cube->tile_size[_X_], cube->tile_size[_Y_]);
  printf("Number of pixels per tile: %d (X) x %d (Y) = %d\n", cube->dim_tile_pixels.cols, cube->dim_tile_pixels.rows, cube->dim_tile_pixels.cells);

  return;
}


/** This function prints multiple datacubes
--- multicube: multiple datacubes
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_multicube(multicube_t *multicube){
int c;

  printf("Multicube contains %d datacubes\n", multicube->n);
  for (c=0; c<multicube->n; c++){
    printf("\nCube %d is covered with data: %d\n", c, multicube->cover[c]);
    print_datacube(multicube->cube[c]);
  }

  return;
}


/** This function updates the extent and number of tiles in datacube
--- cube:   datacube
--- tminx:  minimzm tile X-ID
--- tmaxx:  maximum tile X-ID
--- tminy:  minimzm tile Y-ID
--- tmaxy:  maximum tile Y-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void update_datacube_extent(cube_t *cube, int tminx, int tmaxx, int tminy, int tmaxy){

  // extent
  cube->tile_extent[_X_][_MIN_] = tminx;
  cube->tile_extent[_X_][_MAX_] = tmaxx;
  cube->tile_extent[_Y_][_MIN_] = tminy;
  cube->tile_extent[_Y_][_MAX_] = tmaxy;

  // number of tiles
  cube->dim_tiles.cols = cube->tile_extent[_X_][_MAX_] - cube->tile_extent[_X_][_MIN_] + 1;
  cube->dim_tiles.rows = cube->tile_extent[_Y_][_MAX_] - cube->tile_extent[_Y_][_MIN_] + 1;
  cube->dim_tiles.cells = cube->dim_tiles.cols * cube->dim_tiles.rows;

  return;
}


/** This function updates the resolution and number of pixels in tile and 
+++ chunk in datacube
--- cube:   datacube
--- res:    resolution
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void update_datacube_res(cube_t *cube, double res){
double tol = 5e-3;


  if (fmod(cube->tile_size[_X_], res) > tol ||
      fmod(cube->tile_size[_Y_], res) > tol){
    printf("TILE_SIZE must be a multiple of RESOLUTION.\n");
    exit(1);
  }

  // resolution
  cube->resolution = res;

  // number of pixels in tile
  cube->dim_tile_pixels.cols = (int)(cube->tile_size[_X_]/cube->resolution);
  cube->dim_tile_pixels.rows = (int)(cube->tile_size[_Y_]/cube->resolution);
  cube->dim_tile_pixels.cells = cube->dim_tile_pixels.cols*cube->dim_tile_pixels.rows;


  return;
}


/** Write spatial definition of datacube
+++ This function writes a text file with all spatial specs of the data
+++ cube. The file contains (1) projection as WKT string, (2) origin of 
+++ the tile system as geographic Longitude, (3) origin of the tile system
+++ as geographic Latitude, (4) width of the tiles in projection units.
--- cube:    datacube
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int write_datacube_def(cube_t *cube){
char *lock = NULL;
FILE *fp = NULL;

  concat_string_2(cube->def_path, NPOW_10, cube->dir_path, "datacube-definition.prj", "/");

  if (!fileexist(cube->def_path)){

    if ((lock = (char*)lock_file(cube->def_path, 60)) == NULL) return FAILURE;

    if ((fp = fopen(cube->def_path, "w")) == NULL){
      printf("Unable to open %s. ", cube->def_path); return FAILURE;}

    fprintf(fp, "PROJECTION = %s\n", cube->projection);
    fprintf(fp, "ORIGIN_GEO_X = %f\n", cube->origin_geo.x);
    fprintf(fp, "ORIGIN_GEO_Y = %f\n", cube->origin_geo.y);
    fprintf(fp, "ORIGIN_MAP_X = %f\n", cube->origin_map.x);
    fprintf(fp, "ORIGIN_MAP_Y = %f\n", cube->origin_map.y);
    fprintf(fp, "TILE_SIZE_X = %f\n", cube->tile_size[_X_]);
    fprintf(fp, "TILE_SIZE_Y = %f\n", cube->tile_size[_Y_]);

    fclose(fp);

    unlock_file(lock);

  }

  return SUCCESS;
}


/** Read spatial definition of datacube
+++ This function read the text file with all spatial specs of the data
+++ cube. This function is for backward compatibility with old
+++ datacube-definition.prj files that had a different format.
--- d_read: Input directory
+++ Return: datacube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
char ***read_datacube_def_deprecated(char *fname, int *nrows){
char ***tagval = NULL;
int nlines_input = 6;
int nlines_output = 7;
char buffer[NPOW_10] = "\0";
FILE *fp = NULL;

char tags_output[7][NPOW_10] = {
  "PROJECTION", 
  "ORIGIN_GEO_X",
  "ORIGIN_GEO_Y",
  "ORIGIN_MAP_X",
  "ORIGIN_MAP_Y",
  "TILE_SIZE_Y",
  "TILE_SIZE_X",
};

  alloc_3D((void****)&tagval, nlines_output, _TV_LENGTH_, NPOW_10, sizeof(char));

  if ((fp = fopen(fname, "r")) == NULL){
    printf("Unable to open %s. ", fname); 
    exit(FAILURE);
  }

  for (int line=0; line<nlines_input; line++){
    if (fgets(buffer, NPOW_10, fp) == NULL){
      printf("Unable to read line %d. ", line);
      exit(FAILURE);
    } else {
      buffer[strcspn(buffer, "\r\n#")] = 0;
      copy_string(tagval[line][_TV_TAG_], NPOW_10, tags_output[line]);
      copy_string(tagval[line][_TV_VAL_], NPOW_10, buffer);
    }
  }

  copy_string(tagval[6][_TV_TAG_], NPOW_10, tags_output[6]);
  copy_string(tagval[6][_TV_VAL_], NPOW_10, tagval[5][_TV_VAL_]);

  fclose(fp);

  *nrows = nlines_output;
  return tagval;
}


/** Read spatial definition of datacube
+++ This function read the text file with all spatial specs of the data
+++ cube.
--- d_read: Input directory
+++ Return: datacube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
cube_t *read_datacube_def(char *d_read){
cube_t *cube = NULL;
char ***input = NULL;
int received = 0, expected = 7;
int nrows = 0;

  if ((cube = allocate_datacube()) == NULL) return NULL;


  copy_string(cube->dir_path, NPOW_10, d_read);
  concat_string_2(cube->def_path, NPOW_10, cube->dir_path, "datacube-definition.prj", "/");

  // read datacube definition
  input = read_tagvalue(cube->def_path, &nrows);

  // if reading with new function failed, try deprecated function
  if (input == NULL){
    printf("WARNING: Reading new datacube definition failed. Try deprecated function.\n");
    input = read_datacube_def_deprecated(cube->def_path, &nrows);
  }

  // parse datacube definition
  for (int line=0; line<nrows; line++){

    if (strcmp(input[line][_TV_TAG_], "PROJECTION") == 0){
      copy_string(cube->projection, NPOW_10, input[line][_TV_VAL_]);
      received++;
    } else if (strcmp(input[line][_TV_TAG_], "ORIGIN_GEO_X") == 0){
      cube->origin_geo.x = atof(input[line][_TV_VAL_]);
      received++;
    } else if (strcmp(input[line][_TV_TAG_], "ORIGIN_GEO_Y") == 0){
      cube->origin_geo.y = atof(input[line][_TV_VAL_]);
      received++;
    } else if (strcmp(input[line][_TV_TAG_], "ORIGIN_MAP_X") == 0){
      cube->origin_map.x = atof(input[line][_TV_VAL_]);
      received++;
    } else if (strcmp(input[line][_TV_TAG_], "ORIGIN_MAP_Y") == 0){
      cube->origin_map.y = atof(input[line][_TV_VAL_]);
      received++;
    } else if (strcmp(input[line][_TV_TAG_], "TILE_SIZE_X") == 0){
      cube->tile_size[_X_] = atof(input[line][_TV_VAL_]);
      received++;
    } else if (strcmp(input[line][_TV_TAG_], "TILE_SIZE_Y") == 0){
      cube->tile_size[_Y_] = atof(input[line][_TV_VAL_]);
      received++;
    } else {
      printf("Unknown tag in datacube definition: %s. ", input[line][_TV_TAG_]);
      free_datacube(cube); free_3D((void***)input, nrows, _TV_LENGTH_);
      return NULL;
    }

  }

  free_3D((void***)input, nrows, _TV_LENGTH_);
  input = NULL;
  nrows = 0;

  if (received != expected){
    printf("Incomplete datacube definition. Expected %d tags, received %d.\n", expected, received);
    free_datacube(cube);
    return NULL;
  }
  
  #ifdef FORCE_DEBUG
  print_datacube(cube);
  #endif

  return cube;
}


/** Copy spatial definition of datacube
+++ This function copies the text file with all spatial specs of the data
+++ cube.
--- d_read:        Input directory
--- d_write:       Output directory
+++ Return:        datacube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
cube_t *copy_datacube_def(char *d_read, char *d_write){
cube_t *cube = NULL;


  if ((cube = read_datacube_def(d_read)) == NULL){
    printf("Reading datacube definition failed. "); return NULL;}

  copy_string(cube->dir_path, NPOW_10, d_write);

  if (write_datacube_def(cube) == FAILURE){
    printf("Writing datacube definition failed. "); 
    free_datacube(cube); return NULL;}

  return cube;
}


/** Tile finder
+++ This function returns the tile ID and tile upper left coordinate of
+++ the requestedcoordinate.
--- x:       x coordinate
--- y:       y coordinate
--- tilex:   upper left x coordinate of tile (returned)
--- tiley:   upper left y coordinate of tile (returned)
--- idx:     x tile coordinate ID (returned)
--- idy:     y tile coordinate ID (returned)
--- cube:    datacube
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_find(double x, double y, double *tilex, double *tiley, int *idx, int *idy, cube_t *cube){
double tx = cube->origin_map.x, ty = cube->origin_map.y;
int px = 0, py = 0;

  // if x is right of tile-origin
  if (x > tx){

    // shift tile-x to the right until x is left of tile-x
    while (x > tx){
      tx += cube->tile_size[_X_];
      px++;
    }
    // step back one tile; x should be right of tile-x
    tx -= cube->tile_size[_X_];
    px--;

  // if x is left of tile-origin
  } else if (x < tx){

    // shift tile-x to the left until x is right of tile-x
    while (x < tx){
      tx -= cube->tile_size[_X_];
      px--;
    }
  }
  // if x is tile-origin, nothing to do

  // if y is below tile-origin
  if (y < ty){

    // shift tile-y to the bottom until y is above tile-y
    while (y < ty){
      ty -= cube->tile_size[_Y_];
      py++;
    }
    // step back one tile; y should be below tile-y
    ty += cube->tile_size[_Y_];
    py--;
 
  // if y is above tile-origin
  } else if (y > ty){

    // shift tile-y to the top until y is below tile-y
    while (y > ty){
      ty += cube->tile_size[_Y_];
      py--;
    }
  }
  // if y is tile-origin, nothing to do

  #ifdef FORCE_DEBUG
  printf("ul x/y of first tile (X%04d_Y%04d): %.2f/%.2f\n", px, py, tx, ty);
  #endif

  *idx   = px; // tile id x
  *idy   = py; // tile id y
  *tilex = tx; // ulx of tile
  *tiley = ty; // uly of tile

  return SUCCESS;
}


/** Align coordinates with tiling scheme
+++ This function takes a coordinate, and returns the nearest coordinate
+++ (to the topleft) that aligns with the tiling scheme. This way, e.g.
+++ a warping operation can exactly match the tiles
--- cube:    datacube
--- ulx:     upper left x coordinate of image
--- uly:     upper left y coordinate of image
--- new_ulx: aligned upper left x coordinate of image (returned)
--- new_uly: aligned upper left y coordinate of image (returned)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_align(cube_t *cube, double ulx, double uly, double *new_ulx, double *new_uly){
double t0x, t0y; // ul x/y of first tile
int idx, idy;   // id of first tile


  // ul coordinate of 1st tile: t0x, t0y
  tile_find(ulx, uly, &t0x, &t0y, &idx, &idy, cube);
  
  // shift t0x to the right until it is right of image-ul, 
  // then step back one pixel and assign as new image-ul
  while (t0x < ulx) t0x += cube->resolution;
  t0x -= cube->resolution;
  *new_ulx = t0x;

  // shift t0y to the bottom until it is below image-ul, 
  // then step back one pixel and assign as new image-ul
  while (t0y > uly) t0y -= cube->resolution;
  t0y += cube->resolution;
  *new_uly = t0y;

  return SUCCESS;
}

