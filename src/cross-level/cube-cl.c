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

  if (cube->tx != NULL) free((void*)cube->tx);
  if (cube->ty != NULL) free((void*)cube->ty);
  cube->tx = NULL;
  cube->ty = NULL;
  cube->tn = 0;

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

  cube->dname[0] = '\0';
  cube->proj[0]  = '\0';
  cube->origin_geo.x = 0;
  cube->origin_geo.y = 0;
  cube->origin_map.x = 0;
  cube->origin_map.y = 0;
  cube->tilesize     = 0;
  cube->chunksize    = 0;
  cube->res          = 0;
  cube->nx           = 0;
  cube->ny           = 0;
  cube->nc           = 0;
  cube->cx           = 0;
  cube->cy           = 0;
  cube->cc           = 0;
  cube->cn           = 0;
  cube->tminx        = 0;
  cube->tmaxx        = 0;
  cube->tminy        = 0;
  cube->tmaxy        = 0;
  cube->tnx          = 0;
  cube->tny          = 0;
  cube->tnc          = 0;
  cube->tn           = 0;
  cube->tx           = NULL;
  cube->ty           = NULL;

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
  printf("Path: %s\n", cube->dname);
  printf("Projection: %s\n", cube->proj);
  printf("Active resolution: %.2f\n", cube->res);
  printf("X-origin (geo): %.2f\n",  cube->origin_geo.x);
  printf("Y-origin (geo): %.2f\n",  cube->origin_geo.y);
  printf("X-origin (map): %.2f\n",  cube->origin_map.x);
  printf("Y-origin (map): %.2f\n",  cube->origin_map.y);
  printf("Tile size: %.2f\n", cube->tilesize);
  printf("Tile pixels: %d x %d = %d\n", cube->nx, cube->ny, cube->nc);
  printf("Tiles X%04d to X%04d\n", cube->tminx, cube->tmaxx);
  printf("Tiles Y%04d to Y%04d\n", cube->tminy, cube->tmaxy);
  printf("Chunk size: %.2f\n", cube->chunksize);
  printf("Chunk pixels: %d x %d = %d\n", cube->cx, cube->cy, cube->cc);
  printf("Number of chunks per tile: %d\n", cube->cn);
  printf("Number of tiles (x): %d\n", cube->tnx);
  printf("Number of tiles (y): %d\n", cube->tny);
  printf("Number of tiles: %d\n", cube->tnc);
  printf("Number of white-listed tiles: %d\n", cube->tn);

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
  cube->tminx = tminx;
  cube->tmaxx = tmaxx;
  cube->tminy = tminy;
  cube->tmaxy = tmaxy;

  // number of tiles
  cube->tnx = cube->tmaxx-cube->tminx+1;
  cube->tny = cube->tmaxy-cube->tminy+1;
  cube->tnc = cube->tnx*cube->tny;
  
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


  if (fmod(cube->tilesize, res) > tol){
    printf("TILE_SIZE must be a multiple of RESOLUTION.\n");
    exit(1);
  }
  if (fmod(cube->chunksize, res) > tol){
    printf("BLOCK_SIZE must be a multiple of RESOLUTION.\n");
    exit(1);
  }

  // resolution
  cube->res = res;
  

  // number of pixels in tile
  cube->nx = cube->ny = (int)(cube->tilesize/cube->res);
  cube->nc = cube->nx*cube->ny;

  // number of pixels in chunk
  cube->cx = cube->nx;
  cube->cy = (int)(cube->chunksize/cube->res);
  cube->cc = cube->cx*cube->cy;
  
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
char fname[NPOW_10];
int nchar;
char *lock = NULL;
FILE *fp = NULL;


  nchar = snprintf(fname, NPOW_10, "%s/datacube-definition.prj", cube->dname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

  if (!fileexist(fname)){
    
    if ((lock = (char*)lock_file(fname, 60)) == NULL) return FAILURE;

    if ((fp = fopen(fname, "w")) == NULL){
      printf("Unable to open %s. ", fname); return FAILURE;}

    fprintf(fp, "%s\n", cube->proj);
    fprintf(fp, "%f\n", cube->origin_geo.x);
    fprintf(fp, "%f\n", cube->origin_geo.y);
    fprintf(fp, "%f\n", cube->origin_map.x);
    fprintf(fp, "%f\n", cube->origin_map.y);
    fprintf(fp, "%f\n", cube->tilesize);
    fprintf(fp, "%f\n", cube->chunksize);

    fclose(fp);

    unlock_file(lock);

  }

  return SUCCESS;
}


/** Read spatial definition of datacube
+++ This function read the text file with all spatial specs of the data
+++ cube.
--- d_read: Input directory
+++ Return: datacube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
cube_t *read_datacube_def(char *d_read){
cube_t *cube = NULL;
char fname[NPOW_10];
char buffer[NPOW_10] = "\0";
int nchar;
FILE *fp = NULL;


  if ((cube = allocate_datacube()) == NULL) return NULL;

  nchar = snprintf(fname, NPOW_10, "%s/datacube-definition.prj", d_read);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  
  copy_string(cube->dname, NPOW_10, d_read);

  if ((fp = fopen(fname, "r")) == NULL){
    printf("Unable to open %s. ", fname); 
    free_datacube(cube); return NULL;
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read projection. ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    copy_string(cube->proj, NPOW_10, buffer);
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read X-origin (geo). ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    cube->origin_geo.x = atof(buffer);
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read Y-origin (geo). ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    cube->origin_geo.y = atof(buffer);
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read X-origin (map). ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    cube->origin_map.x = atof(buffer);
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read Y-origin (map). ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    cube->origin_map.y = atof(buffer);
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read tile size. ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    cube->tilesize = atof(buffer);
    //printf("%s %f\n", buffer, cube->tilesize);
  }

  if (fgets(buffer, NPOW_10, fp) == NULL){
    printf("Unable to read chunk size. ");
    free_datacube(cube); return NULL;
  } else {
    buffer[strcspn(buffer, "\r\n#")] = 0;
    cube->chunksize = atof(buffer);
    //printf("%s %f\n", buffer, cube->chunksize);
  }

  fclose(fp);

  cube->cn = (int)(cube->tilesize/cube->chunksize);

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
--- blockoverride: use a different chunk size than in datacube definition?
+++ Return:        datacube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
cube_t *copy_datacube_def(char *d_read, char *d_write, double blockoverride){
cube_t *cube = NULL;
double tol = 5e-3;


  if ((cube = read_datacube_def(d_read)) == NULL){
    printf("Reading datacube definition failed. "); return NULL;}

  if (blockoverride > 0){
    
    if (fmod(cube->tilesize, blockoverride) > tol){
      printf("TILE_SIZE must be a multiple of BLOCK_SIZE. ");
      return NULL;
    }
   
    cube->cn = (int)(cube->tilesize/blockoverride);
    cube->chunksize = cube->tilesize/cube->cn;

  }

  copy_string(cube->dname, NPOW_10, d_write);

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
      tx += cube->tilesize;
      px++;
    }
    // step back one tile; x should be right of tile-x
    tx -= cube->tilesize;
    px--;

  // if x is left of tile-origin
  } else if (x < tx){

    // shift tile-x to the left until x is right of tile-x
    while (x < tx){
      tx -= cube->tilesize;
      px--;
    }
  }
  // if x is tile-origin, nothing to do

  // if y is below tile-origin
  if (y < ty){

    // shift tile-y to the bottom until y is above tile-y
    while (y < ty){
      ty -= cube->tilesize;
      py++;
    }
    // step back one tile; y should be below tile-y
    ty += cube->tilesize;
    py--;
 
  // if y is above tile-origin
  } else if (y > ty){

    // shift tile-y to the top until y is below tile-y
    while (y > ty){
      ty += cube->tilesize;
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
#ifdef FORCE_DEBUG
printf("new function tile_align. test. THOROUGHLY!!!!\n");
#endif

  // ul coordinate of 1st tile: t0x, t0y
  tile_find(ulx, uly, &t0x, &t0y, &idx, &idy, cube);
  
  // shift t0x to the right until it is right of image-ul, 
  // then step back one pixel and assign as new image-ul
  while (t0x < ulx) t0x += cube->res;
  t0x -= cube->res;
  *new_ulx = t0x;

  // shift t0y to the bottom until it is below image-ul, 
  // then step back one pixel and assign as new image-ul
  while (t0y > uly) t0y -= cube->res;
  t0y += cube->res;
  *new_uly = t0y;

  return SUCCESS;
}

