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
This file contains functions for sampling of features
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "sample-hl.h"


/** private functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

double **parse_coord_list(char *fname, size_t *ncoord);


/** This function reads coordinates from a text file. Put X-coords in 1st
+++ column and Y-coords in 2nd column. Coordinates must be in geographic 
+++ decimal degree notation (South and West coordinates are negative). Do
+++ not use a header.
--- fname:  text file containing the coordinates
--- ncoord: number of coordinate pairs (returned)
+++ Return: array with coordinates
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double **parse_coord_list(char *fname, size_t *ncoord){
FILE *fp;
char  buffer[NPOW_10] = "\0";
char *tag = NULL;
const char *separator = " \t";
double **coords = NULL;
size_t k = 0;
size_t bufsize = NPOW_10;


  alloc_2D((void***)&coords, 3, bufsize, sizeof(double));

  // open file
  if (!(fp = fopen(fname, "r"))){
    printf("unable to open coordinate file. "); return NULL;}

  // process line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){
    
    tag = strtok(buffer, separator);

    coords[0][k] = atof(tag); tag = strtok(NULL, separator);
    coords[1][k] = atof(tag); tag = strtok(NULL, separator);
    coords[2][k] = atof(tag); tag = strtok(NULL, separator);
    k++;

    // if extremely large size, attempt to increase buffer size
    if (k >= bufsize) {
      //printf("reallocate.. %lu %lu\n", k, bufsize);
      re_alloc_2D((void***)&coords, 3, bufsize, 3, bufsize*2, sizeof(coord_t));
      bufsize *= 2;
    }

  }

  fclose(fp);


  *ncoord = k;
  return coords;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the sampling module
--- features: input features
--- mask:      mask image
--- nf:        number of features
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output stacks (returned)
+++ Return:    empty stacks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **sample_points(ard_t *features, stack_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct){
small *mask_ = NULL;
int f, s, i, j, p;
size_t ns;
int cx, cy, chunk, tx, ty;
double res;
double **smp = NULL;
coord_t smp_map, smp_map_ul;
int smp_tx, smp_ty, smp_chunk;
int smp_tj, smp_ti;
int error = 0, added = 0;
bool *found = NULL, valid;
short **sample = NULL;
float *response = NULL;
FILE *fp = NULL;
coord_t geo_ul, geo_ur, geo_lr, geo_ll;
coord_t geo_upper, geo_lower, geo_left, geo_right;
double minx, maxx, miny, maxy; 


  // import stacks
  cx = get_stack_chunkncols(features[0].DAT);
  cy = get_stack_chunknrows(features[0].DAT);
  res = get_stack_res(features[0].DAT);
  chunk = get_stack_chunk(features[0].DAT);
  tx = get_stack_tilex(features[0].DAT);
  ty = get_stack_tiley(features[0].DAT);

//  nodata = get_stack_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }


  // read samples
  smp = parse_coord_list(phl->smp.f_coord, &ns);


  alloc((void**)&found, ns, sizeof(bool));
  alloc_2D((void***)&sample, nf, ns, sizeof(short));
  alloc((void**)&response, ns, sizeof(float));

  
  if (phl->smp.projected){

    // corner coordinates of chunk
    minx = get_stack_x(features[0].DAT, 0);
    maxx = get_stack_x(features[0].DAT, cx);
    miny = get_stack_y(features[0].DAT, (chunk+1)*cy);
    maxy = get_stack_y(features[0].DAT, chunk*cy);

  } else {

    // corner coordinates of chunk
    get_stack_geo(features[0].DAT, 0,    chunk*cy,        &geo_ul.x, &geo_ul.y);
    get_stack_geo(features[0].DAT, cx,   chunk*cy,        &geo_ur.x, &geo_ur.y);
    get_stack_geo(features[0].DAT, 0,    (chunk+1)*cy,    &geo_ll.x, &geo_ll.y);
    get_stack_geo(features[0].DAT, cx,   (chunk+1)*cy,    &geo_lr.x, &geo_lr.y);

    // min/max coordinates
    minx = MIN(geo_ul.x, geo_ll.x);
    maxx = MAX(geo_ur.x, geo_lr.x);
    miny = MIN(geo_ll.y, geo_lr.y);
    maxy = MAX(geo_ul.y, geo_ur.y);

    // edge coordinates of chunk
    get_stack_geo(features[0].DAT, cx/2, chunk*cy,        &geo_upper.x, &geo_upper.y);
    get_stack_geo(features[0].DAT, cx/2, (chunk+1)*cy,    &geo_lower.x, &geo_lower.y);
    get_stack_geo(features[0].DAT, 0,    chunk*cy + cy/2, &geo_left.x,  &geo_left.y);
    get_stack_geo(features[0].DAT, cx,   chunk*cy + cy/2, &geo_right.x, &geo_right.y);

    // if edge coordinates are outside of corner min/max, expand the box a bit
    if (geo_left.x  < minx) minx -= 2*fabs(geo_left.x -minx);
    if (geo_right.x > maxx) maxx += 2*fabs(geo_right.x-maxx);
    if (geo_lower.y < miny) miny -= 2*fabs(geo_lower.y-miny);
    if (geo_upper.y > maxy) maxy += 2*fabs(geo_upper.y-maxy);

  }


  #pragma omp parallel private(smp_map,smp_map_ul,smp_tx,smp_ty,smp_tj,smp_ti,smp_chunk,j,i,p,f,valid) shared(tx,ty,chunk,res,cx,cy,nf,ns,minx,maxx,miny,maxy,found,sample,response,smp,features,mask_,cube,phl) reduction(+: error, added) default(none)
  {

    #pragma omp for
    for (s=0; s<(int)ns; s++){

      if (smp[0][s] < minx) continue;
      if (smp[0][s] > maxx) continue;
      if (smp[1][s] < miny) continue;
      if (smp[1][s] > maxy) continue;
      
      if (phl->smp.projected){

        smp_map.x = smp[0][s];
        smp_map.y = smp[1][s];

      } else {

        // get target coordinates in target css coordinates
        if ((warp_geo_to_any(smp[0][s], smp[1][s], &smp_map.x, &smp_map.y, cube->proj)) == FAILURE){
          printf("Computing target coordinates in dst_srs failed!\n"); 
          error++;
          continue;
        }

      }

      // find the tile the sample falls into
      tile_find(smp_map.x, smp_map.y, &smp_map_ul.x, &smp_map_ul.y, &smp_tx, &smp_ty, cube);

      // if not in current tile, skip
      if (smp_tx != tx || smp_ty != ty) continue;

      // find pixel in tile
      smp_tj = (int)((smp_map.x-smp_map_ul.x)/res);
      smp_ti = (int)((smp_map_ul.y-smp_map.y)/res);

      // find chunk in tile
      smp_chunk = (int)(smp_ti/cy);

      // if not in current chunk, skip
      if (smp_chunk != chunk) continue;

      // find pixel in chunk
      j = smp_tj;
      i = smp_ti - chunk*cy;
      p = i*cx+j;

      // skip pixels that are masked
      if (mask_ != NULL && !mask_[p]) continue;

      // extract
      for (f=0, valid=true; f<nf; f++){
        sample[f][s] = features[f].dat[0][p];
        if (!features[f].msk[p] && phl->ftr.exclude) valid = false;
      }
      response[s] = smp[_Z_][s];
      
      if (!valid) continue;

      // we are done with this sample
      found[s] = true;
      added++;

    }

  }

  if (error > 0) printf("there were %d errors in coordinate conversion..\n", error);


  if (!(fp = fopen(phl->smp.f_sample, "a"))){
    printf("unable to open sample file. "); return NULL;}

  for (s=0; s<(int)ns; s++){

    // if sample was taken, skip
    if (found[s]){
      fprintf(fp, "%d", sample[0][s]);
      for (f=1; f<nf; f++) fprintf(fp, " %d", sample[f][s]);
      fprintf(fp, "\n");
    }

  }

  fclose(fp);
  
  
  if (!(fp = fopen(phl->smp.f_response, "a"))){
    printf("unable to open response file. "); return NULL;}

  for (s=0; s<(int)ns; s++){

    // if sample was taken, skip
    if (found[s]) fprintf(fp, "%f\n", response[s]);

  }

  fclose(fp);


  if (!(fp = fopen(phl->smp.f_coords, "a"))){
    printf("unable to open coordinates file. "); return NULL;}

  for (s=0; s<(int)ns; s++){

    // if sample was taken, skip
    if (found[s]) fprintf(fp, "%f %f\n", smp[_X_][s], smp[_Y_][s]);

  }

  fclose(fp);


  #ifdef FORCE_DEBUG
  if (added > 0) printf("Added %d samples in Tile X%04d_Y%04d Chunk %03d.\n", 
    added, tx, ty, chunk);
  #endif

  free_2D((void**)smp, 3);
  free((void*)found);
  free((void*)response);
  free_2D((void**)sample, nf);

  *nproduct = 0;
  return NULL;
}

