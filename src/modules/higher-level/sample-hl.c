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
This file contains functions for sampling of features
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "sample-hl.h"


/** private functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))


void append_table(char *fname, bool *allow, double **tab, int nrow, int ncol, int decimals);

void append_table(char *fname, bool *allow, double **tab, int nrow, int ncol, int decimals){
int row, col;
FILE *fp = NULL;


  if (!(fp = fopen(fname, "a"))){
    printf("unable to open %s. ", fname); exit(FAILURE);}

  for (row=0; row<nrow; row++){

    if (allow[row]){

      fprintf(fp, "%.*f", decimals, tab[row][0]);
      for (col=1; col<ncol; col++) fprintf(fp, " %.*f", decimals, tab[row][col]);
      fprintf(fp, "\n");
    }

  }

  fclose(fp);

  return;
}






/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the sampling module
--- features: input features
--- mask:      mask image
--- nf:        number of features
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    empty bricks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **sample_points(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, table_t *smp, cube_t *cube, int *nproduct){
small *mask_ = NULL;
int f, r, s, i, j, p;
int cx, cy, chunk[2], tile[2];
int nr;
double res;
coord_t smp_map, smp_map_ul;
int smp_tx, smp_ty, smp_chunk[2];
int smp_tj, smp_ti;
int error = 0, found = 0, added = 0;
bool *copied = NULL, valid;
double **smp_features = NULL;
double **smp_response = NULL;
coord_t geo_ul, geo_ur, geo_lr, geo_ll;
coord_t geo_upper, geo_lower, geo_left, geo_right;
double minx, maxx, miny, maxy; 


  // if no sample is left, skip all
  if (smp->n_active_rows == 0){
    *nproduct = 0;
    return NULL;
  }


  // import bricks
  cx    = get_brick_chunkncols(features[0].DAT);
  cy    = get_brick_chunknrows(features[0].DAT);
  res   = get_brick_res(features[0].DAT);
  chunk[_X_] = get_brick_chunkx(features[0].DAT);
  chunk[_Y_] = get_brick_chunky(features[0].DAT);
  tile[_X_]  = get_brick_tilex(features[0].DAT);
  tile[_Y_]  = get_brick_tiley(features[0].DAT);

//  nodata = get_brick_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }


  nr = smp->ncol-2;

  alloc((void**)&copied,           smp->nrow,     sizeof(bool));
  alloc_2D((void***)&smp_features, smp->nrow, nf, sizeof(double));
  alloc_2D((void***)&smp_response, smp->nrow, nr, sizeof(double));


  if (phl->smp.projected){

    // corner coordinates of chunk
    minx = get_brick_x(features[0].DAT, chunk[_X_]*cx);
    maxx = get_brick_x(features[0].DAT, (chunk[_X_]+1)*cx);
    miny = get_brick_y(features[0].DAT, (chunk[_Y_]+1)*cy);
    maxy = get_brick_y(features[0].DAT, chunk[_Y_]*cy);

  } else {

    // corner coordinates of chunk
    get_brick_geo(features[0].DAT, chunk[_X_]*cx,     chunk[_Y_]*cy,     &geo_ul.x, &geo_ul.y);
    get_brick_geo(features[0].DAT, (chunk[_X_]+1)*cx, chunk[_Y_]*cy,     &geo_ur.x, &geo_ur.y);
    get_brick_geo(features[0].DAT, chunk[_X_]*cx,     (chunk[_Y_]+1)*cy, &geo_ll.x, &geo_ll.y);
    get_brick_geo(features[0].DAT, (chunk[_X_]+1)*cx, (chunk[_Y_]+1)*cy, &geo_lr.x, &geo_lr.y);

    // min/max coordinates
    minx = MIN(geo_ul.x, geo_ll.x);
    maxx = MAX(geo_ur.x, geo_lr.x);
    miny = MIN(geo_ll.y, geo_lr.y);
    maxy = MAX(geo_ul.y, geo_ur.y);

    // edge coordinates of chunk
    get_brick_geo(features[0].DAT, chunk[_X_]*cx + cx/2, chunk[_Y_]*cy,        &geo_upper.x, &geo_upper.y);
    get_brick_geo(features[0].DAT, chunk[_X_]*cx + cx/2, (chunk[_Y_]+1)*cy,    &geo_lower.x, &geo_lower.y);
    get_brick_geo(features[0].DAT, chunk[_X_]*cx,        chunk[_Y_]*cy + cy/2, &geo_left.x,  &geo_left.y);
    get_brick_geo(features[0].DAT, (chunk[_X_]+1)*cx,    chunk[_Y_]*cy + cy/2, &geo_right.x, &geo_right.y);

    // if edge coordinates are outside of corner min/max, expand the box a bit
    if (geo_left.x  < minx) minx -= 2*fabs(geo_left.x -minx);
    if (geo_right.x > maxx) maxx += 2*fabs(geo_right.x-maxx);
    if (geo_lower.y < miny) miny -= 2*fabs(geo_lower.y-miny);
    if (geo_upper.y > maxy) maxy += 2*fabs(geo_upper.y-maxy);

  }


  #pragma omp parallel private(smp_map,smp_map_ul,smp_tx,smp_ty,smp_tj,smp_ti,smp_chunk,j,i,p,f,r,valid) shared(tile,chunk,res,cx,cy,nf,nr,minx,maxx,miny,maxy,copied,smp_features,smp_response,smp,features,mask_,cube,phl) reduction(+: error, found, added) default(none)
  {

    #pragma omp for
    for (s=0; s<smp->nrow; s++){

      if (!smp->row_mask[s]) continue;

      if (smp->data[s][_X_] < minx) continue;
      if (smp->data[s][_X_] > maxx) continue;
      if (smp->data[s][_Y_] < miny) continue;
      if (smp->data[s][_Y_] > maxy) continue;
      
      if (phl->smp.projected){

        smp_map.x = smp->data[s][_X_];
        smp_map.y = smp->data[s][_Y_];

      } else {

        // get target coordinates in target css coordinates
        if ((warp_geo_to_any(smp->data[s][_X_], smp->data[s][_Y_], &smp_map.x, &smp_map.y, cube->projection)) == FAILURE){
          printf("Computing target coordinates in dst_srs failed!\n"); 
          error++;
          continue;
        }

      }

      // find the tile the sample falls into
      tile_find(smp_map.x, smp_map.y, &smp_map_ul.x, &smp_map_ul.y, &smp_tx, &smp_ty, cube);

      // if not in current tile, skip
      if (smp_tx != tile[_X_] || smp_ty != tile[_Y_]) continue;

      // find pixel in tile
      smp_tj = (int)((smp_map.x-smp_map_ul.x)/res);
      smp_ti = (int)((smp_map_ul.y-smp_map.y)/res);

      // find chunk in tile
      smp_chunk[_X_] = (int)(smp_tj/cx);
      smp_chunk[_Y_] = (int)(smp_ti/cy);

      // if not in current chunk, skip
      if (smp_chunk[_X_] != chunk[_X_] || smp_chunk[_Y_] != chunk[_Y_]) continue;

      // find pixel in chunk
      j = smp_tj - chunk[_X_]*cx;
      i = smp_ti - chunk[_Y_]*cy;
      p = i*cx+j;

      // skip pixels that are masked
      if (mask_ != NULL && !mask_[p]) continue;

      // extract
      for (f=0, valid=true; f<nf; f++){
        smp_features[s][f] = features[f].dat[0][p];
        if (!features[f].msk[p] && phl->ftr.exclude) valid = false;
      }
     for (r=0; r<nr; r++) smp_response[s][r] = smp->data[s][_Z_+r];


      // we are done with this sample
      smp->row_mask[s] = false;
      found++;

      if (!valid) continue;

      copied[s] = true;
      added++;

    }

  }

  if (error > 0) printf("there were %d errors in coordinate conversion..\n", error);

  smp->n_active_rows -= found;


  if (added > 0){
    append_table(phl->smp.f_sample,   copied, smp_features, smp->nrow, nf, 0);
    append_table(phl->smp.f_response, copied, smp_response, smp->nrow, nr, 6);
    append_table(phl->smp.f_coords,   copied, smp->data,    smp->nrow, 2,  6);
  }


  #ifdef FORCE_DEBUG
  if (added > 0) printf("Added %d samples in Tile X%04d_Y%04d Chunk %02d / %02d.\n", 
    added, tile[_X_], tile[_Y_], chunk[_X_], chunk[_Y_]);
  #endif


  free((void*)copied);
  free_2D((void**)smp_response, smp->nrow);
  free_2D((void**)smp_features, smp->nrow);

  *nproduct = 0;
  return NULL;
}

