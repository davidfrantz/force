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
This file contains functions for Level 2 ImproPhing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "l2-improphe-hl.h"


brick_t **compile_l2i(ard_t *ard, l2i_t *l2i, int nt, par_hl_t *phl, cube_t *cube, int *nproduct);
brick_t *compile_l2i_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the bricks, in which L2I results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- l2i:      pointer to instantly useable L2I image arrays
--- nt:       number of ARD products over time
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for L2I results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_l2i(ard_t *ard, l2i_t *l2i, int nt, par_hl_t *phl, cube_t *cube, int *nproduct){
brick_t **L2I = NULL;
int o, nprod = nt;
int error = 0;
int prodlen;
char prodname[NPOW_02] = "IMP";
short ****ptr = NULL;


  alloc((void**)&ptr,       nprod, sizeof(short***));
  alloc((void**)&l2i->imp_, nprod, sizeof(short**));
  for (o=0; o<nprod; o++) ptr[o] = &l2i->imp_[o];


  alloc((void**)&L2I, nprod, sizeof(brick_t*));
  prodlen = get_brick_nbands(ard[0].DAT);
  

  for (o=0; o<nprod; o++){

    if ((L2I[o] = compile_l2i_brick(ard[o].DAT, prodlen, true, prodname, phl)) == NULL || (*ptr[o] = get_bands_short(L2I[o])) == NULL){
      printf("Error compiling %s product. ", prodname); error++;
    }

  }

  free((void*)ptr);

  if (error > 0){
    printf("%d compiling L2I product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(L2I[o]);
    free((void*)L2I);
    free((void*)l2i->imp_);
    return NULL;
  }

  *nproduct = nprod;
  return L2I;
}


/** This function compiles a L2I brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for L2I result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2i_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
char sensor[NPOW_10];
char date[NPOW_04];
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;


  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Texture");
  set_brick_product(brick, prodname);

  get_brick_compactdate(from, 0, date, NPOW_04);
  get_brick_sensor(from, 0, sensor, NPOW_10);

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);

  if (write){
    set_brick_open(brick, OPEN_CHUNK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, &phl->gdalopt);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
  }

  return brick;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the Level 2 ImproPhe module
--- ard_hr:    high   resolution ARD
--- ard_mr:    medium resolution ARD
--- mask:      mask image
--- nt_hr:     number of high   resolution ARD products over time
--- nt_mr:     number of medium resolution ARD products over time
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with L2I results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **level2_improphe(ard_t *ard_hr, ard_t *ard_mr, brick_t *mask, int nt_hr, int nt_mr, par_hl_t *phl, cube_t *cube, int *nproduct){
l2i_t l2i;
brick_t **L2I;
small *mask_ = NULL;
float **hr_ = NULL;
float **mr_ = NULL;
float *hr_tex_ = NULL;
float **mr_tex_ = NULL;
short **seasonal_avg_ = NULL;
int nprod = 0;
int t, i, j, p, b, nx, ny, nc, nb_hr, nb_mr, npc;
short nodata;
float **KDIST = NULL; // kernel distance
int width, nk, mink; // number of kernel pixels, and minimum number of pixels for good prediction
bool is_empty;


  cite_me(_CITE_IMPROPHE_);

  if (nt_hr < 1 || nt_mr < 1){
    *nproduct = 0;
    return NULL;
  }

  // import bricks
  nx = get_brick_chunkncols(ard_hr[0].DAT);
  ny = get_brick_chunknrows(ard_hr[0].DAT);
  nc = get_brick_chunkncells(ard_hr[0].DAT);
  nodata = get_brick_nodata(ard_hr[0].DAT, 0);

  nb_hr = get_brick_nbands(ard_hr[0].DAT);
  nb_mr = get_brick_nbands(ard_mr[0].DAT);

  
  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }


  // compile products + bricks
  if ((L2I = compile_l2i(ard_mr, &l2i, nt_mr, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile L2I products!\n"); 
    *nproduct = 0;
    return NULL;
  }


  // compute average per seasonal window
  if ((seasonal_avg_ = average_season(ard_hr, mask_, nb_hr, nc, nt_hr, nodata, phl->imp.nwin, phl->imp.dwin, -1, &is_empty)) == NULL){
    printf("error in computing window averages.\n");
    *nproduct = 0;
    return NULL;}


  // if there is no valid target data, copy over and skip
  if (is_empty){
    
    for (t=0; t<nt_mr; t++){
      #pragma omp parallel private(b) shared(l2i,ard_mr,nc,nb_mr,t) default(none)
      {
        #pragma omp for
        for (p=0; p<nc; p++){
          for (b=0; b<nb_mr; b++) l2i.imp_[t][b][p] = ard_mr[t].dat[b][p];
        }
      }
    }

    *nproduct = nprod;
    return L2I;

  }


  // reduce HR dimensionality using principal component analysis
  if ((hr_ = pca(seasonal_avg_, mask_, phl->imp.nwin*nb_hr, nc, nodata, 0.975, &npc)) == NULL){
    printf("error in computing PCA.\n");
    *nproduct = 0;
    return NULL;}
  free_2D((void**)seasonal_avg_, phl->imp.nwin*nb_hr);

  // normalize PCs
  for (b=0; b<npc; b++){
    if (standardize_float(hr_[b], nodata, nc) != SUCCESS){
      printf("error in standardizing PCA.\n");
      *nproduct = 0;
      return NULL;}
  }

  // HR texture
  if ((hr_tex_ = focal_sd(hr_, nodata, phl->imp.ksd, nx, ny, npc, 0)) == NULL){
    printf("error in computing high-res texture.\n");
    *nproduct = 0;
    return NULL;}

 
  
  // pre-compute kernel distances
  // determine minimum # of neighbors that make a good prediction
  width = phl->imp.ksize*2+1;
  distance_kernel(width, &KDIST);
  nk = width*width;
  mink = (int)((nk*M_PI/4.0)*0.5/100.0);
  #ifdef FORCE_DEBUG
  printf("radius/width/nkernel/minnum: %d/%d/%d/%d\n", phl->imp.ksize, width, nk, mink);
  #endif


  for (t=0; t<nt_mr; t++){

    //printf("doing MR image %d\n", t);

    alloc((void**)&mr_tex_, nb_mr, sizeof(float*));
    alloc_2D((void***)&mr_, nb_mr, nc, sizeof(float));


    #pragma omp parallel private(b) shared(mask_,l2i,ard_mr,mr_,nc,nb_mr,t,nodata) default(none)
    {

      #pragma omp for
      for (p=0; p<nc; p++){
        if (mask_ != NULL && !mask_[p]){
          for (b=0; b<nb_mr; b++){
            l2i.imp_[t][b][p]  = nodata;
            mr_[b][p] = nodata;
          }
        } else if (!ard_mr[t].msk[p]){
          for (b=0; b<nb_mr; b++){
            l2i.imp_[t][b][p] = ard_mr[t].dat[b][p];
            mr_[b][p] = nodata;
          }
        } else {
          for (b=0; b<nb_mr; b++) mr_[b][p] = ard_mr[t].dat[b][p];
        }
      }
      
    }

    // MR texture
    for (b=0; b<nb_mr; b++){
      if ((mr_tex_[b] = focal_sd(mr_, nodata, phl->imp.ksd, nx, ny, 1, b)) == NULL){
        printf("error in computing med-res texture.\n");
        *nproduct = 0;
        return NULL;}
    }

    #pragma omp parallel private(p) shared(hr_,hr_tex_,mr_,mr_tex_,l2i,t,ny,nx,npc,nb_mr,nodata,KDIST,nk,mink,phl) default(none)
    {

      #pragma omp for collapse(2) schedule(guided)
      for (i=0; i<ny; i++){
      for (j=0; j<nx; j++){

        p = i*nx+j;

        if (mr_[0][p] == nodata) continue;

        improphe(hr_, hr_tex_, mr_, mr_tex_, l2i.imp_[t], KDIST, nodata, nodata, i, j, p, 
          nx, ny, phl->imp.ksize, npc, nb_mr, nk, mink);

      }
      }

    }
    
    free_2D((void**)mr_,     nb_mr);
    free_2D((void**)mr_tex_, nb_mr);

  }

  
  free_2D((void**)hr_, npc);
  free((void*)hr_tex_);
  free_2D((void**)KDIST, width);
  free((void*)l2i.imp_);

  *nproduct = nprod;
  return L2I;
}

