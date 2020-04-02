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
This file contains functions for Continuous Field  ImproPhing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "cf-improphe-hl.h"


stack_t **compile_cfi(ard_t *ard, cfi_t *cfi, int nt, par_hl_t *phl, cube_t *cube, int *nproduct);
stack_t *compile_cfi_stack(stack_t *ard, int nb, bool write, char *bname, char *prodname, par_hl_t *phl);


/** This function compiles the stacks, in which CFI results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- cf:       coarse resolution continuous field
--- cfi:      pointer to instantly useable CFI image arrays
--- ncf:      number of CF products
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output stacks (returned)
+++ Return:   stacks for CFI results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **compile_cfi(ard_t *cf, cfi_t *cfi, int ncf, par_hl_t *phl, cube_t *cube, int *nproduct){
stack_t **CFI = NULL;
int b, o, nprod = ncf;
int error = 0;
date_t date;
char fdate[NPOW_10];
int nchar;
int prodlen;
char bname[NPOW_10];
short ****ptr = NULL;


  alloc((void**)&ptr,       nprod, sizeof(short***));
  alloc((void**)&cfi->imp_, nprod, sizeof(short**));
  for (o=0; o<nprod; o++) ptr[o] = &cfi->imp_[o];


  alloc((void**)&CFI, nprod, sizeof(stack_t*));
  prodlen = phl->cfi.nyears;


  for (o=0; o<nprod; o++){
    
    basename_without_ext(phl->con.fname[o], bname, NPOW_10);

    if ((CFI[o] = compile_cfi_stack(cf[o].DAT, prodlen, true, bname, "IMP", phl)) == NULL || (*ptr[o] = get_bands_short(CFI[o])) == NULL){
      printf("Error compiling %s product. ", bname); error++;
    } else {
      
      init_date(&date);
      set_date(&date, 2000, 1, 1);
      
      for (b=0; b<prodlen; b++){
        set_date_year(&date, phl->cfi.years[o]);
        nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
        if (nchar < 0 || nchar >= NPOW_10){ 
          printf("Buffer Overflow in assembling domain\n"); error++;}
        set_stack_domain(CFI[o],   b, fdate);
        set_stack_bandname(CFI[o], b, fdate);
      }
    }

  }

  free((void*)ptr);

  if (error > 0){
    printf("%d compiling CFI product errors.\n", error);
    for (o=0; o<nprod; o++) free_stack(CFI[o]);
    free((void*)CFI);
    free((void*)cfi->imp_);
    return NULL;
  }

  *nproduct = nprod;
  return CFI;
}


/** This function compiles a CFI stack
--- from:      stack from which most attributes are copied
--- nb:        number of bands in stack
--- write:     should this stack be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    stack for CFI result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *compile_cfi_stack(stack_t *from, int nb, bool write, char *bname, char *prodname, par_hl_t *phl){
int b;
stack_t *stack = NULL;
char dname[NPOW_10];
char fname[NPOW_10];
int nchar;


  if ((stack = copy_stack(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_stack_name(stack, "FORCE Continuous Field ImproPhe");
  set_stack_product(stack, prodname);

  //printf("dirname should be assemlbed in write_stack, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_stack_tilex(stack), get_stack_tiley(stack));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_stack_dirname(stack, dname);


  nchar = snprintf(fname, NPOW_10, "%s_%s", bname, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  set_stack_filename(stack, fname);

  if (write){
    set_stack_open(stack, OPEN_BLOCK);
  } else {
    set_stack_open(stack, OPEN_FALSE);
  }
  set_stack_format(stack, phl->format);
  set_stack_explode(stack, phl->explode);
  set_stack_par(stack, phl->params->log);

  for (b=0; b<nb; b++){
    set_stack_save(stack, b, true);
  }

  return stack;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the Continuous Field ImproPhe module
--- ard:       ARD
--- cf:        coarse resolution continuous field
--- mask:      mask image
--- nt:        number of ARD products over time
--- ncf:       number of CF products
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output stacks (returned)
+++ Return:    stacks with CFI results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **confield_improphe(ard_t *ard, ard_t *cf, stack_t *mask, int nt, int ncf, par_hl_t *phl, cube_t *cube, int *nproduct){
cfi_t cfi;
stack_t **CFI;
small *mask_ = NULL;
float **ard_ = NULL;
float **cf_ = NULL;
float *ard_tex_ = NULL;
float **cf_tex_ = NULL;
short **seasonal_avg_ = NULL;
short **pred_ = NULL;
int nprod = 0;
int f, i, j, p, b, nx, ny, nc, nb_ard, npc;
int ny_cf, b_cf, y;
short ard_nodata, cf_nodata;
float **KDIST = NULL; // kernel distance
int width, nk, mink; // number of kernel pixels, and minimum number of pixels for good prediction


  cite_me(_CITE_IMPROPHE_);

  if (nt < 1 || ncf < 1){
    *nproduct = 0;
    return NULL;
  }

  // import stacks
  nx = get_stack_chunkncols(ard[0].DAT);
  ny = get_stack_chunknrows(ard[0].DAT);
  nc = get_stack_chunkncells(ard[0].DAT);
  
  ard_nodata = get_stack_nodata(ard[0].DAT, 0);
  cf_nodata  = get_stack_nodata(cf[0].DAT, 0);

  nb_ard = get_stack_nbands(ard[0].DAT);
  ny_cf = get_stack_nbands(cf[0].DAT);

  for (y=0; y<phl->cfi.nyears; y++){
    if ((b_cf = phl->cfi.years[y]-phl->cfi.y0) >= ny_cf){
      printf("requested years and number of bands in file do not match.\n");
      *nproduct = 0;
      return NULL;
    }
  }
    
  
  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }


  // compile products + stacks
  if ((CFI = compile_cfi(cf, &cfi, ncf, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile CFI products!\n"); 
    *nproduct = 0;
    return NULL;
  }
  
  // pre-compute kernel distances
  // determine minimum # of neighbors that make a good prediction
  width = phl->imp.ksize*2+1;
  distance_kernel(width, &KDIST);
  nk = width*width;
  mink = (int)((nk*M_PI/4.0)*0.5/100.0);
  #ifdef FORCE_DEBUG
  printf("radius/width/nkernel/minnum: %d/%d/%d/%d\n", phl->imp.ksize, width, nk, mink);
  #endif


  
  for (y=0; y<phl->cfi.nyears; y++){

    b_cf = phl->cfi.years[y]-phl->cfi.y0;

    // compute average per seasonal window
    if ((seasonal_avg_ = average_season(ard, mask_, nb_ard, nc, nt, ard_nodata, phl->imp.nwin, phl->imp.dwin, phl->cfi.years[y])) == NULL){
      printf("error in computing window averages.\n");
      *nproduct = 0;
      return NULL;}

    // reduce HR dimensionality using principal component analysis
    if ((ard_ = pca(seasonal_avg_, mask_, phl->imp.nwin*nb_ard, nc, ard_nodata, 0.975, &npc)) == NULL){
      printf("error in computing PCA.\n");
      *nproduct = 0;
      return NULL;}
    free_2D((void**)seasonal_avg_, phl->imp.nwin*nb_ard);

    // normalize PCs
    for (b=0; b<npc; b++){
      if (standardize_float(ard_[b], ard_nodata, nc) != SUCCESS){
        printf("error in standardizing PCA.\n");
        *nproduct = 0;
        return NULL;}
    }

    // HR texture
    if ((ard_tex_ = focal_sd(ard_, ard_nodata, phl->imp.ksd, nx, ny, npc, 0)) == NULL){
      printf("error in computing high-res texture.\n");
      *nproduct = 0;
      return NULL;}

   
    alloc((void**)&cf_tex_, ncf, sizeof(float*));
    alloc_2D((void***)&cf_, ncf, nc, sizeof(float));
    alloc_2D((void***)&pred_, ncf, nc, sizeof(short));

    for (f=0; f<ncf; f++){

      //printf("doing MR image %d\n", f);

      #pragma omp parallel shared(mask_,cf,cf_,pred_,nc,f,cf_nodata,b_cf) default(none)
      {

        #pragma omp for
        for (p=0; p<nc; p++){
          if (mask_ != NULL && !mask_[p]){
            pred_[f][p]  = cf_nodata;
            //cfi.imp_[f][y][p]  = cf_nodata;
            cf_[f][p] = cf_nodata;
          } else if (!cf[f].msk[p]){
            pred_[f][p]  = cf[f].dat[b_cf][p];
            //cfi.imp_[f][y][p] = cf[f].dat[b_cf][p];
            cf_[f][p] = cf_nodata;
          } else {
            cf_[f][p] = cf[f].dat[b_cf][p];
          }
        }
        
      }

      // MR texture
      if ((cf_tex_[f] = focal_sd(cf_, cf_nodata, phl->imp.ksd, nx, ny, 1, f)) == NULL){
        printf("error in computing med-res texture.\n");
        *nproduct = 0;
        return NULL;}

    }

    #pragma omp parallel private(p,f) shared(ard_,ard_tex_,cf_,cf_tex_,pred_,cfi,ncf,ny,nx,npc,ard_nodata,cf_nodata,KDIST,nk,mink,phl,y) default(none)
    {

      #pragma omp for collapse(2) schedule(guided)
      for (i=0; i<ny; i++){
      for (j=0; j<nx; j++){

        p = i*nx+j;

        if (cf_[0][p] == cf_nodata) continue;

        improphe(ard_, ard_tex_, cf_, cf_tex_, pred_, KDIST, ard_nodata, cf_nodata, i, j, p, 
          nx, ny, phl->imp.ksize, npc, ncf, nk, mink);

        for (f=0; f<ncf; f++) cfi.imp_[f][y][p] = pred_[f][p];

      }
      }

    }
    

    free_2D((void**)ard_, npc);
    free((void*)ard_tex_);
    free_2D((void**)cf_, ncf);
    free_2D((void**)cf_tex_, ncf);
    free_2D((void**)pred_, ncf);

  }

  free((void*)cfi.imp_); // check this here
  free_2D((void**)KDIST, width);

  *nproduct = nprod;
  return CFI;
}

