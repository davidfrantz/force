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
This file contains functions for Level 3 processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "level3-hl.h"


stack_t *compile_level3_stack(stack_t *ard, int nb, bool fullres, char *prodname, par_hl_t *phl);
stack_t **compile_level3(ard_t *ard, level3_t *l3, par_hl_t *phl, cube_t *cube, int *nproduct);


/** This function compiles the stacks, in which L3 results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- l3:       pointer to instantly useable L3 image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output stacks (returned)
+++ Return:   stacks for L3 results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **compile_level3(ard_t *ard, level3_t *l3, par_hl_t *phl, cube_t *cube, int *nproduct){
stack_t **LEVEL3 = NULL;
int nb, nbands;
int o, nprod = 4;
int error = 0;
char prodname[4][NPOW_02] = { "BAP", "INF", "SCR", "OVV" };
bool fullres[4] = { true, true, true, false };
int prodlen[4] = { 0, 6, 7, 3 };
bool enable[4] = { phl->bap.obap, phl->bap.oinf, phl->bap.oscr, phl->bap.oovv };
short ***ptr[4] = { &l3->bap, &l3->inf, &l3->scr, &l3->ovv };

  nb = get_stack_nbands(ard[0].DAT);

  alloc((void**)&LEVEL3, nprod, sizeof(stack_t*));


  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((nbands = prodlen[o]) == 0) nbands = nb;
      if ((LEVEL3[o] = compile_level3_stack(ard[0].DAT, nbands, fullres[o], prodname[o], phl)) == NULL || (  *ptr[o] = get_bands_short(LEVEL3[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;}
    } else {
      LEVEL3[o] = NULL;
      *ptr[o]   = NULL;
    }
  }
  
  if (error > 0){
    for (o=0; o<nprod; o++) free_stack(LEVEL3[o]);
    free((void*)LEVEL3);
    return NULL;
  }

  *nproduct = nprod;
  return LEVEL3;
}


/** This function compiles a L3 stack
--- from:      stack from which most attributes are copied
--- nb:        number of bands in stack
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    stack for L3 result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *compile_level3_stack(stack_t *from, int nb, bool fullres, char *prodname, par_hl_t *phl){
int b, m, d;
stack_t *stack = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;
double res, res_;
int nx, ny, nx_, ny_; 
int cx, cy, cx_, cy_, cc_; 


  if (phl->bap.score_type == _SCR_SIG_DES_) doy2md(phl->bap.Dt[0], &m, &d);
  if (phl->bap.score_type == _SCR_GAUSS_)   doy2md(phl->bap.Dt[1], &m, &d);
  if (phl->bap.score_type == _SCR_SIG_ASC_) doy2md(phl->bap.Dt[2], &m, &d);
  
  date.year = phl->bap.Yt;
  date.month = m;
  date.day = d;
  date.doy = md2doy(date.month, date.day);
  date.week = doy2week(date.doy);
  date.ce  = doy2ce(date.doy, date.year);
  date.hh    = 0;
  date.mm    = 0;
  date.ss    = 0;
  date.tz    = 0;

  res = get_stack_res(from);
  nx = get_stack_ncols(from);
  ny = get_stack_nrows(from);
  cx = get_stack_chunkncols(from);
  cy = get_stack_chunknrows(from);

  if (fullres){
    
    stack = copy_stack(from, nb, _DT_SHORT_);
    
  } else {

    if (res > 150) res_ = res; else res_ = 150;
    nx_ = nx*res/res_;
    ny_ = ny*res/res_;
    cx_ = cx*res/res_;
    cy_ = cy*res/res_;
    cc_ = cx_*cy_;
    stack = copy_stack(from, nb, _DT_NONE_);
    //set_stack_format(stack, _FMT_JPEG_);
    set_stack_res(stack, res_);
    set_stack_ncols(stack, nx_);
    set_stack_nrows(stack, ny_);
    set_stack_chunkncols(stack, cx_);
    set_stack_chunknrows(stack, cy_);
    allocate_stack_bands(stack, nb, cc_, _DT_SHORT_);

  }

  set_stack_name(stack, "FORCE Level 3 Processing System");
  set_stack_product(stack, prodname);
  
  //printf("dirname should be assemlbed in write_stack, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_stack_tilex(stack), get_stack_tiley(stack));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_stack_dirname(stack, dname);

  nchar = snprintf(fname, NPOW_10, "%04d%02d%02d_LEVEL3_%s_%s", 
    date.year, date.month, date.day, phl->sen.target, prodname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_stack_filename(stack, fname);

  set_stack_open(stack, OPEN_BLOCK);
  set_stack_format(stack, phl->format);
  set_stack_par(stack, phl->params->log);

  for (b=0; b<nb; b++){
    set_stack_save(stack, b, true);
    set_stack_date(stack, b, date);
  }

  return stack;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the Level 3 module
--- ard:       ARD
--- lsp:       LSP
--- mask:      mask image
--- nt:        number of ARD products over time
--- nlsp:      number of LSP products (should be 3)
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output stacks (returned)
+++ Return:    stacks with L3 results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **level3(ard_t *ard, ard_t *lsp, stack_t *mask, int nt, int nlsp, par_hl_t *phl, cube_t *cube, int *nproduct){
level3_t l3;
stack_t **LEVEL3;
small *mask_ = NULL;
int nprod = 0;
int p, nx, ny, nc, nb;
double res;
short nodata;
short lsp_nodata = -32767;
target_t *target = NULL;
int *tdist = NULL;
par_scr_t *score = NULL;
float **cor = NULL;
float hmean, hsd;
bool water;


  cite_me(_CITE_BAP_);

  if (phl->bap.pac.lsp && lsp == NULL){
    printf("PAC requested, but LSP not available.\n");
    *nproduct = 0;
    return NULL;
  }
  
  // compile products + stacks
  if ((LEVEL3 = compile_level3(ard, &l3, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile L3 products!\n"); 
    *nproduct = 0;
    return NULL;
  }


  // import stacks
  nx  = get_stack_chunkncols(ard[0].DAT);
  ny  = get_stack_chunknrows(ard[0].DAT);
  nc  = get_stack_chunkncells(ard[0].DAT);
  res = get_stack_res(ard[0].DAT);
  nb  = get_stack_nbands(ard[0].DAT);

  nodata     = get_stack_nodata(ard[0].DAT, 0);
  if (phl->bap.pac.lsp) lsp_nodata = get_stack_nodata(lsp[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return NULL;}
  }

  if (phl->bap.pac.lsp && nlsp > 3){
    printf("more than 3 LSP files were given, only the first three will be used. You should check if this is correct.\n");
  } else if (phl->bap.pac.lsp && nlsp < 3){
    printf("less than 3 LSP files were given. Cannot perform PAC with this.\n");
    return NULL;
  }


  #pragma omp parallel private(tdist,score,cor,hmean,hsd,water) firstprivate(target) shared(ard,lsp,l3,nt,nb,nc,lsp_nodata,nodata,mask_,phl,LEVEL3) default(none)
  {

    if (phl->bap.w.r > 0) alloc_2D((void***)&cor, nt, nt, sizeof(float));
    alloc((void**)&tdist, nt, sizeof(int));
    alloc((void**)&score,  nt, sizeof(par_scr_t));

    // static target
    if (!phl->bap.pac.lsp) target = compile_target_static(&phl->bap);

    #pragma omp for
    for (p=0; p<nc; p++){

      // skip pixels that are masked
      if (mask_ != NULL && !mask_[p]) continue;
      
      if (phl->bap.pac.lsp) target = compile_target_adaptive(&phl->bap, lsp, p, lsp_nodata);

      // water or land pixel?
      water = pixel_is_water(ard, nt, p);

      // compute correlation matrix
      if (phl->bap.w.r > 0) corr_matrix(ard, nt, nb, p, cor);

      // compute parametric scores
      parametric_score(ard, nt, p, target, cor, score, tdist, &phl->bap);

      // override total score in case of water
      if (water) water_score(ard, nt, p, score);

      // mean and sd of haze score
      haze_stats(ard, nt, p, score, &phl->bap, &hmean, &hsd);

      // compute BAP
      bap_compositing(ard, &l3, nt, nb, nodata, p, score, tdist, hmean, hsd, water, &phl->bap);

      if (phl->bap.pac.lsp) free((void*)target);

    }
    

    // clean
    if (!phl->bap.pac.lsp) free((void*)target);
    if (phl->bap.w.r > 0) free_2D((void**)cor, nt);
    free((void*)tdist);
    free((void*)score);

  }
  
  bap_overview(&l3, nx, ny, nb, res, nodata);
  

  *nproduct = nprod;
  return LEVEL3;
}

