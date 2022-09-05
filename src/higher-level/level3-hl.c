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


brick_t *compile_level3_brick(brick_t *ard, int nb, bool explode, bool fullres, char *prodname, par_hl_t *phl);
brick_t **compile_level3(ard_t *ard, level3_t *l3, par_hl_t *phl, cube_t *cube, int *nproduct);


/** This function compiles the bricks, in which L3 results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- l3:       pointer to instantly useable L3 image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for L3 results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_level3(ard_t *ard, level3_t *l3, par_hl_t *phl, cube_t *cube, int *nproduct){
brick_t **LEVEL3 = NULL;
int b, nb, nbands;
int o, nprod = 4;
int error = 0;
enum { _ref_, _inf_, _scr_, _ovv_ };
char prodname[4][NPOW_02] = { "BAP", "INF", "SCR", "OVV" };
bool fullres[4] = { true, true, true, false };
bool explode[4] = { false, phl->explode, phl->explode, false };
int prodlen[4] = { 0, _INF_LENGTH_, _SCR_LENGTH_, _RGB_LENGTH_ };
int prodtype[4] = { _ref_, _inf_, _scr_, _ovv_ };
bool enable[4] = { phl->bap.obap, phl->bap.oinf, phl->bap.oscr, phl->bap.oovv };
short ***ptr[4] = { &l3->bap, &l3->inf, &l3->scr, &l3->ovv };

  nb = get_brick_nbands(ard[0].DAT);

  alloc((void**)&LEVEL3, nprod, sizeof(brick_t*));


  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((nbands = prodlen[o]) == 0) nbands = nb;
      if ((LEVEL3[o] = compile_level3_brick(ard[0].DAT, nbands, explode[o], fullres[o], prodname[o], phl)) == NULL || (  *ptr[o] = get_bands_short(LEVEL3[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      } else {
        for (b=0; b<prodlen[prodtype[o]]; b++){
          switch (prodtype[o]){
            case _ref_:
              break;
            case _inf_:
              set_brick_domain(LEVEL3[o], b, _TAGGED_ENUM_INF_[b].tag);
              set_brick_bandname(LEVEL3[o], b, _TAGGED_ENUM_INF_[b].tag);
              break;
            case _scr_:
              set_brick_domain(LEVEL3[o], b, _TAGGED_ENUM_SCR_[b].tag);
              set_brick_bandname(LEVEL3[o], b, _TAGGED_ENUM_SCR_[b].tag);
              break;
            case _ovv_:
              set_brick_domain(LEVEL3[o], b, _TAGGED_ENUM_RGB_[b].tag);
              set_brick_bandname(LEVEL3[o], b, _TAGGED_ENUM_RGB_[b].tag);
              break;
            default:
              printf("unknown level3 type.\n"); error++;
              break;
          }
        }
      }
    } else {
      LEVEL3[o] = NULL;
      *ptr[o]   = NULL;
    }
  }
  
  if (error > 0){
    for (o=0; o<nprod; o++) free_brick(LEVEL3[o]);
    free((void*)LEVEL3);
    return NULL;
  }

  *nproduct = nprod;
  return LEVEL3;
}


/** This function compiles a L3 brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for L3 result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_level3_brick(brick_t *from, int nb, bool explode, bool fullres, char *prodname, par_hl_t *phl){
int b, m, d;
brick_t *brick = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;
double res, res_;
int nx, ny, nx_, ny_; 
int cx, cy, cx_, cy_, cc_; 


  if (phl->bap.score_type == _SCR_TYPE_SIG_DES_) doy2md(phl->bap.Dt[0], &m, &d);
  if (phl->bap.score_type == _SCR_TYPE_GAUSS_)   doy2md(phl->bap.Dt[1], &m, &d);
  if (phl->bap.score_type == _SCR_TYPE_SIG_ASC_) doy2md(phl->bap.Dt[2], &m, &d);
  
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

  res = get_brick_res(from);
  nx = get_brick_ncols(from);
  ny = get_brick_nrows(from);
  cx = get_brick_chunkncols(from);
  cy = get_brick_chunknrows(from);

  if (fullres){
    
    brick = copy_brick(from, nb, _DT_SHORT_);
    
  } else {

    if (res > 150) res_ = res; else res_ = 150;
    nx_ = nx*res/res_;
    ny_ = ny*res/res_;
    cx_ = cx*res/res_;
    cy_ = cy*res/res_;
    cc_ = cx_*cy_;
    brick = copy_brick(from, nb, _DT_NONE_);
    //set_brick_format(brick, _FMT_JPEG_);
    set_brick_res(brick, res_);
    set_brick_ncols(brick, nx_);
    set_brick_nrows(brick, ny_);
    set_brick_chunkncols(brick, cx_);
    set_brick_chunknrows(brick, cy_);
    allocate_brick_bands(brick, nb, cc_, _DT_SHORT_);

  }

  set_brick_name(brick, "FORCE Level 3 Processing System");
  set_brick_product(brick, prodname);
  
  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_parentname(brick, phl->d_higher);
  set_brick_dirname(brick, dname);

  nchar = snprintf(fname, NPOW_10, "%04d%02d%02d_LEVEL3_%s_%s", 
    date.year, date.month, date.day, phl->sen.target, prodname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);

  set_brick_open(brick, OPEN_BLOCK);
  set_brick_format(brick, &phl->gdalopt);
  set_brick_explode(brick, explode);
  set_brick_par(brick, phl->params->log);

  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
    set_brick_date(brick, b, date);
  }

  return brick;
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
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with L3 results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **level3(ard_t *ard, ard_t *lsp, brick_t *mask, int nt, int nlsp, par_hl_t *phl, cube_t *cube, int *nproduct){
level3_t l3;
brick_t **LEVEL3;
small *mask_ = NULL;
int nprod = 0;
int p, nx, ny, nc, nb;
double res;
short nodata;
short lsp_nodata = SHRT_MIN;
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
  
  // compile products + bricks
  if ((LEVEL3 = compile_level3(ard, &l3, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile L3 products!\n"); 
    *nproduct = 0;
    return NULL;
  }


  // import bricks
  nx  = get_brick_chunkncols(ard[0].DAT);
  ny  = get_brick_chunknrows(ard[0].DAT);
  nc  = get_brick_chunkncells(ard[0].DAT);
  res = get_brick_res(ard[0].DAT);
  nb  = get_brick_nbands(ard[0].DAT);

  nodata     = get_brick_nodata(ard[0].DAT, 0);
  if (phl->bap.pac.lsp) lsp_nodata = get_brick_nodata(lsp[0].DAT, 0);

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

