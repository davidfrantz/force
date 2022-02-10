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

/** The following code is a modified version of the Fmask code, openly 
+++ shared by Zhe Zhu, Department of Geosciences, Texas Tech University:
+++ https://github.com/prs021/fmask
+++ Fmask Copyright (C) 2012-2015 Zhe Zhu, Curtis Woodcock
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for cloud and cloud shadow identification
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "cloud-ll.h"

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing


float finalize_cloud(par_ll_t *pl2, int npix, atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *DEM, small *fcld_, small *fshd_);
int potential_cloud(par_ll_t *pl2, int *npix, int *nclear, int *nland, brick_t *TOA, brick_t *QAI, brick_t *EXP, small **PCP, small **CLR, small **LND, small **BRT, short **VAR);
float land_probability(int nc, int nclear, int nland, int npix, float cldprob, float *lowt, float *hight, small *lnd_, small *clr_, short *temp_, short *var_, brick_t *QAI, float **PROB);
float water_probability(int nc, float cldprob, short *temp_, short *sw1_, short *sw2_, brick_t *QAI, float **PROB);
int cloud_probability(int nthread, int npix, int nclear, int nland, int *ncloud, float cldprob, float *cc, float *lowt, float *hight, brick_t *TOA, brick_t *QAI, small *pcp_, small *clr_, small *lnd_, small *brt_, short *var_, small **CLD);
int shadow_probability(int nthread, int nland, atc_t *atc, brick_t *TOA, brick_t *QAI, small *lnd_, small *cld_, short **SPR);
int cloud_parallax(int nclear, int nland, int npix, int *ncloud, float *cc, brick_t *TOA, brick_t *QAI, small *pcp_, small *clr_, small *lnd_, small *brt_, short *var_, small **CLD);
int shadow_position(float h, int x, int y, float res, int g, float **sun, float **view, int *newx, int *newy);
int shadow_matching(float shdprob, float lowtemp, float hightemp, atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *EXP, small *cld_, short *spr_, small **SHD);


/** This function builds the final cloud and cloud shadow classification.
--- npix:   number of valid image pixels
--- atc:    atmospheric correction factors
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information (modified)
--- DEM:    DEM
--- fcld_:  cloud mask
--- fshd_:  shadow mask
+++ Return: total cloud cover in %
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float finalize_cloud(par_ll_t *pl2, int npix, atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *DEM, small *fcld_, small *fshd_){
int p, nx, ny, nc, k = 0;
float res, pct;
float z, cir_thr;
small *dem_    = NULL;
short *blue_   = NULL;
short *cirrus_ = NULL;
int cld_buf, shd_buf;

  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);
  nc  = get_brick_ncells(QAI);
  res = get_brick_res(QAI);

  if ((dem_   = get_band_small(DEM, 0)) == NULL) return FAILURE;
  if ((blue_  = get_domain_short(TOA, "BLUE")) == NULL) return FAILURE;
  cirrus_     = get_domain_short(TOA, "CIRRUS");

  #ifdef FORCE_DEBUG
  small *cir_ = NULL; alloc((void**)&cir_, nc, sizeof(small));
  #endif

  /** set confident cloud **/

  #pragma omp parallel shared(nc, QAI, fcld_) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)) continue;
      if (fcld_[p]) set_cloud(QAI, p, 2);
    }
  }

  /** buffer clouds **/
  cld_buf = pl2->cldbuf/res;
  #ifdef CMIX_FAS_2
  cld_buf = 80/res;
  #endif
  buffer_(fcld_, nx, ny, cld_buf);

  /** buffer shadows **/
  shd_buf = pl2->shdbuf/res;
  #ifdef CMIX_FAS_2
  shd_buf = 40/res;
  #endif
  buffer_(fshd_, nx, ny, shd_buf);

  #pragma omp parallel private(z, cir_thr) shared(nc, atc, QAI, dem_, fcld_, fshd_, cirrus_, blue_) reduction(+: k) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)) continue;
      if (fcld_[p]){
        if (get_cloud(QAI, p) == 0) set_cloud(QAI, p, 1); 
      } else if (cirrus_  != NULL && !get_snow(QAI, p) && cirrus_[p] > 100){
        z = atc->dem.min+atc->dem.step/2.0 + dem_[p]*atc->dem.step;
        if ((cir_thr = 70 + 70*z*z) < 100) cir_thr = 100; // Baetens et al. 2019
        if (cirrus_[p] > cir_thr) set_cloud(QAI, p, 3);
      }
      if (fshd_[p]) set_shadow(QAI, p, true);
      if (get_cloud(QAI, p) > 0 || get_shadow(QAI, p)) k++;
    }

  }

  /** total cloud / cloud shadow cover in % **/
  pct = 100.0*k/(float)npix;
  
  #ifdef FORCE_DEBUG
  brick_t *BRICK = NULL; small *brick_ = NULL;
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_CLOUD-BUFFERED");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, fcld_,  nc*sizeof(small));
  set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_SHADOW-BUFFERED");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, fshd_,  nc*sizeof(small));
  set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  for (p=0; p<nc; p++) if (get_cloud(QAI, p) == 3) cir_[p] = true;
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_CIRRUS");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, cir_,  nc*sizeof(small));
  set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK); free((void*)cir_);
  #endif

  #ifdef CMIX_FAS
  #ifndef FORCE_DEBUG
  brick_t *BRICK = NULL; small *brick_ = NULL;
  #endif
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CMIX-FAS");
  brick_ = get_band_small(BRICK, 0); 
  for (p=0; p<nc; p++){
    if (get_off(QAI, p)){
      brick_[p] = 0;
    } else if (get_cloud(QAI, p) == 1 || get_cloud(QAI, p) == 2){
      brick_[p] = 3;
    } else if (get_cloud(QAI, p) == 3){
      brick_[p] = 2;
    } else if (get_shadow(QAI, p)){
      brick_[p] = 4;
    } else if (get_illumination(QAI, p) == 3){
      brick_[p] = 7;
    } else if (get_snow(QAI, p)){
      brick_[p] = 6;
    } else if (get_water(QAI, p)){
      brick_[p] = 5;
    } else {
      brick_[p] = 1;
    }
  }
  set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("finalized cloud mask", TIME);
  #endif

  return pct;
}


/** This function computes the potential cloud pixels.
--- npix:   number of valid image pixels (returned)
--- nclear: number of clear-sky pixels (returned)
--- nland:  number of clear-sky land pixels (returned)
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information (modified)
--- EXP:    Topographic slope
--- PCP:    Potential Cloud Pixels (returned)
--- CLR:    Clear-sky pixels (returned)
--- LND:    Clear-sky land pixels (returned)
--- BRT:    Bright pixels (returned)
--- VAR:    Variability probability (returned)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int potential_cloud(par_ll_t *pl2, int *npix, int *nclear, int *nland, brick_t *TOA, brick_t *QAI, brick_t *EXP, small **PCP, small **CLR, small **LND, small **BRT, short **VAR){
int p, nx, ny, nc;
int non = 0, nsnw = 0, nwtr = 0, nclr = 0, nlnd = 0;
float ndvi, ndsi, hot, vis, white, tmp, r45, vprob;
bool basic;
ushort *slp_   = NULL;
small  *snw_   = NULL;
small  *pcp_   = NULL;
small  *clr_   = NULL;
small  *lnd_   = NULL;
small  *brt_   = NULL;
short  *var_   = NULL;
short  *blue_  = NULL;
short  *green_ = NULL;
short  *red_   = NULL;
short  *nir_   = NULL;
short  *sw1_   = NULL;
short  *sw2_   = NULL;
short  *temp_  = NULL;
int snw_buf;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);
  nc = get_brick_ncells(QAI);
  
  if ((slp_   = get_band_ushort(EXP, ZEN))      == NULL) return FAILURE;
  if ((blue_  = get_domain_short(TOA, "BLUE"))  == NULL) return FAILURE; 
  if ((green_ = get_domain_short(TOA, "GREEN")) == NULL) return FAILURE; 
  if ((red_   = get_domain_short(TOA, "RED"))   == NULL) return FAILURE; 
  if ((nir_   = get_domain_short(TOA, "NIR"))   == NULL) return FAILURE; 
  if ((sw1_   = get_domain_short(TOA, "SWIR1")) == NULL) return FAILURE; 
  if ((sw2_   = get_domain_short(TOA, "SWIR2")) == NULL) return FAILURE; 
  temp_       = get_domain_short(TOA, "TEMP");
  
  alloc((void**)&snw_, nc, sizeof(small));
  alloc((void**)&pcp_, nc, sizeof(small));
  alloc((void**)&clr_, nc, sizeof(small));
  alloc((void**)&lnd_, nc, sizeof(small));
  alloc((void**)&brt_, nc, sizeof(small));
  alloc((void**)&var_, nc, sizeof(short));


  #pragma omp parallel private(tmp, ndvi, ndsi, basic, vis, r45, white, hot, vprob) shared(nc, QAI, slp_, pcp_, snw_, clr_, lnd_, brt_, var_, blue_, green_, red_, nir_, sw1_, sw2_, temp_) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      if (get_off(QAI, p)) continue;


      if ((tmp = nir_[p]+red_[p]) != 0){
        ndvi = (nir_[p]-red_[p])/tmp;
      } else ndvi = 0.01;

      if ((tmp = green_[p]+sw1_[p]) != 0){
        ndsi = (green_[p]-sw1_[p])/tmp;
      } else ndsi = 0.01;

      // Zhu et al., 2015 update, use 283K instead of 277K
      if (temp_ != NULL && ndsi > 0.15 && temp_[p] < 28300 && 
                         nir_[p] > 1100 && green_[p] > 1000){
        snw_[p] = true; 
      } else if (ndsi > 0.4 && sw2_[p] < 1100 &&
                         nir_[p] > 1100 && green_[p] > 1000){
        snw_[p] = true; 
      }

      if (((ndvi < 0.01 && nir_[p] < 1100) || 
           (ndvi < 0.1 && ndvi > 0 && nir_[p] < 500)) &&
            slp_[p] < 870){
        set_water(QAI, p, true);
      }

      if (ndsi < 0.8 && ndvi < 0.8 && sw2_[p] > 300){
        basic = true;
        if (temp_ != NULL && temp_[p] > 30015) basic = false;
      } else basic = false;

      if ((vis = (blue_[p]+green_[p]+red_[p])/3.0) > 1500) brt_[p] = true;

      white = (fabs(blue_[p]-vis) + fabs(green_[p]-vis) + fabs(red_[p]-vis))/vis;
      if (white > 1) white = 1.0;

      hot = blue_[p] - 0.5*red_[p] - 800;

      if (sw1_[p] != 0) r45 = nir_[p]/(float)sw1_[p]; else r45 = 0.0;

      if (get_saturation(QAI, p)){
        pcp_[p]   = true;
      } else if (basic && white < 0.7 && brt_[p] && hot > 0 && r45 > 0.75){
        pcp_[p] = true;
      }

      // MODIFY NDVI & NDSI
      if (get_saturation(QAI, p) && ndvi > 0) ndvi = 0.0; else ndvi = fabs(ndvi);
      if (get_saturation(QAI, p) && ndsi < 0) ndsi = 0.0; else ndsi = fabs(ndsi);

      // variability probability
      if (ndvi >= ndsi){
        if (ndvi >= white) vprob = 1.0 - ndvi; else vprob = 1.0 - white;
      } else {
        if (ndsi >= white) vprob = 1.0 - ndsi; else vprob = 1.0 - white;
      }
      var_[p] = (short)(vprob*10000);

      // PERCENT CLEAR
      if (!pcp_[p]) clr_[p] = true;
      if (!pcp_[p] && !get_water(QAI, p)) lnd_[p] = true;

    }

  } 

  // buffer snow layer
  snw_buf = pl2->snwbuf/get_brick_res(QAI);
  #ifdef CMIX_FAS_2
  snw_buf = 20/get_brick_res(QAI);
  #endif
  buffer_(snw_, nx, ny, snw_buf);

  #pragma omp parallel shared(nc, QAI, pcp_, snw_, clr_, lnd_) reduction(+: nwtr, nsnw, non, nclr, nlnd) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)) continue;
      //if (get_water(QAI, p)){ snw_[p] = false; nwtr++;}
      if (get_water(QAI, p)) nwtr++;
      //if (snw_[p]){ set_snow(QAI, p, true); pcp_[p] = false; nsnw++;}
      if (snw_[p]){ 
        if (clr_[p]) clr_[p] = false; // bc of snow buffer
        if (lnd_[p]) lnd_[p] = false; // bc of snow buffer
        set_snow(QAI, p, true);
        nsnw++;
      }
      if (clr_[p])  nclr++;
      if (lnd_[p])  nlnd++;
      non++;
    }
    
  }
  
  #ifdef FORCE_DEBUG
  brick_t *BRICK = NULL; small *brick_ = NULL; short *brick__ = NULL;
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_SNW");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, snw_,  nc*sizeof(small));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_PCP");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, pcp_,  nc*sizeof(small));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  BRICK = copy_brick(QAI, 1, _DT_SHORT_); set_brick_filename(BRICK, "CLD_VPROB");
  brick__ = get_band_short(BRICK, 0);  memmove(brick__, var_,  nc*sizeof(short));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif

  free((void*)snw_);

  printf("Data  cover (%): %6.2f%\n", non        / (float)nc  * 100.0);
  printf("Water cover (%): %6.2f%\n", nwtr       / (float)non * 100.0);
  printf("Snow  cover (%): %6.2f%\n", nsnw       / (float)non * 100.0);
  printf("PCP   cover (%): %6.2f%\n", (non-nclr) / (float)non * 100.0);

  #ifdef FORCE_CLOCK
  proctime_print("potential cloud identification", TIME);
  #endif

  *PCP    = pcp_;
  *CLR    = clr_;
  *LND    = lnd_;
  *BRT    = brt_;
  *VAR    = var_;
  *npix   = non;
  *nclear = nclr;
  *nland  = nlnd;
  return SUCCESS;
}


/** This function computes the land probability
--- nc:      number of cells
--- nclear:  number of clear-sky pixels (returned)
--- nland:   number of clear-sky land pixels (returned)
--- npix:    number of valid image pixels (returned)
--- cldprob: fixed cloud probability threshold
--- lowt:    low temperature threshold
--- hight:   high temperature threshold
--- lnd_:    Clear-sky land pixels
--- clr_:    Clear-sky pixels
--- temp_:   temperature
--- var_:    Variability probability
--- QAI:     Quality Assurance Information (modified)
--- PROB:    cloud probability
+++ Return:  adaptive cloud probability threshold
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float land_probability(int nc, int nclear, int nland, int npix, float cldprob, float *lowt, float *hight, small *lnd_, small *clr_, short *temp_, short *var_, brick_t *QAI, float **PROB){
int p, k = 0;
float lclr_max;
float lo = 0.175, hi = 0.825;
float lowtemp = 0.0, hightemp = 0.0;
float A, B;
float tempprob;
float *LANDPROB = NULL;
float *CLEARLPROB = NULL;
float *CLEARTEMP = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  /** clear all or clear land temp + temperature quantiles **/

  if ((100.0*nland/(float)npix) >= 0.1){

    alloc((void**)&CLEARTEMP, nland, sizeof(float));

    for (p=0, k=0; p<nc; p++){
      if (lnd_[p]) CLEARTEMP[k++] = temp_[p];
    }

    lowtemp  = quantile(CLEARTEMP, nland, lo);
    hightemp = quantile(CLEARTEMP, nland, hi);

  } else {

    alloc((void**)&CLEARTEMP, nclear, sizeof(float));

    for (p=0, k=0; p<nc; p++){
      if (clr_[p]) CLEARTEMP[k++] = temp_[p];
    }

    lowtemp  = quantile(CLEARTEMP, nclear, lo);
    hightemp = quantile(CLEARTEMP, nclear, hi);

  }

  free((void*)CLEARTEMP);
  

  /** cloud probability over land **/

  A = hightemp+400.0;
  B = hightemp+400.0-lowtemp+400.0;

  alloc((void**)&LANDPROB, nc, sizeof(float));

  for (p=0; p<nc; p++){

    if (get_off(QAI, p)){ LANDPROB[p] = 0; continue;}

    if ((tempprob = (A-temp_[p])/B) < 0) tempprob = 0.0;
    // note: not using the Zhu et al. 2015 ciirus probability
    LANDPROB[p] = tempprob*(var_[p]/10000.0);

  }


  // dynamic cloud detection threshold over land
  alloc((void**)&CLEARLPROB, nland, sizeof(float));

  for (p=0, k=0; p<nc; p++){
    if (lnd_[p]) CLEARLPROB[k++] = LANDPROB[p];
  }

  lclr_max = quantile(CLEARLPROB, nland, hi) + cldprob;
  free((void*)CLEARLPROB);
  
  #ifdef FORCE_CLOCK
  proctime_print("land probability", TIME);
  #endif

  *lowt  = lowtemp;
  *hight = hightemp;
  *PROB = LANDPROB;
  return lclr_max;
}


/** This function computes the water probability
--- nc:      number of cells
--- cldprob: fixed cloud probability threshold
--- temp_:   temperature
--- sw1_:    SWIR1
--- sw2_:    SWIR2
--- QAI:     Quality Assurance Information (modified)
--- PROB:    cloud probability
+++ Return:  adaptive cloud probability threshold
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float water_probability(int nc, float cldprob, short *temp_, short *sw1_, short *sw2_, brick_t *QAI, float **PROB){
int p, k = 0;
float wclr_max;
float hi = 0.825;
float brightprob, tempprob;
float wtrtemp = 0.0;
int nCLEARWTR = 0;
bool  *CLEARWTR = NULL;
float *WATERPROB = NULL;
float *CLEARWPROB = NULL;
float *CLEARTEMP = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  /** cloud probability over water **/

  alloc((void**)&CLEARWTR, nc, sizeof(bool));

  for (p=0; p<nc; p++){
    if (get_water(QAI, p) && sw2_[p] <= 300){ CLEARWTR[p] = true; nCLEARWTR++;}
  }

  // test if there is clear water, if not skip and give water prob = -1
  if (nCLEARWTR > 0){

    alloc((void**)&CLEARTEMP, nCLEARWTR, sizeof(float));

    for (p=0, k=0; p<nc; p++){
      if (CLEARWTR[p])CLEARTEMP[k++] = temp_[p];
    }

    wtrtemp = quantile(CLEARTEMP, nCLEARWTR, hi);
    free((void*)CLEARTEMP);


    alloc((void**)&WATERPROB, nc, sizeof(float));

    for (p=0; p<nc; p++){

      if (get_off(QAI, p)){ WATERPROB[p] = 0; continue;}

      tempprob = (wtrtemp-temp_[p])/400.0;
      if ((brightprob = sw1_[p]/1100.0) > 1.0) brightprob = 1.0;
      // note: not using the Zhu et al. 2015 ciirus probability
      WATERPROB[p] = tempprob*brightprob;

    }

    // dynamic cloud detection threshold over water
    alloc((void**)&CLEARWPROB, nCLEARWTR, sizeof(float));

    for (p=0, k=0; p<nc; p++){
      if (CLEARWTR[p]) CLEARWPROB[k++] = WATERPROB[p];
    }

    wclr_max = quantile(CLEARWPROB, nCLEARWTR, hi) + cldprob;
    free((void*)CLEARWPROB);

  } else {

    alloc((void**)&WATERPROB, nc, sizeof(float));
    wclr_max = 1;

  }

  free((void*)CLEARWTR);
  

  #ifdef FORCE_CLOCK
  proctime_print("water probability", TIME);
  #endif

  *PROB = WATERPROB;
  return wclr_max;
}


/** This function computes the cloud probability.
--- nthread: number of threads
--- npix:    number of valid image pixels
--- nclear:  number of clear-sky pixels
--- nland:   number of clear-sky land pixels
--- ncloud:  number of cloud pixels (returned)
--- cldprob: fixed cloud probability threshold
--- cc:      revised total cloud cover in % (returned)
--- lowt:    lower  temperature quantile (returned)
--- hight:   higher temperature quantile (returned)
--- TOA:     TOA reflectance
--- QAI:     Quality Assurance Information
--- pcp_:    Potential Cloud Pixels
--- clr_:    Clear-sky pixels 
--- lnd_:    Clear-sky land pixels
--- brt_:    Bright pixels 
--- var_:    Variability probability
--- CLD:     Cloud mask (returned)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int cloud_probability(int nthread, int npix, int nclear, int nland, int *ncloud, float cldprob, float *cc, float *lowt, float *hight, brick_t *TOA, brick_t *QAI, small *pcp_, small *clr_, small *lnd_, small *brt_, short *var_, small **CLD){
int p, nx, ny, nc;
int ncld = 0;

float lclr_max, wclr_max;
float lowtemp, hightemp;
float *LANDPROB = NULL;
float *WATERPROB = NULL;


short *sw1_  = NULL;
short *sw2_  = NULL;
short *temp_ = NULL;
small *cld_  = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);
  nc = get_brick_ncells(QAI);
  

  if ((sw1_  = get_domain_short(TOA, "SWIR1")) == NULL) return FAILURE;
  if ((sw2_  = get_domain_short(TOA, "SWIR2")) == NULL) return FAILURE;
  if ((temp_ = get_domain_short(TOA, "TEMP"))  == NULL) return FAILURE;
  
  alloc((void**)&cld_, nc, sizeof(small));
  
  if (nthread == 1){
    lclr_max = land_probability(nc, nclear, nland, npix, cldprob, &lowtemp, &hightemp, 
                                lnd_, clr_, temp_, var_, QAI, &LANDPROB);
    wclr_max =  water_probability(nc, cldprob, temp_, sw1_, sw2_, QAI, &WATERPROB);
  } else {

    #pragma omp parallel num_threads(2) shared(nc, nclear, nland, npix, cldprob, lowtemp, hightemp, lnd_, clr_, temp_, var_, sw1_, sw2_, QAI, lclr_max, wclr_max, LANDPROB, WATERPROB) default(none)
    {

      if (omp_get_thread_num() == 0){
        lclr_max = land_probability(nc, nclear, nland, npix, cldprob, &lowtemp, &hightemp, 
                                    lnd_, clr_, temp_, var_, QAI, &LANDPROB);
      } else {
        wclr_max =  water_probability(nc, cldprob, temp_, sw1_, sw2_, QAI, &WATERPROB);
      }

    }
    
  }
  
  

  #ifdef FORCE_DEBUG
  printf("\n\nclr_max land/water/bt-cold: %f/%f/%f\n", lclr_max, wclr_max, lowtemp-3500);
  #endif

  /**  final cloud layer = 
       cloud over land thin OR 
       cloud over water OR 
       high prob cloud (land) 
       OR extremly cold cloud 
       AND white**/
    
  #pragma omp parallel shared(nc, lclr_max, wclr_max, lowtemp, cld_, pcp_, temp_, brt_, LANDPROB, WATERPROB, QAI) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){
      // Zhu et al., 2015 modification :::
      // Frantz et al., 2015 modification :::
      cld_[p] = ((pcp_[p] && !get_water(QAI, p) && LANDPROB[p]  > lclr_max) ||
                  (pcp_[p] &&  get_water(QAI, p) && WATERPROB[p] > wclr_max) ||
                   temp_[p] < (lowtemp-3500) || 
                   get_saturation(QAI, p)) &&
                   brt_[p];
    }
    
  }

  free((void*)LANDPROB);
  free((void*)WATERPROB);


  // set clear pixel to cloud if 5 or more cloud pixels in neighborhood + 
  // set boundary to clear
  majorfill_(cld_, nx, ny);

  #pragma omp parallel shared(nc, cld_, QAI) reduction(+: ncld) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){
      //if (get_snow(QAI, p)) cld_[p] = false;
      if (cld_[p]) ncld++;
    }

  }
  
  #ifdef FORCE_DEBUG
  brick_t *BRICK = NULL; small *brick_ = NULL;
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_CLOUD");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, cld_,  nc*sizeof(small));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("cloud probability computation", TIME);
  #endif


  // revised cloud coverage
  *CLD = cld_;
  *ncloud = ncld;
  *cc = 100.0 * ncld/(float)npix;
  *lowt  = lowtemp;
  *hight = hightemp;
  return SUCCESS;
}



/** This function computes the shadow probability and return potential 
+++ shadow pixels.
--- nthread: number of threads
--- nland:   number of clear-sky land pixels
--- atc:     atmospheric correction factors
--- TOA:     TOA reflectance
--- QAI:     Quality Assurance Information
--- lnd_:    Clear-sky land pixels 
--- cld_:    Cloud mask
--- SPR:     Potential Shadow Pixels (returned)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int shadow_probability(int nthread, int nland, atc_t *atc, brick_t *TOA, brick_t *QAI, small *lnd_, small *cld_, short **SPR){
int i, j, p, k = 0, nx, ny, nc, b, nb = 2;
int nthr;
int err = 0;
float res;
float maxdist;
short bck;
float lo = 0.175;
short *spr_  = NULL;
short *toa_    = NULL;
short *mask_   = NULL;
short *marker_ = NULL;
float *clear_  = NULL;
ushort *dist_;
char domains[2][NPOW_10] = { "NIR", "SWIR1" };


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nx  = get_brick_ncols(QAI);
  ny  = get_brick_nrows(QAI);
  nc  = get_brick_ncells(QAI);
  res = get_brick_res(QAI);

  alloc((void**)&spr_, nc, sizeof(short));
  
  
  // compute shadow probability only for pixels that 
  // are close enough to clouds
  if ((dist_ = dist_transform_(cld_, nx, ny)) == NULL){
    printf("distance transform failed.\n"); return FAILURE;}
  maxdist = 12000 * tan(acos(atc->cosszen[0]))/res;

  #ifdef FORCE_DEBUG
  printf("maximum distance for cloud shadows: %.0f", maxdist);
  #endif


  #pragma omp parallel shared(nc, spr_) reduction(+: err) default(none)
  {
    #pragma omp for
    for (p=0; p<nc; p++) spr_[p] = SHRT_MAX;
  }


  // Zhu et al., 2015 modification
  // use NIR and SWIR
  
  if (nthread == 1) nthr = 1; else nthr = nb;
  
  #pragma omp parallel num_threads(nthr) private(i, j, p, k, mask_, marker_, clear_, toa_, bck) shared(nb, nx, ny, nc, nland, domains, lo, maxdist, dist_, lnd_, spr_, TOA, QAI) reduction(+: err) default(none)
  {

    alloc((void**)&mask_,   nc,    sizeof(short));
    alloc((void**)&marker_, nc,    sizeof(short));

    #pragma omp for
    for (b=0; b<nb; b++){
      
      if ((toa_  = get_domain_short(TOA, domains[b]))   == NULL){ err++; continue;}

      memmove(mask_, toa_, nc*sizeof(short));

      alloc((void**)&clear_,  nland, sizeof(float));
      for (p=0, k=0; p<nc; p++){
        if (lnd_[p]) clear_[k++] = mask_[p];
      }
      bck = (short)quantile(clear_, nland, lo);
      free((void*)clear_);

      for (i=0, p=0; i<ny; i++){
      for (j=0; j<nx; j++, p++){
          if (i == 0 || i == ny-1 || j == 0 || j == nx-1 || 
              get_off(QAI, p) || dist_[p] > maxdist) mask_[p] = bck;
      }
      }

      greyscale_reconstruction_(mask_, marker_, nx, ny);
      for (p=0; p<nc; p++) marker_[p]  -= mask_[p];

    }

    #pragma omp critical
    {
      for (p=0; p<nc; p++){
        if (marker_[p] < spr_[p]) spr_[p] = marker_[p];
      }
    }

    free((void*)mask_);
    free((void*)marker_);


  }
  
  if (err > 0){ printf("error in shadow probability. "); return FAILURE;}
  
  free((void*)dist_);


  #ifdef FORCE_CLOCK
  proctime_print("shadow probability computation", TIME);
  #endif

  *SPR = spr_;
  return SUCCESS;
}


/** This function separates PCPs on the basis of parallax effects in Sen-
+++ tinel-2 images.
--- nclear: number of clear-sky pixels
--- nland:  number of clear-sky land pixels
--- npix:   number of valid image pixels
--- ncloud: number of cloud pixels (returned)
--- cc:     revised total cloud cover in % (returned)
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information
--- pcp_:   Potential Cloud Pixels
--- clr_:   Clear-sky pixels 
--- lnd_:   Clear-sky land pixels (deallocated within)
--- brt_:   Bright pixels (is deallocated within)
--- var_:   Variability probability
--- CLD:    Cloud mask (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int cloud_parallax(int nclear, int nland, int npix, int *ncloud, float *cc, brick_t *TOA, brick_t *QAI, small *pcp_, small *clr_, small *lnd_, small *brt_, short *var_, small **CLD){
int i, i_, j, j_, p, p_, ii, jj, ni, nj, np, np_;
int nx,  ny,  nc;
int nx_, ny_, nc_;
int ncld = 0;
int nodata = -9999;
int nk = 5;
float **kernel = NULL;
double sum_re3, sum_nir, sum_bnir, sum_var, num, min_var;
double mx, my, varx, vary, vx, vy, cv, k;
float *gauss_ = NULL;
float *var2_ = NULL;
float *var3_ = NULL;
float **ratio_ = NULL;
float *cdi_ = NULL;
bool pcp, valid;
small *clouds_ = NULL;
queue_t fifo;
short *re3_  = NULL;
short *bnir_ = NULL;
short *nir_  = NULL;
small *cld_  = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nx = get_brick_ncols(QAI);  nx_ = nx/2;
  ny = get_brick_nrows(QAI);  ny_ = ny/2;
  nc = get_brick_ncells(QAI); nc_ = nx_*ny_;

  if ((re3_  = get_domain_short(TOA, "REDEDGE3")) == NULL) return FAILURE;
  if ((bnir_ = get_domain_short(TOA, "BROADNIR")) == NULL) return FAILURE;
  if ((nir_  = get_domain_short(TOA, "NIR"))      == NULL) return FAILURE;
  

  alloc((void**)&cld_, nc, sizeof(small));


  /** Convolute BNIR with gaussian kernel **/

  if (gauss_kernel(nk, 0.5, &kernel) != SUCCESS){
    printf("Could not generate kernel. "); return FAILURE;}

  alloc((void**)&gauss_, nc, sizeof(float));

  #pragma omp parallel private (j, p, ii, jj, ni, nj, np, sum_bnir, num) shared(nx, ny, nk, kernel, bnir_, QAI, gauss_) default(none)
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      
      p = i*nx+j;
      
      if (get_off(QAI, p)) continue;

      sum_bnir = num = 0;

      for (ii=0; ii<nk; ii++){
      for (jj=0; jj<nk; jj++){

        ni = -(nk-1)/2 + ii + i; nj = -(nk-1)/2 + jj + j;
        if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;
        np = ni*nx+nj;
        
        if (get_off(QAI, np)) continue;

        sum_bnir += bnir_[np]*kernel[ii][jj];
        num += kernel[ii][jj];

      }
      }

      if (num > 0) gauss_[p] = (float)(sum_bnir/num);

    }
    }
    
  }

  free_2D((void**)kernel, nk);


  /** Compute NIR Ratios at reduced spatial resolution.
  +++  And Variability Probability
  +++  This is only done for PCPs. **/

  alloc_2D((void***)&ratio_, 2, nc_, sizeof(float));
  alloc((void**)&var2_, nc_, sizeof(float));

  #pragma omp parallel private (j, p, p_, ii, jj, ni, nj, np, sum_re3, sum_nir, sum_bnir, sum_var, num, pcp) shared(nx, ny, nx_, re3_, nir_, pcp_, QAI, gauss_, var_, var2_, ratio_) default(none)
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i+=2){
    for (j=0; j<nx; j+=2){
      
      p = i*nx+j;
      p_ = (i/2)*nx_+(j/2);

      if (get_off(QAI, p)) continue;

      sum_re3 = sum_nir = sum_bnir = sum_var = num = 0;
      pcp = false;

      for (ii=0; ii<2; ii++){
      for (jj=0; jj<2; jj++){

        ni = i+ii; nj = j+jj;
        if (ni >= ny || nj >= nx) continue;
        np = ni*nx+nj;

        if (get_off(QAI, np)) continue;

        sum_bnir += gauss_[np];
        sum_nir  += nir_[np];
        sum_re3  += re3_[np];
        sum_var  += var_[np]/10000.0;
        num++;
        
        //hot = (hot_[np] <= 10000.0) ? hot_[np]/10000.0 : 1.0;
        //var = (var_[np] <= 10000.0) ? var_[np]/10000.0 : 1.0;
        //vsum += hot*var;

        if (pcp_[np]) pcp = true;

      }
      }

      if (num > 0 && pcp){
//        ratio_[0][p_] = (float)re3_[p]   / (float)nir_[p];
//        ratio_[1][p_] = (float)(sum/num) / (float)nir_[p];
//        var2_[p_] = vsum/num;
        ratio_[0][p_] = (float)((sum_re3/num)  / (sum_nir/num));
        ratio_[1][p_] = (float)((sum_bnir/num) / (sum_nir/num));
        var2_[p_] = (float)(sum_var/num);
      }

    }
    }
    
  }

  free((void*)gauss_);

  #ifdef FORCE_DEBUG
  brick_t *BRICK = NULL; float *brick_ = NULL;
  BRICK = copy_brick(QAI, 1, _DT_NONE_); set_brick_filename(BRICK, "CLD_R8A8");
  set_brick_ncols(BRICK, nx_); set_brick_nrows(BRICK, ny_); allocate_brick_bands(BRICK, 1, nc_, _DT_FLOAT_);
  brick_ = get_band_float(BRICK, 0);  memmove(brick_, ratio_[0],  nc_*sizeof(float));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  BRICK = copy_brick(QAI, 1, _DT_NONE_); set_brick_filename(BRICK, "CLD_R8A7");
  set_brick_ncols(BRICK, nx_); set_brick_nrows(BRICK, ny_); allocate_brick_bands(BRICK, 1, nc_, _DT_FLOAT_);
  brick_ = get_band_float(BRICK, 0);  memmove(brick_, ratio_[1],  nc_*sizeof(float));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif
  


  /** Compute Cloud Displacement Index CDI
  +++ Texture (variance) is obtained from both ratios and a normalied 
  +++ differenced variance ratio (CDI) is computed thereof. **/
  /** min of variability probability **/

  alloc((void**)&cdi_,  nc_, sizeof(float));
  alloc((void**)&var3_, nc_, sizeof(float));

  #pragma omp parallel private(j_, p_, ii, jj, ni, nj, np_, mx, my, vx, vy, cv, k, varx, vary, min_var) shared(nx_, ny_, cdi_, ratio_, var2_, var3_, nodata) default(none)
  {

    #pragma omp for schedule(guided)
    for (i_=0; i_<ny_; i_++){
    for (j_=0; j_<nx_; j_++){
      
      p_ = i_*nx_+j_;

      cdi_[p_] = nodata;

      if (ratio_[0][p_] == 0) continue;

      mx = my = vx = vy = cv = k = 0;
     // min_var = INT_MAX;

      for (ii=-3; ii<=3; ii++){
      for (jj=-3; jj<=3; jj++){

        ni = i_+ii; nj = j_+jj;
        if (ni < 0 || ni >= ny_ || nj < 0 || nj >= nx_) continue;
        np_ = ni*nx_+nj;

        if (ratio_[0][np_] == 0) continue;

        if (++k == 1){
          mx = ratio_[0][np_]; my = ratio_[1][np_];
        } else {
          covar_recurrence(ratio_[0][np_], ratio_[1][np_],
            &mx, &my, &vx, &vy, &cv, k);
        }

        //if (var2_[np_] < min_var) min_var = var2_[np_];

      }
      }

      if (k > 2){
        varx = variance(vx, k);
        vary = variance(vy, k);
        cdi_[p_] = (varx-vary)/(varx+vary);
      }
      
      
      min_var = INT_MAX;
      
      for (ii=-7; ii<=7; ii++){
      for (jj=-7; jj<=7; jj++){

        ni = i_+ii; nj = j_+jj;
        if (ni < 0 || ni >= ny_ || nj < 0 || nj >= nx_) continue;
        np_ = ni*nx_+nj;

        if (ratio_[0][np_] == 0) continue;

        if (var2_[np_] < min_var) min_var = var2_[np_];

      }
      }

      if (min_var != INT_MAX) var3_[p_] = min_var;

    }
    }
    
  }

  free_2D((void**)ratio_, 2);
  free((void*)var2_);
  
  #ifdef FORCE_DEBUG
  BRICK = copy_brick(QAI, 1, _DT_NONE_); set_brick_filename(BRICK, "CLOUD_CDI");
  set_brick_ncols(BRICK, nx_); set_brick_nrows(BRICK, ny_); allocate_brick_bands(BRICK, 1, nc_, _DT_FLOAT_);
  brick_ = get_band_float(BRICK, 0);  memmove(brick_, cdi_,  nc_*sizeof(float));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);

  BRICK = copy_brick(QAI, 1, _DT_NONE_); set_brick_filename(BRICK, "CLOUD_S2PROBMIN");
  set_brick_ncols(BRICK, nx_); set_brick_nrows(BRICK, ny_); allocate_brick_bands(BRICK, 1, nc_, _DT_FLOAT_);
  brick_ = get_band_float(BRICK, 0);  memmove(brick_, var3_,  nc_*sizeof(float));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif


  /** Separate clouds from built-up using CDI + local variability probability
  +++ Clouds are eroded by one pixel **/

  alloc((void**)&clouds_, nc_, sizeof(small));

  #pragma omp parallel private(j_, p_, ii, jj, ni, nj, np_, valid) shared(nx_, ny_, cdi_, var3_, clouds_, nodata) default(none)
  {

    #pragma omp for schedule(guided)
    for (i_=0; i_<ny_; i_++){
    for (j_=0; j_<nx_; j_++){

      p_ = i_*nx_+j_;

      if (cdi_[p_] == nodata) continue;

      if (cdi_[p_] <= -0.5 || (var3_[p_] > 0.8 && (cdi_[p_] < 0))){

        valid = true;

        for (ii=-1; ii<=1; ii++){
        for (jj=-1; jj<=1; jj++){

          ni = i_+ii; nj = j_+jj;
          if (ni < 0 || ni >= ny_ || nj < 0 || nj >= nx_) continue;
          np_ = ni*nx_+nj;

          if (cdi_[np_] == nodata || (cdi_[np_] >  -0.5 && var3_[np_] < 0.8)) valid = false;
          //if (cdi_[np_] == nodata || cdi_[np_] >  -0.5) valid = false;

        }
        }

        if (valid) clouds_[p_] = true;
        
      } /**else {

        vmin = INT_MAX;

        for (ii=-3; ii<=3; ii++){
        for (jj=-3; jj<=3; jj++){

          ni = i_+ii; nj = j_+jj;
          if (ni < 0 || ni >= ny_ || nj < 0 || nj >= nx_) continue;
          np_ = ni*nx_+nj;

          if (cdi_[np_] == nodata) continue;
          if (var2_[np_] < vmin) vmin = var2_[np_];

        }
        }

        if (vmin > 7500) clouds_[p_] = true;

      }**/

    }
    }

  }
  
  free((void*)var3_);


  /** Region-growing 1. Put all cloud edge pixels, from which
  +++ clouds can be grown in FIFO **/
  
  if ((create_queue(&fifo, nc_)) == FAILURE){
    printf("failed to create new queue!\n"); return FAILURE;}

  for (i_=0, p_=0; i_<ny_; i_++){
  for (j_=0; j_<nx_; j_++, p_++){

    if (!clouds_[p_]) continue;

      for (ii=-1; ii<=1; ii++){
      for (jj=-1; jj<=1; jj++){

        if (ii==0 && jj==0) continue;

        ni = i_+ii; nj = j_+jj;
        if (ni < 0 || ni >= ny_ || nj < 0 || nj >= nx_) continue;
        np_ = ni*nx_+nj;

        if (!clouds_[np_] && cdi_[np_] < -0.25 && cdi_[np_] >= -1){
          if (enqueue(&fifo, j_, i_) == FAILURE){
            printf("Failed to allocate enqueue memory\n"); return FAILURE;}
          ii = jj = 10;
        }

      }
      }

  }
  }


  /** Region-growing 2. Use FIFO to grow cloud objects.
  +++ All connected pixels with fairly low CDI become clouds. **/

  while (dequeue(&fifo, &j_, &i_) == SUCCESS){

    p_ = i_*nx_+j_;

    for (ii=-1; ii<=1; ii++){
    for (jj=-1; jj<=1; jj++){

      if (ii==0 && jj==0) continue;

      ni = i_+ii; nj = j_+jj;
      if (ni < 0 || ni >= ny_ || nj < 0 || nj >= nx_) continue;
      np_ = ni*nx_+nj;

      if (!clouds_[np_] && cdi_[np_] < -0.25 && cdi_[np_] >= -1){
        clouds_[np_] = true;
        if (enqueue(&fifo, nj, ni) == FAILURE){
          printf("Failed to allocate enqueue memory\n"); return FAILURE;}
      }

    }
    }

  }

  free((void*)cdi_);
  destroy_queue(&fifo);


  /** Restore original resolution **/

  #pragma omp parallel private(j_, p_, i, j) shared(nx_, ny_, nx, clouds_, cld_) default(none)
  {

    #pragma omp for schedule(guided)
    for (i_=0; i_<ny_; i_++){
    for (j_=0; j_<nx_; j_++){
      
      p_ = i_*nx_+j_;

      if (clouds_[p_]){

        for (i=i_*2; i<=(i_*2+1); i++){
        for (j=j_*2; j<=(j_*2+1); j++){
            cld_[i*nx+j] = true;
        }
        }
      
      }
      
    }
    }
    
  }

  free((void*)clouds_);


  /** Add saturated, remove dark pixels */

  #pragma omp parallel shared(nc, cld_, brt_, QAI) reduction(+: ncld) default(none)
  {

    #pragma omp for schedule(static)
    for (p=0; p<nc; p++){

      // Frantz et al., 2015 modification :::
      if (get_saturation(QAI, p)) cld_[p] = true;
      if (cld_[p] && !brt_[p]) cld_[p] = false;
      if (cld_[p]) ncld++;

    }
    
  }

  
  #ifdef FORCE_DEBUG
  small *brick__ = NULL;
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_CLOUD");
  brick__ = get_band_small(BRICK, 0);  memmove(brick__, cld_,  nc*sizeof(small));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("cloud from CDI computation", TIME);
  #endif


  // revised cloud coverage
  *CLD = cld_;
  *ncloud = ncld;
  *cc = 100.0 * ncld/(float)npix;
  return SUCCESS;
}


/** Due to the scanning geometry, a cloud is projected to the image plane
+++ in a ~perpendicular angle to the nadir line, i.e. the viewing azimuth.
+++ This is not exactly correct for detectors with multiple lines, but it
+++ should be sufficiently precise for this purpose. Thus, the cloud needs
+++ to be shifted towards the nadir line in across-scan direction in order
+++ to find the 'real' cloud position. The distance depends on the assumed
+++ cloud pixel height and viewing zenith. The shift is larger for large 
+++ zenith angles and high clouds.
+++ Afterwards, the cloud needs to be projected to the image plane in sun
+++ azimuth direction. The distance depends on the assumed cloud pixel 
+++ height and sun zenith. The shift is larger for large zenith angles and
+++ high clouds. The function returns the position of the casted shadow in 
+++ x/y pixel coords.
--- h:      height of object (in m above ground) 
--- x:      location of object in image
--- y:      location of object in image
--- res:    resolution
--- g:      coarse grid cell
--- sun:    sun angles
--- view:   view angles
--- newx:   location of object in real world (returned)
--- newy:   location of object in real world (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int shadow_position(float h, int x, int y, float res, int g, float **sun, float **view, int *newx, int *newy){
float dist_across, dist_cast;
float dx_across, dx_cast;
float dy_across, dy_cast;


  dist_across = h * view[tZEN][g] / res;
  dist_cast   = h * sun[tZEN][g] / res;

  dx_across = dist_across * view[sAZI][g];
  dx_cast   = dist_cast   * sun[sAZI][g];

  dy_across = dist_across * view[cAZI][g];
  dy_cast   = dist_cast   * sun[cAZI][g];

  *newx = (int) round(x + dx_across - dx_cast);
  *newy = (int) round(y - dy_across + dy_cast);

  return SUCCESS;
}


/** Knowing the view and solar geometry (and cloud height), we can predict
+++ the location where the shadow is projected to the ground. This projec-
+++ tion is matched with the shadow probability, and if it matches, a sha-
+++ dow is drawn. As the cloud height is not known precisely, the possible
+++ cloud base height range is iterated. The base height range can be nar-
+++ rowed down with a temperature-based prediction. The individual cloud
+++ pixel height is also predicted from temperature, i.e. a cloud DEM is 
+++ built before projecting the cloud to the image plane. If a temperature
+++ band is not available, the full cloud base height range is tested and
+++ the clouds are treated as flat planes.
--- shdprob: shadow probability
--- lowtemp:  lower  temperature quantile (returned)
--- hightemp: higher temperature quantile (returned)
--- atc:      atmospheric correction factors
--- TOA:      TOA reflectance
--- QAI:      Quality Assurance Information
--- EXP:      Topographic slope
--- cld_:     Cloud mask
--- spr_:     Potential Shadow Pixels
--- SHD:      Shadow mask (returned)
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int shadow_matching(float shdprob, float lowtemp, float hightemp, atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *EXP, small *cld_, short *spr_, small **SHD){
int i, j, p, nx, ny, nc, nf, ne, g, k;
int skip, influence = 8;
int nobj, id, size_max = 0, size;
int *P = NULL, *best_P = NULL;
float *qtemp = NULL;
float res, radius, core, basetemp;
float base_min, base_max, base_step, base;
float best_match, match, shadow, total;
int x, y;
float height;
float wlapse = 0.65, dlapse = 0.98, rlapse = 0.1; // wet, dry and reducced adiabatic lapse rate in 100*kelvin/m
int    *CCL        = NULL;
int    *SIZE       = NULL;
int   **array_x    = NULL;
int   **array_y    = NULL;
short **array_temp = NULL;
float **sun_       = NULL;
float **view_      = NULL;
ushort *slp_       = NULL;
small  *shd_       = NULL;
short  *temp_      = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  nx   = get_brick_ncols(QAI);
  ny   = get_brick_nrows(QAI);
  nc   = get_brick_ncells(QAI);
  res  = get_brick_res(QAI);
  
  if ((sun_  =  get_bands_float(atc->xy_sun))  == NULL) return FAILURE;
  if ((view_ =  get_bands_float(atc->xy_view)) == NULL) return FAILURE;
  if ((slp_  = get_band_ushort(EXP, ZEN))       == NULL) return FAILURE;
  temp_      = get_domain_short(TOA, "TEMP");

  alloc((void**)&shd_, nc, sizeof(small));
  
  #ifdef FORCE_DEV
  printf("shadow match increased to .95, was .85. 2 occurences.\n");
  #endif



  /** make cloud objects and delete small clouds **/
  binary_to_objects(cld_, nx, ny, 3, &CCL, &SIZE, &nobj);

  printf("Number of cloud objects: %d\n", nobj);


  /** skip shadow matching if there is no cloud left **/
  if (nobj == 0){

    #ifdef FORCE_CLOCK
    proctime_print("shadow matching", TIME);
    #endif

    #ifdef FORCE_DEBUG
    printf("\nno remaining cloud object. skip shadow matching.\n");
    #endif

    *SHD = shd_;
    return SUCCESS;
  }

  
  if ((array_x = (int**) malloc(nobj*sizeof(int*))) == NULL){
    printf("unable to allocate memory!\n"); exit(1);}
  for (id=0; id<nobj; id++){
    if ((array_x[id] = (int*) malloc(SIZE[id]*sizeof(int))) == NULL){
      printf("unable to allocate memory!\n"); exit(1);}
  }

  if ((array_y = (int**) malloc(nobj*sizeof(int*))) == NULL){
    printf("unable to allocate memory!\n"); exit(1);}
  for (id=0; id<nobj; id++){
    if ((array_y[id] = (int*) malloc(SIZE[id]*sizeof(int))) == NULL){
      printf("unable to allocate memory!\n"); exit(1);}
  }

  if ((array_temp = (short**) malloc(nobj*sizeof(short*))) == NULL){
    printf("unable to allocate memory!\n"); exit(1);}
  for (id=0; id<nobj; id++){
    if ((array_temp[id] = (short*) malloc(SIZE[id]*sizeof(short))) == NULL){
      printf("unable to allocate memory!\n"); exit(1);}
  }

  int *K;
  alloc((void**)&K, nobj, sizeof(int));
  

  // copy relevant info to smaller arrays
  skip = floor(30/res);
  for (i=0, p=0; i<ny; i++){
  for (j=0; j<nx; j++, p++){
    if ((id = CCL[p]) > 0){
      // only do for pixels that are >30m apart
      if (res < 30 && (i % skip != 0 || j % skip != 0)) continue;
      id--;
      array_x[id][K[id]] = j;
      array_y[id][K[id]] = i;
      if (temp_ != NULL) array_temp[id][K[id]] = temp_[p];
      K[id]++;
    }
  }
  }

  // copy new size
  if (res < 30){
    for (id=0; id<nobj; id++) SIZE[id] = K[id];
  }
  
  free((void*)K);


  /** Determine the step size for the base height iteration
  +++ Pick height that coincides with a horizontal shift of 50m.
  +++ Use the sun zenith at scene center as approximation. **/
  nf = get_brick_ncols(atc->xy_sun);
  ne = get_brick_nrows(atc->xy_sun);
  base_step = 50 / sun_[tZEN][(ne/2)*nf+(nf/2)];

 
  #ifdef FORCE_DEBUG
  printf("\nbase height iteration in %.2fm steps\n", base_step);
  #endif

  for (id=0; id<nobj; id++){
    if (SIZE[id] > size_max) size_max = SIZE[id];
  }

  #ifdef FORCE_DEBUG
  printf("max. object size: %d\n", size_max);
  #endif





  #pragma omp parallel private(k, g, x, y, p, size, radius, core, basetemp, base_min, base_max, base, height, shadow, total, match, best_match, qtemp, P, best_P) shared(nx, ny, res, nobj, influence, lowtemp, hightemp, dlapse, rlapse, wlapse, base_step, size_max, spr_, shdprob, cld_, slp_, atc, sun_, view_, QAI, CCL, shd_, array_x, array_y, array_temp, SIZE, temp_) default(none) 
  {

    if (temp_ != NULL) alloc((void**)&qtemp, size_max, sizeof(float));
    alloc((void**)&P,       size_max, sizeof(int));
    alloc((void**)&best_P,  size_max, sizeof(int));

    #pragma omp for schedule(guided)
    for (id=0; id<nobj; id++){

      // assume object is round
      size = SIZE[id];
      radius = sqrt(size/M_PI);

      // percent of cloud core area
      if (radius > influence){
        core = (radius-influence)*(radius-influence)/(radius*radius);
      } else {
        core = 0.0;
      }


      /** Predict min and max cloud base height:
      +++ The cloud temperature and the dry adiabatic lapse rate are used
      +++ to narrow down the base height range for better performance. If 
      +++ there is no temperature band, use the full range. **/
      if (temp_ != NULL){

        /** edge of the cloud is influenced by the warm surface
        +++ Pixels that are too warm get the value from the cloud core. **/
        for (k=0; k<size; k++) qtemp[k] = array_temp[id][k];
        basetemp = quantile(qtemp, size, core);
        for (k=0; k<size; k++){
          if (array_temp[id][k] > basetemp) array_temp[id][k] = (short)basetemp;
        }

        /** base heigt is estimated from the warmest cloud temperatures. **/
        if ((base_min = (lowtemp- 400-basetemp)/dlapse) < 200)   base_min = 200;
        if ((base_max = (hightemp+400-basetemp)/rlapse) > 12000) base_max = 12000;
        if (base_min > base_max) continue;

      } else {
        basetemp = 0;
        base_min = 200;
        base_max = 12000;
      }


      /** Base height iteration:
      +++ Lift the cloud up across the possible base height range and match
      +++ the casted shadow with the potential shadow layer. **/
      for (base=base_min, best_match=0; base<=base_max; base+=base_step){

        for (k=0, shadow=0, total=0; k<size; k++){

          /** Predict cloud pixel height:
          +++ A cloud DEM is used for a more exact calculation of projected
          +++ shadow position. The height is predicted using the temperature
          +++ band and the wet adiabatic lapse rate, plus the cloud base
          +++ height. If there is no temperature band, the cloud is assumed
          +++ to be a flat plate. **/
          if (temp_ != NULL){
            height = (basetemp-array_temp[id][k])/wlapse + base;
          } else {
            height = base;
          }

          /** Position of projected shadow:
          +++ Copmpute the position of the projected shadow as a function of
          +++ view and sun geometry. **/
          g = convert_brick_ji2p(QAI, atc->xy_sun, array_y[id][k], array_x[id][k]);
          shadow_position(height, array_x[id][k], array_y[id][k], res, g, sun_, view_, &x, &y);

          if (y < 0 || y >= ny || x < 0 || x >= nx){
            P[k] = -1; continue;}

          P[k] = p = nx*y + x; 


          /** Simplified match:
          +++ A shadow is matched if it is a potential shadow, but not a 
          +++ cloud. The shadow matching 'runs' into big clouds if clouds
          +++ are also permitted. The match is measured relative to the
          +++ complete shifted object, excluding the original cloud. **/
          if (spr_[p] > (shdprob*10000) && !cld_[p] && !(slp_[p] == 0 && get_water(QAI, p))) shadow += spr_[p]/750.0;
          if (CCL[p] != id+1) total++;

        }

        /** Matching measure:
        +++ The match is expressed in fractions between 0...1 **/
        if (total > 0) match = shadow/total; else match = 0;


        /** Excellent match:
        +++ If the match is better than 95%, draw a shadow. This is a greedy
        +++ implementation when compared to the original Fmask, but we found
        +++ that the shadow matching terminated too early in many cases. We
        +++ do not terminate the matching once the match decreases again.**/
        if (match >= 0.85){
          for (k=0; k<size; k++){
            if ((p = P[k]) != -1){
              shd_[p] = true;
            }
          }
        }

        /** Best match:
        +++ Record the position and value of the best match. **/
        if (match > best_match){
          best_match = match;
          for (k=0; k<size; k++) best_P[k] = P[k];
        }

      }


      /** Best match:
      +++ If best match is less than 85%, draw the shadow anyway, even if
      +++ the match is not very high. This is a greedy implementation when
      +++ compared to the original Fmask, but every cloud has a shadow, so
      +++ we assume that the best match is in the right position. If the
      +++ best match is better than 85%, it was already drawn. **/
      if (best_match < 0.85 && best_match > 0){
        for (k=0; k<size; k++){
          if ((p = best_P[k]) != -1){
            shd_[p] = true;
          }
        }
      }

    }

    if (temp_ != NULL) free((void*)qtemp);
    free((void*)P);
    free((void*)best_P);

  } // end omp parallel
  

  #ifdef FORCE_DEBUG
  brick_t *BRICK = NULL; small *brick_ = NULL;
  BRICK = copy_brick(QAI, 1, _DT_SMALL_); set_brick_filename(BRICK, "CLD_SHADOW");
  brick_ = get_band_small(BRICK, 0);  memmove(brick_, shd_,  nc*sizeof(small));
  print_brick_info(BRICK); set_brick_open(BRICK, OPEN_CREATE); write_brick(BRICK); free_brick(BRICK);
  #endif


  for (id=0; id<nobj; id++){
    free((void*)array_x[id]);
    free((void*)array_y[id]);
    free((void*)array_temp[id]);
  }
  free((void*)array_x);
  free((void*)array_y);
  free((void*)array_temp);



  free((void*)CCL);
  free((void*)SIZE);

  
  #ifdef FORCE_CLOCK
  proctime_print("shadow matching", TIME);
  #endif

  *SHD = shd_;
  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the 1st core module of FORCE L2PS, and is a modified 
+++ implementation of the Function of Mask code, better known as Fmask. 
+++ Cloud and cloud shadow detection is done in this module.
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ Zhu, Z. & Woodcock, C.E. (2012). Object-based cloud and cloud shadow d
+++ etection in Landsat imagery. Remote Sensing of Environment, 118, 83-94
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
--- pl2:     L2 parameters
--- mission: mission ID
--- atc:     atmospheric correction factors
--- TOA:     TOA reflectance
--- DEM:     DEM
--- EXP:     Exposition
--- QAI:     Quality Assurance Information (modified)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int detect_clouds(par_ll_t *pl2, int mission, atc_t *atc, brick_t *TOA, brick_t *DEM, brick_t *EXP, brick_t *QAI){
int npix, nclear, nland, ncloud, nc, p;
float lowtemp = -1.0, hightemp = -1.0;
float cc;
small *pcp_   = NULL;
small *clr_   = NULL;
small *lnd_   = NULL;
small *brt_   = NULL;
short *var_   = NULL;
short *spr_   = NULL;
small *cld_   = NULL;
small *shd_   = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  cite_me(_CITE_CLOUD_);


  nc = get_brick_ncells(QAI);

  printf("\nCloud Detection :::\n");

  /** Potential Cloud Pixels **/
  if (potential_cloud(pl2, &npix, &nclear, &nland, 
        TOA, QAI, EXP, &pcp_, &clr_, &lnd_, &brt_, &var_) == FAILURE){
    printf("error in PCP module.\n"); return FAILURE;}


  // more than 0.1% clear pixels? -> compute probabilities
  if ((100.0*nclear/(float)npix) > 0.1){

    /** Cloud Probability **/

    if (mission == LANDSAT){
      if (cloud_probability(pl2->nthread, npix, nclear, nland, &ncloud, pl2->cldprob, &cc, &lowtemp, &hightemp,
          TOA, QAI, pcp_, clr_, lnd_, brt_, var_, &cld_) == FAILURE){
        printf("error in cloud probability module.\n"); return FAILURE;}
    } else if (mission == SENTINEL2){
      if (cloud_parallax(nclear, nland, npix, &ncloud, &cc, TOA, QAI, pcp_, clr_, lnd_, brt_, var_, &cld_) == FAILURE){
        printf("error in cloud parallax module.\n"); return FAILURE;}
    }
    free((void*)pcp_); free((void*)clr_); free((void*)brt_); free((void*)var_);

    // less than 80% of max. allowable cloud cover? -> shadow matching
    if (cc <= pl2->maxcc*0.8){

      // if there is no cloud, there is no shadow
      if (ncloud > 0){
        
        shadow_probability(pl2->nthread, nland, atc, TOA, QAI, lnd_, cld_, &spr_);
        free((void*)lnd_);
        
        shadow_matching(pl2->shdprob, lowtemp, hightemp, atc, TOA, QAI, EXP, cld_, spr_, &shd_);
        free((void*)spr_);

      } else {
        
        free((void*)lnd_); free((void*)spr_);
        alloc((void**)&shd_, nc, sizeof(small));
        
      }

      // create the cloud/shadow mask and calculate distance
      atc->cc = finalize_cloud(pl2, npix, atc, TOA, QAI, DEM, cld_, shd_);

    // more than 80% of max. allowable cloud cover? -> everything is cloud or shadow
    } else {

      alloc((void**)&shd_, nc, sizeof(small));
      free((void*)lnd_);

      for (p=0; p<nc; p++) shd_[p] = true;

      atc->cc = finalize_cloud(pl2, npix, atc, TOA, QAI, DEM, cld_, shd_);
      atc->cc = 100;

    }


   // less than 0.1% clear pixels? -> everything is cloud or shadow
  } else {

    alloc((void**)&cld_, nc, sizeof(small));
    alloc((void**)&shd_, nc, sizeof(small));

    for (p=0; p<nc; p++){ cld_[p] = pcp_[p]; shd_[p] = true;}
    free((void*)pcp_); free((void*)clr_); free((void*)brt_); free((void*)var_);

    atc->cc = finalize_cloud(pl2, npix, atc, TOA, QAI, DEM, cld_, shd_);
    atc->cc = 100;

  }
  
  free((void*)cld_); free((void*)shd_);

  #ifdef FORCE_DEBUG
  print_brick_info(QAI); set_brick_open(QAI, OPEN_CREATE); write_brick(QAI);
  #endif

  printf("Cloud cover (%): %6.2f%\n", atc->cc);

  #ifdef CMIX_FAS
  exit(1);
  #endif

  if (atc->cc > pl2->maxcc){ 
    printf("Cloud cover evaluation: abort\n");
    return CANCEL;
  } else {
    printf("Cloud cover evaluation: proceed\n");
  }

  #ifdef FORCE_CLOCK
  proctime_print("cloud module", TIME);
  #endif

  return SUCCESS;
}


/** Compute distance to the next cloud, cloud shadow or snow pixel in m
--- QAI:    Quality Assurance Information
--- nodata: nodata value
--- DIST:   Distance (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int cloud_distance(brick_t *QAI, int nodata, short *DIST){
int p, nx, ny, nc, k=0;
float res, dist;
small  *TO_DIST    = NULL;
ushort *DIST_PIX = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);
  nc  = get_brick_ncells(QAI);
  res = get_brick_res(QAI);


  /** build binary mask **/
  alloc((void**)&TO_DIST, nc, sizeof(small));
  for (p=0; p<nc; p++){
    if (get_cloud(QAI, p) > 0 || get_shadow(QAI, p) || get_snow(QAI, p)){
      TO_DIST[p] = true; 
      k++;
    }
  }

  /** compute pixel distance **/
  if (k > 0){
    DIST_PIX = dist_transform_(TO_DIST, nx, ny);
  } else {
    alloc((void**)&DIST_PIX, nc, sizeof(ushort));
    for (p=0; p<nc; p++) DIST_PIX[p] = ny+nx;
  }
  free((void*)TO_DIST);

  /** compute distance in meters, capped by 32767 **/
  for (p=0; p<nc; p++){
    if (get_off(QAI, p)){
      DIST[p] = nodata;
    } else if ((dist = DIST_PIX[p]*res) < SHRT_MAX){
      DIST[p] = (short)dist;
    } else {
      DIST[p] = SHRT_MAX;
    }
  }
  free((void*)DIST_PIX);

  
  #ifdef FORCE_CLOCK
  proctime_print("cloud distance", TIME);
  #endif

  return SUCCESS;
}

