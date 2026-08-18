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
This file contains functions for screening quality bit files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "quality-hl.h"


bool use_this_pixel(brick_t *qai, int p, par_qai_t *qai_rule, bool is_ard);


/** Decide whether to use this pixel
+++ This function checks the QAI layer against the user-defined QAI crite-
+++ ria.
--- qai:      Quality Assurance Information
--- p:        pixel
--- qai_rule: ruleset for QAI filtering
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool use_this_pixel(brick_t *qai, int p, par_qai_t *qai_rule, bool is_ard){
  
  if (!is_ard            && get_off(qai, p))               return false;

  if (qai_rule->off      && get_off(qai, p))               return false;
  if (qai_rule->cld_unc  && get_cloud(qai, p) == 1)        return false;
  if (qai_rule->cld_opq  && get_cloud(qai, p) == 2)        return false;
  if (qai_rule->cld_cir  && get_cloud(qai, p) == 3)        return false;
  if (qai_rule->shd      && get_shadow(qai, p))            return false;
  if (qai_rule->snw      && get_snow(qai, p))              return false;
  if (qai_rule->wtr      && get_water(qai, p))             return false;
  if (qai_rule->aod_int  && get_aerosol(qai, p) == 1)      return false;
  if (qai_rule->aod_high && get_aerosol(qai, p) == 2)      return false;
  if (qai_rule->aod_fill && get_aerosol(qai, p) == 3)      return false;
  if (qai_rule->sub      && get_subzero(qai, p))           return false;
  if (qai_rule->sat      && get_saturation(qai, p))        return false;
  if (qai_rule->sun      && get_lowsun(qai, p))            return false;
  if (qai_rule->ill_low  && get_illumination(qai, p) == 1) return false;
  if (qai_rule->ill_poor && get_illumination(qai, p) == 2) return false;
  if (qai_rule->ill_shd  && get_illumination(qai, p) == 3) return false;
  if (qai_rule->slp      && get_slope(qai, p))             return false;
  if (qai_rule->wvp      && get_vaporfill(qai, p))         return false;

  return true;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function generates a processing mask (true/false) for each ARD
+++ dataset based on the user-defined QAI criteria.
+++ ria.
--- ard:      ARD
--- nt:       number of ARD products over time
--- qai_rule: ruleset for QAI filtering
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int screen_qai(ard_t *ard, int nt, brick_t *mask, par_qai_t *qai_rule, int input_level){
int t, p, nc;
int error = 0;
bool is_ard = false;
small *mask_ = NULL;



  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return FAILURE;}
  }

  
  if (input_level == _INP_ARD_ || input_level == _INP_QAI_) is_ard = true;


  #pragma omp parallel shared(ard,nt) reduction(+: error) default(none)
  {

    #pragma omp for
    for (t=0; t<nt; t++){
      if ((ard[t].MSK = copy_brick(ard[t].QAI, 1, _DT_SMALL_)) == NULL || 
          (ard[t].msk = get_band_small(ard[t].MSK, 0)) == NULL){
        printf("Error compiling screened QAI brick."); error++;}
    }

  }

  if (error > 0){
    printf("%d screening QAI errors. ", error); 
    return FAILURE;
  }


  nc = get_brick_chunkncells(ard[0].MSK);

  #pragma omp parallel private(t) shared(ard,mask_,nt,nc,qai_rule,is_ard) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (t=0; t<nt; t++) ard[t].msk[p] = false;
      } else {
        for (t=0; t<nt; t++) ard[t].msk[p] = use_this_pixel(ard[t].QAI, p, qai_rule, is_ard);
      }

    }

  }
  

  #ifdef FORCE_CLOCK
  proctime_print("screen QAI", TIME);
  #endif

  return SUCCESS;
}


/** This function re-evaluated the quality masks of the ARD, and removes
+++ outliers (that are larger than the time series noise), and restores
+++ inliers (that are well within the time series noise).
--- ard:      ARD
--- nt:       number of ARD products over time
--- qai_rule: ruleset for QAI filtering
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int screen_noise(ard_t *ard, int nt, brick_t *mask, par_qai_t *qai_rule){
int p, nc;
int b = 0; // use shortest wavelength
int t, t_left, t_mid, t_right, t_first, t_last, t_max, n;
bool valid_left, valid_right;
int *ce = NULL;
short nodata;
int nout, nadd;
float y_hat;
int t_maxr;
float maxr;
float ssqr;
float noise;
float rel_noise;
bool *removed = NULL;
small *mask_ = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  if (qai_rule->above_noise == 0 && qai_rule->below_noise == 0) return CANCEL;
  if (nt < 3) return CANCEL;

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return FAILURE;}
  }

  nc = get_brick_chunkncells(ard[0].MSK);
  nodata = get_brick_nodata(ard[0].DAT, 0);

  alloc((void**)&ce, nt, sizeof(int));
  for (t_mid=0; t_mid<nt; t_mid++) ce[t_mid] = get_brick_ce(ard[t_mid].DAT, 0);


  #pragma omp parallel private(t,t_mid,t_left,t_right,t_first,t_last,t_max,n,valid_left,valid_right,y_hat,maxr,t_maxr,ssqr,noise,rel_noise,removed,nout,nadd) shared(ard,mask_,nc,nt,ce,nodata,qai_rule,b) default(none)
  {

    alloc((void**)&removed, nt, sizeof(bool));

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]) continue;
      
      rel_noise = INT_MAX;
      noise = INT_MAX;
      n = nt;
      
      nout = 0;
      nadd = 0;
      
      memset(removed, 0, nt*sizeof(bool));
      
      while (rel_noise > qai_rule->above_noise && n > 2){

        t_left  = 0;
        t_right = 0;
        t_first = 0;
        t_last  = 0;
        n = 0;
        maxr = 0;
        ssqr = 0;
        t_maxr = 0;

        for (t_mid=1; t_mid<(nt-1); t_mid++){

          if (!ard[t_mid].msk[p]) continue;

          valid_left  = false;
          valid_right = false;

          // find previous and next point
          for (t=t_left; t<nt; t++){

            if (!ard[t].msk[p]) continue;

            if (t < t_mid){
              t_left = t;
              if (!valid_left) t_first = t;
              valid_left = true;
            } else if (t == t_mid){
              continue;
            } else if (t > t_mid){
              if (!valid_right){
                t_right = t;
                t_last  = t;
                valid_right = true;
              } else {
                t_last  = t;
                break;
              }
            }

          }

          if (!valid_left || !valid_right) continue;


          if (ce[t_right] == ce[t_left]){
            y_hat = fabs(ard[t_mid].dat[b][p] - 
                        (ard[t_right].dat[b][p]-ard[t_left].dat[b][p]) / 2.0);
          } else {
            y_hat = fabs(ard[t_mid].dat[b][p] - 
                        (ard[t_right].dat[b][p]-ard[t_left].dat[b][p]) / 
                        (ce[t_right]-ce[t_left]) * (ce[t_mid]-ce[t_left]) - 
                        ard[t_left].dat[b][p]);
          }
          t_max = t_mid;
          
          if (t_left == t_first &&
             fabs(ard[t_left].dat[b][p] - ard[t_right].dat[b][p]) >
             fabs(ard[t_mid].dat[b][p]  - ard[t_right].dat[b][p])){
              t_max = t_left;

          } else if (t_right == t_last &&
              fabs(ard[t_left].dat[b][p] - ard[t_right].dat[b][p]) >
              fabs(ard[t_mid].dat[b][p]  - ard[t_left].dat[b][p])){
              t_max = t_right;
          }

          if (y_hat > maxr){
            maxr = y_hat;
            t_maxr = t_max;
          }
          
          ssqr += y_hat*y_hat;
          n++;

        }
        

        if (n < 2) continue;

        noise = sqrt(ssqr/n);
        rel_noise = maxr/noise;
        
        //printf("max. residual is %f at time %d, relative to noise %f\n", maxr, t_maxr, rel_noise);

        if (rel_noise > qai_rule->above_noise){ 
          ard[t_maxr].msk[p] = false; 
          removed[t_maxr] = true; 
          nout++;
        }

      }
      
      
      if (noise == INT_MAX) continue;

      t_left = 0;

      for (t_mid=1; t_mid<(nt-1); t_mid++){

        if (ard[t_mid].msk[p] || ard[t_mid].dat[b][p] == nodata || removed[t_mid]) continue;

        valid_left  = false;
        valid_right = false;

        // find previous and next point
        for (t=t_left; t<nt; t++){

          if (!ard[t].msk[p]) continue;

          if (t < t_mid){
            t_left = t;
            valid_left = true;
          } else if (t == t_mid){
            continue;
          } else if (t > t_mid){
            t_right = t;
            valid_right = true;
            break;
          }

        }

        if (!valid_left || !valid_right) continue;

        if (ce[t_right] == ce[t_left]){
          y_hat = fabs(ard[t_mid].dat[b][p] - 
                      (ard[t_right].dat[b][p]-ard[t_left].dat[b][p]) / 2.0);
        } else {
          y_hat = fabs(ard[t_mid].dat[b][p] - 
                      (ard[t_right].dat[b][p]-ard[t_left].dat[b][p]) / 
                      (ce[t_right]-ce[t_left]) * (ce[t_mid]-ce[t_left]) - 
                      ard[t_left].dat[b][p]);
        }

        rel_noise = y_hat/noise;

        if (rel_noise < qai_rule->below_noise){ ard[t_mid].msk[p] = true; nadd++;}

      }

      //printf("removed/added %d/%d observations\n", nout, nadd);
      
    }


    free((void*)removed);
  
  }


  free((void*)ce);

  #ifdef FORCE_CLOCK
  proctime_print("screen QAI", TIME);
  #endif


  return SUCCESS;
}


/** This function re-codes the quality masks of the ARD, and removes
+++ dates that are outside of the secondary adaptive date range data.
--- ard:                 ARD
--- adaptive:            secondary input: adaptive date range data
--- nt:                  number of ARD products over time
--- n_adaptive:          number of adaptive date range products
--- adaptive_date_range: parameters
--- is_qai:               true if the input only contains QAI
+++ Return:              SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int screen_adaptive_date_range(ard_t *ard, ard_t *adaptive, int nt, int n_adaptive, brick_t *mask, par_adr_t *adaptive_date_range, int input_level1){


  // just to be sure
  if (!adaptive_date_range->use) return SUCCESS;

  if (adaptive == NULL || n_adaptive == 0){
    printf("ADAPTIVE_RANGE requested, but data are not available.\n");
    return FAILURE;
  }

  short adaptive_nodata = get_brick_nodata(adaptive[0].DAT, 0);

  if (n_adaptive != 2){
    printf("more than 2 ADAPTIVE_RANGE files were given. Cannot work with this.\n");
    return FAILURE;
  } else if (n_adaptive < 2){
    printf("less than 2 ADAPTIVE_RANGE files were given. Cannot work with this.\n");
    return FAILURE;
  }

  int n_window = get_brick_nbands(adaptive[0].DAT);
  if (n_window != get_brick_nbands(adaptive[1].DAT)){
    printf("ADAPTIVE_RANGE files have different number of bands. Cannot work with this.\n");
    return FAILURE;
  }

  int nc;
  if (input_level1 == _INP_QAI_){
    nc = get_brick_chunkncells(ard[0].QAI);
  } else {
    nc = get_brick_chunkncells(ard[0].DAT);
  }
  if (nc != get_brick_chunkncells(adaptive[0].DAT)){
    printf("ADAPTIVE_RANGE files have different dimensions than ARD files. Cannot work with this.\n");
    return FAILURE;
  }


  // import mask (if available)
  small *mask_ = NULL;
  
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return FAILURE;}
  }


  #pragma omp parallel shared(ard,nt,adaptive,n_adaptive,mask_,nc,n_window,adaptive_nodata,adaptive_date_range,input_level1) default(none)
  {

    #pragma omp for
    for (int p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]) continue;

      int t = 0;
      int ce_image;

      if (input_level1 == _INP_QAI_){
        ce_image = get_brick_ce(ard[t].QAI, 0);
      } else {
        ce_image = get_brick_ce(ard[t].DAT, 0);
      }

      for (int w=0; w<n_window; w++){

        // skip if the window is invalid
        if (adaptive[0].dat[w][p] == adaptive_nodata || 
            adaptive[1].dat[w][p] == adaptive_nodata){
          continue;
        }

        // convert adaptive date range to ce
        int ce_start = adaptive[0].dat[w][p] + adaptive_date_range->start - 1;
        int ce_end   = adaptive[1].dat[w][p] + adaptive_date_range->start - 1;
        

        // disable all images before the start of the window
        while (t < nt && ce_image < ce_start){
          ard[t].msk[p] = false;
          t++;
          if (t < nt){
            if (input_level1 == _INP_QAI_){
              ce_image = get_brick_ce(ard[t].QAI, 0);
            } else {
              ce_image = get_brick_ce(ard[t].DAT, 0);
            }
          }
        }

        // fast forward to the end of the window
        while (t < nt && ce_image <= ce_end){
          t++;
          if (t < nt){
            if (input_level1 == _INP_QAI_){
              ce_image = get_brick_ce(ard[t].QAI, 0);
            } else {
              ce_image = get_brick_ce(ard[t].DAT, 0);
            }
          }
        }

      }

      // disable all images after the end of the last window
      while (t < nt){
        ard[t].msk[p] = false;
        t++;
      }

    }

  }

  return SUCCESS;
}
