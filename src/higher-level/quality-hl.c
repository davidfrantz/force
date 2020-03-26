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
This file contains functions for screening quality bit files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "quality-hl.h"


bool use_this_pixel(stack_t *qai, int p, par_qai_t *qai_rule, bool is_ard);


/** Decide whether to use this pixel
+++ This function checks the QAI layer against the user-defined QAI crite-
+++ ria.
--- qai:      Quality Assurance Information
--- p:        pixel
--- qai_rule: ruleset for QAI filtering
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool use_this_pixel(stack_t *qai, int p, par_qai_t *qai_rule, bool is_ard){
  
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
int screen_qai(ard_t *ard, int nt, par_qai_t *qai_rule, int input_level){
int t, p, nc;
int error = 0;
bool is_ard = false;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  if (input_level == _INP_ARD_ || input_level == _INP_QAI_) is_ard = true;


  #pragma omp parallel shared(ard,nt) reduction(+: error) default(none)
  {

    #pragma omp for
    for (t=0; t<nt; t++){
      if ((ard[t].MSK = copy_stack(ard[t].QAI, 1, _DT_SMALL_)) == NULL || 
          (ard[t].msk = get_band_small(ard[t].MSK, 0)) == NULL){
        printf("Error compiling screened QAI stack."); error++;}
    }

  }

  if (error > 0){
    printf("%d screening QAI errors. ", error); 
    return FAILURE;
  }


  nc = get_stack_chunkncells(ard[0].MSK);

  #pragma omp parallel private(t) shared(ard,nt,nc,qai_rule,is_ard) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){
      for (t=0; t<nt; t++) ard[t].msk[p] = use_this_pixel(ard[t].QAI, p, qai_rule, is_ard);
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
int screen_noise(ard_t *ard, int nt, par_qai_t *qai_rule){
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


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  if (qai_rule->above_noise == 0 && qai_rule->below_noise == 0) return CANCEL;
  if (nt < 3) return CANCEL;

  nc = get_stack_chunkncells(ard[0].MSK);
  nodata = get_stack_nodata(ard[0].DAT, 0);

  alloc((void**)&ce, nt, sizeof(int));
  for (t_mid=0; t_mid<nt; t_mid++) ce[t_mid] = get_stack_ce(ard[t_mid].DAT, 0);


  #pragma omp parallel private(t,t_mid,t_left,t_right,t_first,t_last,t_max,n,valid_left,valid_right,y_hat,maxr,t_maxr,ssqr,noise,rel_noise,removed,nout,nadd) shared(ard,nc,nt,ce,nodata,qai_rule,b) default(none)
  {

    alloc((void**)&removed, nt, sizeof(bool));

    #pragma omp for
    for (p=0; p<nc; p++){
      
      
      rel_noise = INT_MAX;
      noise = INT_MAX;
      n = nt;
      
      nout = 0;
      nadd = 0;
      
      memset(removed, 0, nt*sizeof(bool));
      
      while (rel_noise > qai_rule->above_noise && n > 2){

        t_left  = 0;
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
              t_right = t;
              if (!valid_right) t_last = t;
              valid_right = true;
              break;
            }

          }

          if (!valid_left || !valid_right) continue;

          y_hat = fabs(ard[t_mid].dat[b][p] - 
                      (ard[t_right].dat[b][p]-ard[t_left].dat[b][p]) / 
                      (ce[t_right]-ce[t_left]) * (ce[t_mid]-ce[t_left]) - 
                       ard[t_left].dat[b][p]);
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


        y_hat = fabs(ard[t_mid].dat[b][p] - 
                    (ard[t_right].dat[b][p]-ard[t_left].dat[b][p]) / 
                    (ce[t_right]-ce[t_left]) * (ce[t_mid]-ce[t_left]) - 
                     ard[t_left].dat[b][p]);

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

