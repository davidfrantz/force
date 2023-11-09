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
This file contains functions for spectral temporal metrics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "stm-hl.h"


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function computes Spectral Temporal Metrics
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- ni:     number of interpolation steps
--- nodata: nodata value
--- stm:    STM parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_stm(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_stm_t *stm){
int t, b, n, p, q;
short minimum, maximum;
short q25_, q75_;
double mean, var;
double skew, kurt;
double skewscaled, kurtscaled;
double *q_array = NULL; // need to be double for GSL quantile function
bool alloc_q_array = false;


  if (ts->stm_ == NULL) return CANCEL;
  
  cite_me(_CITE_STM_);
  
  if (stm->sta.quantiles || stm->sta.iqr > -1) alloc_q_array = true;
  

  #pragma omp parallel private(t,b,minimum,maximum,q,q_array,mean,var,skew,kurt,n,skewscaled,kurtscaled,q25_,q75_) shared(mask_,ts,nc,ni,nodata,stm,alloc_q_array) default(none)
  {

    // initialize stats
    if (alloc_q_array) alloc((void**)&q_array, ni, sizeof(double));

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (b=0; b<stm->sta.nmetrics; b++) ts->stm_[b][p] = nodata;
        continue;
      }

      mean = var = skew = kurt = n = 0;
      minimum = SHRT_MAX; maximum = SHRT_MIN;
      q25_ = q75_ = SHRT_MIN;

      for (t=0; t<ni; t++){

        if (ts->tsi_[t][p] == nodata) continue;

        // range metrics
        if (ts->tsi_[t][p] < minimum) minimum = ts->tsi_[t][p];
        if (ts->tsi_[t][p] > maximum) maximum = ts->tsi_[t][p];

        // quantile metrics
        if (alloc_q_array) q_array[n] = ts->tsi_[t][p];

        n++;

        // moments metrics
        kurt_recurrence(ts->tsi_[t][p], &mean, &var, 
                          &skew, &kurt, n);

      }

  
      if (n > 0){
        skewscaled = skewness(var, skew, n)*1000;
        kurtscaled = (kurtosis(var, kurt, n)-3)*1000;
        if (skewscaled < -30000) skewscaled = -30000;
        if (skewscaled >  30000) skewscaled =  30000;
        if (kurtscaled < -30000) kurtscaled = -30000;
        if (kurtscaled >  30000) kurtscaled =  30000;
        if (stm->sta.num > -1) ts->stm_[stm->sta.num][p] = n;
        if (stm->sta.min > -1) ts->stm_[stm->sta.min][p] = minimum;
        if (stm->sta.max > -1) ts->stm_[stm->sta.max][p] = maximum;
        if (stm->sta.rng > -1) ts->stm_[stm->sta.rng][p] = maximum-minimum;
        if (stm->sta.avg > -1) ts->stm_[stm->sta.avg][p] = (short)mean;
        if (stm->sta.std > -1) ts->stm_[stm->sta.std][p] = (short)standdev(var, n);
        if (stm->sta.skw > -1) ts->stm_[stm->sta.skw][p] = (short)skewscaled;
        if (stm->sta.krt > -1) ts->stm_[stm->sta.krt][p] = (short)kurtscaled;

        if (stm->sta.quantiles){
          for (q=0; q<stm->sta.nquantiles; q++){
            ts->stm_[stm->sta.qxx[q]][p] = (short)quantile(q_array, n, stm->sta.q[q]);
            if (stm->sta.q[q] == 0.25) q25_ = ts->stm_[stm->sta.qxx[q]][p];
            if (stm->sta.q[q] == 0.75) q75_ = ts->stm_[stm->sta.qxx[q]][p];
          }
        }

        if (stm->sta.iqr > -1){
          if (q25_ == SHRT_MIN) q25_ = (short)quantile(q_array, n, 0.25);
          if (q75_ == SHRT_MIN) q75_ = (short)quantile(q_array, n, 0.75);
          ts->stm_[stm->sta.iqr][p] = q75_-q25_;
        }
      } else {
        for (b=0; b<stm->sta.nmetrics; b++) ts->stm_[b][p] = nodata;
      }
      
    }

    if (alloc_q_array) free((void*)q_array);
    
  }

  return SUCCESS;
}

