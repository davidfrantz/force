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
This file contains functions for folding time series
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "fold-hl.h"


enum { _YEAR_, _QUARTER_, _MONTH_, _WEEK_, _DOY_ };

int fold(short **tsi_, date_t *d_tsi, small *mask_, int nc, int ni, short **fld_, date_t *d_fld, int nf, short nodata, int by, par_hl_t *phl);


/** This function folds the time series in a given aggregation period
--- tsi_:   interpolated image array
--- d_tsi_: interpolation dates
--- mask:   mask image
--- nc:     number of cells
--- ni:     number of interpolation steps
--- fld_:   folded image array (returned)
--- d_fld:  dates of folded time series
--- nf:     number of folds
--- nodata: nodata value
--- by:     aggregation period
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int fold(short **tsi_, date_t *d_tsi, small *mask_, int nc, int ni, short **fld_, date_t *d_fld, int nf, short nodata, int by, par_hl_t *phl){
int f, t, n, p;
short minimum, maximum;
double mean, var;
double skew, kurt;
double skewscaled, kurtscaled;
float *q_array = NULL;
bool alloc_q_array = false;


  if (fld_ == NULL) return CANCEL;

  if ((phl->tsa.fld.type >= _STA_Q01_ && phl->tsa.fld.type <= _STA_Q99_) ||
       phl->tsa.fld.type == _STA_IQR_) alloc_q_array = true;
  

  #pragma omp parallel private(f,t,minimum,maximum,q_array,mean,var,skew,kurt,n,skewscaled,kurtscaled) shared(mask_,tsi_,fld_,d_fld,d_tsi,nc,ni,nf,by,nodata,phl,alloc_q_array) default(none)
  {
    
    // initialize _STAts
    if (alloc_q_array) alloc((void**)&q_array, ni, sizeof(float));

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (f=0; f<nf; f++) fld_[f][p] = nodata;
        continue;
      }


      for (f=0; f<nf; f++){

        mean = var = skew = kurt = n = 0;
        minimum = SHRT_MAX; maximum = SHRT_MIN;

        // compute _STAts
        for (t=0; t<ni; t++){

          switch (by){
            case _YEAR_:
              if (d_tsi[t].year != d_fld[f].year) continue;
              break;
            case _QUARTER_:
              if (d_tsi[t].quarter != d_fld[f].quarter) continue;
    //printf("%d %d - Q FLD: %d, Q TSI: %d\n", f, t, d_fld[f].quarter, d_tsi[t].quarter);
    //print_date(&d_fld[f]);
    //print_date(&d_tsi[t]);
              break;
            case _MONTH_:
              if (d_tsi[t].month != d_fld[f].month) continue;
              break;
            case _WEEK_:
              if (d_tsi[t].week  != d_fld[f].week) continue;
    //printf("%d %d - W FLD: %d, W TSI: %d\n", f, t, d_fld[f].week, d_tsi[t].week);
    //print_date(&d_fld[f]);
    //print_date(&d_tsi[t]);
              break;
            case _DOY_:
              if (d_tsi[t].doy != d_fld[f].doy) continue;
              break;
          }

          if (tsi_[t][p] == nodata) continue;

          
          // range metrics
          if (tsi_[t][p] < minimum) minimum = tsi_[t][p];
          if (tsi_[t][p] > maximum) maximum = tsi_[t][p];

          // quantile metrics
          if (alloc_q_array) q_array[n] = tsi_[t][p];

          n++;

          // moments metrics
          kurt_recurrence(tsi_[t][p], &mean, &var, 
                            &skew, &kurt, n);

        }

        
        // fold by mean (0), min (1), max (2)
        if (n > 0){

          switch (phl->tsa.fld.type){
            case _STA_NUM_:
              fld_[f][p] = n;
              break;
            case _STA_AVG_:
              fld_[f][p] = (short)mean;
              break;
            case _STA_MIN_:
              fld_[f][p] = minimum;
              break;
            case _STA_MAX_:
              fld_[f][p] = maximum;
              break;
            case _STA_RNG_:
              fld_[f][p] = maximum-minimum;
              break;
            case _STA_STD_:
              fld_[f][p] = (short)standdev(var, n);
              break;
            case _STA_SKW_:
              skewscaled = skewness(var, skew, n)*1000;
              if (skewscaled < -30000) skewscaled = -30000;
              if (skewscaled >  30000) skewscaled =  30000;
              fld_[f][p] = (short)skewscaled;
              break;
            case _STA_KRT_:
              kurtscaled = (kurtosis(var, kurt, n)-3)*1000;
              if (kurtscaled < -30000) kurtscaled = -30000;
              if (kurtscaled >  30000) kurtscaled =  30000;
              fld_[f][p] = (short)kurtscaled;
              break;
            case _STA_IQR_:
              fld_[f][p] = (short)(quantile(q_array, n, 0.75)-quantile(q_array, n, 0.25));
              break;
          }

          if (phl->tsa.fld.type >= _STA_Q01_ && phl->tsa.fld.type <= _STA_Q99_){
            fld_[f][p] = (short)quantile(q_array, n, (phl->tsa.fld.type-_STA_Q01_+1)/100.0);
          }

        } else {

          fld_[f][p] = nodata;

        }

      }
      
    }
   
    if (alloc_q_array) free((void*)q_array);
   
  }
  

  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function folds the time series in different aggregation periods
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- ni:     number of interpolation steps
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_fold(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_hl_t *phl){


  fold(ts->tsi_, ts->d_tsi, mask_, nc, ni, ts->fby_, ts->d_fby, phl->ny, nodata, _YEAR_,    phl);
  fold(ts->tsi_, ts->d_tsi, mask_, nc, ni, ts->fbq_, ts->d_fbq, phl->nq, nodata, _QUARTER_, phl);
  fold(ts->tsi_, ts->d_tsi, mask_, nc, ni, ts->fbm_, ts->d_fbm, phl->nm, nodata, _MONTH_,   phl);
  fold(ts->tsi_, ts->d_tsi, mask_, nc, ni, ts->fbw_, ts->d_fbw, phl->nw, nodata, _WEEK_,    phl);
  fold(ts->tsi_, ts->d_tsi, mask_, nc, ni, ts->fbd_, ts->d_fbd, phl->nd, nodata, _DOY_,     phl);

  return SUCCESS;
}

