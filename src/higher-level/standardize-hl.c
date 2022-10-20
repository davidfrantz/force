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
This file contains functions for standardizing time series
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "standardize-hl.h"


int standardize_timeseries(short **ts_, small *mask_, int nc, int nt, short nodata, int method);


/** This function standardizes a specific time series
--- ts_:    image array
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
--- nodata: nodata value
--- method: standardization method
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int standardize_timeseries(short **ts_, small *mask_, int nc, int nt, short nodata, int method){
double avg, var, std = 1.0, k, tmp;
int t, p;


  if (method == _STD_NONE_) return SUCCESS;
  
  
  #pragma omp parallel private(t,avg,var,k,tmp) firstprivate(std) shared(mask_,ts_,nt,nc,nodata,method) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]) continue;

      avg = var = k = 0.0;

      // compute stats
      for (t=0; t<nt; t++){
        
        if (ts_[t][p] == nodata) continue;
        
        k++;
        if (k == 1){
          avg = ts_[t][p];
        } else {
          var_recurrence(ts_[t][p], &avg, &var, k);
        }

      }

      if (k < 2) continue;


      if (method == _STD_NORMAL_) std = standdev(var, k)/10000; // scale



      // center or standardize time series
      for (t=0; t<nt; t++){
        
        if (ts_[t][p] == nodata) continue;

        tmp = (ts_[t][p] - avg) / std;
        if (tmp > SHRT_MAX) tmp = SHRT_MAX;
        if (tmp < SHRT_MIN) tmp = SHRT_MIN;
        ts_[t][p] = (short)tmp;;

      }

    }

  }

  return SUCCESS;
}



/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function standardizes the time series around the mean over time,
+++ and optionally to one standard deviation.
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of ARD products over time
--- ni:     number of interpolation steps
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_standardize(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata, par_hl_t *phl){
int l, nlsp = 26;


  if (phl->tsa.otss)     standardize_timeseries(ts->tss_, mask_, nc, nt,      nodata, phl->tsa.standard);
  if (phl->tsa.tsi.otsi) standardize_timeseries(ts->tsi_, mask_, nc, ni,      nodata, phl->tsa.tsi.standard);
  if (phl->tsa.fld.ofby) standardize_timeseries(ts->fby_, mask_, nc, phl->ny, nodata, phl->tsa.fld.standard);
  if (phl->tsa.fld.ofbq) standardize_timeseries(ts->fbq_, mask_, nc, phl->nq, nodata, phl->tsa.fld.standard);
  if (phl->tsa.fld.ofbm) standardize_timeseries(ts->fbm_, mask_, nc, phl->nm, nodata, phl->tsa.fld.standard);
  if (phl->tsa.fld.ofbw) standardize_timeseries(ts->fbw_, mask_, nc, phl->nw, nodata, phl->tsa.fld.standard);
  if (phl->tsa.fld.ofbd) standardize_timeseries(ts->fbd_, mask_, nc, phl->nd, nodata, phl->tsa.fld.standard);
  
  if (phl->tsa.lsp.ocat){
    for (l=0; l<nlsp; l++){
      standardize_timeseries(ts->lsp_[l], mask_, nc, phl->tsa.lsp.ny, nodata, phl->tsa.lsp.standard);
    }
  }

  return SUCCESS;
}

