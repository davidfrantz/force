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
This file contains functions for time series trends
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "trend-hl.h"


int trend(short **fld_, date_t *d_fld, small *mask_, int nc, int nf, short **trd_, short nodata, int by, bool in_ce, par_trd_t *trd);
int cat(short **fld_, date_t *d_fld, small *mask_, int nc, int nf, short **cat_, short nodata, int by, bool in_ce, par_trd_t *trd);


/** This function computes a trend analysis for any time series. Currently
+++ implemented trend parameters are mean, intercept, slope, R-squared,
+++ significance of slope, RMSE, MAE, max. absolute residual and # of obs.
--- fld_:   folded image array
--- d_fld:  dates of folded time series
--- mask:   mask image
--- nc:     number of cells
--- nf:     number of folds
--- trd_:   trend image array (returned)
--- nodata: nodata value
--- by:     aggregation period of fold
--- in_ce:  are the values given in continuous days?
--- trd:    trend parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int trend(short **fld_, date_t *d_fld, small *mask_, int nc, int nf, short **trd_, short nodata, int by, bool in_ce, par_trd_t *trd){
int p, f, x, b, sig;
double mx, my, vx, vy, cv, k;
double off, slp, rsq, yhat;
double e, ssqe, sae;
double sxsq, maxe, seb;
double mae, rmse;


  if (trd_ == NULL) return CANCEL;


  #pragma omp parallel private(b,f,x,mx,my,vx,vy,cv,k,ssqe,sae,sxsq,maxe,seb,mae,rmse,off,slp,rsq,yhat,e,sig) shared(mask_,fld_,d_fld,trd_,nc,nf,by,nodata,in_ce,trd) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (b=0; b<_TRD_LENGTH_; b++) trd_[b][p] = nodata;
        continue;
      }


      x = mx = my = vx = vy = cv = k  = 0;
      ssqe = sae = sxsq = 0;
      maxe = SHRT_MIN;

      // compute stats
      for (f=0; f<nf; f++){

        if (fld_[f][p] == nodata) continue;

        switch (by){
          case _FLD_YEAR_:
            x = d_fld[f].year - d_fld[0].year;
            break;
          case _FLD_QUARTER_:
            x = d_fld[f].quarter - d_fld[0].quarter;
            break;
          case _FLD_MONTH_:
            x = d_fld[f].month - d_fld[0].month;
            break;
          case _FLD_WEEK_:
            x = d_fld[f].week - d_fld[0].week;
            break;
          case _FLD_DOY_:
            x = d_fld[f].doy - d_fld[0].doy;
            break;
        }

        k++;
        if (k == 1){
          mx = x;
          my = fld_[f][p];
        } else {
          covar_recurrence(x, fld_[f][p], &mx, &my, &vx, &vy, &cv, k);
        }

      }

      if (k < 3){
        for (b=0; b<_TRD_LENGTH_; b++) trd_[b][p] = nodata;
        continue;
      }


      // compute trend coefficients and R-squared
      cv = covariance(cv, k);
      vx = variance(vx, k);
      vy = variance(vy, k);
      linreg_coefs(mx, my, cv, vx, &slp, &off);
      linreg_rsquared(cv, vx, vy, &rsq);
      rsq *= 10000;

      // compute residual metrics
      for (f=0; f<nf; f++){

        if (fld_[f][p] == nodata) continue;

        switch (by){
          case _FLD_YEAR_:
            x = d_fld[f].year - d_fld[0].year;
            break;
          case _FLD_QUARTER_:
            x = d_fld[f].quarter - d_fld[0].quarter;
            break;
          case _FLD_MONTH_:
            x = d_fld[f].month - d_fld[0].month;
            break;
          case _FLD_WEEK_:
            x = d_fld[f].week - d_fld[0].week;
            break;
          case _FLD_DOY_:
            x = d_fld[f].doy - d_fld[0].doy;
            break;
        }

        linreg_predict(x, slp, off, &yhat);
        e = fld_[f][p] - yhat;
        ssqe += e*e;
        sae += fabs(e);
        sxsq += (x-mx)*(x-mx);
        if (fabs(e) > maxe) maxe = fabs(e);

      }



      // account for values given in continuous days
      if (in_ce){
        my  -= 365*(nf-1)/2;
        slp -= 365;
      }

      // standard error of slope, and significance of slope
      seb = sqrt(1.0/(k-2)*ssqe)/sqrt(sxsq);
      sig = slope_significant(1-trd->conf, trd->tail, k, slp, 0.0, seb);

      // account for values given in continuous days
      if (in_ce) slp *= 1000;

      rmse = sqrt(ssqe/k);
      mae = sae/k;


      trd_[_TRD_MEAN_][p]   = (short)my;
      trd_[_TRD_OFFSET_][p] = (short)off;
      trd_[_TRD_SLOPE_][p]  = (short)slp;
      trd_[_TRD_GAIN_][p]   = (short)(slp*nf/off*1000);
      trd_[_TRD_RSQ_][p]    = (short)rsq;
      trd_[_TRD_SIG_][p]    = (short)sig;
      trd_[_TRD_RMSE_][p]   = (short)rmse;
      trd_[_TRD_MAE_][p]    = (short)mae;
      trd_[_TRD_MAXE_][p]   = (short)maxe;
      trd_[_TRD_NUM_][p]    = (short)k;

    }

  }

  return SUCCESS;
}


/** This function computes a trend analysis for any time series. Currently
+++ implemented trend parameters are mean, intercept, slope, R-squared,
+++ significance of slope, RMSE, MAE, max. absolute residual and # of obs.
+++ Trend parameters are computed for three parts of the time series: be-
+++ fore/after the change, and for the full time series. Change parameters
+++ are magnitude of greatest change, and time of greates change.
--- fld_:   folded image array
--- d_fld:  dates of folded time series
--- mask:   mask image
--- nc:     number of cells
--- nf:     number of folds
--- cat_:   change/trend image array (returned)
--- nodata: nodata value
--- by:     aggregation period of fold
--- in_ce:  are the values given in continuous days?
--- trd:    trend parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int cat(short **fld_, date_t *d_fld, small *mask_, int nc, int nf, short **cat_, short nodata, int by, bool in_ce, par_trd_t *trd){
int p, f, f_pre, f_post, f_change, f_min[3], f_max[3], f_len[3];
float change, change_now, change_real;
int x, b, part, sig;
double mx, my, vx, vy, cv, k;
double off, slp, rsq, yhat;
double e, ssqe, sae;
double sxsq, maxe, seb;
double mae, rmse;



  if (cat_ == NULL) return CANCEL;


  #pragma omp parallel private(b,part,f,f_pre,f_post,f_change,f_min,f_max,f_len,change,change_now,change_real,x,mx,my,vx,vy,cv,k,ssqe,sae,sxsq,maxe,seb,mae,rmse,off,slp,rsq,yhat,e,sig) shared(mask_,fld_,d_fld,cat_,nc,nf,by,nodata,in_ce,trd) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      // fill with nodata
      for (b=0; b<_CAT_LENGTH_; b++) cat_[b][p] = nodata;

      if (mask_ != NULL && !mask_[p]) continue;


      f_change = 0;
      change = SHRT_MIN;

      // find largest change
      for (f=1; f<nf; f++){

        if (fld_[f][p] == nodata) continue;

        f_pre = f-1;
        while (f_pre >= 0 && fld_[f_pre][p] == nodata) f_pre--;
        if (f_pre < 0 || fld_[f_pre][p] == nodata) continue;

        change_now  = change_real = (float)(fld_[f_pre][p] - fld_[f][p]);

        if (change_now > 0 && trd->penalty){

          f_post = f+1;
          while (f_post < nf && fld_[f_post][p] == nodata) f_post++;
          if (f_post >= nf || fld_[f_post][p] == nodata) continue;
  
          change_now *= (float)(fld_[f_pre][p] - fld_[f_post][p]) / 1e4;

        }

        if (change_now > change){
          change = change_real;
          f_change = f;
        }

      }

      if (change == SHRT_MIN) continue;


      switch (by){
        case _FLD_YEAR_:
          x = d_fld[f_change].year;
          break;
        case _FLD_QUARTER_:
          x = d_fld[f_change].quarter;
          break;
        case _FLD_MONTH_:
          x = d_fld[f_change].month;
          break;
        case _FLD_WEEK_:
          x = d_fld[f_change].week;
          break;
        case _FLD_DOY_:
          x = d_fld[f_change].doy;
          break;
        default:
          x = 0;
          break;
      }

      cat_[_CAT_CHANGE_][p] = (short)change;
      cat_[_CAT_YEAR_][p] = (short)x;


      // split the time series into three parts
      f_min[_PART_TOTAL_]  = 0;        f_max[_PART_TOTAL_]  = nf;         // complete
      f_min[_PART_BEFORE_] = 0;        f_max[_PART_BEFORE_] = f_change-1; // before
      f_min[_PART_AFTER_]  = f_change; f_max[_PART_AFTER_]  = nf;         // after

      f_len[_PART_TOTAL_]  = f_max[_PART_TOTAL_]  - f_min[_PART_TOTAL_];  // complete
      f_len[_PART_BEFORE_] = f_max[_PART_BEFORE_] - f_min[_PART_BEFORE_]; // before
      f_len[_PART_AFTER_]  = f_max[_PART_AFTER_]  - f_min[_PART_AFTER_];  // after


      for (part=0; part<_PART_LENGTH_; part++){

        mx = my = vx = vy = cv = k = 0.0;

        // compute stats
        for (f=f_min[part]; f<f_max[part]; f++){

          if (fld_[f][p] == nodata) continue;

          switch (by){
            case _FLD_YEAR_:
              x = d_fld[f].year - d_fld[0].year;
              break;
            case _FLD_QUARTER_:
              x = d_fld[f].quarter - d_fld[0].quarter;
              break;
            case _FLD_MONTH_:
              x = d_fld[f].month - d_fld[0].month;
              break;
            case _FLD_WEEK_:
              x = d_fld[f].week - d_fld[0].week;
              break;
            case _FLD_DOY_:
              x = d_fld[f].doy - d_fld[0].doy;
              break;
          }

          k++;
          if (k == 1){
            mx = x;
            my = fld_[f][p];
          } else {
            covar_recurrence(x, fld_[f][p], &mx, &my, &vx, &vy, &cv, k);
          }

        }

        if (k < 1) continue;

        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_MEAN_][p] = (short)my;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_NUM_][p]  = (short)k;

        if (k < 3) continue;

        // compute trend coefficients and R-squared
        cv = covariance(cv, k);
        vx = variance(vx, k);
        vy = variance(vy, k);
        linreg_coefs(mx, my, cv, vx, &slp, &off);
        linreg_rsquared(cv, vx, vy, &rsq);
        rsq *= 10000;


        // compute residual metrics
        ssqe = sae = sxsq = 0; maxe = SHRT_MIN;

        for (f=f_min[part]; f<f_max[part]; f++){

          if (fld_[f][p] == nodata) continue;

          switch (by){
            case _FLD_YEAR_:
              x = d_fld[f].year - d_fld[0].year;
              break;
            case _FLD_QUARTER_:
              x = d_fld[f].quarter - d_fld[0].quarter;
              break;
            case _FLD_MONTH_:
              x = d_fld[f].month - d_fld[0].month;
              break;
            case _FLD_WEEK_:
              x = d_fld[f].week - d_fld[0].week;
              break;
            case _FLD_DOY_:
              x = d_fld[f].doy - d_fld[0].doy;
              break;
          }

          linreg_predict(x, slp, off, &yhat);
          e = fld_[f][p] - yhat;
          ssqe += e*e;
          sae += fabs(e);
          sxsq += (x-mx)*(x-mx);
          if (fabs(e) > maxe) maxe = fabs(e);

        }


        // account for values given in continuous days
        if (in_ce){
          my  -= 365*(nf-1)/2;
          slp -= 365; slp *= 1000;
        }

        // standard error of slope, and significance of slope
        seb = sqrt(1.0/(k-2)*ssqe)/sqrt(sxsq);
        sig = slope_significant(1-trd->conf, trd->tail, k, slp, 0.0, seb);

        rmse = sqrt(ssqe/k);
        mae = sae/k;

        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_OFFSET_][p] = (short)off;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_SLOPE_][p]  = (short)slp;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_GAIN_][p]   = (short)(slp*f_len[part]/off*1000);
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_RSQ_][p]    = (short)rsq;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_SIG_][p]    = (short)sig;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_RMSE_][p]   = (short)rmse;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_MAE_][p]    = (short)mae;
        cat_[_CAT_YEAR_+1+_TRD_LENGTH_*part+_TRD_MAXE_][p]   = (short)maxe;

      }

      // loss, i.e. change relative to total offset
      if (cat_[_CAT_YEAR_+1+_TRD_LENGTH_*_PART_TOTAL_+_TRD_OFFSET_][p] > 0){
        cat_[_CAT_LOSS_][p] = (short)((float)cat_[_CAT_CHANGE_][p]/
                                      cat_[_CAT_YEAR_+1+_TRD_LENGTH_*_PART_TOTAL_+_TRD_OFFSET_][p]*
                                      1000);
      } else {
        cat_[_CAT_LOSS_][p] = 0;
      }


    }

  }


  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function computes time series trend
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_trend(tsa_t *ts, small *mask_, int nc, short nodata, par_hl_t *phl){
int l;
bool in_ce = false;


  trend(ts->fby_, ts->d_fby, mask_, nc, phl->ny, ts->try_, nodata, _FLD_YEAR_,    in_ce, &phl->tsa.trd);
  trend(ts->fbq_, ts->d_fbq, mask_, nc, phl->nq, ts->trq_, nodata, _FLD_QUARTER_, in_ce, &phl->tsa.trd);
  trend(ts->fbm_, ts->d_fbm, mask_, nc, phl->nm, ts->trm_, nodata, _FLD_MONTH_,   in_ce, &phl->tsa.trd);
  trend(ts->fbw_, ts->d_fbw, mask_, nc, phl->nw, ts->trw_, nodata, _FLD_WEEK_,    in_ce, &phl->tsa.trd);
  trend(ts->fbd_, ts->d_fbd, mask_, nc, phl->nd, ts->trd_, nodata, _FLD_DOY_,     in_ce, &phl->tsa.trd);


  if (phl->tsa.lsp.otrd){
    for (l=0; l<_LSP_LENGTH_; l++){
      if (l < 7) in_ce = true; else in_ce = false;
      trend(ts->lsp_[l], ts->d_lsp, mask_, nc, phl->tsa.lsp.ny, ts->trp_[l], nodata, _FLD_YEAR_, in_ce, &phl->tsa.trd);
    }
  }
  
  if (phl->tsa.pol.otrd){
    for (l=0; l<_POL_LENGTH_; l++){
      if (l < 9) in_ce = true; else in_ce = false;
      trend(ts->pol_[l], ts->d_pol, mask_, nc, phl->tsa.pol.ny, ts->tro_[l], nodata, _FLD_YEAR_, in_ce, &phl->tsa.trd);
    }
  }

  return SUCCESS;
}


/** This function computes time series trend and change
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_cat(tsa_t *ts, small *mask_, int nc, short nodata, par_hl_t *phl){
int l;
bool in_ce = false;


  cite_me(_CITE_CAT_);


  cat(ts->fby_, ts->d_fby, mask_, nc, phl->ny, ts->cay_, nodata, _FLD_YEAR_,    in_ce, &phl->tsa.trd);
  cat(ts->fbq_, ts->d_fbq, mask_, nc, phl->nq, ts->caq_, nodata, _FLD_QUARTER_, in_ce, &phl->tsa.trd);
  cat(ts->fbm_, ts->d_fbm, mask_, nc, phl->nm, ts->cam_, nodata, _FLD_MONTH_,   in_ce, &phl->tsa.trd);
  cat(ts->fbw_, ts->d_fbw, mask_, nc, phl->nw, ts->caw_, nodata, _FLD_WEEK_,    in_ce, &phl->tsa.trd);
  cat(ts->fbd_, ts->d_fbd, mask_, nc, phl->nd, ts->cad_, nodata, _FLD_DOY_,     in_ce, &phl->tsa.trd);


  if (phl->tsa.lsp.ocat){
    for (l=0; l<_LSP_LENGTH_; l++){
      if (l < 7) in_ce = true; else in_ce = false;
      cat(ts->lsp_[l], ts->d_lsp, mask_, nc, phl->tsa.lsp.ny, ts->cap_[l], nodata, _FLD_YEAR_, in_ce, &phl->tsa.trd);
    }
  }
  
  if (phl->tsa.pol.ocat){
    for (l=0; l<_POL_LENGTH_; l++){
      if (l < 9) in_ce = true; else in_ce = false;
      cat(ts->pol_[l], ts->d_pol, mask_, nc, phl->tsa.pol.ny, ts->cao_[l], nodata, _FLD_YEAR_, in_ce, &phl->tsa.trd);
    }
  }

  return SUCCESS;
}

