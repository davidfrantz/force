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

/** The RBF interpolation was adpoted from the python routine 
+++ 'RBF_fitting_large_scale.py', Copyright (C) 2018 Andreas Rabe
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for time series interpolation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "interpolate-hl.h"

/** GNU Scientific Library (GSL) **/
#include <gsl/gsl_multifit.h> // Linear Least Squares Fitting


int interpolate_none(tsa_t *ts, int nc, int nt);
int interpolate_linear(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata);
int interpolate_moving(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata, par_tsi_t *tsi);
int interpolate_rbf(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata, par_tsi_t *tsi);
rbf_t *rbf_kernel(par_tsi_t *tsi);
void free_rbf(rbf_t *rbf);


/** This function "interpolates" the time series with the NONE option:
+++ it essentially copies the time series brick to the interpolation
+++ brick
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int interpolate_none(tsa_t *ts, int nc, int nt){
int t, p;

  #pragma omp parallel private(t) shared(ts,nc,nt) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      for (t=0; t<nt; t++) ts->tsi_[t][p] = ts->tss_[t][p];
      
    }

  }


  return SUCCESS;
}


/** This function interpolates the time series linearly.
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
--- ni:     number of interpolation steps
--- nodata: nodata value
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int interpolate_linear(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata){
int t, t_left, i, p;
float x_left, x_right, x;
float y_left, y_right, y;


  #pragma omp parallel private(t,t_left,i,x_left,x_right,x,y_left,y_right,y) shared(mask_,ts,nc,nt,ni,nodata) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (i=0; i<ni; i++) ts->tsi_[i][p] = nodata;
        continue;
      }


      // interpolate for each equidistant timestep
      for (i=0, t_left=0; i<ni; i++){

        // current time
        x = ts->d_tsi[i].ce;

        x_left = x_right = INT_MIN;
        y_left = y_right = nodata;

        // find previous and next point
        for (t=t_left; t<nt; t++){

          if (ts->tss_[t][p] == nodata) continue;

          if (ts->d_tss[t].ce < x){
            x_left = ts->d_tss[t].ce;
            y_left = ts->tss_[t][p];
            t_left = t;
          } else if (ts->d_tss[t].ce == x){
            x_left = x_right = x;
            y_left = y_right = ts->tss_[t][p];
            t_left = t;
            break;
          } else if (ts->d_tss[t].ce > x){
            x_right = ts->d_tss[t].ce;
            y_right = ts->tss_[t][p];
            break;
          }

        }

        // set nodata, copy value or interpolate
        if (x_left < 0 && x_right < 0){
          y = nodata;
        } else if (x_left < 0){
          y = y_right;
        } else if (x_right < 0){
          y = y_left;
        } else if (x_left == x_right){
          y = (y_left+y_right)/2.0;
        } else {
          y = (y_left*(x_right-x) + y_right*(x-x_left))/(x_right-x_left);
        }

        ts->tsi_[i][p] = (short)y;

      }
      
    }
    
  }


  return SUCCESS;
}


/** This function interpolates the time series using a moving average.
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
--- ni:     number of interpolation steps
--- nodata: nodata value
--- tsi:    interpolation parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int interpolate_moving(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata, par_tsi_t *tsi){
int t, t_left, i, p;
float x, x_;
double sum, num;


  #pragma omp parallel private(t,t_left,i,x, x_,sum,num) shared(mask_,ts,nc,nt,ni,tsi,nodata) default(none)
  {

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (i=0; i<ni; i++) ts->tsi_[i][p] = nodata;
        continue;
      }


      // interpolate for each equidistant timestep
      for (i=0, t_left=0; i<ni; i++){

        // current time
        x = ts->d_tsi[i].ce;

        sum = num = 0.0;

        // use all points within temporal window
        for (t=t_left; t<nt; t++){

          if (ts->tss_[t][p] == nodata) continue;

          x_ = ts->d_tss[t].ce;

          if (x-x_ > tsi->mov_max){ // earlier than window
            t_left = t;
            continue;
          } else if (x_-x > tsi->mov_max){ // later than window
            break;
          } else { // in window
            sum += ts->tss_[t][p]; 
            num++;
          }

        }

        // interpolate with moving mean, or use nodata
        if (num > 0){
           ts->tsi_[i][p] = (short)(sum/num);
        } else {
           ts->tsi_[i][p] = nodata;
        }

      }
      
    }
    
  }


  return SUCCESS;
}


/** RBF interpolation
+++ This function interpolates and smoothes a time series using ensembles
+++ of radial basis functions. The ensembles are weighted according to da-
+++ ta availability.
+++ The RBF interpolation was adpoted from the python routine 
+++ 'RBF_fitting_large_scale.py', Copyright (C) 2018 Andreas Rabe
--- MASK:   Analysis mask (optional)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function interpolates the time series using ensembles of radial 
+++ basis functions. The ensembles are weighted according to data 
+++ availability. The RBF interpolation was adpoted from the python 
+++ routine 'RBF_fitting_large_scale.py', Copyright (C) 2018 Andreas Rabe
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
--- ni:     number of interpolation steps
--- nodata: nodata value
--- tsi:    interpolation parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int interpolate_rbf(tsa_t *ts, small *mask_, int nc, int nt, int ni, short nodata, par_tsi_t *tsi){
int t, *t_left = NULL, i, k, p;
int x, x_;
float y;
double *sum_yw = NULL, *sum_w = NULL;
double sum_kd, sum_d;
rbf_t *rbf = NULL;


  rbf = rbf_kernel(tsi);

  #pragma omp parallel private(k,i,t,t_left,x, x_,sum_yw,sum_w,sum_kd,sum_d,y) shared(mask_,ts,nc,nt,ni,rbf,tsi,nodata) default(none)
  {

    alloc((void**)&t_left, rbf->nk, sizeof(int));
    alloc((void**)&sum_yw, rbf->nk, sizeof(double));
    alloc((void**)&sum_w,  rbf->nk, sizeof(double));


    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (i=0; i<ni; i++) ts->tsi_[i][p] = nodata;
        continue;
      }

      for (k=0; k<rbf->nk; k++) t_left[k] = 0;


      // interpolate for each equidistant timestep
      for (i=0; i<ni; i++){

        // current time
        x = ts->d_tsi[i].ce;

        for (k=0; k<rbf->nk; k++){

          sum_yw[k] = sum_w[k] = 0.0;

          // use all points within temporal window
          for (t=t_left[k]; t<nt; t++){

            x_ = ts->d_tss[t].ce;

            if (x-x_ > rbf->max_ce[k]){ // earlier than window
              t_left[k] = t;
              continue;
            } else if (x_-x > rbf->max_ce[k]){ // later than window
              break;
            } else { // in window
              if (ts->tss_[t][p] == nodata) continue;
              sum_yw[k] += ts->tss_[t][p]*rbf->kernel[k][x_-x+rbf->hbin];
              sum_w[k]  += rbf->kernel[k][x_-x+rbf->hbin];
            }

          }

        }

        // compute weighted average + 
        // weight the kernels with their data availability (weighted with 
        //        the same kernel as the time series)
        for (k=0, sum_kd=0, sum_d=0; k<rbf->nk; k++){
          if (sum_w[k] > 0){
            // weighted average * weighted data availability (weighted fractions of days)
            // sum_kd += sum_yw[k]/sum_w[k] * sum_w[k]/max_w[k];
            // this reduces to:
            sum_kd += sum_yw[k]/rbf->max_w[k];
            sum_d  += sum_w[k]/rbf->max_w[k];
          }
        }

        // ensemble fit
        if (sum_d > 0){
          y = sum_kd/sum_d;
        } else {
          y = nodata;
        }

        ts->tsi_[i][p] = (short)y;

      }
      
    }

    // clean
    free((void*)t_left);
    free((void*)sum_yw);
    free((void*)sum_w);
    
  }

  
  free_rbf(rbf);
  
  
  return SUCCESS;
}


/** This function compiles the RBF kernels.
--- tsi:    interpolation parameters
+++ Return: RBF kernels (must be free with free_rbf)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
rbf_t *rbf_kernel(par_tsi_t *tsi){
rbf_t *rbf = NULL;
int k, bin, dce;
float sigma, sigma2;
float sum;

  alloc((void**)&rbf, 1, sizeof(rbf_t));

  // number of kernels
  rbf->nk = tsi->rbf_nk;

  // number of maximum bins (days) in kernel
  if ((rbf->nbin = 8*tsi->rbf_sigma[rbf->nk-1]) % 2 == 0) rbf->nbin++;

  // allocate kernels etc
  alloc_2D((void***)&rbf->kernel, rbf->nk, rbf->nbin, sizeof(float));
  alloc((void**)&rbf->max_ce,  rbf->nk, sizeof(int));
  alloc((void**)&rbf->max_w,  rbf->nk, sizeof(double));

  // max temporal distance (days) to be considered (half kernel size)
  rbf->hbin = (rbf->nbin-1)/2;

  // compile kernels
  for (k=0; k<rbf->nk; k++){

    sum = 0.0;

    sigma  = tsi->rbf_sigma[k];
    sigma2 = sigma*sigma;

    // compute kernel weights, cutoff (alpha=0.1) + sum of weights
    for (bin=0; bin<rbf->nbin; bin++){
      
      dce = bin-rbf->hbin;

      rbf->kernel[k][bin] = 1/(sqrt(2*M_PI) * sigma) * 
          exp(-0.5*(dce)*(dce)/(sigma2));
          
      if (dce >= 0){

        sum += rbf->kernel[k][bin];

        if (sum <= tsi->rbf_cutoff/2.0){
          rbf->max_ce[k] = dce;
          rbf->max_w[k] += rbf->kernel[k][bin];
        }

      }

    }

    rbf->max_w[k] *= 2;

    #ifdef FORCE_DEBUG
    printf("sigma: %.0f, kernel width: %d, max. distance: %d, max sign. distance: %d, max weight: %.2f\n", sigma, rbf->nbin, rbf->hbin, rbf->max_ce[k], rbf->max_w[k]);
    #endif

  }

  return rbf;
}


/** This function frees the RBF kernels.
--- rbf:    RBF kernels
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_rbf(rbf_t *rbf){

  if (rbf == NULL) return;

  if (rbf->kernel != NULL){ free_2D((void**)rbf->kernel, rbf->nk); rbf->kernel = NULL;}
  if (rbf->max_ce != NULL){ free((void*)rbf->max_ce); rbf->max_ce = NULL;}
  if (rbf->max_w  != NULL){ free((void*)rbf->max_w);  rbf->max_w  = NULL;}

  free((void*)rbf); rbf = NULL;
  
  return;  
}


int irls_fit(const gsl_matrix *X, const gsl_vector *y,
             gsl_vector *c, gsl_matrix *cov){
int s;


  gsl_multifit_robust_workspace *work = 
    gsl_multifit_robust_alloc(gsl_multifit_robust_bisquare, X->size1, X->size2);

  //gsl_multifit_robust_maxiter(1000, work);

  s = gsl_multifit_robust(X, y, c, cov, work);

  //if (s == GSL_EMAXITER) printf("max iter reached.\n");

  gsl_multifit_robust_free(work);

  return s;
}


/** This function interpolates the time series using harmonic models
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
--- ni:     number of interpolation steps
--- nodata: nodata value
--- tsi:    interpolation parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int interpolate_harmonic(tsa_t *ts, small *mask_, int nc, int nt, int nr, int ni, short nodata, par_tsi_t *tsi){
int t, i, k, nk, p;
int n_coef, i_coef;
double y_pred, y_err;
gsl_matrix *x = NULL, *cov = NULL;
gsl_vector *y = NULL, *c = NULL, *x_pred = NULL;
enum { _TERM_OFF_, _TERM_TREND_, _TERM_UNI_, _TERM_BI_, _TERM_TRI_, _TERM_LEN_ };
bool model_terms[_TERM_LEN_];


  n_coef = 0;
  memset(model_terms, 0, _TERM_LEN_);


  // add offset
  model_terms[_TERM_OFF_] = true;
  n_coef++;

  // add trend
  if (tsi->harm_trend){
    model_terms[_TERM_TREND_] = true;
    n_coef++;
  }

  // add uni-modal frequency
  if (tsi->harm_nmodes >= 1){
    model_terms[_TERM_UNI_] = true;
    n_coef += 2;
  }

  // add bi-modal frequency
  if (tsi->harm_nmodes >= 2){
    model_terms[_TERM_BI_] = true;
    n_coef += 2;
  }
  
  // add tri-modal frequency
  if (tsi->harm_nmodes >= 3){
    model_terms[_TERM_TRI_] = true;
    n_coef += 2;
  }

  // at least have offset and uni-modal frequency
  if (n_coef < 3){
      printf("not enough coefficients for harmonic fitting. "
             "Something is wrong. Abort.\n");
      return FAILURE;
  }

  gsl_set_error_handler_off();

  #pragma omp parallel private(i,t,k,nk,i_coef,x,cov,y,c,x_pred,y_pred,y_err) shared(mask_,ts,nc,nt,nr,ni,tsi,nodata,n_coef,model_terms) default(none)
  {

    c = gsl_vector_alloc (n_coef);
    cov = gsl_matrix_alloc (n_coef, n_coef);
    x_pred = gsl_vector_alloc(n_coef);

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (i=0; i<ni; i++) ts->tsi_[i][p] = nodata;
        continue;
      }

      for (t=0, nk=0; t<nt; t++){

        if (ts->tss_[t][p] == nodata || 
            ts->d_tss[t].ce < tsi->harm_fit_range[_MIN_].ce ||
            ts->d_tss[t].ce > tsi->harm_fit_range[_MAX_].ce){
          continue;
        } else {
          nk++;
        }

      }

      //if (nk == 0){
      if (nk < n_coef){
        for (i=0; i<ni; i++) ts->tsi_[i][p] = nodata;
        continue;
      }

//printf("nk: %d\n", nk);
      x = gsl_matrix_alloc(nk, n_coef);
      y = gsl_vector_alloc(nk);


      for (t=0, k=0; t<nt; t++){

        if (ts->tss_[t][p] == nodata || 
            ts->d_tss[t].ce < tsi->harm_fit_range[_MIN_].ce ||
            ts->d_tss[t].ce > tsi->harm_fit_range[_MAX_].ce){

          continue;

        } else {

          i_coef = 0;

          if (model_terms[_TERM_OFF_]) gsl_matrix_set(x, k, i_coef++, 1.0);

          if (model_terms[_TERM_TREND_]) gsl_matrix_set(x, k, i_coef++, ts->d_tss[t].ce);

          if (model_terms[_TERM_UNI_]){
            gsl_matrix_set(x, k, i_coef++, cos(2 * M_PI / 365 * ts->d_tss[t].ce));
            gsl_matrix_set(x, k, i_coef++, sin(2 * M_PI / 365 * ts->d_tss[t].ce));
          }

          if (model_terms[_TERM_BI_]){
            gsl_matrix_set(x, k, i_coef++, cos(4 * M_PI / 365 * ts->d_tss[t].ce));
            gsl_matrix_set(x, k, i_coef++, sin(4 * M_PI / 365 * ts->d_tss[t].ce));
          }

          if (model_terms[_TERM_TRI_]){
            gsl_matrix_set(x, k, i_coef++, cos(6 * M_PI / 365 * ts->d_tss[t].ce));
            gsl_matrix_set(x, k, i_coef++, sin(6 * M_PI / 365 * ts->d_tss[t].ce));
          }

          gsl_vector_set(y, k, ts->tss_[t][p]);

          k++;

        }


      }

      // Iteratively Reweighted Least Squares (IRLS)
      irls_fit(x, y, c, cov);

      // interpolate for each equidistant timestep
      for (i=0; i<ni; i++){

        i_coef = 0;
        
        if (model_terms[_TERM_OFF_]) gsl_vector_set(x_pred, i_coef++, 1.0);

        if (model_terms[_TERM_TREND_]) gsl_vector_set(x_pred, i_coef++, ts->d_tsi[i].ce);

        if (model_terms[_TERM_UNI_]){
          gsl_vector_set(x_pred, i_coef++, cos(2 * M_PI / 365 * ts->d_tsi[i].ce));
          gsl_vector_set(x_pred, i_coef++, sin(2 * M_PI / 365 * ts->d_tsi[i].ce));
        }

        if (model_terms[_TERM_BI_]){
          gsl_vector_set(x_pred, i_coef++, cos(4 * M_PI / 365 * ts->d_tsi[i].ce));
          gsl_vector_set(x_pred, i_coef++, sin(4 * M_PI / 365 * ts->d_tsi[i].ce));
        }

        if (model_terms[_TERM_TRI_]){
          gsl_vector_set(x_pred, i_coef++, cos(6 * M_PI / 365 * ts->d_tsi[i].ce));
          gsl_vector_set(x_pred, i_coef++, sin(6 * M_PI / 365 * ts->d_tsi[i].ce));
        }

        gsl_multifit_robust_est(x_pred, c, cov, &y_pred, &y_err);
        ts->tsi_[i][p] = (short)y_pred;

      }


      if (tsi->onrt){

        for (t=0, k=0; t<nt; t++){

          if (ts->d_tss[t].ce <= tsi->harm_fit_range[_MAX_].ce) continue;

          if (ts->tss_[t][p] == nodata){

            ts->nrt_[k][p] = (short)nodata;

          } else {

            i_coef = 0;

            if (model_terms[_TERM_OFF_]) gsl_vector_set(x_pred, i_coef++, 1.0);

            if (model_terms[_TERM_TREND_]) gsl_vector_set(x_pred, i_coef++, ts->d_tss[t].ce);

            if (model_terms[_TERM_UNI_]){
              gsl_vector_set(x_pred, i_coef++, cos(2 * M_PI / 365 * ts->d_tss[t].ce));
              gsl_vector_set(x_pred, i_coef++, sin(2 * M_PI / 365 * ts->d_tss[t].ce));
            }

            if (model_terms[_TERM_BI_]){
              gsl_vector_set(x_pred, i_coef++, cos(4 * M_PI / 365 * ts->d_tss[t].ce));
              gsl_vector_set(x_pred, i_coef++, sin(4 * M_PI / 365 * ts->d_tss[t].ce));
            }

            if (model_terms[_TERM_TRI_]){
              gsl_vector_set(x_pred, i_coef++, cos(6 * M_PI / 365 * ts->d_tss[t].ce));
              gsl_vector_set(x_pred, i_coef++, sin(6 * M_PI / 365 * ts->d_tss[t].ce));
            }

            gsl_multifit_robust_est(x_pred, c, cov, &y_pred, &y_err);
            ts->nrt_[k][p] = (short)(ts->tss_[t][p] - y_pred);

          }

          k++;

        }

      }

      

      gsl_matrix_free (x);
      gsl_vector_free (y);

    }

    gsl_vector_free (c);
    gsl_matrix_free (cov);
    gsl_vector_free (x_pred);

  }

  gsl_set_error_handler(NULL);
  
  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function interpolates the time series
--- ts:     pointer to instantly useable TSA image arrays
--- mask:   mask image
--- nc:     number of cells
--- nt:     number of time steps
--- ni:     number of interpolation steps
--- nodata: nodata value
--- tsi:    interpolation parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_interpolation(tsa_t *ts, small *mask_, int nc, int nt, int nr, int ni, short nodata, par_tsi_t *tsi){


  if (ts->tsi_ == NULL) return CANCEL;


  switch (tsi->method){
    case _INT_NONE_:
      interpolate_none(ts, nc, nt);
      break;
    case _INT_LINEAR_:
      interpolate_linear(ts, mask_, nc, nt, ni, nodata);
      break;
    case _INT_MOVING_:
      interpolate_moving(ts, mask_, nc, nt, ni, nodata, tsi);
      break;
    case _INT_RBF_:
      cite_me(_CITE_RBF_);
      interpolate_rbf(ts, mask_, nc, nt, ni, nodata, tsi);
      break;
    case _INT_HARMONIC_:
      cite_me(_CITE_HARMONIC_);
      interpolate_harmonic(ts, mask_, nc, nt, nr, ni, nodata, tsi);
      break;
    default:
      printf("unknown INTERPOLATE\n");
      break;    
  }
  
  return SUCCESS;
}

