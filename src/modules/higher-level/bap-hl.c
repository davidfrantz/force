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
This file contains functions BAP selection
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "bap-hl.h"

/** GNU Scientific Library (GSL) **/
#include <gsl/gsl_multimin.h>          // minimization functions 


double fun_logistic_target(const gsl_vector *v, void *params);
int optimize_logistic_target(float *param, float *a, float *b);
void target_function(par_bap_t *bap, target_t *target);


/** Sigmoid minimizer function
+++ This minimizer function computes the RMSE between computed function 
+++ values and predefined function values (s0, s1, s2).
+++ Return: RMSE between computed and defined function values
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double fun_logistic_target(const gsl_vector *v, void *params){
float a, b, *p = (float *)params;
double y, tmp;
int f;


  a = gsl_vector_get(v, 0);
  b = gsl_vector_get(v, 1);

  for (f=0, tmp=0; f<3; f++){
    y = p[6]/(1+exp(a*p[f]+b));
    tmp += (p[f+3]-y)*(p[f+3]-y);
  }

  return sqrt(tmp/3);
}


/** Sigmoid function optimization
+++ This estimates the sigmoid function parameters using a simplex fit.
+++ Return: function values for a sigmoid function
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int optimize_logistic_target(float *param, float *a, float *b){
const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
gsl_multimin_fminimizer *s = NULL;
gsl_vector *ss, *x;
gsl_multimin_function minex_func;
size_t iter = 0;
int status;
double size;


  // start value
  x = gsl_vector_alloc (2);
  gsl_vector_set (x, 0, 0);
  gsl_vector_set (x, 1, 0);

  // initial step size of simplex
  ss = gsl_vector_alloc (2);
  gsl_vector_set (ss, 0, 0.01);
  gsl_vector_set (ss, 1, 1.0);

  // initialize method and iterate
  minex_func.n = 2;
  minex_func.f = fun_logistic_target;
  minex_func.params = param;

  s = gsl_multimin_fminimizer_alloc(T, 2);
  gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

  do {

    iter++;
    if ((status = gsl_multimin_fminimizer_iterate(s))) break;

    size = gsl_multimin_fminimizer_size(s);
    status = gsl_multimin_test_size(size, 1e-2);

  }
  while (status == GSL_CONTINUE && iter < 100);

  *a = gsl_vector_get(s->x, 0);
  *b = gsl_vector_get(s->x, 1);

  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free (s);

  return SUCCESS;
}


/** This function estimates the function parameters for the temporal sco-
+++ ring functions, i.e. the sigmas (left and right tail) of the Gaussian
+++ or a and b for the sigmoid functions. Parameters are estimated for all
+++ pixels and years. If the static approach is used, one global set is 
+++ estimated. If the phenology-adative approach is used, parameters are 
+++ estimated for each pixel on the basis of the phenology. 
--- bap:    bap parameters
--- target: target information
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void target_function(par_bap_t *bap, target_t *target){
int f, y;
float param[7];


  if (bap->score_type == _SCR_TYPE_SIG_DES_ || bap->score_type == _SCR_TYPE_SIG_ASC_){
    for (f=0; f<3; f++) param[f+3] = bap->Ds[f];
    if (bap->score_type == _SCR_TYPE_SIG_DES_) param[6] = bap->Ds[0];
    if (bap->score_type == _SCR_TYPE_SIG_ASC_) param[6] = bap->Ds[2];
  }

  for (y=0; y<bap->Yn; y++){

    // center the dates
    param[0] = target[y].ce[0] - target[y].ce[1];
    param[1] = 0;
    param[2] = target[y].ce[2] - target[y].ce[1];

    if (bap->score_type == _SCR_TYPE_GAUSS_){
      target[y].a = fabs((param[0])/sqrt(-2*log(bap->Ds[0]/bap->Ds[1])));
      target[y].b = fabs((param[2])/sqrt(-2*log(bap->Ds[2]/bap->Ds[1])));
    } else if (bap->score_type == _SCR_TYPE_SIG_DES_ || bap->score_type == _SCR_TYPE_SIG_ASC_){
      optimize_logistic_target(param, &target[y].a, &target[y].b);
    }

  }

  return;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function compiles the temporal target, at which the composites
+++ are anchored. The static values from the parameter file are used. 
+++ Based on the temporal target, the function parameters for the 
+++ Gaussian or sigmoidal scoring functions are estimated.
--- bap:    bap parameters
+++ Return: target information
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
target_t *compile_target_static(par_bap_t *bap){
int f, y;
int ce;
target_t *target;


  alloc((void**)&target, bap->Yn, sizeof(target_t));

  // Compile temporal target in continuous time
  for (y=0; y<bap->Yn; y++){
    for (f=0; f<3; f++){
      if (f == 0 && bap->Dt[f] > bap->Dt[f+1]){
        ce = doy2ce(bap->Dt[f], bap->Yt-bap->Yr+y-1);
      } else if (f == 2 && bap->Dt[f] < bap->Dt[f-1]){
        ce = doy2ce(bap->Dt[f], bap->Yt-bap->Yr+y+1);
      } else {
        ce = doy2ce(bap->Dt[f], bap->Yt-bap->Yr+y);
      }
      target[y].ce[f] = ce;
    }
  }


  // Estimate function values for temporal scoring functions
  target_function(bap, target);


  return target;
}


/** This function compiles the temporal target, at which the composites
+++ are anchored. The phenology-adaptive approach is used. Based on the 
+++ temporal target, the function parameters for the Gaussian or sigmo-
+++ idal scoring functions are estimated.
--- bap:    bap parameters
--- lsp:    LSP
--- p:      pixel
+++ Return: target information
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
target_t *compile_target_adaptive(par_bap_t *bap, ard_t *lsp, int p, short nodata){
int f, nf = 3, ny;
target_t *target;
double mx, my, vx, vy, cv, n, intercept, slope, yhat, sum, num, rmse;
int y0 = bap->pac.y0, y1 = bap->Yt-bap->Yr;
int yr, y;
int lsp_yr;
float *lsp_;


  ny = get_brick_nbands(lsp[0].DAT);

  alloc((void**)&target, bap->Yn, sizeof(target_t));
  alloc((void**)&lsp_,   ny,      sizeof(float));
  

  for (f=0; f<nf; f++){
    
    // convert to ce
    for (y=0; y<ny; y++){
      if (lsp[f].dat[y][p] == nodata){
        lsp_[y] = nodata; continue;}
      lsp_[y] = lsp[f].dat[y][p] + bap->pac.start - 1;
    }


    // linear regression parameters
    mx = my = vx = vy = cv = n = 0.0;

    for (y=0, yr=y0; y<ny; y++, yr++){

      // remove spurious values
      lsp_yr = ce2year(lsp_[y]);
      if (lsp_yr < yr-1 || lsp_yr > yr+1) lsp_[y]  = nodata;
      
      if (lsp_[y] == nodata) continue;

      if (++n == 1){
        mx = (double)yr;
        my = (double)lsp_[y];
      } else {
        covar_recurrence((double)yr, (double)lsp_[y], 
          &mx, &my, &vx, &vy, &cv, n);
      }

    }


    // RMSE
    num = sum = yhat = 0;

    if (n > 1){
      linreg_coefs(mx, my, cv, vx, &slope, &intercept);
    } else if (n == 1){
      intercept = my-mx*365;
      slope     = 365;
    }
    
    for (y=0, yr=y0; y<ny; y++, yr++){

      if (lsp_[y] == nodata) continue;

      linreg_predict((double)yr, slope, intercept, &yhat);
      sum += (lsp_[y]-yhat)*
             (lsp_[y]-yhat);
      num++;

    }
    rmse = sqrt(sum/num);


    // interpolate nodata and large residuals
    for (y=0, yr=y0; y<ny; y++, yr++){

      if (lsp_[y] == nodata || rmse > bap->pac.rmse){
        linreg_predict((double)yr, slope, intercept, &yhat);
        if (yhat < 0 || yhat > INT_MAX || isnan(yhat)) yhat = 0;
        lsp_[y] = yhat;
      }

    }


    // clip and/or extend the time series to the compositing period,
    // copy to target
    for (y=0, yr=y1; y<bap->Yn; y++, yr++){
      
      if (yr < y0 || yr >= y0+ny){
        linreg_predict((double)yr, slope, intercept, &yhat);
        if (yhat < 0 || yhat > INT_MAX || isnan(yhat)) yhat = 0;
        target[y].ce[f] = yhat;
      } else {
        target[y].ce[f] = lsp_[y+(y1-y0)];
      }

    }

  }

  // Estimate function values for temporal scoring functions
  target_function(bap, target);
  
  free((void*)lsp_);

  return target;
}


/** This function tests if a pixel is mostly water over time.
--- ard:    ARD
--- nt:     number of ARD products over time
--- p:      pixel
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool pixel_is_water(ard_t *ard, int nt, int p){
bool water = false;
float n = 0, nwater = 0;
int t;


  for (t=0; t<nt; t++){
    if (!ard[t].msk[p]) continue;
    if (get_water(ard[t].QAI, p)) nwater++;
    n++;
  }

  if (n > 0 && nwater/n > 0.9) water = true;

  return water;
}


/** This function computes the correlation matrix between all potential 
+++ observations of a given pixel to devaluate data artifacts or transient
+++ phenomena during compositing. The function exits gracefully if the 
+++ correlation score was disabled, i.e. the correlation score weight was
+++ set to 0. The correlation is set to _NODATA_ for incomplete pairs.
--- ard:    ARD
--- nt:     number of ARD products over time
--- nb:     number of bands
--- p:      pixel
--- cor:    correlation matrix of all observations (modified)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int corr_matrix(ard_t *ard, int nt, int nb, int p, float **cor){
int t, u, b, nodata = -9999;
double xm, ym, xv, yv, cv;

  for (t=0; t<nt; t++) memset(cor[t], 0, sizeof(float)*nt);

  for (t=0;     t<nt; t++){
  for (u=(t+1); u<nt; u++){

    if (!ard[t].msk[p] || !ard[u].msk[p]){

      cor[t][u] = cor[u][t] = nodata;

    } else {

      for (b=0, xm=0, ym=0; b<nb; b++){
        xm += ard[t].dat[b][p];
        ym += ard[u].dat[b][p];
      }
      xm /= nb; ym /= nb;

      for (b=0, cv=0, xv=0, yv=0; b<nb; b++){
        cv += (ard[t].dat[b][p]-xm)*(ard[u].dat[b][p]-ym);
        xv += (ard[t].dat[b][p]-xm)*(ard[t].dat[b][p]-xm);
        yv += (ard[u].dat[b][p]-ym)*(ard[u].dat[b][p]-ym);
      }

      if (sqrt(xv*yv) == 0){
        cor[t][u] = cor[u][t] = nodata;
      } else {
        cor[t][u] = cor[u][t] = (float)(cv/sqrt(xv*yv));
      }

    }

  }
  }

  for (t=0;             t<nt; t++){
  for (u=0, xm=0, ym=0; u<nt; u++){

    if (t == u || cor[t][u] == nodata) continue;

    xm += fabs(cor[t][u]);
    ym++;
    cor[t][0] = (float)(xm/ym);

  }
  }

  return SUCCESS;
}


/** This function computes haze statistics, which are used to discard very
+++ hazy pixels.
--- ard:    ARD
--- nt:     number of ARD products over time
--- p:      pixel
--- score:  score parameters
--- bap:    bap parameters
--- mean:   mean of HOT
--- sd:     std. dev. of HOT
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int haze_stats(ard_t *ard, int nt, int p, par_scr_t *score, par_bap_t *bap, float *mean, float *sd){
double n = 0, m = 0, v = 0;
int t;

  for (t=0; t<nt; t++){
    
    if (!ard[t].msk[p] || 
       (!bap->offsea && score[t].d < 0.01)) continue;

    if (dequal(++n, 1)){
      m = score[t].h;
    } else {
      var_recurrence(score[t].h, &m, &v, n);
    }

  }
  
  if (n > 0){
    *mean = (float)m;
    *sd   = (float)standdev(v, n);
  } else {
    *mean = -1;
    *sd   = -1;
  }

  return SUCCESS;
}


/** For water pixels, the compositing algorithm is switched to favor low
+++ SWIR, while making sure NIR is larger than SWIR
--- ard:    ARD
--- nt:     number of ARD products over time
--- p:      pixel
--- score:  score parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int water_score(ard_t *ard, int nt, int p, par_scr_t *score){
int t, sw2, nir;
float scale;


  if ((sw2 = find_domain(ard[0].DAT, "SWIR2"))   < 0){
    printf("no SWIR2 available for min-SWIR2 compositing.\n");
    return FAILURE;
  }
  if ((nir = find_domain(ard[0].DAT, "NIR"))   < 0){
    printf("no NIR available for regularizing min-SWIR2 compositing.\n");
    return FAILURE;
  }
  
  scale = get_brick_scale(ard[0].DAT, 0);
  
  for (t=0; t<nt; t++){
    
    if (!ard[t].msk[p]) continue;

    // low SWIR gets high score
    score[t].t = (scale-ard[t].dat[sw2][p])/scale;
    
    // decrease score if NIR is lower than SWIR2
    if (ard[t].dat[nir][p] < ard[t].dat[sw2][p]) score[t].t /= 2.0;

  }
  
  return SUCCESS;
}


/** This function computes the compositing scores for each observation,
+++ i.e. the total sum (weighted sum of temporal and auxilliary scores),
+++ temporal scores (DOY and Year scores), and auxilliary scores (cloud/
+++ shadow, haze, correlation view zenith scores).
--- ard:    ARD
--- nt:     number of ARD products over time
--- p:      pixel
--- target: target information
--- cor:    correlation matrix
--- score:  score parameters
--- tdist:  temporal distance to target
--- bap:    bap parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parametric_score(ard_t *ard, int nt, int p, target_t *target, float **cor, par_scr_t *score, int *tdist, par_bap_t *bap){
int t, y, y_;
int ce;
int slice, dy;
int diff, ce_dist;
float vz;


  for (t=0; t<nt; t++){

    if (bap->combine == _SCR_COMB_ADD_){
      score[t].t = 0;
    } else if (bap->combine == _SCR_COMB_MUL_){
      score[t].t = 1;
    }

    // skip if no data
    if (!ard[t].msk[p]){
      score[t].t = 0;
      continue;
    } 

    // get date in continuous time
    ce = get_brick_ce(ard[t].DAT, 0);

    // get closest year
    ce_dist = INT_MAX; y = 0;
    for (y_=0; y_<bap->Yn; y_++){
      if ((diff = abs(ce-target[y_].ce[1])) < ce_dist){ ce_dist = diff; y = y_;}
    }
    ce_dist = ce-target[y].ce[1];

    // if target is corrupt, skip pixel
    if (target[y].ce[0] < 1900*365 || 
        target[y].ce[1] < 1900*365 || 
        target[y].ce[2] < 1900*365) continue;

    // difference between acquisition and target year
    dy = abs(get_brick_year(ard[t].DAT, 0) - bap->Yt);


    /** Seasonal suitability: DOY score
    +++ Annual   suitability: Year score
    +** *****************************************************************/

    if (bap->score_type == _SCR_TYPE_GAUSS_){

      // use left tail of Gaussian if acquisition before target
      if (ce < target[y].ce[1]){

        // DOY score
        score[t].d = bap->Ds[1] * exp(ce_dist*ce_dist/
                                (-2*target[y].a*target[y].a));

        // Year score
        slice = (target[y].ce[1]-target[y].ce[0])/((bap->Yr+1)*bap->Yf);
        score[t].y = bap->Ds[1] * exp(dy*slice*dy*slice/
                                (-2*target[y].a*target[y].a));

      // use right tail of Gaussian if acquisition before target
      } else {

        // DOY score
        score[t].d = bap->Ds[1] * exp(ce_dist*ce_dist/
                                (-2*target[y].b*target[y].b));

        // Year score
        slice = (target[y].ce[2]-target[y].ce[1])/((bap->Yr+1)*bap->Yf);
        score[t].y = bap->Ds[1] * exp(dy*slice*dy*slice/
                                (-2*target[y].b*target[y].b));

      }

      // difference between acquisition and target
      tdist[t] = ce_dist;

    } else if (bap->score_type == _SCR_TYPE_SIG_DES_){

      // use 0 weight if acquisition before target
      if (ce < target[y].ce[0]){

        score[t].d = score[t].y = 0;

      // use descending sigmoid if acquisition after target
      } else {

        // DOY score
        score[t].d = bap->Ds[0] / (1 + 
                     exp(target[y].a * ce_dist + target[y].b));

        // Year score
        slice = (target[y].ce[1]-target[y].ce[0])/((bap->Yr+1)*bap->Yf);
        score[t].y = bap->Ds[0] / (1 + exp(target[y].a * 
                    (target[y].ce[0]+dy*slice-target[y].ce[1]) + target[y].b));

      }
      
      // difference between acquisition and target
      tdist[t] = ce - target[y].ce[0];

    } else if (bap->score_type == _SCR_TYPE_SIG_ASC_){

      // use 0 weight if acquisition after target
      if (ce > target[y].ce[2]){

        score[t].d = score[t].y = 0;

      // use ascending sigmoid if acquisition before target
      } else {

      // DOY score
        score[t].d = bap->Ds[2] / (1 + 
                     exp(target[y].a * ce_dist + target[y].b));

        // Year score
        slice = (target[y].ce[2]-target[y].ce[1])/((bap->Yr+1)*bap->Yf);
        score[t].y = bap->Ds[2] / (1 + exp(target[y].a * 
                     (target[y].ce[2]-dy*slice-target[y].ce[1]) + target[y].b));

      }

      // difference between acquisition and target
      tdist[t] = ce - target[y].ce[2];

    }


    /** Auxilliary scores
    +** *****************************************************************/

    // doesn't make sense, but to be sure
    if (dequal(bap->w.d, 0)) score[t].d = 0.0;
    if (dequal(bap->w.y, 0)) score[t].y = 0.0;

    // cloud distance score
    if (bap->w.c > 0){
      score[t].c = 1.0 / (1.0 + exp((-10.0/bap->dreq) * 
                ((float)ard[t].aux[bap->band_dst][p] - (bap->dreq/2.0)) ));
    } else score[t].c = 0.0;

    // haze score
    if (bap->w.h > 0){
      score[t].h = 1.0 / (1.0 + exp((10.0/200.0) * 
                ((float)ard[t].aux[bap->band_hot][p] + (bap->hreq*(-1.0))) ));
    } else score[t].h = 0.0;

    // correlation score
    if (bap->w.r > 0){
      score[t].r = 1.0 / (1.0 + exp((-10.0/(0.5*2.0/3.0)) *
                (cor[t][0] - (0.5*4.0/3.0))));
    } else score[t].r = 0.0;

    // view zenith score
    if (bap->w.v > 0){
      vz = ((float)ard[t].aux[bap->band_vzn][p])*0.01;
      if (vz < 0 || vz > 180){ score[t].v = 0.0;
      } else { score[t].v = 1.0 / (1.0 + exp(10.0/bap->vreq * 
                          (vz - (bap->vreq/2.0))));
      }
    } else score[t].v = 0.0;


    /** Total score
    +** *****************************************************************/

    if (bap->combine == _SCR_COMB_ADD_){
      if (bap->w.d > 0) score[t].t += (bap->w.d * score[t].d);
      if (bap->w.y > 0) score[t].t += (bap->w.y * score[t].y);
      if (bap->w.c > 0) score[t].t += (bap->w.c * score[t].c);
      if (bap->w.h > 0) score[t].t += (bap->w.h * score[t].h);
      if (bap->w.r > 0) score[t].t += (bap->w.r * score[t].r);
      if (bap->w.v > 0) score[t].t += (bap->w.v * score[t].v);
      score[t].t /=  bap->w.t;
    } else if (bap->combine == _SCR_COMB_MUL_){
      if (bap->w.d > 0) score[t].t *= (bap->w.d * score[t].d);
      if (bap->w.y > 0) score[t].t *= (bap->w.y * score[t].y);
      if (bap->w.c > 0) score[t].t *= (bap->w.c * score[t].c);
      if (bap->w.h > 0) score[t].t *= (bap->w.h * score[t].h);
      if (bap->w.r > 0) score[t].t *= (bap->w.r * score[t].r);
      if (bap->w.v > 0) score[t].t *= (bap->w.v * score[t].v);
    }
    
    //printf("d weight %lf, score %lf\n", bap->w.d, score[t].d);
    //printf("y weight %lf, score %lf\n", bap->w.y, score[t].y);
    //printf("c weight %lf, score %lf\n", bap->w.c, score[t].c);
    //printf("h weight %lf, score %lf\n", bap->w.h, score[t].h);
    //printf("r weight %lf, score %lf\n", bap->w.r, score[t].r);
    //printf("v weight %lf, score %lf\n", bap->w.v, score[t].v);
    //printf("t            score %lf\n",           score[t].t);

  }

  return SUCCESS;
}


/** This function does the best available pixel compositing.
--- ard:    ARD
--- l3:     pointer to instantly useable L3 image arrays
--- nt:     number of ARD products over time
--- nb:     number of bands
--- nodata: nodata value
--- p:      pixel
--- score:  score parameters
--- tdist:  temporal distance to target
--- hmean:  mean of HOT
--- hsd:    std. dev. of HOT
--- water:  is pixel water?
--- bap:    bap parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int bap_compositing(ard_t *ard, level3_t *l3, int nt, int nb, short nodata, int p, par_scr_t *score, int *tdist, float hmean, float hsd, bool water, par_bap_t *bap){
int t, max_t = -1, n = 0, b;
double max_score = -1;


  // go through time
  for (t=0; t<nt; t++){

    if (!ard[t].msk[p]) continue;

    if (!bap->offsea && score[t].d < 0.01) continue;
    if (!bap->use_hazy && bap->w.h  > 0 && score[t].h  < 0.01 && 
        hmean > 0.01 && hsd > 0.01) continue;
    if (!bap->use_cloudy && bap->w.c > 0 && score[t].c < 0.01) continue;

    n++;

    // find best score
    if (score[t].t > max_score){
      max_score = score[t].t;
      max_t = t;
    }

  }

  
  if (n > 0){

    // compositing scores
    if (l3->scr != NULL){
      l3->scr[_SCR_TOTAL_][p]    = (short) (score[max_t].t * 10000);
      if (!water){
        l3->scr[_SCR_DOY_][p]    = (short) (score[max_t].d * 10000);
        l3->scr[_SCR_YEAR_][p]   = (short) (score[max_t].y * 10000);
        l3->scr[_SCR_DST_][p]    = (short) (score[max_t].c * 10000);
        l3->scr[_SCR_HAZE_][p]   = (short) (score[max_t].h * 10000);
        l3->scr[_SCR_CORREL_][p] = (short) (score[max_t].r * 10000);
        l3->scr[_SCR_VZEN_][p]   = (short) (score[max_t].v * 10000);
      }
    }

    // compositing information
    if (l3->inf != NULL){
      l3->inf[_INF_QAI_][p]  = ard[max_t].qai[p];
      l3->inf[_INF_NUM_][p]  = n;
      l3->inf[_INF_DOY_][p]  = get_brick_doy(ard[max_t].DAT, 0);
      l3->inf[_INF_YEAR_][p] = get_brick_year(ard[max_t].DAT, 0);
      l3->inf[_INF_DIFF_][p] = tdist[max_t];
      l3->inf[_INF_SEN_][p]  = get_brick_sensorid(ard[max_t].DAT)+1;
    }

    // best available pixel composite
    if (l3->bap != NULL){
      for (b=0; b<nb; b++) l3->bap[b][p] = ard[max_t].dat[b][p];
    }

  } else {

    if (l3->scr != NULL){
      for (b=0; b<_SCR_LENGTH_; b++) l3->scr[b][p] = nodata;
    }
    if (l3->inf != NULL){
      l3->inf[_INF_QAI_][p] = 1;
      l3->inf[_INF_NUM_][p] = 0;
      for (b=2; b<_INF_LENGTH_; b++) l3->inf[b][p] = nodata;
    }
    if (l3->bap != NULL){
      for (b=0; b<nb; b++) l3->bap[b][p] = nodata;
    }

  }


  return SUCCESS;
}


/** This function does the best available pixel compositing.
--- ard:    ARD
--- l3:     pointer to instantly useable L3 image arrays
--- nt:     number of ARD products over time
--- nb:     number of bands
--- nodata: nodata value
--- p:      pixel
--- score:  score parameters
--- tdist:  temporal distance to target
--- hmean:  mean of HOT
--- hsd:    std. dev. of HOT
--- water:  is pixel water?
--- bap:    bap parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int bap_weighting(ard_t *ard, level3_t *l3, int nt, int nb, short nodata, int p, par_scr_t *score, int *tdist, float hmean, float hsd, bool water, par_bap_t *bap){
int t, b, n = 0;
double *sum_reflection = NULL;
double sum_total = 0;
double sum_d = 0, sum_y = 0;
double sum_c = 0, sum_h = 0;
double sum_r = 0, sum_v = 0;


  // allocate 0-initialized
  alloc((void**)&sum_reflection, nb, sizeof(double));

  // go through time
  for (t=0; t<nt; t++){

    if (!ard[t].msk[p]) continue;

    if (!bap->offsea && score[t].d < 0.01) continue;
    if (!bap->use_hazy && bap->w.h  > 0 && score[t].h  < 0.01 && 
        hmean > 0.01 && hsd > 0.01) continue;
    if (!bap->use_cloudy && bap->w.c > 0 && score[t].c < 0.01) continue;


    for (b=0; b<nb; b++) sum_reflection[b] += (ard[t].dat[b][p] * score[t].t);
    sum_total += score[t].t;

    sum_d += score[t].d;
    sum_y += score[t].y;
    sum_c += score[t].c;
    sum_h += score[t].h;
    sum_r += score[t].r;
    sum_v += score[t].v;

    n++;
    
    //printf("score_total: %lf, sum_total: %lf, n: %d\n", score[t].t, sum_total, n);
    //printf("score_d:     %lf, sum_d:     %lf, n: %d\n", score[t].d, sum_d,     n);
    //printf("score_y:     %lf, sum_y:     %lf, n: %d\n", score[t].y, sum_y,     n);
    //printf("score_c:     %lf, sum_c:     %lf, n: %d\n", score[t].c, sum_c,     n);
    //printf("score_h:     %lf, sum_h:     %lf, n: %d\n", score[t].h, sum_h,     n);
    //printf("score_r:     %lf, sum_r:     %lf, n: %d\n", score[t].r, sum_r,     n);
    //printf("score_v:     %lf, sum_v:     %lf, n: %d\n", score[t].v, sum_v,     n);

  }

  if (sum_total > 0){

    // weighted average
    if (l3->bap != NULL){
      for (b=0; b<nb; b++) sum_reflection[b] /= sum_total;
    }

    // information    
    if (l3->inf != NULL){
      if ((sum_c / n) < 0.01) l3->inf[_INF_QAI_][p] |= (short)(1 << _QAI_BIT_CLD_);
      if ((sum_h / n) < 0.01) l3->inf[_INF_QAI_][p] |= (short)(1 << _QAI_BIT_SHD_);
      l3->inf[_INF_NUM_][p] = n;
      for (b=2; b<_INF_LENGTH_; b++) l3->inf[b][p] = nodata;
    }

    // compositing scores
    if (l3->scr != NULL){
      l3->scr[_SCR_TOTAL_][p]    = (short) (sum_total / n * 10000);
      if (!water){
        l3->scr[_SCR_DOY_][p]    = (short) (sum_d / n * 10000);
        l3->scr[_SCR_YEAR_][p]   = (short) (sum_y / n * 10000);
        l3->scr[_SCR_DST_][p]    = (short) (sum_c / n * 10000);
        l3->scr[_SCR_HAZE_][p]   = (short) (sum_h / n * 10000);
        l3->scr[_SCR_CORREL_][p] = (short) (sum_r / n * 10000);
        l3->scr[_SCR_VZEN_][p]   = (short) (sum_v / n * 10000);
      }
    }


  } else {

    if (l3->bap != NULL){
      for (b=0; b<nb; b++) sum_reflection[b] = nodata;
    }

    if (l3->inf != NULL){
      l3->inf[_INF_QAI_][p] = 1;
      l3->inf[_INF_NUM_][p] = 0;
      for (b=2; b<_INF_LENGTH_; b++) l3->inf[b][p] = nodata;
    }

    if (l3->scr != NULL){
      for (b=0; b<_SCR_LENGTH_; b++) l3->scr[b][p] = nodata;
    }

  }

  for (b=0; b<nb; b++) l3->bap[b][p] = (short)sum_reflection[b];

  free((void*)sum_reflection);

  return SUCCESS;
}


/** This function builds overview images for the best available pixel 
+++ composite.
--- l3:     pointer to instantly useable L3 image arrays
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int bap_overview(level3_t *l3, int nx, int ny, int nb, double res, short nodata){
int i, j, p, i_, j_, p_;
int nx_, ny_;
double  res_, step, scale;
enum { B, G, R };
enum { VV, VH };
int is_radar = false;
float ratio;

  
  if (l3->ovv == NULL || l3->bap == NULL) return SUCCESS;


  if (res > 150) res_ = res; else res_ = 150;
  nx_ = nx*res/res_;
  ny_ = ny*res/res_;

  step = res_/res;
  scale = 2500;
  
  // if there are 2 bands, we assume radar
  if (nb == 2) is_radar = true;


  #pragma omp parallel private(j_,p_,i,j,p,ratio) shared(nx,nx_,ny_,step,scale,l3,is_radar,nodata) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i_=0; i_<ny_; i_++){
    for (j_=0; j_<nx_; j_++){
      
      p_ = i_*nx_+j_;

      i = i_*step;
      j = j_*step;
      p = i*nx+j;
      
      if (!is_radar){
        
        if (l3->bap[R][p] < 0 || l3->bap[R][p] == nodata){
          l3->ovv[_RGB_R_][p_] = 0;
        } else if (l3->bap[R][p] > scale){
          l3->ovv[_RGB_R_][p_] = 255;
        } else {
          l3->ovv[_RGB_R_][p_] = (small)(l3->bap[R][p]/scale*255);
        }
        if (l3->bap[G][p] < 0 || l3->bap[G][p] == nodata){
          l3->ovv[_RGB_G_][p_] = 0;
        } else if (l3->bap[G][p] > scale){
          l3->ovv[_RGB_G_][p_] = 255;
        } else {
          l3->ovv[_RGB_G_][p_] = (small)(l3->bap[G][p]/scale*255);
        }
        if (l3->bap[B][p] < 0 || l3->bap[B][p] == nodata){
          l3->ovv[_RGB_B_][p_] = 0;
        } else if (l3->bap[B][p] > scale){
          l3->ovv[_RGB_B_][p_] = 255;
        } else {
          l3->ovv[_RGB_B_][p_] = (small)(l3->bap[B][p]/scale*255);
        }
        
      } else {

        if (l3->bap[VV][p] < -1600 || l3->bap[VV][p] == nodata){
          l3->ovv[_RGB_R_][p_] = 0;
        } else if (l3->bap[VV][p] > -600){
          l3->ovv[_RGB_R_][p_] = 255;
        } else {
          l3->ovv[_RGB_R_][p_] = (small)((l3->bap[VV][p] - -1600)/1000.0*255);
        }
        if (l3->bap[VH][p] < -2200 || l3->bap[VH][p] == nodata){
          l3->ovv[_RGB_G_][p_] = 0;
        } else if (l3->bap[VH][p] > -1200){
          l3->ovv[_RGB_G_][p_] = 255;
        } else {
          l3->ovv[_RGB_G_][p_] = (small)((l3->bap[VH][p] - -2200)/1000.0*255);
        }
        
        ratio = (float)l3->bap[VV][p] / (float)l3->bap[VH][p];
        
        if (ratio < 0.2 || l3->bap[VV][p] == nodata|| l3->bap[VH][p] == nodata){
          l3->ovv[_RGB_B_][p_] = 0;
        } else if (l3->bap[VH][p] > 1.0){
          l3->ovv[_RGB_B_][p_] = 255;
        } else {
          l3->ovv[_RGB_B_][p_] = (small)((ratio - 0.2)/0.8*255);
        }

      }

    }
    }
  }

  
  return SUCCESS;
}

