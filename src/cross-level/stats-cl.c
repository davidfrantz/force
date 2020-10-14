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

/** The t-score calculations were obtained from SurfStat Australia's 
+++ t-distribution Calculator, online available at:
+++ https://surfstat.anu.edu.au/surfstat-home/tables/t.php
+++ t-distribution Calculator (C) 2017 Keith Dear
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for statistics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "stats-cl.h"


void quantile_swap(float *x, int from, int to);
int comp(const void *a, const void *b);


/** One-pass variance and covariance estimation
+++ This function implements a one-pass estimation of variance and covari-
+++ ance based on recurrence formulas. It can be used to estimate mean of 
+++ x and y, variance or standard deviation of x and y, covariance between
+++ x and y. This function can also be used to compute one-pass linear re-
+++ gressions. Use this function in a loop.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- x:      current x-value
--- y:      current y-value
--- mx:     last estimate of mean of x (is updated)
--- my:     last estimate of mean of y (is updated)
--- vx:     last estimate of variance of x (is updated)
--- vy:     last estimate of variance of y (is updated)
--- cv:     last estimate of covariance between x and y (is updated)
--- n:      number of observations
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void covar_recurrence(double   x, double   y, 
                      double *mx, double *my, 
                      double *vx, double *vy, 
                      double *cv, double n){
double oldmx = *mx, oldmy = *my;
double oldvx = *vx, oldvy = *vy;
double oldcv = *cv;

  *mx = oldmx + (x-oldmx)/n;
  *my = oldmy + (y-oldmy)/n;
  *vx = oldvx + (x-oldmx)*(x-*mx);
  *vy = oldvy + (y-oldmy)*(y-*my);
  *cv = oldcv + (n-1)/n*(x-oldmx)*(y-oldmy);

  return;
}


/** One-pass covariance estimation
+++ This function implements a one-pass estimation of covariance based on
+++ recurrence formulas. It can be used to estimate mean of x and y, and 
+++ covariance between x and y. Use this function in a loop.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- x:      current x-value
--- y:      current y-value
--- mx:     last estimate of mean of x (is updated)
--- my:     last estimate of mean of y (is updated)
--- cv:     last estimate of covariance between x and y (is updated)
--- n:      number of observations
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void cov_recurrence(double   x, double   y, 
                    double *mx, double *my, 
                    double *cv, double n){
double oldmx = *mx, oldmy = *my;
double oldcv = *cv;

  *mx = oldmx + (x-oldmx)/n;
  *my = oldmy + (y-oldmy)/n;
  *cv = oldcv + (n-1)/n*(x-oldmx)*(y-oldmy);

  return;
}


/** One-pass skewness and kurtosis estimation
+++ This function implements a one-pass estimation of skewness and kurto-
+++ sis based on recurrence formulas. It can be used to estimate mean of 
+++ x, variance or standard deviation of x, skewness of x, kurtosis of x. 
+++ Use this function in a loop.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- x:      current x-value
--- mx:     last estimate of mean of x (is updated)
--- vx:     last estimate of variance of x (is updated)
--- sx:     last estimate of skewness of x (is updated)
--- kx:     last estimate of kurtosis of x (is updated)
--- n:      number of observations
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void kurt_recurrence(double   x, double *mx, 
                     double *vx, double *sx,
                     double *kx, double n){
double delta, delta_n, delta_n2, tmp;

  delta = x-(*mx);
  delta_n = delta/n;
  delta_n2 = delta_n*delta_n;
  tmp = delta*delta_n*(n-1);

  *mx = *mx + delta_n;
  *kx = *kx + tmp*delta_n2*(n*n-3*n+3) + 6*delta_n2*(*vx) - 4*delta_n*(*sx);
  *sx = *sx + tmp*delta_n*(n-2) - 3*delta_n*(*vx);
  *vx  = *vx + tmp;

  return;
}


/** One-pass skewness and kurtosis estimation
+++ This function implements a one-pass estimation of skewness based on 
+++ recurrence formulas. It can be used to estimate mean of x, variance or
+++ standard deviation of x, skewness of x. Use this function in a loop.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- x:      current x-value
--- mx:     last estimate of mean of x (is updated)
--- vx:     last estimate of variance of x (is updated)
--- sx:     last estimate of skewness of x (is updated)
--- n:      number of observations
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void skew_recurrence(double   x, double *mx, 
                     double *vx, double *sx,
                     double n){
double delta, delta_n, tmp;

  delta = x-(*mx);
  delta_n = delta/n;
  tmp = delta*delta_n*(n-1);

  *mx = *mx + delta_n;
  *sx = *sx + tmp*delta_n*(n-2) - 3*delta_n*(*vx);
  *vx  = *vx + tmp;

  return;
}


/** One-pass variance estimation
+++ This function implements a one-pass estimation of variance based on 
+++ recurrence formulas. It can be used to estimate mean of x, variance or
+++ standard deviation of x. Use this function in a loop.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- x:      current x-value
--- mx:     last estimate of mean of x (is updated)
--- vx:     last estimate of variance of x (is updated)
--- n:      number of observations
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void var_recurrence(double   x, double *mx, 
                    double *vx, double n){
double oldmx = *mx;
double oldvx = *vx;

  *mx = oldmx + (x-oldmx)/n;
  *vx = oldvx + (x-oldmx)*(x-*mx);

  return;
}


/** Compute kurtosis
+++ This function computes kurtosis based on the estimates of the recur-
+++ rence formulas above.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- var:    recurrence estimate of variance
--- kurt:   recurrence estimate of kurtosis
--- n:      number of observations
+++ Return: kurtosis
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double kurtosis(double var, double kurt, double n){

  return(kurt/(n*variance(var, n+1)*variance(var, n+1)));
}


/** Compute skewness
+++ This function computes skewness based on the estimates of the recur-
+++ rence formulas above.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- var:    recurrence estimate of variance
--- skew:   recurrence estimate of skewness
--- n:      number of observations
+++ Return: skewness
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double skewness(double var, double skew, double n){

  return(skew/(n*pow(standdev(var, n+1),3)));
}


/** Compute variance
+++ This function computes variance based on the estimates of the recur-
+++ rence formulas above.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- var:    recurrence estimate of variance
--- n:      number of observations
+++ Return: variance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double variance(double var, double n){

  return(var/(n-1));
}


/** Compute standard deviation
+++ This function computes standard deviation based on the estimates of 
+++ the recurrence formulas above.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- var:    recurrence estimate of variance
--- n:      number of observations
+++ Return: standard deviation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double standdev(double var, double n){

  return(sqrt(variance(var, n)));
}


/** Compute covariance
+++ This function computes covariance based on the estimates of the recur-
+++ rence formulas above.
+++-----------------------------------------------------------------------
+++ P. Pébay. SANDIA REPORT SAND2008-6212 (2008). Formulas for Robust, 
+++ One-Pass Parallel Computation of Co- variances and Arbitrary-Order 
+++ Statistical Moments.
+++-----------------------------------------------------------------------
--- cov:    recurrence estimate of covariance
--- n:      number of observations
+++ Return: covariance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double covariance(double cov, double n){

  return(cov/(n-1));
}


/** Compute slope of linear regression
+++ This function computes the slope of a linear regression.
--- cov:    covariance between x and y
--- varx:   variance of x
--- slope:  slope (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void linreg_slope(double cov, double varx, double *slope){

  *slope = cov/varx;

  return;
}


/** Compute intercept of linear regression
+++ This function computes the intercept of a linear regression.
--- slope:     slope of the regression
--- mx:        mean of x
--- my:        mean of y
--- intercept: intercept (returned)
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void linreg_intercept(double slope, double mx, double my, double *intercept){

  *intercept = my-slope*mx;

  return;
}


/** Compute regression coefficients of linear regression
+++ This function computes the slope and intercept of a linear regression.
--- mx:        mean of x
--- my:        mean of y
--- cov:       covariance between x and y
--- varx:      variance of x
--- slope:     slope of the regression (returned)
--- intercept: intercept (returned)
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void linreg_coefs(double mx, double my, double cov, double varx, 
                  double *slope, double *intercept){

  linreg_slope(cov, varx, slope);
  linreg_intercept(*slope, mx, my, intercept);

  return;
}


/** Compute correlation coefficient of a linear regression
+++ This function computes the r of a linear regression.
--- cov:    covariance between x and y
--- varx:   variance of x
--- vary:   variance of y
--- r:      correlation coefficient (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void linreg_r(double cov, double varx, double vary, double *r){

  *r = cov/sqrt(varx*vary);

  return;
}


/** Compute coefficient of determination of a linear regression
+++ This function computes the R^2 of a linear regression.
--- cov:    covariance between x and y
--- varx:   variance of x
--- vary:   variance of y
--- rsq:    coefficient of determination (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void linreg_rsquared(double cov, double varx, double vary, double *rsq){

  *rsq = cov*cov/(varx*vary);

  return;
}


/** Predict a value based on a linear regression
+++ This function predicts y values.
--- x:         x value
--- slope:     slope of the regression
--- intercept: intercept
--- y:         predicted y value (returned)
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void linreg_predict(double x, double slope, double intercept, double *y){

  *y = intercept + slope*x;

  return;
}


/** Test for significance of slope of regression line
+++ This function tests if the slope of a linear regression is different
+++ from a given value. A t-test is performed.
--- p:        probability (significance level)
--- tailtype: left (-1), twotail (0), right (1)
--- n:        number of samples
--- b:        slope
--- b0:       slope to test against, e.g. 0
--- seb:      standard error of slope
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int slope_significant(float p, int tailtype, int n, float b, float b0, float seb){
float t, tcrit;
int df;

  df = n-2;
  if (tscore(p, df, tailtype, &tcrit) == FAILURE){
    return 0;}
  t = (b-b0)/seb;

  if (tailtype == _TAIL_LEFT_  && t <=  tcrit) return -1;
  if (tailtype == _TAIL_RIGHT_ && t >=  tcrit) return  1;
  if (tailtype == _TAIL_TWO_   && t <= -tcrit) return -1;
  if (tailtype == _TAIL_TWO_   && t >=  tcrit) return  1;

  return 0;
}


/** Critical t-score
+++ This function computes a critical t-score.
--- p:        probability (significance level)
--- df:       degrees of freedom
--- tailtype: left (-1), twotail (0), right (1)
--- tscore:   critical t-score
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tscore(float p, int df, int tailtype, float *tscore){
float p0, p1, diff, t;
bool negative;

  if (p <= 0 || p >= 1 || df < 1){
    printf("error computing t-score.\n"); return FAILURE;}

  if ((tailtype == _TAIL_LEFT_  && p < 0.5) || 
      (tailtype == _TAIL_RIGHT_ && p > 0.5)){
    negative = true;
  } else negative = false;

  p0 = tscore_tail2twotail(p, tailtype, negative);
  p1 = p0;
  diff = 1.0;

  while (fabs(diff) > 0.0001){
    t = tscore_Hills_inv_t(p1, df); // initial rough value
    diff = tscore_T_p(t, df) - p0;  // compare result with forward fn
    p1 -= diff;                     // small adjustment to p1
  }

  if (negative) t = -t;

  *tscore = t;
  return SUCCESS;
}


/** Convert any tailtype to 'left'
+++ This function converts any tailtype's p to 'left' tailtype
--- p:        probability (significance level)
--- tail:     left (-1), twotail (0), right (1)
--- negative: negative part of t-distribution
+++ Return:   adjusted p-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_tail2left(float p, int tail, bool negative){

  if (tail == _TAIL_LEFT_)  return p;
  if (tail == _TAIL_RIGHT_) return (1.0-p);

  p = 1.0 - p/2.0;
  if (negative) p = 1.0-p;

  return p;
}


/** Convert 'left' tailtype to 'twotail'
+++ This function convert a 'left' tailtype's p to 'twotail' tailtype
--- p:        probability (significance level)
--- negative: negative part of t-distribution
+++ Return:   adjusted p-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_left2twotail(float p, bool negative){

  if (negative) p = 1.0-p;      // corrects to p>0.5;

  return 2.0*(1.0-p);
}


/** Convert any tailtype to 'twotail'
+++ This function convert any tailtype's p to 'twotail' tailtype
--- p:        probability (significance level)
--- tail:     left (-1), twotail (0), right (1)
--- negative: negative part of t-distribution
+++ Return:   adjusted p-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_tail2twotail(float p, int tail, bool negative){
float p0;

  p0 = tscore_tail2left(p, tail, negative);
  p0 = tscore_left2twotail(p0, negative);

  return p0;
}


/** Two-tailed standard normal probability of z
--- z:        z-value
+++ Return:   p-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_Norm_p(float z){
float az, p;
float a1 = 0.0000053830, a2 = 0.0000488906, a3 = 0.0000380036;
float a4 = 0.0032776263, a5 = 0.0211410061, a6 = 0.0498673470;

  az = fabs(z);
  p = (((((a1*az+a2)*az+a3)*az+a4)*az+a5)*az+a6)*az+1;
  p = pow(p, -16);

  return p;
}


/** z-value
--- z:        p-value
+++ Return:   z-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_Norm_z(float p){
float a0 = 2.5066282,  a1 =-18.6150006, a2 = 41.3911977, a3 =-25.4410605;
float b1 =-8.4735109,  b2 = 23.0833674, b3 =-21.0622410, b4 =  3.1308291;
float c0 =-2.7871893,  c1 = -2.2979648, c2 =  4.8501413, c3 =  2.3212128;
float d1 = 3.5438892,  d2 =   1.6370678;
float r, z;
 
  if (p>0.42){
      r=sqrt(-log(0.5-p));
      z=(((c3*r+c2)*r+c1)*r+c0)/((d2*r+d1)*r+1);
  } else {
      r=p*p;
      z=p*(((a3*r+a2)*r+a1)*r+a0)/((((b4*r+b3)*r+b2)*r+b1)*r+1);
  }

  return z;
}


/** Approximate t-value
+++ Computes approximate t-value with given df and two-tail probability.
+++ Hill's approx. inverse t-dist.: Comm. of A.C.M Vol.13 No.10 1970.
--- p:        p-value
--- idf:      degrees of freedom
+++ Return:   t-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_Hills_inv_t(float p, int idf){
float a, b, c, d, t, x, y;
float df = (float)idf;

  if (df == 1){
    t = cos(p*M_PI/2)/sin(p*M_PI/2);
  } else if (df == 2){
    t = sqrt(2/(p*(2 - p)) - 2);
  } else {
    a = 1/(df - 0.5);
    b = 48/(a*a);
    c = ((20700*a/b - 98)*a - 16)*a + 96.36;
    d = ((94.5/(b + c) - 3)/b + 1)*sqrt(a*M_PI*0.5)*df;
    x = d*p;
    y = pow(x, 2.0/df);
    if (y > 0.05 + a){
      x = tscore_Norm_z(0.5*(1 - p)); 
      y = x*x;
      if (df < 5) c = c + 0.3*(df - 4.5)*(x + 0.6);
      c = (((0.05*d*x - 5)*x - 7)*x - 2)*x + b + c;
      y = (((((0.4*y + 6.3)*y + 36)*y + 94.5)/c - y - 3)/b + 1)*x;
      y = a*y*y;
      if (y > 0.002){
        y = exp(y) - 1;
      } else {
        y = 0.5*y*y + y;
      }
      t = sqrt(df*y);
    } else {
      y = ((1/(((df + 6.0)/(df*y) - 0.089*d - 0.822)*(df + 2.0)*3) +
        0.5/(df + 4.0))*y - 1.0)*(df + 1.0)/(df + 2.0) + 1.0/y;
      t = sqrt(df*y);
    }
  }

  return t;
}


/** Approximate z-value
+++ Converts a t-value to an approximate z-value with given df and two-
+++ tail probability.
--- t:        t-value
--- df:       degrees of freedom
+++ Return:   z-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_T_z(float t, int df){
float A9, B9, T9, Z8, P7, B7, z;

  A9 = df - 0.5;
  B9 = 48*A9*A9,
  T9 = t*t/df;

  if (T9 >= 0.04){
    Z8 = A9*log(1+T9);
  } else {
    Z8 = A9*(((1 - T9*0.75)*T9/3 - 0.5)*T9 + 1)*T9;
  }
  P7 = ((0.4*Z8 + 3.3)*Z8 + 24)*Z8 + 85.5;
  B7 = 0.8*pow(Z8, 2) + 100 + B9;
  z = (1 + (-P7/B7 + Z8 + 3)/B9)*sqrt(Z8);

  return z;
}


/** Two-tail p-value
+++ computes a p-value with given t-value, df and two-tail probability.
--- t:        t-value
--- df:       degrees of freedom
+++ Return:   p-value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float tscore_T_p(float t, int df){
float abst, tsq, p, z;

  abst = fabs(t);
  tsq = t*t;

  if        (df == 1){ p = 1 - 2*atan(abst)/M_PI;
  } else if (df == 2){ p = 1 - abst/sqrt(tsq + 2);
  } else if (df == 3){ p = 1 - 2*(atan(abst/sqrt(3)) + abst*sqrt(3)/(tsq + 3))/M_PI;
  } else if (df == 4){ p = 1 - abst*(1 + 2/(tsq + 4))/sqrt(tsq + 4);
  } else {
  // finds the z equivalent of t and df st they yield same probs.
    z = tscore_T_z(abst, df);
    if (df>4){
      p = tscore_Norm_p(z);
    }  else {
      p = tscore_Norm_p(z); // small non-integer df
    }
  }

  return p;
}


/** Quantile
+++ This function computes a quantile of an array. The quick select algo-
+++ rithm is used for this purpose. Caution: the array will be screwed up.
+++ Copy the array before calling the quantile function.
--- x:      array
--- n:      length of array
--- p:      probability [0,1]
+++ Return: quantile
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float quantile(float *x, int n, float p){
int left = 0, right = n - 1, r, w, k;
float piv;


  // compute position from probability
  k = (int) (n-1)/(1/p);

  // do until left and right converge at k
  while (left < right){

    r = left ;        // reader
    w = right;        // writer
    piv = x[(r+w)/2]; // pivot

    // do until reader and writer are equal
    while (r < w){

      if (x[r] >= piv){
        // if larger than pivot, put value to the end, decrease writer
        quantile_swap(x, r, w);
        w--;
      } else {
        // if smaller than pivot, skip, increase reader
        r++;
      }
    }

    // if elements are equal, array consists of one unique value
    if (x[left] == x[right]) return (x[k]);

    // if reader was increased, decrese reader
    if (x[r] > piv) r--;

    // reader is on the end of the first k elements
    if (k <= r){
      right = r;
    } else {
      left = r + 1;
    }

  }

  return x[k];
}


/** Function for swapping two array elements
+++ This function is used in the quick select algorithm for quantiles.
+++--------------------------------------------------------------------**/
void quantile_swap(float *x, int from, int to){
float tmp = x[from];

  x[from] = x[to]; x[to] = tmp;

  return;
}


/** Mode
+++ This function computes the mode of an array. The array is completely
+++ sorted. Caution: the array will be screwed up.
+++ Copy the array before calling the mode function.
--- x:      array
--- n:      length of array
+++ Return: mode
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int mode(int *x, int n){
int i;
int max = 1;
int mod = x[0];
int now = 1;


  qsort(x, n, sizeof(int), comp);

  for (i=1; i<n; i++){
    if (x[i] == x[i-1]){
      now++;
    } else {
      if (now > max){
          max = now;
          mod = x[i-1];
      }
      now = 1;
    }
  }

  if (now > max){
    max = now;
    mod = x[n-1];
  }


  return mod;
}


/** Number of unique values
+++ This function computes the number of unique values of an array. 
--- x:      array
--- n:      length of array
+++ Return: number
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int n_uniq(int *x, int n){
int i;
int *arr = NULL;
int num = 1;


  alloc((void**)&arr, n, sizeof(int));
  for (i=0; i<n; i++) arr[i] = x[i];

  qsort(arr, n, sizeof(int), comp);

  for (i=1; i<n; i++){
    if (arr[i] != arr[i-1]) num++;
  }

  free((void*)arr);


  return num;
}


/** Histogram
+++ This function computes a histogram.
--- x:      array
--- n:      length of array
+++ Return: histogram
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int **histogram(int *x, int n, int *n_uniq){
int i, k;
int *arr = NULL;
int **hist = NULL;
int num = 1;


  alloc((void**)&arr, n, sizeof(int));
  for (i=0; i<n; i++) arr[i] = x[i];

  qsort(arr, n, sizeof(int), comp);

  for (i=1; i<n; i++){
    if (arr[i] != arr[i-1]) num++;
  }

  alloc_2D((void***)&hist, 2, num, sizeof(int));

  k = 0;

  hist[0][k] = arr[0];
  hist[1][k]++;

  for (i=1; i<n; i++){

    if (arr[i] != arr[i-1]){
      k++;
      hist[0][k] = arr[i];
    }

    hist[1][k]++;
  }


  free((void*)arr);

  *n_uniq = num;
  return hist;
}


/** Function for comparing two array elements
+++ This function is used mode algorithm
+++--------------------------------------------------------------------**/
int comp(const void *a, const void *b){
  return ( *(int*)a - *(int*)b );
}

