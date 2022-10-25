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
Statistics header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef STATS_CL_H
#define STATS_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h> // boolean data type
#include <math.h>    // common mathematical functions

#include "../cross-level/const-cl.h"
#include "../cross-level/alloc-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

void covar_recurrence(double   x, double   y, double *mx, double *my, double *vx, double *vy, double *cv, double n);
void cov_recurrence(double   x, double   y, double *mx, double *my, double *cv, double n);
void kurt_recurrence(double   x,    double *mx, double *vx,    double *sx,double *kx, double n);
void skew_recurrence(double   x,    double *mx, double *vx,    double *sx,double n);
void var_recurrence(double   x, double *mx, double *vx, double n);
double kurtosis(double var, double kurt, double n);
double skewness(double var, double skew, double n);
double variance(double var, double n);
double standdev(double var, double n);
double covariance(double cov, double n);
void linreg_slope(double cov, double varx, double *slope);
void linreg_intercept(double slope, double mx, double my, double *intercept);
void linreg_coefs(double mx, double my, double cov, double varx, double *slope, double *intercept);
void linreg_r(double cov, double varx, double vary, double *r);
void linreg_rsquared(double cov, double varx, double vary, double *rsq);
void linreg_predict(double x, double slope, double intercept, double *y);
int slope_significant(float p, int tailtype, int n, float b, float b0, float seb);
int tscore(float p, int df, int tailtype, float *tscore);
float tscore_tail2left(float p, int tail, bool negative);
float tscore_left2twotail(float p, bool negative);
float tscore_tail2twotail(float p, int tail, bool negative);
float tscore_Norm_p(float z);
float tscore_Norm_z(float p);
float tscore_Hills_inv_t(float p, int idf);
float tscore_T_z(float t, int df);
float tscore_T_p(float t, int df);
float quantile(float *x, int n, float p);
int mode(int *x, int n);
int n_uniq(int *x, int n);
int **histogram(int *x, int n, int *n_uniq);

#ifdef __cplusplus
}
#endif

#endif

