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
Coregistration functions header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef COREGFUNS_LL_H
#define COREGFUNS_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions
#include <math.h>    // common mathematical functions
#include <stdbool.h>  // boolean data type

#include "../cross-level/const-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

void LSMatching_SAM(short *imagel, int coll, int rowl, short *imager, int colr, int rowr, int img_coll, int img_rowl, float x1, float y1, float *ptr_x2, float *ptr_y2, float *ptr_cr, short shtMeanDiffThreshold);
void SAM(short *img1, int col1, int row1, short *img2, int col2, int row2, int x1, int y1, int x2, int y2, int width, int height, float *ptr_cr);
int resampling(short *image, int col, int row, short *img, int img_col, int img_row, int x1, int y1, double *par_x, double *par_y);
double corr2(short *x, short *y, int n);
int imsub(short *im, int ncol, int nrow, int x, int y, int h, short *sub);
short GetMeanDiff(short *img1, int col1, int row1, short *img2, int col2, int row2, int x1, int y1, int x2, int y2, int width, int height);
double Mean1(double *x, int n);
void RMSE1(double *x, int n, int t, double *ptr_rmse, double *ptr_avg);
float GetStd(short *pshtData, int n, small *pucMask);
void ApplyMask(short *pshtData, int n, small *pucMask, short shtFillValue);
void FitTranslationTransform(double *x1, double *y1, double *x2, double *y2, int n, double Coefs[], double *ptr_Errors, double *ptr_errors_mean, double *ptr_fitting_rmse);
void FitAffineTransform(double *x1, double *y1, double *x2, double *y2, int n, double U[], double *ptr_Errors, double *ptr_errors_mean, double *ptr_fitting_rmse);
void FitPolynomialTransform(double *x1, double *y1, double *x2, double *y2, int n, double Coefs[], double *ptr_Errors, double *ptr_errors_mean, double *ptr_fitting_rmse);
void GetTransformedCoords(double x1, double y1, int iTransformationType, double *Coefs, double *ptr_x2, double *ptr_y2);
int INVSQR1(double *A, double *B, int n);
void CalcGradient2D(float* pImageTemp, float* pImageX, float* pImageY, int iWidth, int iHeight);
void CalcMultiply(float* pImageTemp1, float* pImageTemp2, float* pImageNew, int iSize);
void CalcDivide(float* pImageTemp1, float* pImageTemp2, float* pImageNew, int iSize);
void CalcAdd(float* pImageTemp1, float* pImageTemp2, float* pImageNew, int iSize);
void CalcSubtract(float* pImageTemp1, float* pImageTemp2, float* pImageNew, int iSize);
void CalcMinMaxMeanWithMask(float* pImg, small *pucMask, float *pdMin, float *pdMax, float *pdMean, int iSize);
void AddConst(float *pBuffer, float c, int iSize);
float* GetGaussian(double dSigma, int *iFilterWidth);
void Conv2same(short *pImg, short *pImgNew, int iWidth, int iHeight, short nodata, float* dFilter, int w, int step);
void Conv2same_FLT_T(float* pImg, float* pImgNew, int iWidth, int iHeight, float* dFilter, int w);
bool FindTargetValueInWindow(short *pImg, int iWidth, int iHeight, int iTargetCol, int iTargetRow, int w, short targetValue);

#ifdef __cplusplus
}
#endif

#endif

