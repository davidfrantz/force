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
This file contains functions for supporting coregistration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** The following code was adopted from the LSReg code, developed by Lin 
+++ Yan and David Roy at the Geospatial Sciences Center of Excellence, 
+++ South Dakota State University under NASA grant NNX17AB34G. 
+++ LSReg, Version: 2.0
+++ Copyright (C) 2018 Lin Yan & David Roy
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ Yan, L., Roy, D.P., Zhang, H.K., Li, J., Huang, H. (2016). An automated
+++ approach for sub-pixel registration of Landsat-8 Operational Land 
+++ Imager (OLI) and Sentinel-2 Multi Spectral Instrument (MSI) imagery. 
+++ Remote Sensing, 8(6), 520. 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "coregfuns-ll.h"


#define DELTA_LIMIT      (0.001f)
#define ABS(x)        ((x>=0)? (x):-(x))
#define MAX(a,b) (((a)>(b))? (a) : (b))
#define MIN(a,b) (((a)<(b))? (a) : (b))


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


void LSMatching_SAM(short *imagel, int coll, int rowl, short *imager,
int colr, int rowr, int img_coll, int img_rowl,
float x1, float y1, float *ptr_x2, float *ptr_y2, float *ptr_cr, short shtMeanDiffThreshold){
int i, j;
int x, y, x10, y10, x20, y20;
double A[8], N[8][8], l, U[8];
double gx, gy;
double h0, h1;          // two radiometric parameters
double par_x[3], par_y[3];    // six geometric parameters
double par[8];          // total above eight parameters to be fitted by least squares
float cr_current, cr_new;    // similarity metric, can be SAM or correlation; SAM is used here
int  img_rowr, img_colr;      // size of resampled image patch on right image; note it is (img_rowl + 2) x (img_coll + 2) pixels rather than img_rowl x img_coll pixels
short *imgl0, *imgr0, *imgr1;
int k, r;
short *imgl = NULL, *imgr = NULL;

  
  img_rowr = img_rowl + 2;
  img_colr = img_coll + 2;

  imgl = (short *)malloc(img_rowl*img_coll *sizeof(short));
  imgr = (short *)malloc(img_rowr*img_colr *sizeof(short));

  // get left image patch
  x10 = (int)(x1 + 1e-10);
  par_x[0] = (double)(x10);
  par_x[1] = 1;
  par_x[2] = 0;
  y10 = (int)(y1 + 0.000001);
  par_y[0] = (double)(y10);
  par_y[1] = 0;
  par_y[2] = 1;
  if (!resampling(imagel, coll, rowl, imgl, img_coll, img_rowl, x10, y10, par_x, par_y))
  {
    *ptr_cr = -1;
    free(imgl);
    free(imgr);
    return;
  }

  // get right image patch
  x20 = (int)(*ptr_x2 + 1e-10);
  par_x[0] = (double)(x20);
  par_x[1] = 1;
  par_x[2] = 0;
  y20 = (int)(*ptr_y2 + 1e-10);
  par_y[0] = (double)(y20);
  par_y[1] = 0;
  par_y[2] = 1;
  if (!resampling(imager, colr, rowr, imgr, img_colr, img_rowr, x20, y20, par_x, par_y))
  {
    *ptr_cr = -2;
    free(imgl);
    free(imgr);
    return;
  }

  // check mean difference of two image patches based on shtMeanDiffThreshold
  if (GetMeanDiff(imgl, img_coll, img_rowl, imgr, img_colr, img_rowr, img_coll / 2, img_rowl / 2, img_colr / 2, img_rowr / 2, img_coll, img_rowl) >= shtMeanDiffThreshold)
  {
    // mean difference greater than shtMeanDiffThreshold, return
    *ptr_cr = -4;
    free(imgl);
    free(imgr);
    return;
  }

  // initialize 8 parameters
  par_x[0] = *ptr_x2;
  par_x[1] = 1;
  par_x[2] = 0;
  par_y[0] = *ptr_y2;
  par_y[1] = 0;
  par_y[2] = 1;
  h0 = 0;
  h1 = 1;

  // get initial SAM
  SAM(imgl, img_coll, img_rowl, imgr, img_colr, img_rowr, img_coll / 2, img_rowl / 2, img_colr / 2, img_rowr / 2, img_coll, img_rowl, &cr_current);

  // do least squres fitting
  while (1)
  {
    imgl0 = imgl;
    imgr0 = imgr1 = imgr + img_colr + 1;

    memset(U, 0, 8 * sizeof(double));
    memset(N, 0, 64 * sizeof(double));

    // traverse every pixel in img_rowl x img_coll window
    for (i = 0; i<img_rowl; i++)
    {
      y = (int)(-img_rowl / 2.0 + i + 1e-10);
      for (j = 0; j<img_coll; j++)
      {
        x = (int)(-img_coll / 2.0 + j + 1e-10);
        l = *imgl0 - ((*imgr1)*h1 + h0);
        gx = (double)(*(imgr1 + 1) - *(imgr1 - 1)) / 2;
        gy = (double)(*(imgr1 + img_colr) - *(imgr1 - img_colr)) / 2;

        *A = 1;
        *(A + 1) = *imgr1;
        *(A + 2) = gx;
        *(A + 3) = gx*x;
        *(A + 4) = gx*y;
        *(A + 5) = gy;
        *(A + 6) = gy*x;
        *(A + 7) = gy*y;

        // construct normal equation
        for (k = 0; k < 8; k++)
        {
          for (r = 0; r < 8; r++)
            N[k][r] += A[k] * A[r];
          U[k] += A[k] * l;
        }

        imgl0++;
        imgr1++;
        x++;
      }
      imgr0 += img_colr;
      imgr1 = imgr0;
      y++;
    }

    // solve normal equation Nx = U, x saved in U
    if (!INVSQR1((double*)N, U, 8))
      break;

    // update eight parameters
    memcpy(par, U, 8 * sizeof(double));
    h0 = h0 + *par;
    h1 = h1 + *(par + 1);
    *par_x += *(par + 2);
    *(par_x + 1) += *(par + 3);
    *(par_x + 2) += *(par + 4);
    *par_y += *(par + 5);
    *(par_y + 1) += *(par + 6);
    *(par_y + 2) += *(par + 7);

    // resample right image patch using new geometric coefficients
    if (!resampling(imager, colr, rowr, imgr, img_colr, img_rowr, x20, y20, par_x, par_y))
    {
      *ptr_cr = -3;
      free(imgl);
      free(imgr);
      return;
    }

    // calculate new similarity (SAM) between left image patch and newly resampled right image patch; conventially correlation_coefficient() is used
    SAM(imgl, img_coll, img_rowl, imgr, img_colr, img_rowr, img_coll / 2, img_rowl / 2, img_colr / 2, img_rowr / 2, img_coll, img_rowl, &cr_new);

    // check whether similarity increases
    if (cr_new > cr_current)
      cr_current = cr_new; // similarity inceasing, go on
    else
    {
      // similarity deceasing, reverse parameters to values in previous iteration and stop least squares fitting
      *par_x -= *(par + 2);
      *(par_x + 1) -= *(par + 3);
      *(par_x + 2) -= *(par + 4);
      *par_y -= *(par + 5);
      *(par_y + 1) -= *(par + 6);
      *(par_y + 2) -= *(par + 7);

      break;
    }
  }

  // final similarity
  *ptr_cr = cr_current;

  // final matched position on right image, corresponding to (x1, y1) on left image
  *ptr_x2 = (float)(par_x[0]);
  *ptr_y2 = (float)(par_y[0]);

  free(imgl);
  free(imgr);

  return;
}


int resampling(short *image, int col, int row, short *img, int img_col, int img_row, int x1, int y1, double *par_x, double *par_y){
int i, j, i1, i2, j1, j2;
short g, *image0, *img0;
double x2, y2, p, q;
int    x20, y20;
double   bit;

  img0 = img;
  i1 = -img_row / 2;
  i2 = img_row / 2;
  j1 = -img_col / 2;
  j2 = img_col / 2;
  for (i = i1; i <= i2; i++)
  {
    for (j = j1; j <= j2; j++)
    {
      x2 = par_x[0] + j*par_x[1] + i*par_x[2];    //  x2=x+a0+x*a1+y*a2
      y2 = par_y[0] + j*par_y[1] + i*par_y[2];    //  y2=y+b0+y*b1+x*b2
      x20 = (int)(x2 + 1e-10);
      y20 = (int)(y2 + 1e-10);

      if (x20 < 0 || x20 >= col - 1 || y20 < 0 || y20 >= row - 1)
        return 0;

      image0 = image + y20*col + x20;
      p = x2 - x20;
      q = y2 - y20;
      g = 0;
      bit = 0;
      bit += (*image0)*(1 - p) * (1 - q);
      bit += (*(image0 + 1)) * p * (1 - q);
      bit += (*(image0 + col)) * (1 - p) * q;
      bit += (*(image0 + col + 1)) * p * q;
      g = (short)(bit + 0.5f);
      *img0++ = g;
    }
  }

  return 1;
}


void SAM(short *img1, int col1, int row1, short *img2, int col2, int row2, int x1, int y1, int x2, int y2, int width, int height, float *ptr_cr){
int i, j;
int  height2, width2;
float Sxx, Syy, Sxy;
short *image1, *image10, *image2, *image20;
short pixel1, pixel2;

  height2 = height / 2;
  width2 = width / 2;


  Sxx = 0.0;
  Syy = 0.0;
  Sxy = 0;
  image10 = image1 = img1 + (y1 - height2)*col1 + (x1 - width2);
  image20 = image2 = img2 + (y2 - height2)*col2 + (x2 - width2);

  for (i = 0; i<height; i++)
  {
    for (j = 0; j < width; j++)
    {
      pixel1 = *image1;
      pixel2 = *image2;
      Sxx += pixel1*pixel1;
      Syy += pixel2*pixel2;
      Sxy += pixel1*pixel2;
      image1++;
      image2++;
    }
    image10 += col1;  image1 = image10;
    image20 += col2;  image2 = image20;
  }
  if (Sxx > 0 && Syy > 0)
    *ptr_cr = Sxy / sqrt((float)(Sxx*Syy));
  else
    *ptr_cr = 0;

  return;
}


short GetMeanDiff(short *img1, int col1, int row1, short *img2, int col2, int row2, int x1, int y1, int x2, int y2, int width, int height){
int i, j, n;
int  height2, width2;
short *image1, *image10, *image2, *image20;
short pixel1, pixel2;
float diff;

  height2 = height / 2;
  width2 = width / 2;

  n = width*height;

  image10 = image1 = img1 + (y1 - height2)*col1 + (x1 - width2);
  image20 = image2 = img2 + (y2 - height2)*col2 + (x2 - width2);
  diff = 0;
  for (i = 0; i<height; i++)
  {
    for (j = 0; j < width; j++)
    {
      pixel1 = *image1;
      pixel2 = *image2;
      diff += (pixel1 > pixel2) ? pixel1 - pixel2 : pixel2 - pixel1;
      image1++;
      image2++;
    }
    image10 += col1;  image1 = image10;
    image20 += col2;  image2 = image20;
  }
  diff /= n;

  return (short)(diff + 0.5f);
}


int INVSQR1(double *A, double *B, int n){
int k, i, j, i0 = 0;
double C;
double T;

  for (k = 0; k<n; k++)
  {
    C = 0;
    for (i = k; i<n; i++)
    {
      if (fabs(A[i*n + k]) >= fabs(C))
      {
        C = A[i*n + k];
        i0 = i;
      }
    }
    if (i != k)
    {
      for (j = k; j<n; j++)
      {
        T = A[k*n + j];
        A[k*n + j] = A[i0*n + j];
        A[i0*n + j] = T;
      }
      T = B[k];
      B[k] = B[i0];
      B[i0] = T;
    }
    if (fabs(C) <= 0.0)
      return 0;
    C = 1 / C;
    for (j = k + 1; j<n; j++)
    {
      A[k*n + j] *= C;
      for (i = k + 1; i<n; i++)
      {
        A[i*n + j] = A[i*n + j] - A[i*n + k] * A[k*n + j];
      }
    }
    B[k] *= C;
    for (i = k + 1; i<n; i++)
    {
      B[i] = B[i] - B[k] * A[i*n + k];
    }
  }
  for (i = n - 2; i >= 0; i--)
  {
    for (j = i + 1; j<n; j++)
    {
      B[i] = B[i] - A[i*n + j] * B[j];
    }
  }

  return 1;
}


void RMSE1(double *x, int n, int t, double *ptr_rmse, double *ptr_avg){
int i;

  if (n <= t)
  {
    *ptr_rmse = 9999;
    *ptr_avg = 9999;
    return;
  }

  *ptr_avg = 0;
  *ptr_rmse = 0;

  *ptr_avg = Mean1(x, n);
  for (i = 0; i < n; i++)
    *ptr_rmse += (*(x + i))*(*(x + i));

  *ptr_rmse /= (n - t);
  *ptr_rmse = sqrt(*ptr_rmse);

  return;
}


double Mean1(double *x, int n){
double sum = 0;
int i;

  for (i = 0; i<n; i++)
    sum += x[i];

  return sum / n;
}


void FitAffineTransform(double *x1, double *y1, double *x2, double *y2, int n, double Coefs[], double *ptr_Errors, double *ptr_errors_mean, double *ptr_fitting_rmse){
double U[6];
int iIterNum;
int i, k, r;
double L;
double A[6], N[6][6];
double xdif, ydif;
double fitting_rmse_, errors_mean_;

  memset(U, 0, 6 * sizeof(double));
  memset(ptr_Errors, 0, n * sizeof(double));
  *ptr_errors_mean = 0;
  *ptr_fitting_rmse = 0;

  if (n < 6)
    return;

  memset(A, 0, 6 * sizeof(double));
  memset(N, 0, 36 * sizeof(double));
  memset(Coefs, 0, 6 * sizeof(double));
  Coefs[1] = 1;
  Coefs[5] = 1;
  iIterNum = 0;
  while (iIterNum < 10)
  {
    for (i = 0; i < n; i++)
    {
      A[0] = 1;
      A[1] = x1[i];
      A[2] = y1[i];
      A[3] = 0;
      A[4] = 0;
      A[5] = 0;
      L = x2[i] - (Coefs[0] + Coefs[1] * x1[i] + Coefs[2] * y1[i]);
      for (k = 0; k < 6; k++)
      {
        for (r = 0; r < 6; r++)
          N[k][r] += A[k] * A[r];
        U[k] += A[k] * L;
      }
      A[0] = 0;
      A[1] = 0;
      A[2] = 0;
      A[3] = 1;
      A[4] = x1[i];
      A[5] = y1[i];
      L = y2[i] - (Coefs[3] + Coefs[4] * x1[i] + Coefs[5] * y1[i]);
      for (k = 0; k < 6; k++)
      {
        for (r = 0; r < 6; r++)
          N[k][r] += A[k] * A[r];
        U[k] += A[k] * L;
      }
    }

    if (!INVSQR1((double*)N, U, 6))
      break;

    for (k = 0; k < 6; k++)
      Coefs[k] += U[k];

    if (abs(U[0]) < DELTA_LIMIT && abs(U[3]) < DELTA_LIMIT)
      break;

    iIterNum += 1;
  }

  memcpy(U, Coefs, 6 * sizeof(double));
  for (i = 0; i < n; i++)
  {
    xdif = U[0] + U[1] * x1[i] + U[2] * y1[i] - x2[i];
    ydif = U[3] + U[4] * x1[i] + U[5] * y1[i] - y2[i];
    ptr_Errors[i] = sqrt(xdif*xdif + ydif*ydif);
  }

  RMSE1(ptr_Errors, n, 6, &fitting_rmse_, &errors_mean_);

  *ptr_errors_mean = errors_mean_;
  *ptr_fitting_rmse = fitting_rmse_;

  return;
}


void FitTranslationTransform(double *x1, double *y1, double *x2, double *y2, int n, double Coefs[], double *ptr_Errors, double *ptr_errors_mean, double *ptr_fitting_rmse){
double U[2];
int iIterNum;
int i, k, r;
double L;
double A[2], N[2][2];
double xdif, ydif;
double fitting_rmse_, errors_mean_;

  memset(U, 0, 2 * sizeof(double));
  memset(ptr_Errors, 0, n * sizeof(double));
  *ptr_errors_mean = 0;
  *ptr_fitting_rmse = 0;

  if (n < 2)
    return;

  memset(A, 0, 2 * sizeof(double));
  memset(N, 0, 4 * sizeof(double));
  memset(Coefs, 0, 2 * sizeof(double));
  iIterNum = 0;
  while (iIterNum < 10)
  {
    for (i = 0; i < n; i++)
    {
      A[0] = 1;
      A[1] = 0;
      L = x2[i] - (Coefs[0] + x1[i]);
      for (k = 0; k < 2; k++)
      {
        for (r = 0; r < 2; r++)
          N[k][r] += A[k] * A[r];
        U[k] += A[k] * L;
      }
      A[0] = 0;
      A[1] = 1;
      L = y2[i] - (Coefs[1] + y1[i]);
      for (k = 0; k < 2; k++)
      {
        for (r = 0; r < 2; r++)
          N[k][r] += A[k] * A[r];
        U[k] += A[k] * L;
      }
    }

    if (!INVSQR1((double*)N, U, 2))
      break;

    for (k = 0; k < 2; k++)
      Coefs[k] += U[k];

    if (abs(U[0]) < DELTA_LIMIT && abs(U[1]) < DELTA_LIMIT)
      break;

//    break; // theoretically iterations are not needed as a0 and b0 are independent

    iIterNum += 1;
  }

  memcpy(U, Coefs, 2 * sizeof(double));
  for (i = 0; i < n; i++)
  {
    xdif = U[0] + x1[i] - x2[i];
    ydif = U[1] + y1[i] - y2[i];
    ptr_Errors[i] = sqrt(xdif*xdif + ydif*ydif);
  }

  RMSE1(ptr_Errors, n, 2, &fitting_rmse_, &errors_mean_);

  *ptr_errors_mean = errors_mean_;
  *ptr_fitting_rmse = fitting_rmse_;

  return;
}


void FitPolynomialTransform(double *x1, double *y1, double *x2, double *y2, int n, double Coefs[], double *ptr_Errors, double *ptr_errors_mean, double *ptr_fitting_rmse){
double U[12];
int iIterNum;
int i, k, r;
double L;
double A[12], N[12][12];
double xdif, ydif;
double fitting_rmse_, errors_mean_;

  memset(U, 0, 12 * sizeof(double));
  memset(ptr_Errors, 0, n * sizeof(double));
  *ptr_errors_mean = 0;
  *ptr_fitting_rmse = 0;

  if (n < 12)
    return;

  memset(A, 0, 12 * sizeof(double));
  memset(N, 0, 144 * sizeof(double));
  memset(Coefs, 0, 12 * sizeof(double));
  Coefs[1] = 1;
  Coefs[8] = 1;
  iIterNum = 0;
  while (iIterNum < 10)
  {
    for (i = 0; i < n; i++)
    {
      // x2 = a0 + a1*x1 + a2*y1 + a3*x1*x1 + a4*x1*y1 + a5*y1*y1
      A[0] = 1;
      A[1] = x1[i];
      A[2] = y1[i];
      A[3] = x1[i] * x1[i];
      A[4] = x1[i] * y1[i];
      A[5] = y1[i] * y1[i];
      A[6] = 0;
      A[7] = 0;
      A[8] = 0;
      A[9] = 0;
      A[10] = 0;
      A[11] = 0;
      L = x2[i] - (Coefs[0] + Coefs[1] * x1[i] + Coefs[2] * y1[i] + Coefs[3] * x1[i] * x1[i] + Coefs[4] * x1[i] * y1[i] + Coefs[5] * y1[i] * y1[i]);
      for (k = 0; k < 12; k++)
      {
        for (r = 0; r < 12; r++)
          N[k][r] += A[k] * A[r];
        U[k] += A[k] * L;
      }

      // y2 = b0 + b1*x1 + b2*y1 + b3*x1*x1 + b4*x1*y1 + b5*y1*y1
      A[0] = 0;
      A[1] = 0;
      A[2] = 0;
      A[3] = 0;
      A[4] = 0;
      A[5] = 0;
      A[6] = 1;
      A[7] = x1[i];
      A[8] = y1[i];
      A[9] = x1[i] * x1[i];
      A[10] = x1[i] * y1[i];
      A[11] = y1[i] * y1[i];
      L = y2[i] - (Coefs[6] + Coefs[7] * x1[i] + Coefs[8] * y1[i] + Coefs[9] * x1[i] * x1[i] + Coefs[10] * x1[i] * y1[i] + Coefs[11] * y1[i] * y1[i]);
      for (k = 0; k < 12; k++)
      {
        for (r = 0; r < 12; r++)
          N[k][r] += A[k] * A[r];
        U[k] += A[k] * L;
      }
    }

    if (!INVSQR1((double*)N, U, 12))
      break;

    for (k = 0; k < 12; k++)
      Coefs[k] += U[k];

    if (abs(U[0]) < DELTA_LIMIT && abs(U[6]) < DELTA_LIMIT)
      break;

    iIterNum += 1;
  }

  memcpy(U, Coefs, 12 * sizeof(double));
  for (i = 0; i < n; i++)
  {
    xdif = U[0] + U[1] * x1[i] + U[2] * y1[i] + U[3] * x1[i] * x1[i] + U[4] * x1[i] * y1[i] + U[5] * y1[i] * y1[i] - x2[i];
    ydif = U[6] + U[7] * x1[i] + U[8] * y1[i] + U[9] * x1[i] * x1[i] + U[10] * x1[i] * y1[i] + U[11] * y1[i] * y1[i] - y2[i];
    ptr_Errors[i] = sqrt(xdif*xdif + ydif*ydif); // fitting residual
  }

  RMSE1(ptr_Errors, n, 12, &fitting_rmse_, &errors_mean_);

  *ptr_errors_mean = errors_mean_;
  *ptr_fitting_rmse = fitting_rmse_;

  return;
}


void GetTransformedCoords(double x1, double y1, int iTransformationType, double *Coefs, double *ptr_x2, double *ptr_y2){
double x2_, y2_;

  *ptr_x2 = 0;
  *ptr_y2 = 0;

  switch (iTransformationType)
  {
  case 1:
    x2_ = Coefs[0] + x1;
    y2_ = Coefs[1] + y1;
    break;
  case 2:
    x2_ = Coefs[0] + Coefs[1] * x1 + Coefs[2] * y1;
    y2_ = Coefs[3] + Coefs[4] * x1 + Coefs[5] * y1;
    break;
  case 3:
    x2_ = Coefs[0] + Coefs[1] * x1 + Coefs[2] * y1 + Coefs[3] * x1*x1 + Coefs[4] * x1*y1 + Coefs[5] * y1*y1;
    y2_ = Coefs[6] + Coefs[7] * x1 + Coefs[8] * y1 + Coefs[9] * x1*x1 + Coefs[10] * x1*y1 + Coefs[11] * y1*y1;
    break;
  default:
    x2_ = Coefs[0] + Coefs[1] * x1 + Coefs[2] * y1;
    y2_ = Coefs[3] + Coefs[4] * x1 + Coefs[5] * y1;
    break;
  }

  *ptr_x2 = x2_;
  *ptr_y2 = y2_;

  return;
}


int imsub(short *im, int ncol, int nrow, int x, int y, int h, short *sub){
int i, j;
int w;
  
  w = h * 2 + 1;
  memset(sub, 0, w*w*sizeof(short));

  if (x<h || x>ncol - h - 1 || y<h || y>nrow - h - 1)
    return 0;

  for (i = 0; i < w; i++)
  {
    for (j = 0; j < w; j++)
      sub[i*w + j] = im[(y - h + i)*ncol + (x - h + j)];
  }

  return 1;
}


double corr2(short *x, short *y, int n){
double  dSumL1 = 0, dSumL2 = 0, dSumR1 = 0, dSumR2 = 0, dSumLR = 0;
short *pSrc, *pObj;
int     i1;
double  dCoefL, dCoef1, dCoef2, dCoef;
int     MATCH_FULL_SIZE;
short src, obj;

  pSrc = x;
  pObj = y;
  MATCH_FULL_SIZE = n;
  for (i1 = 0; i1<MATCH_FULL_SIZE; i1++)
  {
    src = (*pSrc);
    obj = (*pObj);
    dSumL1 += (src);
    dSumL2 += (double)((src)*(src));
    dSumR1 += (obj);
    dSumR2 += (double)((obj)* (obj));
    dSumLR += (double)((src)* (obj));
    pSrc++;
    pObj++;
  }

  dCoefL = dSumL2 - dSumL1*dSumL1 / MATCH_FULL_SIZE;
  dCoef1 = dSumLR - dSumL1*dSumR1 / MATCH_FULL_SIZE;
  dCoef2 = dCoefL*(dSumR2 - dSumR1*dSumR1 / MATCH_FULL_SIZE);
  if (dCoef2<0.0000001 && dCoef2>-0.0000001)
    dCoef = 0.0;
  else
    dCoef = dCoef1 / sqrt(dCoef2);

  return dCoef;
}


void CalcGradient2D(float* pImg, float* pImgX, float* pImgY, int iWidth, int iHeight){
int i1, j1;
  for (i1 = 1; i1 < iHeight - 1; i1++)
  {
    for (j1 = 1; j1 < iWidth - 1; j1++)
    {
      pImgX[i1*iWidth + j1] = (pImg[i1*iWidth + j1 + 1] - pImg[i1*iWidth + j1 - 1]) / 2;
      pImgY[i1*iWidth + j1] = (pImg[(i1 + 1)*iWidth + j1] - pImg[(i1 - 1)*iWidth + j1]) / 2;
    }
  }
}


void CalcMultiply(float* pImg1, float* pImg2, float* pImgNew, int iSize){
int i1;
  for (i1 = 0; i1 < iSize; i1++)
    pImgNew[i1] = pImg1[i1] * pImg2[i1];
}


void CalcDivide(float* pImg1, float* pImg2, float* pImgNew, int iSize){
int i1;
  for (i1 = 0; i1 < iSize; i1++)
    pImgNew[i1] = (ABS(pImg2[i1]) > 0.1e-10f) ? pImg1[i1] / pImg2[i1] : 0;
}


void CalcAdd(float* pImg1, float* pImg2, float* pImgNew, int iSize){
int i1;
  for (i1 = 0; i1 < iSize; i1++)
    pImgNew[i1] = pImg1[i1] + pImg2[i1];
}


void CalcSubtract(float* pImg1, float* pImg2, float* pImgNew, int iSize){
int i1;
  for (i1 = 0; i1 < iSize; i1++)
    pImgNew[i1] = pImg1[i1] - pImg2[i1];
}


void CalcMinMaxMeanWithMask(float* pImg, small *pucMask, float *pdMin, float *pdMax, float *pdMean, int iSize){
int iValidNum;
int i1;

  *pdMin = 1000000000;
  *pdMax = -1000000000;
  *pdMean = 0;

  iValidNum = 0;

  for (i1 = 0; i1 < iSize; i1++)
  {
    if (ABS(pImg[i1]) < 0.1e-10)
      continue;
    if (pucMask[i1] == 0)
      continue;

    *pdMin = *pdMin < pImg[i1] ? *pdMin : pImg[i1];
    *pdMax = *pdMax > pImg[i1] ? *pdMax : pImg[i1];
    *pdMean += pImg[i1];
    iValidNum += 1;
  }
  if (iValidNum > 0)
    *pdMean /= iValidNum;
  else
    *pdMean = 0;

  return;
}


void AddConst(float *pBuffer, float c, int iSize){
int i;
  for (i = 0; i < iSize; i++)
    pBuffer[i] = pBuffer[i] + c;
}


float* GetGaussian(double dSigma, int *iFilterWidth){
int    i, j;
int    iWidth = (int)(3 * sqrt(8.0)*dSigma + 1e-10);
  if ((iWidth % 2) == 0)  iWidth = iWidth - 1; // to get the odd size
  *iFilterWidth = iWidth;
int    iHalfWidth = (iWidth - 1) / 2, iSize = iWidth*iWidth;

float* dBuffer = (float *)calloc(iSize, sizeof(float));
float* pdBuffer = dBuffer;
double dSum = 0;
double dRef = -0.5 / pow(dSigma, 2);

  for (i = -iHalfWidth; i <= iHalfWidth; i++)
  {
    for (j = -iHalfWidth; j <= iHalfWidth; j++)
    {
      *pdBuffer = (float)(exp(dRef * (i*i + j*j)));
      dSum += *pdBuffer;
      pdBuffer++;
    }
  }

  pdBuffer = dBuffer;
  for (i = 0; i < iSize; i++)
  {
    *pdBuffer = (float)(*pdBuffer / dSum);
    pdBuffer++;
  }
  return dBuffer;
}


void Conv2same(short *pImg, short *pImgNew, int iWidth, int iHeight, short nodata, float* dFilter, int w, int step){
int    h, start;
float  dCurSum, fsum;
int    i, j, k, l;
int k1, k2, l1, l2;

  h = (w - 1) / 2;
  start = step;
  while (start < h) start += step;

  
  memset(pImgNew, 0, iWidth*iHeight*sizeof(short));


  #pragma omp parallel private(j,dCurSum,fsum,k,l) shared(start,step,iHeight,iWidth,pImg,pImgNew,nodata,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (i=start; i<iHeight-h; i+=step){ // old: for (i = 0; i <= iHeight - w; i++)
    for (j=start; j<iWidth -h; j+=step){ // old: for (j = 0; j <= iWidth - w; j++)

      if (pImg[i*iWidth + j] == nodata) continue;

      dCurSum = 0;
      fsum = 0;
      for (k=0; k<w; k++){
      for (l=0; l<w; l++){
        if (pImg[(i + k - h)*iWidth + (j + l - h)] == nodata) continue;
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];  // old: dCurSum += pImg[(i + k)*iWidth + (j + l)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }

      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;
      pImgNew[i*iWidth + j] = (short)(dCurSum + 1e-10);

    }
    }
    
  }


  // Upper part
  #pragma omp parallel private(i,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(step,iHeight,iWidth,pImg,pImgNew,nodata,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (j=0; j<iWidth; j+=step){
    for (i=0; i<h;      i+=step){

      if (pImg[i*iWidth + j] == nodata) continue;

      dCurSum = 0;
      fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        if (pImg[(i + k - h)*iWidth + (j + l - h)] == nodata) continue;
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;
      pImgNew[i*iWidth + j] = (short)(dCurSum + 1e-10);

    }
    }
    
  }
  

  start = iHeight - h;
  while (fmod(start, step) != 0) start++;
  
  // lower part
  #pragma omp parallel private(i,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(start,step,iHeight,iWidth,pImg,pImgNew,nodata,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (j=0;     j<iWidth;  j+=step){
    for (i=start; i<iHeight; i+=step){

      if (pImg[i*iWidth + j] == nodata) continue;

      dCurSum = 0;
      fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k = k1; k<k2; k++){
      for (l = l1; l<l2; l++){
        if (pImg[(i + k - h)*iWidth + (j + l - h)] == nodata) continue;
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;
      pImgNew[i*iWidth + j] = (short)(dCurSum + 1e-10);

    }
    }
  
  }
  
  
  // left part
  #pragma omp parallel private(j,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(start,step,iHeight,iWidth,pImg,pImgNew,nodata,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (i=0; i<iHeight; i+=step){
    for (j=0; j<h;       j+=step){

      if (pImg[i*iWidth + j] == nodata) continue;

      dCurSum = 0;
      fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        if (pImg[(i + k - h)*iWidth + (j + l - h)] == nodata) continue;
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;
      pImgNew[i*iWidth + j] = (short)(dCurSum + 1e-10);

    }
    }
  
  }
  

  start = iWidth - h;
  while (fmod(start, step) != 0) start++;
  
  // right part
  #pragma omp parallel private(j,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(start,step,iHeight,iWidth,pImg,pImgNew,nodata,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (i=0;     i<iHeight; i+=step){
    for (j=start; j<iWidth;  j+=step){
      
      if (pImg[i*iWidth + j] == nodata) continue;

      dCurSum = 0;
      fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        if (pImg[(i + k - h)*iWidth + (j + l - h)] == nodata) continue;
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;
      pImgNew[i*iWidth + j] = (short)(dCurSum + 1e-10);
      
    }
    }
    
  }
  

  return;
}


void Conv2same_FLT_T(float* pImg, float* pImgNew, int iWidth, int iHeight, float* dFilter, int w){
int    h;
float  dCurSum, fsum;
int    i, j, k, l;
int k1, k2, l1, l2;

  h = (w - 1) / 2;

  memset(pImgNew, 0, iWidth*iHeight*sizeof(float));


  #pragma omp parallel private(j,dCurSum,k,l) shared(iHeight,iWidth,pImg,pImgNew,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (i=0; i<iHeight-w; i++){
    for (j=0; j<iWidth -w; j++){
      
      dCurSum = 0;
      for (k=0; k<w; k++){
        for (l = 0; l < w; l++) dCurSum += pImg[(i + k)*iWidth + (j + l)] * dFilter[k*w + l];
      }
      
      pImgNew[(i + h)*iWidth + (j + h)] = dCurSum;
      
    }
    }
  
  }
  

  // Upper part
  #pragma omp parallel private(i,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(iHeight,iWidth,pImg,pImgNew,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (j=0; j<iWidth; j++){
    for (i=0; i<h;      i++){
      
      dCurSum = 0; fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;

      pImgNew[i*iWidth + j] = dCurSum;
      
    }
    }

  }

  
  // lower part
  #pragma omp parallel private(i,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(iHeight,iWidth,pImg,pImgNew,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (j=0;         j<iWidth;  j++){
    for (i=iHeight-h; i<iHeight; i++){
      
      dCurSum = 0; fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;

      pImgNew[i*iWidth + j] = dCurSum;
      
    }
    }
    
  }
  
  
  // left part
  #pragma omp parallel private(j,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(iHeight,iWidth,pImg,pImgNew,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (i=0; i<iHeight; i++){
    for (j=0; j<h;       j++){
      
      dCurSum = 0; fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;

      pImgNew[i*iWidth + j] = dCurSum;
      
    }
    }
    
  }
  
  
  // right part
  #pragma omp parallel private(j,dCurSum,fsum,k,l,k1,k2,l1,l2) shared(iHeight,iWidth,pImg,pImgNew,w,h,dFilter) default(none)
  {

    #pragma omp for
    for (i=0;        i<iHeight; i++){
    for (j=iWidth-h; j<iWidth;  j++){
      
      dCurSum = 0;
      fsum = 0;
      k1 = (h - i)>0 ? (h - i) : 0;
      k2 = (iHeight - i)<w ? (iHeight - i) : w;
      l1 = (h - j)>0 ? (h - j) : 0;
      l2 = (iWidth - j)<w ? (iWidth - j) : w;
      for (k=k1; k<k2; k++){
      for (l=l1; l<l2; l++){
        dCurSum += pImg[(i + k - h)*iWidth + (j + l - h)] * dFilter[k*w + l];
        fsum += dFilter[k*w + l];
      }
      }
      if (ABS(fsum) < 1e-10) continue;

      dCurSum /= fsum;

      pImgNew[i*iWidth + j] = dCurSum;
      
    }
    }
    
  }
  

  return;
}


bool FindTargetValueInWindow(short *pImg, int iWidth, int iHeight, int iTargetCol, int iTargetRow, int w, short targetValue){
int Row, Col;

  for (Row = iTargetRow - w; Row <= iTargetRow + w; Row++)
  {
    for (Col = iTargetCol - w; Col <= iTargetCol + w; Col++)
    {
      if (Row < 0 || Row >= iHeight || Col < 0 || Col >= iWidth)
        continue;

      if (pImg[Row*iWidth + Col] == targetValue)
        return true;
    }
  }

  return false;
}


float GetStd(short *pshtData, int n, small *pucMask){
int i;
float fSum, fMean, fStd;
int n_valid;

  fSum = 0;
  n_valid = 0;
  for (i = 0; i<n; i++)
  {
    if (pucMask[i] > 0)
    {
      fSum += pshtData[i];
      n_valid += 1;
    }
  }

  if (n_valid == 0)
    return -1;

  fMean = fSum / n_valid;
  fSum = 0;
  for (i = 0; i<n; i++)
  {
    if (pucMask[i] > 0)
      fSum += (pshtData[i] - fMean)*(pshtData[i] - fMean);
  }

  fStd = sqrt(fSum / n_valid);

  return fStd;
}


void ApplyMask(short *pshtData, int n, small *pucMask, short shtFillValue){
int i;
  for (i = 0; i<n; i++)
  {
    if (pucMask[i] == 0)
      pshtData[i] = shtFillValue;
  }

  return;
}

