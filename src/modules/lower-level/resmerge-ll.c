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

/** The STARFM implementation included in this file, is adopted from the
+++ version provided provided by the United States Department of Agricul-
+++ ture, Agricultural Research Service:
+++ https://www.ars.usda.gov/research/software/download/?softwareid=432
+++ STARFM Copyright (C) 2005-2014 Feng Gao
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ STARFM License:
+++ The package is now being released at no cost to the public for further
+++ development. ARS is releasing the STARFM so that interested parties 
+++ can use the software tool for their own needs and purposes. ARS does 
+++ not foresee providing monetary or technical support to refine, adapt, 
+++ or use this software tool, and provides no warranty for its use for 
+++ any purpose. ARS does not reserve any rights or interests in the work 
+++ that may be performed by others to refine or adapt it. ARS does reser-
+++ ve the right to continue its own refinement of the current version of 
+++ the software package at a later date, should program needs require it.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions to enhance spatial resolution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "resmerge-ll.h"

/** GNU Scientific Library (GSL) **/
#include <gsl/gsl_multifit.h>          // multi-parameter fitting


/** StarFM candidate struct
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

typedef struct {
  int dx, dy;     // distance
  float coarse;   // coarse data
  float fine;     // fine data
  float r_coarse; // relative difference between two coarse obs.
  float r_fine;   // relative difference between coarse and fine obs.
  float C;        // reversed distance
} starfm_candidate_t;


int resolution_merge_1(brick_t *TOA, brick_t *QAI);
int resolution_merge_2(brick_t *TOA, brick_t *QAI);
int resolution_merge_3(brick_t *TOA, brick_t *QAI);
double rescale_weight_resmerge(double weight, double minweight, double maxweight);
short **improphe_resmerge(int nx, int ny, brick_t *QAI, short **toa_, float **KDIST, int h, int nb_m, int nb_c, int *bands_m, int *bands_c, int nk, int mink);
float *starfm(int nx, int ny, brick_t *QAI, float **coarse, float **fine, int r);


/** Sentinel-2 resolution merge option 1
+++ This function enhances the spatial resolution of the 20m Sentinel-2
+++ bands using a multi-parameter regression of the general form: y = c X,
+++ where y is a vector of n observations (20m band), X is an n-by-p mat-
+++ rix of predictor variables (intercept, green, red, NIR bands), c are
+++ p regression coefficients, i.e. y = c0 + c1 GREEN + c2 RED + c3 NIR.
+++ A least squares fit to a linear model is used by minimizing the cost 
+++ function chi^2 (sum of squares of the residuals from the best-fit). 
+++ The best-fit is found by singular value decomposition of the matrix X
+++ using the modified Golub-Reinsch SVD algorithm, with column scaling to
+++ improve the accuracy of the singular values. Any components which have
+++ zero singular value (to machine precision) are discarded from the fit.
+++ A moving kernel of size n = 5*5 is used based on a sensitivity study 
+++ of Haß et al. (in preparation).
--- TOA:    TOA reflectance (will be altered)
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS / FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int resolution_merge_1(brick_t *TOA, brick_t *QAI){
int b = 0, nb = 6, bands[6], green, red, bnir;
int i, j, p, ii, jj, ni, nj, np, nx, ny;
int r = 2, w, nw, k, nv = 4;
gsl_matrix *X, **cov;
gsl_vector *x, **y, **c;
gsl_multifit_linear_workspace **work;
double chisq, est, err;
short **toa_ = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  nx  = get_brick_ncols(QAI);
  ny  = get_brick_nrows(QAI);

  if ((toa_ = get_bands_short(TOA)) == NULL) return FAILURE;

  if ((green = find_domain(TOA, "GREEN"))    < 0) return FAILURE;
  if ((red   = find_domain(TOA, "RED"))      < 0) return FAILURE;
  if ((bnir  = find_domain(TOA, "BROADNIR")) < 0) return FAILURE;
  
  if ((bands[b++] = find_domain(TOA, "REDEDGE1")) < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "REDEDGE2")) < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "REDEDGE3")) < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "NIR"))      < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "SWIR1"))    < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "SWIR2"))    < 0) return FAILURE;

  // kernel size
  w = r*2+1; nw = w*w;


  #pragma omp parallel private(k,b,j,p,ii,jj,ni,nj,np,X,x,y,c,cov,work,chisq,est,err) shared(r,w,nb,nw,nv,ny,nx,QAI,toa_,bands,green,red,bnir) default(none)
  {

    /** initialize and allocate
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

    // nw-by-nv predictor variables; kernel + central pixel
    X = gsl_matrix_calloc(nw, nv);
    x = gsl_vector_calloc(nv);
    
    // set first column of X to 1 -> intercept c0
    for (k=0; k<nw; k++) gsl_matrix_set(X, k, 0, 1.0);
    gsl_vector_set(x, 0, 1.0);

    // vector of nw observations
    alloc((void**)&y, nb, sizeof(gsl_vector*));
    for (b=0; b<nb; b++) y[b] = gsl_vector_calloc(nw);

    // nv regression coefficients
    alloc((void**)&c, nb, sizeof(gsl_vector*));
    for (b=0; b<nb; b++) c[b] = gsl_vector_calloc(nv);

    // nv-by-nv covariance matrix
    alloc((void**)&cov, nb, sizeof(gsl_matrix*));
    for (b=0; b<nb; b++) cov[b] = gsl_matrix_calloc(nv, nv);

    // workspace
    alloc((void**)&work, nb, sizeof(gsl_multifit_linear_workspace*));
    for (b=0; b<nb; b++) work[b] = gsl_multifit_linear_alloc(nw, nv);


    /** do regression for every valid pixel, and for each 20m band
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

    #pragma omp for schedule(guided)  
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      p = i*nx+j;

      if (get_off(QAI, p) || get_cloud(QAI, p) > 0 || get_shadow(QAI, p)) continue;

      // add central pixel
      gsl_vector_set(x, 1, toa_[green][p]/10000.0);
      gsl_vector_set(x, 2, toa_[red][p]/10000.0);
      gsl_vector_set(x, 3, toa_[bnir][p]/10000.0);

      k = 0;

      // add neighboring pixels
      for (ii=-r; ii<=r; ii++){
      for (jj=-r; jj<=r; jj++){

        ni = i+ii; nj = j+jj;
        if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;
        np = ni*nx+nj;

        if (get_off(QAI, np)) continue;

        gsl_matrix_set(X, k, 1, toa_[green][np]/10000.0);
        gsl_matrix_set(X, k, 2, toa_[red][np]/10000.0);
        gsl_matrix_set(X, k, 3, toa_[bnir][np]/10000.0);
        
        for (b=0; b<nb; b++) gsl_vector_set(y[b], k, toa_[bands[b]][np]/10000.0);

        k++;

      }
      }

      // append zeros, if less than nw neighboring pixels were added
      while (k < nw){
        gsl_matrix_set(X, k, 1, 0.0);
        gsl_matrix_set(X, k, 2, 0.0);
        gsl_matrix_set(X, k, 3, 0.0);
        for (b=0; b<nb; b++) gsl_vector_set(y[b], k, 0.0);
        k++;
      }

      // solve model, and predict central pixel
      for (b=0; b<nb; b++){

        gsl_multifit_linear(X, y[b], c[b], cov[b], &chisq, work[b]);
        gsl_multifit_linear_est(x, c[b], cov[b], &est, &err);
        toa_[bands[b]][p] = (short)(est*10000);

      }

    }
    }


    /** clean
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    gsl_matrix_free (X); gsl_vector_free (x);
    for (b=0; b<nb; b++) gsl_vector_free(y[b]); 
    for (b=0; b<nb; b++) gsl_vector_free (c[b]); 
    for (b=0; b<nb; b++) gsl_matrix_free (cov[b]); 
    for (b=0; b<nb; b++) gsl_multifit_linear_free(work[b]); 
    free((void*)y);      free((void*)c);
    free((void*)cov);    free((void*)work);

  }

  #ifdef FORCE_DEBUG
  set_brick_filename(TOA, "TOA-RESMERGED");
  print_brick_info(TOA); set_brick_open(TOA, OPEN_CREATE); write_brick(TOA);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("Resolution merge", TIME);
  #endif
  
  return SUCCESS;
}


/** Sentinel-2 resolution merge option 2
+++ This function enhances the spatial resolution of the 20m Sentinel-2
+++ bands. The red, green and BNIR bands are used. In principle, the 
+++ ImproPhe code is used, but is parameterized using a spectral in-
+++ stead of a temporal setup. The texture metrics are disabled as they 
+++ work badly with a fine/coarse resolution factor of 2. This version is
+++ faster than resolution_merge_1.
--- TOA:    TOA reflectance (will be altered)
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS / FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int resolution_merge_2(brick_t *TOA, brick_t *QAI){
int mb = 0, cb = 0, nb_m = 3, nb_c = 6;
int bands_m[3], bands_c[6];
int p, nx, ny, nc;
float **KDIST = NULL; // kernel distance
int ksize = 5, nk, mink, h; // number of kernel pixels, and minimum number of pixels for good prediction, radius
short **toa_  = NULL;
short **pred_ = NULL;

  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  nx  = get_brick_ncols(QAI);
  ny  = get_brick_nrows(QAI);
  nc  = get_brick_ncells(QAI);

  if ((toa_ = get_bands_short(TOA)) == NULL) return FAILURE;
  
  if ((bands_m[mb++] = find_domain(TOA, "GREEN"))    < 0) return FAILURE;
  if ((bands_m[mb++] = find_domain(TOA, "RED"))      < 0) return FAILURE;
  if ((bands_m[mb++] = find_domain(TOA, "BROADNIR")) < 0) return FAILURE;

  if ((bands_c[cb++] = find_domain(TOA, "REDEDGE1")) < 0) return FAILURE;
  if ((bands_c[cb++] = find_domain(TOA, "REDEDGE2")) < 0) return FAILURE;
  if ((bands_c[cb++] = find_domain(TOA, "REDEDGE3")) < 0) return FAILURE;
  if ((bands_c[cb++] = find_domain(TOA, "NIR"))      < 0) return FAILURE;
  if ((bands_c[cb++] = find_domain(TOA, "SWIR1"))    < 0) return FAILURE;
  if ((bands_c[cb++] = find_domain(TOA, "SWIR2"))    < 0) return FAILURE;
  
  
  
  // pre-compute kernel distances
  // determine minimum # of neighbors that make a good prediction
  if (distance_kernel(ksize, &KDIST) != SUCCESS){
    printf("error in generating kernel. "); return FAILURE;}
  nk = ksize*ksize;
  h   = (ksize-1)/2;
  mink = 3;//(int)((nk*M_PI/4.0)*0.5/100.0);
  #ifdef FORCE_DEBUG
  printf("ksize/nkernel/minnum: %d/%d/%d\n", ksize, nk, mink);
  #endif


  // predict fine data
  pred_ = improphe_resmerge(nx, ny, QAI, toa_, KDIST,
                            h, nb_m, nb_c, bands_m, bands_c, nk, mink);

  // update toa with prediction
  for (p=0; p<nc; p++){
    if (get_off(QAI, p) || get_cloud(QAI, p) > 0 || get_shadow(QAI, p)) continue;
    for (cb=0; cb<nb_c; cb++) toa_[bands_c[cb]][p] = pred_[cb][p];
  }

  #ifdef FORCE_DEBUG
  set_brick_filename(TOA, "TOA-RESMERGED");
  print_brick_info(TOA); set_brick_open(TOA, OPEN_CREATE); write_brick(TOA);
  #endif


  free_2D((void**)KDIST, ksize);
  free_2D((void**)pred_, nb_c);

  #ifdef FORCE_CLOCK
  proctime_print("Pansharpening", TIME);
  #endif
  
  return SUCCESS;
}


/** Sentinel-2 resolution merge option 3
+++ This function enhances the spatial resolution of the 20m Sentinel-2
+++ bands. A pseudo-PAN (average of green/red) and the BNIR are used.
+++ Coarse and fine image pairs are generated by spatially degrading the
+++ PANs using an approximated Point Spread Function. In principle, STARFM
+++ is used but is parameterized using a spectral instead of a temporal 
+++ setup. Prediction artefcats were observed at object boundaries, but 
+++ this version is faster than resolution_merge_1.
--- TOA:    TOA reflectance (will be altered)
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS / FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int resolution_merge_3(brick_t *TOA, brick_t *QAI){
int b = 0, b_, nb = 6, bands[6], green, red, bnir;
int i, j, p, ii, jj, ni, nj, np, nx, ny, nc;
int nk = 5;
double sum[2], num;
float **coarse = NULL;
float **fine   = NULL;
float  *pred   = NULL;
float **kernel = NULL;
short **toa_ = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  nx  = get_brick_ncols(QAI);
  ny  = get_brick_nrows(QAI);
  nc  = get_brick_ncells(QAI);

  if ((toa_ = get_bands_short(TOA)) == NULL) return FAILURE;
  
  if ((green = find_domain(TOA, "GREEN"))    < 0) return FAILURE;
  if ((red   = find_domain(TOA, "RED"))      < 0) return FAILURE;
  if ((bnir  = find_domain(TOA, "BROADNIR")) < 0) return FAILURE;

  if ((bands[b++] = find_domain(TOA, "REDEDGE1")) < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "REDEDGE2")) < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "REDEDGE3")) < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "NIR"))      < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "SWIR1"))    < 0) return FAILURE;
  if ((bands[b++] = find_domain(TOA, "SWIR2"))    < 0) return FAILURE;

  alloc_2D((void***)&coarse, 3, nc, sizeof(float));
  alloc_2D((void***)&fine,   2, nc, sizeof(float));

  for (p=0; p<nc; p++){
    fine[0][p] = (toa_[green][p]/10000.0+toa_[red][p]/10000.0)/2.0;
    fine[1][p] = toa_[bnir][p]/10000.0;
  }


  /** Convolute PANs with gaussian kernel **/

  if (gauss_kernel(nk, 0.5, &kernel) != SUCCESS){
    printf("Could not generate kernel. "); return FAILURE;}

  for (i=0, p=0; i<ny; i++){
  for (j=0; j<nx; j++, p++){
    
    if (get_off(QAI, p)) continue;

    sum[0] = sum[1] = num = 0;

    for (ii=0; ii<nk; ii++){
    for (jj=0; jj<nk; jj++){

      ni = -(nk-1)/2 + ii + i; nj = -(nk-1)/2 + jj + j;
      if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;
      np = ni*nx+nj;
      
      if (get_off(QAI, np)) continue;

      sum[0] += (toa_[green][np]/10000.0+toa_[red][np]/10000.0)/2.0*kernel[ii][jj];
      sum[1] += toa_[bnir][np]/10000.0*kernel[ii][jj];
      num += kernel[ii][jj];

    }
    }

    if (num > 0){
      coarse[0][p] = sum[0]/num;
      coarse[1][p] = sum[1]/num;
    }

  }
  }

  free_2D((void**)kernel, nk);
  
  #ifdef FORCE_DEBUG
  //db_write_image_2D((void**)coarse,  "PANSHARP-GAUSS", 2, nx, ny, sizeof(float), 4);
  #endif


  /** Reduce spatial resolution, supersample to target resolution **/

  for (i=0, p=0; i<ny; i+=2){
  for (j=0; j<nx; j+=2, p++){
    
    if (get_off(QAI, p)) continue;

    sum[0] = sum[1] = num = 0;

    for (ii=0; ii<2; ii++){
    for (jj=0; jj<2; jj++){

      ni = i+ii; nj = j+jj;
      if (ni >= ny || nj >= nx) continue;
      np = ni*nx+nj;

      if (get_off(QAI, np)) continue;

      sum[0] += coarse[0][np];
      sum[1] += coarse[1][np];
      num++;

    }
    }

    if (num > 0){

      for (ii=0; ii<2; ii++){
      for (jj=0; jj<2; jj++){

        ni = i+ii; nj = j+jj;
        if (ni >= ny || nj >= nx) continue;
        np = ni*nx+nj;

        if (get_off(QAI, np)) continue;

        coarse[0][np] = sum[0]/num;
        coarse[1][np] = sum[1]/num;

      }
      }

    }

  }
  }

  #ifdef FORCE_DEBUG
  //db_write_image_2D((void**)coarse,  "PANSHARP-COARSE", 2, nx, ny, sizeof(float), 4);
  //db_write_image_2D((void**)fine,    "PANSHARP-FINE",   2, nx, ny, sizeof(float), 4);
  #endif


  /** Predict at fine resolution **/

  for (b_=0; b_<nb; b_++){
    
    b = bands[b_];
    
    for (p=0; p<nc; p++) coarse[2][p] = toa_[b][p]/10000.0;

    pred = starfm(nx, ny, QAI, coarse, fine, 1);

    for (p=0; p<nc; p++) toa_[b][p] = (short)(pred[p]*10000);

    free((void*)pred);

  }

  #ifdef FORCE_DEBUG
  set_brick_filename(TOA, "TOA-RESMERGED");
  print_brick_info(TOA); set_brick_open(TOA, OPEN_CREATE); write_brick(TOA);
  #endif

  free_2D((void**)coarse, 3);
  free_2D((void**)fine,   2);

  #ifdef FORCE_CLOCK
  proctime_print("Pansharpening", TIME);
  #endif
  
  return SUCCESS;
}


/** Rescale neighbor pixel weights
+++ This function rescales the neighbor pixel weights (R --> R').
--- weight:    weight R
--- minweight: minimum weight within kernel
--- maxweight: maximum weight within kernel
+++ Return:    rescaled weight R'
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double rescale_weight_resmerge(double weight, double minweight, double maxweight){
double normweight;

  if (weight == 0){
    normweight = 1;
  } else if (minweight == maxweight){
    normweight = 1;
  } else {
    normweight = 1+exp(25 * 
      ((weight-minweight)/(maxweight-minweight))-7.5);
  }

  return normweight;
}


/** Predict Sentinel-2 10m bands (ImproPhe method)
+++ This function is the same as improphe, but deals with Sentinel-2 data
+++ directly. This avoids copying the 16bit TOA reflectance to FLOAT va-
+++ lues, which would significantly increase RAM requirements. Texture is
+++ also disabled as it does not make sense for a 2:1 ratio of spatial re-
+++ solution.
--- nx:      number of X-pixels (3x3 mosaic)
--- ny:      number of Y-pixels (3x3 mosaic)
--- QAI:     Quality Assurance Information
--- TOA:     TOA reflectance
--- KDIST:   kernel distance
--- h:       prediction radius
--- nb_m:    number of bands in MR data
--- nb_c:    number of bands in CR data
--- bands_m: MR bands to use
--- bands_c: CR bands to predict
--- nk:      number of kernel pixels, and 
--- mink:    minimum number of pixels for good prediction
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short **improphe_resmerge(int nx, int ny, brick_t *QAI, short **toa_, float **KDIST, int h, int nb_m, int nb_c, int *bands_m, int *bands_c, int nk, int mink){
int i, j, p;
int ki, kj; // iterator for KDIST
int ii, jj; // iterator for kernel relative to center
int ni, nj, np; // pixel positions
int b, f, s; // iterator for bands, CR parameters, cutoff-threshold
int wn, k; // iterator for neighbours


double S, SS; // spectral similarity + rescaled S
double W;     // pixel weight

// S, T, U, C of all valid neighbours
double *Srecord, **Crecord;
// S, T, U range of all valid neighbours
double Srange[2][5];

double *weightxdata = NULL, weight;

double sum;          // temporary variables to calculate S
float Sthr[5] = { 0.0025, 0.005, 0.01, 0.02, 0.025 }; // cutoff-threshold for S
int Sn[5];  // number of neighbours for each cutoff-threshold
int ns = 5, Sc, *Sclass;             // index for cutoff-threshold

short **pred_ = NULL;



  alloc_2D((void***)&pred_, nb_c, nx*ny, sizeof(short));
  
  #pragma omp parallel private(b, f, s, k, j, p, ii, jj, ni, nj, np, ki, kj, sum, S, SS, W, Sc, Sclass, Srecord, weightxdata, Crecord, Sn, Srange, weight, wn) shared(nx, ny, h, ns, nb_m, nb_c, bands_m, bands_c, nk, mink, Sthr, pred_, toa_, KDIST, QAI) default(none)
  {

    alloc((void**)&Sclass, nk, sizeof(int));
    alloc((void**)&Srecord, nk, sizeof(double));
    alloc((void**)&weightxdata, nb_c, sizeof(double));
    alloc_2DC((void***)&Crecord, nb_c, nk, sizeof(double));

    #pragma omp for schedule(guided)    
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      p = i*nx+j;

      // if nodata or cloud/shadow: skip
      if (get_off(QAI, p) || get_cloud(QAI, p) > 0 || get_shadow(QAI, p)) continue;


      // initialize arrays
      memset(Sclass,  0, nk*sizeof(int));
      memset(Srecord, 0, nk*sizeof(double));
      memset(Crecord[0], 0, nb_c*nk*sizeof(double));
      memset(weightxdata, 0, nb_c*sizeof(double));
      memset(Sn, 0, ns*sizeof(int));
      for (s=0; s<ns; s++){ Srange[0][s] = INT_MAX; Srange[1][s] = INT_MIN;}
      weight = 0.0; wn = 0;


      /** for each neighbour **/
      for (ii=-h, ki=0; ii<=h; ii++, ki++){
      for (jj=-h, kj=0; jj<=h; jj++, kj++){

        ni  = i+ii; nj = j+jj;
        if (ni < 0 || nj < 0 || ni >= ny || nj >= nx){
          continue;}
        np = nx*ni+nj;
        
        // if not in circular kernel, skip
        if (KDIST[ki][kj] > h) continue;

        // if nodata, skip
        if (get_off(QAI, np) || get_cloud(QAI, np) > 0 || get_shadow(QAI, np)) continue;

        // specral distance (MAE)
        for (b=0, sum=0; b<nb_m; b++) sum += fabs(((float)(toa_[bands_m[b]][p]-toa_[bands_m[b]][np]))/10000.0);
        S = sum/(float)nb_m;

        // get cutoff-threshold index for S
        for (s=0, Sc=-1; s<ns; s++){
          if (S < Sthr[s]){ Sc = s; break;}
        }
        if (Sc < 0) continue;
        Sclass[wn] = Sc;

        // keep S + determine range
        Srecord[wn] = S;
        if (S > 0){
          for (s=Sc; s<ns; s++){
            if (S > Srange[1][s]) Srange[1][s] = S;
            if (S < Srange[0][s]) Srange[0][s] = S;
          }
        }

        // keep CR data
        for (f=0; f<nb_c; f++) Crecord[f][wn] = ((float)toa_[bands_c[f]][np])/10000.0;

        wn++; // number of valid neighbours
        for (s=Sc; s<ns; s++) Sn[s]++; // number of valid neighbours in S-class

      }
      }

      // if no valid neighbour... damn.. use CR
      if (wn == 0){
        for (f=0; f<nb_c; f++) pred_[f][p] = toa_[bands_c[f]][p];
        continue;
      }

      // determine the spectral similarity cutoff threshold
      for (s=0, Sc=-1; s<ns; s++){
        if (Sn[s] >= mink){ Sc = s; break;}
      }
      if (Sc < 0) Sc = ns-1;


      // compute pixel weight
      for (k=0; k<wn; k++){

        if (Sclass[k] > Sc) continue;

        SS = rescale_weight_resmerge(Srecord[k], Srange[0][Sc], Srange[1][Sc]);

        W = 1/SS;
        for (f=0; f<nb_c; f++) weightxdata[f] += W*Crecord[f][k];
        weight += W;

      }

      // prediction -> weighted mean
      for (f=0; f<nb_c; f++) pred_[f][p] = (short)((weightxdata[f]/weight)*10000);
      
    }
    }

    // tidy up things
    free((void*)Sclass);
    free((void*)Srecord);
    free((void*)weightxdata);
    free_2DC((void**)Crecord);
    
  }

  return pred_;
}


/** STARFM
+++ This function is a streamlined version of the STARFM code using two
+++ image pairs of coarse/fine data and one coarse image that should be
+++ prediction at the fine scale.
+++-----------------------------------------------------------------------
+++ Gao, F., Masek, J., Schwaller, M., & Hall, F. (2006). On the Blending 
+++ of the Landsat and MODIS Surface Reflectance: Predicting Daily Landsat
+++ Surface Reflectance. IEEE TGRS, 44, 2207-2218.
+++-----------------------------------------------------------------------
--- nx:      number of X-pixels (3x3 mosaic)
--- ny:      number of Y-pixels (3x3 mosaic)
--- QAI:     Quality Assurance Information
--- coarse: coarse resolution data (3 bands, 1-2 base pairs, 3 prediction)
--- fine:   fine data (2 bands, i.e. base pairs)
--- r:      prediction radius in pixels
+++ Return: SUCCESS / FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *starfm(int nx, int ny, brick_t *QAI, float **coarse, float **fine, int r){
int i, j, p, ii, jj, ni, nj, np, c, ncan;
float diff;
float d, D;     // spatial distance, relative spatial distance
float T, S, TS; // temporal, spectral, and combined difference
float C_sum;    // sum of combined inversed weight
float W;        // normalized weight
float min_r_coarse, min_r_fine; // min. of rel. differences of central pix
float unc_coarse_1, unc_coarse_2, unc_fine; // uncertainty coarse and fine
float unc_r_coarse, unc_r_fine;             // uncertainty rel. difference
float unc_all;                              // combined uncertainty
starfm_candidate_t *can = NULL; // candidate pixel struct
float *prediction = NULL; // prediction

#ifdef FORCE_DEBUG
float *percentsample = NULL; // percentage of valid candidates
#endif

int pair, npair = 2;

double sumx, sumx2, num;
float slice_value[2], std, nclass = 40;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  /** compute uncertainty **/

  unc_coarse_1 = unc_coarse_2 = unc_fine  = 0.002;
  unc_r_coarse = sqrt(unc_coarse_1*unc_coarse_1 + unc_coarse_2*unc_coarse_2);
  unc_r_fine = sqrt(unc_coarse_1*unc_coarse_1 + unc_fine*unc_fine);
  unc_all = sqrt(unc_r_coarse*unc_r_coarse +  unc_r_fine* unc_r_fine);

  #ifdef FORCE_DEBUG
  printf("uncertainty coarse:   %.4f\n", unc_r_coarse);
  printf("uncertainty fine:     %.4f\n", unc_r_fine);
  printf("uncertainty combined: %.4f\n", unc_all);
  #endif


  /** compute slice values for spectal similarity **/

  for (pair=0; pair<npair; pair++){

    sumx = sumx2 = num = 0.0;

    for (i=0, p=0; i<ny; i++){
    for (j=0; j<nx; j++, p++){
      
      if (get_off(QAI, p)) continue;

      sumx  += fine[pair][p];
      sumx2 += fine[pair][p]*fine[pair][p];
      num++;

    }
    }

    std = sqrt(sumx2/num - (sumx/num) * (sumx/num));
    slice_value[pair] = std * 2.0 / nclass;
    
    #ifdef FORCE_DEBUG
    printf("std %.4f, slice value: %.4f (fine #%d)\n", std, slice_value[pair], pair);
    #endif

  }


  /** predict fine resolution data **/

  alloc((void**)&can, (r*2+1)*(r*2+1)*npair+1, sizeof(starfm_candidate_t));
  alloc((void**)&prediction, nx*ny, sizeof(float));
  #ifdef FORCE_DEBUG
  alloc((void**)&percentsample, nx*ny, sizeof(float));
  #endif

  for (i=0, p=0; i<ny; i++){
  for (j=0; j<nx; j++, p++){

    if (get_off(QAI, p)) continue;


    /** add original fine observation to candidate list (Index 0) **/

    c = 0;
    can[c].dx = 0;
    can[c].dy = 0;
    can[c].coarse = coarse[npair][p];
    c++;


    /** add base pairs of central pixel to candidates and
    +++ estimate minimum relative differences **/

    min_r_coarse = min_r_fine = SHRT_MAX;

    for (pair=0; pair<npair; pair++){

      can[c].dx = 0;
      can[c].dy = 0;
      can[c].coarse = coarse[pair][p];
      can[c].fine   = fine[pair][p];

      can[c].r_coarse = coarse[pair][p] - coarse[npair][p];
      can[c].r_fine   = coarse[pair][p] - fine[pair][p];

      if (fabs(can[c].r_coarse) < min_r_coarse) min_r_coarse = fabs(can[c].r_coarse);
      if (fabs(can[c].r_fine)   < min_r_fine)   min_r_fine   = fabs(can[c].r_fine);
      c++;

    }


    /** add all base observations within prediction kernel to candidates
    +++ if they are more similar than the minimum relative difference
    +++ at the central pixel. This includes the central pixel (reset index
    +++ to 1). **/

    c = 1;

    for (ii=-r; ii<=r; ii++){
    for (jj=-r; jj<=r; jj++){

      ni = i+ii; nj = j+jj;
      if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;
      np = ni*nx+nj;

      if (get_off(QAI, np)) continue;
      
      for (pair=0; pair<npair; pair++){

        diff = fabs(fine[pair][np] - fine[pair][p]);
        if (diff > slice_value[pair]) continue;

        can[c].dx = jj;
        can[c].dy = ii;
        can[c].coarse = coarse[pair][np];
        can[c].fine   = fine[pair][np];

        can[c].r_coarse = coarse[pair][np] - coarse[npair][np];
        can[c].r_fine   = coarse[pair][np] - fine[pair][np];

        if (fabs(can[c].r_coarse) < (min_r_coarse+unc_r_coarse) ||
            fabs(can[c].r_fine)   < (min_r_fine+unc_r_fine)) c++;

      }

    }
    }


    /** predict fine data from candidates based on 
    +++ spatial distance,
    +++ spectral distance (coarse-fine difference of input pairs) and 
    +++ 'temporal distance' (coarse-coarse difference of one base to
    +++ target image).
    +++ If there is no candidate, use coarse data. **/

    C_sum = 0.0;
    
    if ((ncan = c) > 1){

      for (c=1; c<ncan; c++){

          // location distance to central pixel
          d = sqrt(can[c].dx*can[c].dx + can[c].dy*can[c].dy);

          // temporal, spectral and combined distance
          // add a small uncertainty to avoid zero in T or S (avoid a 
          // biased large weight)

          T = fabs(can[c].r_coarse) + 0.0001;
          S = fabs(can[c].r_fine)   + 0.0001;
          TS = T*S;


          // use max weight if very close or compute combined inverted weight
          if (TS < unc_all){
            can[c].C = 1.0;
          } else {
            D = 1.0+d/r;
            can[c].C = 1.0/(TS*D);
          }

          // sum of all inverted weights
          C_sum += can[c].C;

      }

      for (c=1; c<ncan; c++){

        // normalized weight
        W = can[c].C/C_sum;

        // predict fine data using weighting function
        prediction[p] += W * (can[c].fine - can[c].r_coarse);

        #ifdef FORCE_DEBUG
        percentsample[p] = (ncan/((r*2+1)*(r*2+1)*npair));
        #endif

      }
      
    } else {

      // retain coarse data if no suitable candidate
      prediction[p] = can[0].coarse;

    }


  }
  }

  free((void*)can);

  #ifdef FORCE_DEBUG
  //db_write_image_1D((void**)percentsample,  "PANSHARP-PSAM", nx, ny, sizeof(float), 4);
  free((void*)percentsample);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("STARFM", TIME);
  #endif

  return prediction;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** Sentinel-2 resolution merge controller
+++ This function controlls the Sentinel-2 resolution merging of the 20m 
+++ Sentinel-2 bands. Based on parameterization, different methods are 
+++ used. The function will exit gracefully if not Sentinel-2.
--- mission: mission ID
--- resmerge: resolution merge option
--- TOA:      TOA reflectance (will be altered)
--- QAI:      Quality Assurance Information
+++ Return:   SUCCESS / FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int resolution_merge(int mission, int resmerge, brick_t *TOA, brick_t *QAI){


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  if (mission != SENTINEL2) return SUCCESS;
  
  //cite_me(_CITE_RESMERGE_);

  
  switch (resmerge){
    case _RES_MERGE_NONE_:
      return SUCCESS;
    case _RES_MERGE_REGRESSION_:
      cite_me(_CITE_REGRESSION_);
      if (resolution_merge_1(TOA, QAI) != SUCCESS){
        printf("unable to merge resolutions (1). "); return FAILURE;}
      break;
    case _RES_MERGE_IMPROPHE_:
      cite_me(_CITE_IMPROPHE_);
      if (resolution_merge_2(TOA, QAI) != SUCCESS){
        printf("unable to merge resolutions (2). "); return FAILURE;}
      break;
    case _RES_MERGE_STARFM_:
      cite_me(_CITE_STARFM_);
      if (resolution_merge_3(TOA, QAI) != SUCCESS){
        printf("unable to merge resolutions (3). "); return FAILURE;}
      break;
    default:
      printf("unknown resolution merge option. "); return FAILURE;
  }
  

  #ifdef FORCE_CLOCK
  proctime_print("resolution merge", TIME);
  #endif

  return SUCCESS;
}

