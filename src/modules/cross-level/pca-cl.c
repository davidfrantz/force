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
This file contains functions for principal components analysis
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "pca-cl.h"

/** GNU Scientific Library (GSL) **/
#include "gsl/gsl_blas.h"
#include "gsl/gsl_eigen.h"


/** Compute Principal Components
+++ This function computes Principal Components. The input data may be in-
+++ complete, a nodata value must be given. The PCs can be truncated using
+++ a percenatge of total variance.
--- INP:    input image
--- mask_:  mask image
--- nb:     number of bands
--- nc:     number of cells
--- nodata: nodata value
--- minvar: amount of retained variance [0...1]
--- newnb:  number of PC bands (returned)
+++ Return: PC rotated data 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float **pca(short **INP, small *mask_, int nb, int nc, short nodata, float minvar, int *newnb){
int p, k, nk, b;
double *mean = NULL;
float totalvar = 0, cumvar = 0, pctvar;
int numcomp = nb;
bool *NODATA = NULL;
float **PCA  = NULL;
gsl_matrix *GINP = NULL;
gsl_matrix *GPCA = NULL;
gsl_matrix *covm = NULL;
gsl_matrix *evec = NULL;
gsl_vector *eval = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  // compile nodata image for computing PCA with valld data only
  alloc((void**)&NODATA, nc, sizeof(bool));
  alloc((void**)&mean,   nb, sizeof(double));

  for (p=0, nk=0; p<nc; p++){

    if (mask_ != NULL && !mask_[p]){
      NODATA[p] = true; continue;}

    for (b=0; b<nb; b++){
      if (INP[b][p] == nodata){
        NODATA[p] = true;
      } else {
        mean[b] += INP[b][p];
      }
    }

    if (!NODATA[p]) nk++;
  }

  for (b=0; b<nb; b++) mean[b] /= nk;

  #ifdef FORCE_DEBUG
  printf("number of cells %d, number of valid cells %d\n", nc, nk);
  #endif


  // allocate GSL matrices for original and projected data
  GINP = gsl_matrix_alloc(nk, nb);
  GPCA = gsl_matrix_alloc(nk, nb);

  // allocate covariance matrix, eigen-values and eigen-vectors
  covm = gsl_matrix_calloc(nb, nb);
  eval = gsl_vector_alloc(nb);
  evec = gsl_matrix_alloc(nb, nb);


  // center each band around mean
  for (b=0; b<nb; b++){

    for (p=0, k=0; p<nc; p++){
      if (!NODATA[p]) gsl_matrix_set(GINP, k++, b, INP[b][p]-mean[b]);
    }

    #ifdef FORCE_DEBUG
    printf("mean band %d: %f\n", b, mean[b]);
    #endif

  }
  
  free((void*)mean);


  // compute covariance matrix and scale
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, GINP, GINP, 0, covm);
  gsl_matrix_scale(covm, 1.0/(double)(nk - 1));

  #ifdef FORCE_DEBUG
  int bb;
  printf("Covariance Matrix:\n");
  for (b=0;  b<nb;  b++){
  for (bb=0; bb<nb; bb++){
    printf("%8.2f ", gsl_matrix_get(covm,b,bb));
    if (bb==nb-1) printf("\n");
  }
  }
  #endif


  // find eigen-values and eigen-vectors
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(nb);
  gsl_eigen_symmv(covm, eval, evec, w);
  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_VAL_DESC);

  #ifdef FORCE_DEBUG
  printf("Eigen values:\n");
  for (b=0; b<nb; b++) printf("%10.4f ", gsl_vector_get(eval,b));
  printf("\n\nEigen Vector Matrix Values:\n");
  for (b=0;  b<nb;  b++){
  for (bb=0; bb<nb; bb++){
    printf("%8.5f ", gsl_matrix_get(evec,b,bb));
    if (bb==nb-1) printf("\n");
  }
  }
  #endif


  // find how many components to keep
  if (minvar < 1){
    for (b=0; b<nb; b++) totalvar += gsl_vector_get(eval,b);
    for (b=0; b<nb; b++){
      cumvar += gsl_vector_get(eval,b);
      pctvar = cumvar/totalvar;
      if (pctvar > minvar){
        numcomp = b+1;
        break;
      }
    }
  } else numcomp = nb;

  #ifdef FORCE_DEBUG
  printf("%d components are retained\n", numcomp);
  #endif


  // allocate projected and truncated data
  alloc_2D((void***)&PCA, numcomp, nc, sizeof(float));

  // project original data to principal components
  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, GINP, evec, 0.0, GPCA);

  // restructure data
  for (b=0; b<numcomp; b++){
    for (p=0, k=0; p<nc; p++){
      if (NODATA[p]){ 
        PCA[b][p] = nodata;
      } else {
        PCA[b][p] = gsl_matrix_get(GPCA, k++, b);
      }
    }
  }


  // clean
  gsl_vector_free(eval);
  gsl_matrix_free(covm);
  gsl_matrix_free(evec);
  gsl_matrix_free(GINP);
  gsl_matrix_free(GPCA);
  free((void*)NODATA);

  #ifdef FORCE_CLOCK
  proctime_print("computing PCA", TIME);
  #endif


  *newnb = numcomp;
  return PCA;
}

