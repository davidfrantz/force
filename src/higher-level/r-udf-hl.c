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
This file contains functions for plugging-in R UDFs into FORCE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "r-udf-hl.h"

#include <Rinternals.h>
#include <Rembedded.h>
//#include <Python.h>
//#include <numpy/ndarrayobject.h>
//#include <numpy/ndarraytypes.h>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


/**typedef struct {
  npy_intp dim_nt[1];
  npy_intp dim_nb[1];
  PyArrayObject* year;
  PyArrayObject* month;
  PyArrayObject* day;
  PyArrayObject* sensor;
  PyArrayObject* bandname;
  PyArray_Descr *desc_sensor;
  PyArray_Descr *desc_bandname;
} py_dimlab_t;**/


//py_dimlab_t python_label_dimensions(ard_t *ard, tsa_t *ts, int submodule, char *idx_name, int nb, int nt, par_udf_t *udf);
//int date_from_bandname(date_t *date, char *bandname);

/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


void init_snowfall(int n){
SEXP parallel;
SEXP ncpu;


  PROTECT(parallel = allocVector(LGLSXP, 1));
  LOGICAL(parallel)[0] = true;

  PROTECT(ncpu = allocVector(REALSXP, 1));
  REAL(ncpu)[0] = n;

  R_tryEval(lang2(install("library"), mkString("snowfall")), R_GlobalEnv, NULL);
  R_tryEval(lang3(install("sfInit"), parallel, ncpu), R_GlobalEnv, NULL);

  //R_tryEval(lang2(install("print"), parallel), R_GlobalEnv, NULL);
  //R_tryEval(lang2(install("print"), ncpu),     R_GlobalEnv, NULL);

  UNPROTECT(2);

  return;
}


void source_rstats(const char *fname){


  R_tryEval(lang2(install("source"), mkString(fname)), R_GlobalEnv, NULL);

  return;
}


/** This function initializes the R interpreter, and defines a 
+++ function for wrapping the UDF code
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_rstats(par_hl_t *phl){
par_udf_t *udf;
int r_argc = 2;
char *r_argv[] = { "R", "--silent" };


  #ifdef FORCE_DEBUG
  printf("starting to register R interface\n");
  #endif

  // choose module
  if (phl->tsa.rsp.out){
    udf = &phl->tsa.rsp;
  } else if (phl->udf.rsp.out){
    udf = &phl->udf.rsp;
  } else {
    return;
  }


  Rf_initEmbeddedR(r_argc, r_argv);
  // make functions available?
  // todo

  source_rstats(udf->f_code);

  findFun(install("force_rstats_init"), R_GlobalEnv);


  if (udf->type == _UDF_PIXEL_){
    init_snowfall(phl->cthread);
    findFun(install("force_rstats_pixel"), R_GlobalEnv);
    // wrapper fun
   } else if (udf->type == _UDF_BLOCK_){
    findFun(install("force_rstats_block"), R_GlobalEnv);
    // wrapper fun
  } else {
    printf("unknown UDF type.\n"); 
    exit(FAILURE);
  }

  #ifdef FORCE_DEBUG
  printf("finished to register R interface\n");
  #endif

  return;
}


/** This function cleans up the R interpreter
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void deregister_rstats(par_hl_t *phl){
par_udf_t *udf;


  #ifdef FORCE_DEBUG
  printf("starting to deregister R interface\n");
  #endif

  if (phl->tsa.pyp.out){
    udf = &phl->tsa.pyp;
  } else if (phl->udf.pyp.out){
    udf = &phl->udf.pyp;
  } else {
    return;
  }


  if (udf->type == _UDF_PIXEL_){
     R_tryEval(lang1(install("sfStop")), R_GlobalEnv, NULL);
  }

  if (udf->out) Rf_endEmbeddedR(0);


  #ifdef FORCE_DEBUG
  printf("finished to deregister R interface\n");
  #endif

  return;
}

