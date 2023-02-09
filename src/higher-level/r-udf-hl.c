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
#include <R_ext/Parse.h>


void rstats_label_dimensions(ard_t *ard, tsa_t *ts, int submodule, char *idx_name, int nb, int nt, par_udf_t *udf);
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


void parse_rstats(const char *string){
ParseStatus status;
SEXP cmdexpr;
int i;


  cmdexpr = PROTECT(R_ParseVector(mkString(string), -1, &status, R_NilValue));
  if (status != PARSE_OK){
    UNPROTECT(1);
    printf("error in UDF module when parsing internal string\n");
    exit(FAILURE);
  }

  for(i=0; i<length(cmdexpr); i++) eval(VECTOR_ELT(cmdexpr, i), R_GlobalEnv);
  UNPROTECT(1);

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


  printf("starting to register R interface\n");
  #ifdef FORCE_DEBUG
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


  // parse once to make sure that functions are available
  source_rstats(udf->f_code);
  findFun(install("force_rstats_init"), R_GlobalEnv);

  parse_rstats(
    "force_rstats_init_ <- function(){                       \n"
    "  dates_str <- paste(years, months, days, sep='-')      \n"
    "  dates <- as.Date(dates_str, format='%Y-%m-%d')        \n"
    "  bands <- force_rstats_init(dates, sensors, bandnames) \n"
    "  return(as.character(bands))                           \n"
    "}                                                       \n"
  );
  findFun(install("force_rstats_init_"), R_GlobalEnv); // make sure parsing worked

  R_tryEval(lang1(install("force_rstats_init")), R_GlobalEnv, NULL);


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

  if (phl->tsa.rsp.out){
    udf = &phl->tsa.rsp;
  } else if (phl->udf.rsp.out){
    udf = &phl->udf.rsp;
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

/** This function initializes the R udf
--- ard:       ARD
--- ts:        pointer to instantly useable TSA image arrays
--- submodule: HLPS submodule
--- idx_name:  name of index for TSA submodule
--- nb:        number of bands
--- nt:        number of products over time
--- udf:       user-defined code parameters
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_rsp(ard_t *ard, tsa_t *ts, int submodule, char *idx_name, int nb, int nt, par_udf_t *udf){

//PyObject *py_fun      = NULL;
//PyObject *py_return   = NULL;
//PyObject *py_bandname = NULL;
//PyObject *py_encoded  = NULL;
SEXP bandnames;
char *bandname = NULL;
date_t date;
int b;


  printf("starting to initialize R interface\n");
  #ifdef FORCE_DEBUG
  #endif

  //make sure bandnames and dates are NULL-initialized
  udf->bandname = NULL;
  udf->date     = NULL;

  if (!udf->out){
    udf->nb = 1;
    return;
  }

printf("here?\n");
  rstats_label_dimensions(ard, ts, submodule, idx_name, nb, nt, udf);
printf("here2?\n");

  PROTECT(bandnames = R_tryEval(lang1(install("force_rstats_init_")), R_GlobalEnv, NULL));

  R_tryEval(lang2(install("print"), bandnames), R_GlobalEnv, NULL);


  UNPROTECT(1);



/**

  if (py_return == Py_None){
    printf("None returned from forcepy_init_. Check the R UDF code!\n");
    exit(FAILURE);}

  if (!PyList_Check(py_return)){
    printf("forcepy_init_ did not return a list. Check the R UDF code!\n");
    exit(FAILURE);}


  udf->nb = (int)PyList_Size(py_return);

  alloc_2D((void***)&udf->bandname, udf->nb, NPOW_10, sizeof(char));
  alloc((void**)&udf->date, udf->nb, sizeof(date_t));

  for (b=0; b<udf->nb; b++){

    py_bandname = PyList_GetItem(py_return, b);
    py_encoded  = PyUnicode_AsEncodedString(py_bandname, "UTF-8", "strict");
    if ((bandname = PyBytes_AsString(py_encoded)) == NULL){
      printf("forcepy_init_ did not return a list of strings. Check the R UDF code!\n");
      exit(FAILURE);}
    Py_DECREF(py_encoded);

    copy_string(udf->bandname[b], NPOW_10, bandname);
    
    date_from_bandname(&date, bandname);
    copy_date(&date, &udf->date[b]);

    #ifdef FORCE_DEBUG
    printf("bandname # %d: %s\n", b, udf->bandname[b]);
    print_date(&udf->date[b]);
    #endif

  }

  **/

  #ifdef FORCE_DEBUG
  printf("finished to initialize R interface\n");
  #endif

  return;
}


/** This function terminates the R udf
--- udf:    user-defined code parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void term_rsp(par_udf_t *udf){


  #ifdef FORCE_DEBUG
  printf("starting to terminate R interface\n");
  #endif

  if (udf->bandname != NULL){
    free_2D((void**)udf->bandname, udf->nb); 
    udf->bandname = NULL;
  }

  if (udf->date != NULL){
    free((void*)udf->date); 
    udf->date = NULL;
  }

  #ifdef FORCE_DEBUG
  printf("finished to terminate R interface\n");
  #endif

  return;
}


/** This function labels the dimension of the UDF input data (time, band, sensor)
--- ard:       ARD
--- ts:        pointer to instantly useable TSA image arrays
--- submodule: HLPS submodule
--- idx_name:  name of index for TSA submodule
--- nb:        number of bands
--- nt:        number of products over time
--- udf:       user-defined code parameters
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void rstats_label_dimensions(ard_t *ard, tsa_t *ts, int submodule, char *idx_name, int nb, int nt, par_udf_t *udf){
int b, t;
date_t date;
char sensor[NPOW_04];
char bandname[NPOW_10];

SEXP dim_nt, dim_nb;
SEXP years, months, days;
SEXP sensors, bandnames;


  PROTECT(dim_nt = allocVector(INTSXP, 1));
  INTEGER(dim_nt)[0] = nt;
  defineVar(install("nt"), dim_nt, R_GlobalEnv);
  UNPROTECT(1);

  PROTECT(dim_nb = allocVector(INTSXP, 1));
  INTEGER(dim_nb)[0] = nb;
  defineVar(install("nb"), dim_nb, R_GlobalEnv);
  UNPROTECT(1);

  PROTECT(years     = allocVector(INTSXP, nt));
  PROTECT(months    = allocVector(INTSXP, nt));
  PROTECT(days      = allocVector(INTSXP, nt));
  PROTECT(sensors   = allocVector(STRSXP, nt));
  PROTECT(bandnames = allocVector(STRSXP, nb));


  // copy C data to R objects
  if (submodule == _HL_UDF_){

    for (t=0; t<nt; t++){
      date = get_brick_date(ard[t].DAT, 0);
      INTEGER(years)[t]  = date.year;
      INTEGER(months)[t] = date.month;
      INTEGER(days)[t]   = date.day;
      get_brick_sensor(ard[t].DAT, 0, sensor, NPOW_04);
      SET_STRING_ELT(sensors, t, mkChar(sensor));

    }

    for (b=0; b<nb; b++){
      get_brick_bandname(ard[0].DAT, b, bandname, NPOW_10);
      SET_STRING_ELT(bandnames, b, mkChar(bandname));
    }

  } else if (submodule == _HL_TSA_){

    for (t=0; t<nt; t++){
      INTEGER(years)[t]  = ts->d_tsi[t].year;
      INTEGER(months)[t] = ts->d_tsi[t].month;
      INTEGER(days)[t]   = ts->d_tsi[t].day;
      SET_STRING_ELT(sensors, t, mkChar("BLENDED"));
    }

    SET_STRING_ELT(bandnames, 0, mkChar(idx_name));

  } else {
    printf("unknown submodule. ");
    exit(FAILURE);
  }

  defineVar(install("years"),     years,     R_GlobalEnv);
  defineVar(install("months"),    months,    R_GlobalEnv);
  defineVar(install("days"),      days,      R_GlobalEnv);
  defineVar(install("sensors"),   sensors,   R_GlobalEnv);
  defineVar(install("bandnames"), bandnames, R_GlobalEnv);
  UNPROTECT(5);


  return;
}

