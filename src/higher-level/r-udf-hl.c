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


void init_snowfall(int n);
void parse_rstats(const char *string);
void rstats_label_dimensions(ard_t *ard, tsa_t *ts, int submodule, char *idx_name, int nb, int nt, par_udf_t *udf);


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/



/** This function initializes a snowfall cluster in R
--- n:      number of CPUs
+++ Return: void
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


/** This function parses an R file from disc (UDF)
--- fname:  filename of R UDF
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void source_rstats(const char *fname){


  R_tryEval(lang2(install("source"), mkString(fname)), R_GlobalEnv, NULL);

  return;
}


/** This function parses a string containing R code
--- string:  string
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
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


  // parse once to make sure that functions are available
  source_rstats(udf->f_code);

  // wrapper for 'force_rstats_init'
  findFun(install("force_rstats_init"), R_GlobalEnv);
  parse_rstats(
    "force_rstats_init_ <- function(){                             \n"
    "  dates_str <- paste(years, months, days, sep='-')            \n"
    "  dates <- as.Date(dates_str, format='%Y-%m-%d')              \n"
    "  bands <- force_rstats_init(dates, sensors, bandnames)       \n"
    "  if (class(bands) == 'Date') bands <- format(bands, '%Y%m%d')\n"
    "  return(as.character(bands))                                 \n"
    "}                                                             \n"
  );
  findFun(install("force_rstats_init_"), R_GlobalEnv); // make sure parsing worked


  if (udf->type == _UDF_PIXEL_){
    // create snowfall cluster
    init_snowfall(phl->cthread);
    // wrapper for 'force_rstats_pixel'
    findFun(install("force_rstats_pixel"), R_GlobalEnv);
    parse_rstats(
      "force_rstats_ <- function(array){                                     \n"
      "  dates_str <- paste(years, months, days, sep='-')                    \n"
      "  dates <- as.Date(dates_str, format='%Y-%m-%d')                      \n"
      "  array <- replace(array, array == na_value, NA)                      \n"
      "  if (ncpu > 1){                                                      \n"
      "   result <- sfApply(array, c(3,4), force_rstats_pixel,               \n"
      "                     dates, sensors, bandnames, 1)                    \n"
      "  } else {                                                            \n"
      "   result <- apply(array, c(3,4), force_rstats_pixel,                 \n"
      "                     dates, sensors, bandnames, 1)                    \n"
      "  }                                                                   \n"
      "  result <- replace(result, is.na(result), na_value)                  \n"
      "  storage.mode(result) <- 'integer'                                   \n"
      "  return(result)                                                      \n"
      "}                                                                     \n"
    );
    findFun(install("force_rstats_"), R_GlobalEnv); // make sure parsing worked
    // wrapper fun
   } else if (udf->type == _UDF_BLOCK_){
    // wrapper for 'force_rstats_block'
    findFun(install("force_rstats_block"), R_GlobalEnv);
    parse_rstats(
      "force_rstats_ <- function(array){                                     \n"
      "  dates_str <- paste(years, months, days, sep='-')                    \n"
      "  dates <- as.Date(dates_str, format='%Y-%m-%d')                      \n"
      "  array <- replace(array, array == na_value, NA)                      \n"
      "  result <- force_rstats_block(array, dates, sensors, bandnames, ncpu)\n"
      "  result <- replace(result, is.na(result), na_value)                  \n"
      "  storage.mode(result) <- 'integer'                                   \n"
      "  return(result)                                                      \n"
      "}                                                                     \n"
    );
    findFun(install("force_rstats_"), R_GlobalEnv); // make sure parsing worked
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
SEXP bandnames;
date_t date;
int b;


  #ifdef FORCE_DEBUG
  printf("starting to initialize R interface\n");
  #endif

  //make sure bandnames and dates are NULL-initialized
  udf->bandname = NULL;
  udf->date     = NULL;

  if (!udf->out){
    udf->nb = 1;
    return;
  }


  rstats_label_dimensions(ard, ts, submodule, idx_name, nb, nt, udf);


  PROTECT(bandnames = R_tryEval(lang1(install("force_rstats_init_")), R_GlobalEnv, NULL));

  if (isNull(bandnames) || length(bandnames) < 1){
    printf("no bandnames returnded (NULL). Check R UDF code!\n");
    exit(FAILURE);
  } 

  //R_tryEval(lang2(install("print"), bandnames), R_GlobalEnv, NULL);

  udf->nb = length(bandnames);
  alloc_2D((void***)&udf->bandname, udf->nb, NPOW_10, sizeof(char));
  alloc((void**)&udf->date, udf->nb, sizeof(date_t));

  for (b=0; b<udf->nb; b++){

    copy_string(udf->bandname[b], NPOW_10, CHAR(STRING_ELT(bandnames, b)));

    date_from_string(&date, udf->bandname[b]);
    copy_date(&date, &udf->date[b]);

    #ifdef FORCE_DEBUG
    printf("bandname # %d: %s\n", b, udf->bandname[b]);
    print_date(&udf->date[b]);
    #endif

  }

  UNPROTECT(1);


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

/** This function connects FORCE to plug-in R UDFs
--- ard:       pointer to instantly useable ARD image arrays
--- udf:       pointer to instantly useable UDF image arrays
--- ts:        pointer to instantly useable TSA image arrays
--- mask:      mask image
--- submodule: HLPS submodule
--- idx_name:  name of index for TSA submodule
--- nx:        number of columns
--- ny:        number of rows
--- nc:        number of cells
--- nb:        number of bands
--- nt:        number of time steps
--- nodata:    nodata value
--- p_udf:     user-defined code parameters
--- cthread:   number of computing threads
+++ Return:    SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int rstats_udf(ard_t *ard, udf_t *udf_, tsa_t *ts, small *mask_, int submodule, char *idx_name, int nx, int ny, int nc, int nb, int nt, short nodata, par_udf_t *udf, int cthread){
int b, t, i, j, p;
size_t k;
int *array_ = NULL, value;
SEXP na_value, ncpu;
SEXP dim, array;
SEXP rstats_return;
int *rstats_return_ = NULL;


  if (submodule == _HL_UDF_ && udf_->rsp_ == NULL) return CANCEL;
  if (submodule == _HL_TSA_ &&   ts->rsp_ == NULL) return CANCEL;


  #ifdef FORCE_DEBUG
  printf("starting to run R interface\n");
  #endif

  rstats_label_dimensions(ard, ts, submodule, idx_name, nb, nt, udf);


  PROTECT(na_value = allocVector(INTSXP, 1));
  INTEGER(na_value)[0] = nodata;
  defineVar(install("na_value"), na_value, R_GlobalEnv);
  UNPROTECT(1); // na_value

  PROTECT(ncpu = allocVector(INTSXP, 1));
  INTEGER(ncpu)[0] = cthread;
  defineVar(install("ncpu"), ncpu, R_GlobalEnv);
  UNPROTECT(1); // ncpu


  PROTECT(dim = allocVector(INTSXP, 4));
  INTEGER(dim)[0] = nt;
  INTEGER(dim)[1] = nb;
  INTEGER(dim)[2] = ny;
  INTEGER(dim)[3] = nx;

  PROTECT(array = allocArray(INTSXP, dim));
  UNPROTECT(1); // dim
  array_ = INTEGER(array);

  // copy C data to R objects

  k = 0;

  for (j=0; j<nx; j++){
  for (i=0; i<ny; i++){
  for (b=0; b<nb; b++){
  for (t=0; t<nt; t++){

    p = nx*i + j;

    if (submodule == _HL_UDF_){
      if (!ard[t].msk[p]){
        value = nodata;
      } else {
        value = ard[t].dat[b][p];
      }
    } else if (submodule == _HL_TSA_){
      value = ts->tsi_[t][p];
    }

    array_[k++] = value;

  }
  }
  }
  }


  // fire up R
  PROTECT(rstats_return = R_tryEval(lang2(install("force_rstats_"), array), R_GlobalEnv, NULL));

  // copy to output brick
  rstats_return_ = INTEGER(rstats_return);

  k =  0;

  for (j=0; j<nx; j++){
  for (i=0; i<ny; i++){
  for (b=0; b<udf->nb; b++){

      p = nx*i + j;

      if (submodule == _HL_UDF_){
        udf_->rsp_[b][p] = rstats_return_[k++];
      } else if (submodule == _HL_TSA_){
        ts->rsp_[b][p] = rstats_return_[k++];
      }

  }
  }
  }

  UNPROTECT(2); // array, rstats_return


  #ifdef FORCE_DEBUG
  printf("finished to run R interface\n");
  #endif

  return SUCCESS;
}

