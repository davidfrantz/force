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
Parsing parameter header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef PARAM_CL_H
#define PARAM_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h> // boolean data type
#include <string.h>  // string handling functions
#include <limits.h>  // macro constants of the integer types

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/dir-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char   name[NPOW_10];
  bool   set;
  int    type;
  int    *length;
  int    *int_,    **int_vec_;
  float  *float_,  **float_vec_;
  double *double_, **double_vec_;
  date_t *date_,   **date_vec_;
  char   **char_,  ***char_vec_;
  int n_enums;
  tagged_enum_t *enums;
  int    int_range[2];
  float  float_range[2];
  double double_range[2];
  date_t date_range[2];
  int expected_length;
  int char_test;
} par_t;

typedef struct {
  int   n;
  int   nmax;
  par_t *par;
  char  *log;
} params_t;

int length_par(const char *buf);
date_t parse_date(char *str);
int parse_enum(char *str, tagged_enum_t *enums, int n_enums);
params_t *allocate_params();
void reallocate_params(params_t *params);
void free_params(params_t *params);
void allocate_par(par_t *par);
void free_par(par_t *par);
int prescreen_par(FILE *fpar, const char *tag, int *num);
void register_int_par(params_t *params, const char *name, int min, int max, int *ptr);
void register_enum_par(params_t *params, const char *name, const tagged_enum_t *enums, int n_enums, int *ptr);
void register_float_par(params_t *params, const char *name, float min, float max, float *ptr);
void register_double_par(params_t *params, const char *name, double min, double max, double *ptr);
void register_bool_par(params_t *params, const char *name, int *ptr);
void register_date_par(params_t *params, const char *name, const char *min, const char *max, date_t *ptr);
void register_char_par(params_t *params, const char *name, int char_test, char **ptr);
void register_intvec_par(params_t *params, const char *name, int min, int max, int expected_length, int **ptr, int *ptr_length);
void register_enumvec_par(params_t *params, const char *name, const tagged_enum_t *enums, int n_enums, int expected_length, int **ptr, int *ptr_length);
void register_floatvec_par(params_t *params, const char *name, float min, float max, int expected_length, float **ptr, int *ptr_length);
void register_doublevec_par(params_t *params, const char *name, double min, double max, int expected_length, double **ptr, int *ptr_length);
void register_boolvec_par(params_t *params, const char *name, int expected_length, int **ptr, int *ptr_length);
void register_datevec_par(params_t *params, const char *name, const char *min, const char *max, int expected_length, date_t **ptr, int *ptr_length);
void register_charvec_par(params_t *params, const char *name, int char_test, int expected_length, char ***ptr, int *ptr_length);
void parse_parameter(params_t *params, const char *buf);
void print_parameter(params_t *params);
void log_parameter(params_t *params);
int check_int_par(par_t *par);
int check_enum_par(par_t *par);
int check_float_par(par_t *par);
int check_double_par(par_t *par);
int check_date_par(par_t *par);
int check_char_par(par_t *par);
int check_parameter(params_t *params);

#ifdef __cplusplus
}
#endif

#endif

