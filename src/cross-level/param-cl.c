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
This file contains functions for parsing parameter files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "param-cl.h"


/** Number of values
+++ This function takes a parameter in tag and multi-value notation, and
+++ returns the number of values.
--- buf:   buffer that holds tag / value line as read from parameter file
+++ Return: number of values
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int length_par(const char *buf){
char buffer[NPOW_10];
char *ptr = NULL;
const char *separator = " =";
int n = -1; // start at -1 to ignore tag


  copy_string(buffer, NPOW_10, buf);

  buffer[strcspn(buffer, "\r\n#")] = 0;

  ptr = strtok(buffer, separator);

  while (ptr != NULL){
    ptr = strtok(NULL, separator);
    n++;
  }

  return n;
}


/** Parse sequence of char strings
+++ This function takes a parameter in tag and multi-value string notation
--- tag:   parameter tag
--- str:   tag/values string
--- param: parsed parameters (returned)
--- n:     number of parameters (returned)
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_char_seq(const char *tag, char *str, char param[NPOW_10][NPOW_10], int *n){
int num = 0;
char *ptr = NULL;
const char *separator = " =\n";


  ptr = strtok(str, separator);

  while ((ptr = strtok(NULL, separator)) != NULL){
    copy_string(param[num++], NPOW_10, ptr);}

  *n = num;
  return true;
}


/** Parse date value
+++ This function takes a string in YYYYMMDD notation and converts to a
+++ date.
--- str:   string
+++ Return: date
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
date_t parse_date(char *str){
char cy[5], cm[3], cd[3];
int y, m, d;
date_t date;


  strncpy(cy, str,   4); cy[4] = '\0'; y = atoi(cy);
  strncpy(cm, str+5, 2); cm[2] = '\0'; m = atoi(cm);
  strncpy(cd, str+8, 2); cd[2] = '\0'; d = atoi(cd);

  init_date(&date);
  set_date(&date, y, m, d);

  return date;
}


/** Parse enum value
+++ This function takes a string, and compares to an enum defintion. The
+++ enum is returned. If no match was found, INT_MAX is returned.
--- str:     string
--- enums:   enum definition
--- n_enums: number of enums in enum definition
+++ Return:  enum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_enum(char *str, tagged_enum_t *enums, int n_enums){
int e;


  for (e=0; e<n_enums; e++){
    if (strcmp(str, enums[e].tag) == 0) return enums[e].en;
  }

  return INT_MAX;
}


/** This function allocates the parsed parameters
+++ Return: parsed parameters (must be freed with free_params)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
params_t *allocate_params(){
params_t *params = NULL;


  alloc((void**)&params, 1, sizeof(params_t));
  params->n    = 0;
  params->nmax = NPOW_04;

  alloc((void**)&params->par, params->nmax, sizeof(par_t));
  alloc((void**)&params->log, NPOW_14,      sizeof(char));

  return params;
}


/** If necessary, this function increases the allocation for the parsed
+++ parameters
--- params: parsed parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void reallocate_params(params_t *params){


  if (params->n < params->nmax) return;

  re_alloc((void**)&params->par, params->nmax, params->nmax*2, sizeof(par_t));

  params->nmax *= 2;

  return;
}


/** This function frees the parsed parameters
--- params: parsed parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_params(params_t *params){
int i;


  if (params != NULL){

    if (params->par != NULL){
      for (i=0; i<params->n; i++) free_par(&params->par[i]);
      free((void*)params->par);
    }
    if (params->log != NULL){
      free((void*)params->log);
    }

    free((void*)params); params = NULL;

  }

  return;
}


/** This function allocates a parameter vector
--- par:    parameter (must be freed with free_par)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void allocate_par(par_t *par){


  switch (par->type){
    case _PAR_INT_:
      alloc((void**)&par->int_vec_[0], *par->length, sizeof(int));
      break;
    case _PAR_ENUM_:
      alloc((void**)&par->int_vec_[0], *par->length, sizeof(int));
      break;
    case _PAR_FLOAT_:
      alloc((void**)&par->float_vec_[0], *par->length, sizeof(float));
      break;
    case _PAR_DOUBLE_:
      alloc((void**)&par->double_vec_[0], *par->length, sizeof(double));
      break;
    case _PAR_BOOL_:
      alloc((void**)&par->int_vec_[0], *par->length, sizeof(int));
      break;
    case _PAR_DATE_:
      alloc((void**)&par->date_vec_[0], *par->length, sizeof(date_t));
      break;
    case _PAR_CHAR_:
      alloc_2D((void***)&par->char_vec_[0], *par->length, NPOW_10, sizeof(char));
      break;
    default:
      printf("unknown datatype for par..\n");
      break;
  }

  return;
}


/** This function frees a parameter vector
--- par:    parameter
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_par(par_t *par){


  switch (par->type){
    case _PAR_INT_:
      if (par->length != NULL) free((void*)par->int_vec_[0]);
      break;
    case _PAR_ENUM_:
      free((void*)par->enums);
      if (par->length != NULL) free((void*)par->int_vec_[0]);
      break;
    case _PAR_FLOAT_:
      if (par->length != NULL) free((void*)par->float_vec_[0]);
      break;
    case _PAR_DOUBLE_:
      if (par->length != NULL) free((void*)par->double_vec_[0]);
      break;
    case _PAR_BOOL_:
      free((void*)par->enums);
      if (par->length != NULL) free((void*)par->int_vec_[0]);
      break;
    case _PAR_DATE_:
      if (par->length != NULL) free((void*)par->date_vec_[0]);
      break;
    case _PAR_CHAR_:
      if (par->length != NULL){
        free_2D((void**)par->char_vec_[0], *par->length);
      } else {
        free((void*)par->char_[0]);
      }
      break;
    default:
      printf("unknown datatype for par..\n");
      break;
  }

  return;
}


/** This function pre-screens a parameter file, and counts how often a
--- given tag was specified
--- fpar:   parameter filepath
--- tag:    parameter tag
--- num:    number of instances of parameter tag (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int prescreen_par(FILE *fpar, const char *tag, int *num){
char buffer[NPOW_10];
char *ptr = NULL;
const char *separator = " =";
int n = 0;


  while (fgets(buffer, NPOW_10, fpar) != NULL){
    buffer[strcspn(buffer, "\r\n#")] = 0;
    if ((ptr = strtok(buffer, separator)) == NULL) continue;
    if (strcmp(ptr, tag) == 0) n++;
  }
  fseek(fpar, 0, SEEK_SET);

  if (n == 0){
    printf("pre-screening parfile failed. No tag %s was detected. ", tag);
    return FAILURE;
  }

  *num = n;
  return SUCCESS;
}


/** This function registers an integer parameter scalar
--- params: parsed parameters
--- name:   parameter name (tag)
--- min:    minimum valid value for parameter
--- max:    maximum valid value for parameter
--- ptr:    pointer to instantly useable parameter variable
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_int_par(params_t *params, const char *name, int min, int max, int *ptr){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].int_range[_MIN_] = min;
  params->par[params->n].int_range[_MAX_] = max;

  params->par[params->n].type = _PAR_INT_;
  params->par[params->n].length = NULL;

  params->par[params->n].int_ = ptr;

  params->n++;

  return;
}


/** This function registers an enum parameter scalar
--- params:  parsed parameters
--- name:    parameter name (tag)
--- enums:   enum definition
--- n_enums: number of enums in enum definition
--- ptr:     pointer to instantly useable parameter variable
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_enum_par(params_t *params, const char *name, const tagged_enum_t *enums, int n_enums, int *ptr){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].n_enums = n_enums;
  alloc((void**)&params->par[params->n].enums, n_enums, sizeof(tagged_enum_t));
  memmove(params->par[params->n].enums, enums, sizeof(tagged_enum_t)*n_enums);

  params->par[params->n].type = _PAR_ENUM_;
  params->par[params->n].length = NULL;

  params->par[params->n].int_ = ptr;

  params->n++;

  return;
}


/** This function registers a float parameter scalar
--- params: parsed parameters
--- name:   parameter name (tag)
--- min:    minimum valid value for parameter
--- max:    maximum valid value for parameter
--- ptr:    pointer to instantly useable parameter variable
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_float_par(params_t *params, const char *name, float min, float max, float *ptr){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].float_range[_MIN_] = min;
  params->par[params->n].float_range[_MAX_] = max;

  params->par[params->n].type = _PAR_FLOAT_;
  params->par[params->n].length = NULL;

  params->par[params->n].float_ = ptr;

  params->n++;

  return;
}


/** This function registers a double parameter scalar
--- params: parsed parameters
--- name:   parameter name (tag)
--- min:    minimum valid value for parameter
--- max:    maximum valid value for parameter
--- ptr:    pointer to instantly useable parameter variable
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_double_par(params_t *params, const char *name, double min, double max, double *ptr){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].double_range[_MIN_] = min;
  params->par[params->n].double_range[_MAX_] = max;

  params->par[params->n].type = _PAR_DOUBLE_;
  params->par[params->n].length = NULL;

  params->par[params->n].double_ = ptr;

  params->n++;

  return;
}


/** This function registers a bool parameter scalar
--- params: parsed parameters
--- name:   parameter name (tag)
--- ptr:    pointer to instantly useable parameter variable
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_bool_par(params_t *params, const char *name, int *ptr){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].n_enums = 2;
  alloc((void**)&params->par[params->n].enums, 2, sizeof(tagged_enum_t));

  copy_string(params->par[params->n].enums[0].tag, NPOW_04, "FALSE");
  copy_string(params->par[params->n].enums[1].tag, NPOW_04, "TRUE");

  params->par[params->n].enums[0].en = false;
  params->par[params->n].enums[1].en = true;

  params->par[params->n].type = _PAR_BOOL_;
  params->par[params->n].length = NULL;

  params->par[params->n].int_ = ptr;

  params->n++;

  return;
}


/** This function registers a date parameter scalar
--- params: parsed parameters
--- name:   parameter name (tag)
--- min:    minimum valid date for parameter
--- max:    maximum valid date for parameter
--- ptr:    pointer to instantly useable parameter variable
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_date_par(params_t *params, const char *name, const char *min, const char *max, date_t *ptr){
char cmin[NPOW_10];
char cmax[NPOW_10];


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  copy_string(cmin, NPOW_10, min);
  copy_string(cmax, NPOW_10, max);

  params->par[params->n].date_range[_MIN_] =  parse_date(cmin);
  params->par[params->n].date_range[_MAX_] =  parse_date(cmax);

  params->par[params->n].type = _PAR_DATE_;
  params->par[params->n].length = NULL;

  params->par[params->n].date_ = ptr;

  params->n++;

  return;
}


/** This function registers a string parameter scalar
--- params:    parsed parameters
--- name:      parameter name (tag)
--- char_test: test type for string evaluation
--- ptr:       pointer to instantly useable parameter variable
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_char_par(params_t *params, const char *name, int char_test, char **ptr){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].char_test = char_test;

  params->par[params->n].type = _PAR_CHAR_;
  params->par[params->n].length = NULL;

  params->par[params->n].char_ = ptr;
  alloc((void**)&params->par[params->n].char_[0], NPOW_10, sizeof(char));

  params->n++;

  return;
}


/** This function registers an integer parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- min:        minimum valid value for parameter
--- max:        maximum valid value for parameter
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_intvec_par(params_t *params, const char *name, int min, int max, int **ptr, int *ptr_length){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].int_range[_MIN_] = min;
  params->par[params->n].int_range[_MAX_] = max;

  params->par[params->n].type = _PAR_INT_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].int_vec_ = ptr;

  params->n++;

  return;
}


/** This function registers an enum parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- enums:      enum definition
--- n_enums:    number of enums in enum definition
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_enumvec_par(params_t *params, const char *name, const tagged_enum_t *enums, int n_enums, int **ptr, int *ptr_length){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].n_enums = n_enums;
  alloc((void**)&params->par[params->n].enums, n_enums, sizeof(tagged_enum_t));
  memmove(params->par[params->n].enums, enums, sizeof(tagged_enum_t)*n_enums);

  params->par[params->n].type = _PAR_ENUM_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].int_vec_ = ptr;

  params->n++;

  return;
}


/** This function registers a float parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- min:        minimum valid value for parameter
--- max:        maximum valid value for parameter
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_floatvec_par(params_t *params, const char *name, float min, float max, float **ptr, int *ptr_length){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].float_range[_MIN_] = min;
  params->par[params->n].float_range[_MAX_] = max;

  params->par[params->n].type = _PAR_FLOAT_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].float_vec_ = ptr;

  params->n++;

  return;
}


/** This function registers a double parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- min:        minimum valid value for parameter
--- max:        maximum valid value for parameter
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_doublevec_par(params_t *params, const char *name, double min, double max, double **ptr, int *ptr_length){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].double_range[_MIN_] = min;
  params->par[params->n].double_range[_MAX_] = max;

  params->par[params->n].type = _PAR_DOUBLE_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].double_vec_ = ptr;

  params->n++;

  return;
}


/** This function registers a bool parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_boolvec_par(params_t *params, const char *name, int **ptr, int *ptr_length){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].n_enums = 2;
  alloc((void**)&params->par[params->n].enums, 2, sizeof(tagged_enum_t));

  copy_string(params->par[params->n].enums[0].tag, NPOW_04, "FALSE");
  copy_string(params->par[params->n].enums[1].tag, NPOW_04, "TRUE");

  params->par[params->n].enums[0].en = false;
  params->par[params->n].enums[1].en = true;

  params->par[params->n].type = _PAR_BOOL_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].int_vec_ = ptr;

  params->n++;

  return;
}


/** This function registers a date parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- min:        minimum valid date for parameter
--- max:        maximum valid date for parameter
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_datevec_par(params_t *params, const char *name, const char *min, const char *max, date_t **ptr, int *ptr_length){
char cmin[NPOW_10];
char cmax[NPOW_10];


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  copy_string(cmin, NPOW_10, min);
  copy_string(cmax, NPOW_10, max);

  params->par[params->n].date_range[_MIN_] =  parse_date(cmin);
  params->par[params->n].date_range[_MAX_] =  parse_date(cmax);

  params->par[params->n].type = _PAR_DATE_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].date_vec_ = ptr;

  params->n++;

  return;
}


/** This function registers a string parameter vector
--- params:     parsed parameters
--- name:       parameter name (tag)
--- char_test:  test type for string evaluation
--- ptr:        pointer to instantly useable parameter variable
--- ptr_length: pointer to instantly useable parameter variable (holding n)
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_charvec_par(params_t *params, const char *name, int char_test, char ***ptr, int *ptr_length){


  reallocate_params(params);

  copy_string(params->par[params->n].name, NPOW_10, name);

  params->par[params->n].set = false;

  params->par[params->n].char_test = char_test;

  params->par[params->n].type = _PAR_CHAR_;

  params->par[params->n].length = ptr_length;
  *params->par[params->n].length = 0;

  params->par[params->n].char_vec_ = ptr;

  params->n++;

  return;
}


/** This function parses a parameter. One line of the parameter file is
+++ compared to all registered parameters
--- params: parsed parameters
--- buf:    buffer that holds tag / value line as read from parameter file
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void parse_parameter(params_t *params, const char *buf){
int i, n;
char buffer[NPOW_10];
char *tag = NULL;
char *ptr = NULL;
const char *separator = " =";


  copy_string(buffer, NPOW_10, buf);
  buffer[strcspn(buffer, "\r\n#")] = 0;

  ptr = strtok(buffer, separator);
  tag = ptr;

  if (tag == NULL) return;

  for (i=0; i<params->n; i++){

    if (params->par[i].set) continue;

    if (strcmp(tag, params->par[i].name) == 0){

      if ((ptr = strtok(NULL, separator)) == NULL) return;

      params->par[i].set = true;

      // scalar
      if (params->par[i].length == NULL){

        switch (params->par[i].type){
          case _PAR_INT_:
            *params->par[i].int_ = atoi(ptr);
            break;
          case _PAR_ENUM_:
            *params->par[i].int_ = parse_enum(ptr, params->par[i].enums, params->par[i].n_enums);
            break;
          case _PAR_FLOAT_:
            *params->par[i].float_ = atof(ptr);
            break;
          case _PAR_DOUBLE_:
            *params->par[i].double_ = atof(ptr);
            break;
          case _PAR_BOOL_:
            *params->par[i].int_ = parse_enum(ptr, params->par[i].enums, params->par[i].n_enums);
            break;
          case _PAR_DATE_:
            *params->par[i].date_ = parse_date(ptr);
            break;
          case _PAR_CHAR_:
            copy_string(*params->par[i].char_, NPOW_10, ptr);
            break;
          default:
            printf("unknown datatype for par..\n");
            break;
        }

      // vector
      } else {

        if ((*params->par[i].length = length_par(buf)) < 1) return;
        allocate_par(&params->par[i]);

        copy_string(buffer, NPOW_10, buf);
        buffer[strcspn(buffer, "\r\n#")] = 0;

        ptr = strtok(buffer, separator);
        tag = ptr;


        n = 0;

        while ((ptr = strtok(NULL, separator)) != NULL){

          switch (params->par[i].type){
            case _PAR_INT_:
              params->par[i].int_vec_[0][n] = atoi(ptr);
              break;
            case _PAR_ENUM_:
              params->par[i].int_vec_[0][n] = parse_enum(ptr, params->par[i].enums, params->par[i].n_enums);
              break;
            case _PAR_FLOAT_:
              params->par[i].float_vec_[0][n] = atof(ptr);
              break;
            case _PAR_DOUBLE_:
              params->par[i].double_vec_[0][n] = atof(ptr);
              break;
            case _PAR_BOOL_:
              params->par[i].int_vec_[0][n] = parse_enum(ptr, params->par[i].enums, params->par[i].n_enums);
              break;
            case _PAR_DATE_:
              params->par[i].date_vec_[0][n] = parse_date(ptr);
              break;
            case _PAR_CHAR_:
              copy_string(params->par[i].char_vec_[0][n], NPOW_10, ptr);
              break;
            default:
              printf("unknown datatype for par..\n");
              break;
          }

          n++;

        }

      }

    }

  }

  return;
}


/** This function prints all parsed parameters
--- params: parsed parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_parameter(params_t *params){
int i, n;
char eol[1];


  printf("Parameter printout:\n");
  printf("%d parameters were registered.\n", params->n);

  for (i=0; i<params->n; i++){

    printf("%s: ", params->par[i].name);

    // scalar
    if (params->par[i].length == NULL){

      switch (params->par[i].type){
        case _PAR_INT_:
          printf("%d\n", *params->par[i].int_);
          break;
        case _PAR_ENUM_:
          printf("%d\n", *params->par[i].int_);
          break;
        case _PAR_FLOAT_:
          printf("%f\n", *params->par[i].float_);
          break;
        case _PAR_DOUBLE_:
          printf("%f\n", *params->par[i].double_);
          break;
        case _PAR_BOOL_:
          printf("%d\n", *params->par[i].int_);
          break;
        case _PAR_DATE_:
          printf("%04d-%02d-%02d\n", params->par[i].date_->year,
            params->par[i].date_->month, params->par[i].date_->day);
          break;
        case _PAR_CHAR_:
          printf("%s\n", *params->par[i].char_);
          break;
        default:
          printf("unknown datatype for par..\n");
          break;
      }

    // vector
    } else {

      for (n=0; n<*params->par[i].length; n++){

        if (n == *params->par[i].length-1) eol[0] = '\n'; else eol[0] = ' ';

        switch (params->par[i].type){
          case _PAR_INT_:
            printf("%d%s", params->par[i].int_vec_[0][n], eol);
            break;
          case _PAR_ENUM_:
            printf("%d%s", params->par[i].int_vec_[0][n], eol);
            break;
          case _PAR_FLOAT_:
            printf("%f%s", params->par[i].float_vec_[0][n], eol);
            break;
          case _PAR_DOUBLE_:
            printf("%f%s", params->par[i].double_vec_[0][n], eol);
            break;
          case _PAR_BOOL_:
            printf("%d%s", params->par[i].int_vec_[0][n], eol);
            break;
          case _PAR_DATE_:
            printf("%04d-%02d-%02d%s", params->par[i].date_vec_[0][n].year,
              params->par[i].date_vec_[0][n].month, params->par[i].date_vec_[0][n].day, eol);
            break;
          case _PAR_CHAR_:
            printf("%s%s", params->par[i].char_vec_[0][n], eol);
            break;
          default:
            printf("unknown datatype for par..\n");
            break;
        }

      }

    }

  }

  return;
}


/** This function logs all parsed parameters to an internal log slot
--- params: parsed parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void log_parameter(params_t *params){
int i, n;
char *cur = params->log;
char *const end = params->log + NPOW_14;
char eol[1];


  for (i=0; i<params->n; i++){

    if (i > 0){
      if (cur < end){
        cur += snprintf(cur, end-cur, ", ");
      } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
    }

    if (cur < end){
      cur += snprintf(cur, end-cur, "%s: ", params->par[i].name);
    } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}


    // scalar
    if (params->par[i].length == NULL){

      switch (params->par[i].type){
        case _PAR_INT_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%d", *params->par[i].int_);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        case _PAR_ENUM_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%d", *params->par[i].int_);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        case _PAR_FLOAT_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%f", *params->par[i].float_);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        case _PAR_DOUBLE_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%f", *params->par[i].double_);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        case _PAR_BOOL_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%d", *params->par[i].int_);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        case _PAR_DATE_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%04d-%02d-%02d", params->par[i].date_->year,
            params->par[i].date_->month, params->par[i].date_->day);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        case _PAR_CHAR_:
          if (cur < end){
            cur += snprintf(cur, end-cur, "%s", *params->par[i].char_);
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
        default:
          if (cur < end){
            cur += snprintf(cur, end-cur, "unknown datatype for par..");
          } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
          break;
      }

    // vector
    } else {

      for (n=0; n<*params->par[i].length; n++){

        if (n == *params->par[i].length-1) eol[0] = '\0'; else eol[0] = ' ';

        switch (params->par[i].type){
          case _PAR_INT_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%d%s", params->par[i].int_vec_[0][n], eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          case _PAR_ENUM_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%d%s", params->par[i].int_vec_[0][n], eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          case _PAR_FLOAT_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%f%s", params->par[i].float_vec_[0][n], eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          case _PAR_DOUBLE_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%f%s", params->par[i].double_vec_[0][n], eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          case _PAR_BOOL_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%d%s", params->par[i].int_vec_[0][n], eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          case _PAR_DATE_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%04d-%02d-%02d%s", params->par[i].date_vec_[0][n].year,
              params->par[i].date_vec_[0][n].month, params->par[i].date_vec_[0][n].day, eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          case _PAR_CHAR_:
            if (cur < end){
              cur += snprintf(cur, end-cur, "%s%s", params->par[i].char_vec_[0][n], eol);
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
          default:
            if (cur < end){
              cur += snprintf(cur, end-cur, "unknown datatype for par..");
            } else { printf("Buffer Overflow in assembling par log\n"); exit(1);}
            break;
        }

      }

    }

  }

  return;
}


/** This function checks for the validity of an integer parameter
--- par:    parameter
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_int_par(par_t *par){
int error = 0, n;


  if (par->length == NULL){
    if (*par->int_ < par->int_range[_MIN_] ||
        *par->int_ > par->int_range[_MAX_]) error++;
  } else {
    for (n=0; n<*par->length; n++){
      if (par->int_vec_[0][n] < par->int_range[_MIN_] ||
          par->int_vec_[0][n] > par->int_range[_MAX_]) error++;
    }
  }

  if (error > 0){
    printf("parameter %s is out of bounds [%d,%d].\n",
      par->name, par->int_range[_MIN_], par->int_range[_MAX_]);
    return FAILURE;
  }

  return SUCCESS;
}


/** This function checks for the validity of an enum parameter
--- par:    parameter
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_enum_par(par_t *par){
int error = 0, n, e;
bool ok;


  if (par->length == NULL){
    for (e=0, ok=false; e<par->n_enums; e++){
      if (*par->int_ == par->enums[e].en) ok = true;
    }
    if (!ok) error++;
  } else {
    for (n=0; n<*par->length; n++){
      for (e=0, ok=false; e<par->n_enums; e++){
        if (par->int_vec_[0][n] == par->enums[e].en) ok = true;
      }
      if (!ok) error++;
    }
  }

  if (error > 0){
    printf("parameter %s is out of bounds {%s", par->name, par->enums[0].tag);
    for (e=1; e<par->n_enums; e++) printf(",%s", par->enums[e].tag);
    printf("}.\n");
    return FAILURE;
  }

  return SUCCESS;
}


/** This function checks for the validity of a float parameter
--- par:    parameter
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_float_par(par_t *par){
int error = 0, n;


  if (par->length == NULL){
    if (*par->float_ < par->float_range[_MIN_] ||
        *par->float_ > par->float_range[_MAX_]) error++;
  } else {
    for (n=0; n<*par->length; n++){
      if (par->float_vec_[0][n] < par->float_range[_MIN_] ||
          par->float_vec_[0][n] > par->float_range[_MAX_]) error++;
    }
  }

  if (error > 0){
    printf("parameter %s is out of bounds [%f,%f].\n",
      par->name, par->float_range[_MIN_], par->float_range[_MAX_]);
    return FAILURE;
  }

  return SUCCESS;
}


/** This function checks for the validity of a double parameter
--- par:    parameter
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_double_par(par_t *par){
int error = 0, n;


  if (par->length == NULL){
    if (*par->double_ < par->double_range[_MIN_] ||
        *par->double_ > par->double_range[_MAX_]) error++;
  } else {
    for (n=0; n<*par->length; n++){
      if (par->double_vec_[0][n] < par->double_range[_MIN_] ||
          par->double_vec_[0][n] > par->double_range[_MAX_]) error++;
    }
  }

  if (error > 0){
    printf("parameter %s is out of bounds [%f,%f].\n",
      par->name, par->double_range[_MIN_], par->double_range[_MAX_]);
    return FAILURE;
  }

  return SUCCESS;
}


/** This function checks for the validity of a date parameter
--- par:    parameter
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_date_par(par_t *par){
int error = 0, n;


  if (par->length == NULL){
    if (par->date_->ce < par->date_range[_MIN_].ce ||
        par->date_->ce > par->date_range[_MAX_].ce) error++;
  } else {
    for (n=0; n<*par->length; n++){
      if (par->date_vec_[0][n].ce < par->date_range[_MIN_].ce ||
          par->date_vec_[0][n].ce > par->date_range[_MAX_].ce) error++;
    }
  }

  if (error > 0){
    printf("parameter %s is out of bounds [%04d-%02d-%02d,%04d-%02d-%02d].\n",
      par->name,
      par->date_range[_MIN_].year, par->date_range[_MIN_].month, par->date_range[_MIN_].day,
      par->date_range[_MAX_].year, par->date_range[_MAX_].month, par->date_range[_MAX_].day);
    return FAILURE;
  }

  return SUCCESS;
}


/** This function checks for the validity of a string parameter
--- par:    parameter
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_char_par(par_t *par){
int n;


  if (par->length == NULL){

    switch (par->char_test){
      case _CHAR_TEST_NULL_OR_EXIST_:
        if ((strcmp(*par->char_, "NULL") != 0) && !fileexist(*par->char_)){
          printf("parameter %s does not exist in the filesystem.\n", par->name); return FAILURE;}
        break;
      case _CHAR_TEST_EXIST_:
        if (!fileexist(*par->char_)){
          printf("parameter %s does not exist in the filesystem.\n", par->name); return FAILURE;}
        break;
      case _CHAR_TEST_NOT_EXIST_:
        if (fileexist(*par->char_)){
          printf("parameter %s exists in the filesystem. Delete or give new path.\n", par->name); return FAILURE;}
        break;
      case _CHAR_TEST_NULL_OR_BASE_:
        if (strcmp(*par->char_, "NULL") != 0){
          if (strstr(*par->char_, "/") != NULL){
            printf("parameter %s is not a basename. \"/\" detected.\n", par->name); return FAILURE;}
          if (strstr(*par->char_, ".") == NULL){
            printf("parameter %s is not a basename. No file extension detected.\n", par->name); return FAILURE;}
        }
        break;
      case _CHAR_TEST_BASE_:
        if (strstr(*par->char_, "/") != NULL){
          printf("parameter %s is not a basename. \"/\" detected.\n", par->name); return FAILURE;}
        if (strstr(*par->char_, ".") == NULL){
          printf("parameter %s is not a basename. No file extension detected.\n", par->name); return FAILURE;}
        break;
      case _CHAR_TEST_NONE_:
        // nothing to check
        break;
      default:
        printf("unknown char test..\n");
        break;
    }

  } else {
    for (n=0; n<*par->length; n++){

      switch (par->char_test){
        case _CHAR_TEST_NULL_OR_EXIST_:
          if ((strcmp(par->char_vec_[0][n], "NULL") != 0) && !fileexist(par->char_vec_[0][n])){
            printf("parameter %s does not exist in the filesystem.\n", par->name); return FAILURE;}
          break;
        case _CHAR_TEST_EXIST_:
          if (!fileexist(par->char_vec_[0][n])){
            printf("parameter %s does not exist in the filesystem.\n", par->name); return FAILURE;}
          break;
        case _CHAR_TEST_NOT_EXIST_:
          if (fileexist(par->char_vec_[0][n])){
            printf("parameter %s exists in the filesystem. Delete or give new path.\n", par->name); return FAILURE;}
          break;
        case _CHAR_TEST_NULL_OR_BASE_:
          if (strcmp(par->char_vec_[0][n], "NULL") != 0){
            if (strstr(par->char_vec_[0][n], "/") != NULL){
              printf("parameter %s is not a basename. \"/\" detected.\n", par->name); return FAILURE;}
            if (strstr(par->char_vec_[0][n], ".") == NULL){
              printf("parameter %s is not a basename. No file extension detected.\n", par->name); return FAILURE;}
          }
          break;
        case _CHAR_TEST_BASE_:
          if (strstr(par->char_vec_[0][n], "/") != NULL){
            printf("parameter %s is not a basename. \"/\" detected.\n", par->name); return FAILURE;}
          if (strstr(par->char_vec_[0][n], ".") == NULL){
            printf("parameter %s is not a basename. No file extension detected.\n", par->name); return FAILURE;}
          break;
        case _CHAR_TEST_NONE_:
          // nothing to check
          break;
        default:
          printf("unknown char test..\n");
          break;
      }


    }
  }

  return SUCCESS;
}


/** This function checks for the validity of all parsed parameters
--- params: parsed parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_parameter(params_t *params){
int error = 0, i;


  for (i=0; i<params->n; i++){

    if (params->par[i].set == false){
      printf("parameter %s was not set.\n", params->par[i].name);
      error++;
      continue;
    }

    if (params->par[i].length != NULL && *params->par[i].length == 0){
      printf("parameter %s was incorrectly parsed.\n", params->par[i].name);
      error++;
      continue;
    }

    switch (params->par[i].type){
      case _PAR_INT_:
        if (check_int_par(&params->par[i]) == FAILURE) error++;
        break;
      case _PAR_ENUM_:
        if (check_enum_par(&params->par[i]) == FAILURE) error++;
        break;
      case _PAR_FLOAT_:
        if (check_float_par(&params->par[i]) == FAILURE) error++;
        break;
      case _PAR_DOUBLE_:
        if (check_double_par(&params->par[i]) == FAILURE) error++;
        break;
      case _PAR_BOOL_:
        if (check_enum_par(&params->par[i]) == FAILURE) error++;
        break;
      case _PAR_DATE_:
        if (check_date_par(&params->par[i]) == FAILURE) error++;
        break;
      case _PAR_CHAR_:
        if (check_char_par(&params->par[i]) == FAILURE) error++;
        break;
      default:
        printf("unknown datatype for par..\n");
        break;
    }

  }


  if (error > 0) return FAILURE; else return SUCCESS;
}

