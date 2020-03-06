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


#include "param-train-aux.h"


void register_train(params_t *params, par_train_t *train);


/** This function registers training parameters that are parsed from the 
+++ parameter file.
--- params: registered parameters
--- train:  train parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_train(params_t *params, par_train_t *train){
par_enum_t method[_ML_LENGTH_] = {
  { _ML_SVR_, "SVR" }, { _ML_SVC_, "SVC" }, { _ML_RFR_, "RFR" }, { _ML_RFC_, "RFC" }};


  register_char_par(params,     "FILE_FEATURES",         _CHAR_TEST_EXIST_, &train->f_feature);
  register_char_par(params,     "FILE_RESPONSE",         _CHAR_TEST_EXIST_, &train->f_response);
  register_char_par(params,     "FILE_MODEL",            _CHAR_TEST_NONE_,  &train->f_model);
  register_char_par(params,     "FILE_LOG",              _CHAR_TEST_NONE_,  &train->f_log);
  register_float_par(params,    "PERCENT_TRAIN",         0.001, 100, &train->per_train);
  register_bool_par(params,     "RANDOM_SPLIT",          &train->random_split);
  register_enum_par(params,     "ML_METHOD",             method, _ML_LENGTH_, &train->method);
  register_int_par(params,      "RF_NTREE",              0, INT_MAX, &train->rf.ntree);
  register_float_par(params,    "RF_OOB_ACCURACY",       0, INT_MAX, &train->rf.oob_accuracy);
  register_int_par(params,      "RF_NFEATURE",           0, INT_MAX, &train->rf.feature_subset);
  register_bool_par(params,     "RF_FEATURE_IMPORTANCE", &train->rf.feature_importance);
  register_int_par(params,      "RF_DT_MINSAMPLE",       0, INT_MAX, &train->rf.dt.min_sample);
  register_int_par(params,      "RF_DT_MAXDEPTH",        0, INT_MAX, &train->rf.dt.max_depth);
  register_float_par(params,    "RF_DT_REG_ACCURACY",    0, INT_MAX, &train->rf.dt.reg_accuracy);
  register_int_par(params,      "SVM_MAXITER",           0, INT_MAX, &train->sv.max_iter);
  register_float_par(params,    "SVM_ACCURACY",          0, INT_MAX, &train->sv.accuracy);
  register_int_par(params,      "SVM_KFOLD",             1, INT_MAX, &train->sv.kfold);
  register_float_par(params,    "SVM_P",                 0, INT_MAX, &train->sv.P);
  register_floatvec_par(params, "SVM_C_GRID",            0, INT_MAX, &train->sv.Cgrid, &train->sv.nCgrid);
  register_floatvec_par(params, "SVM_GAMMA_GRID",        0, INT_MAX, &train->sv.Gammagrid, &train->sv.nGammagrid);


  return;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function allocates the training parameters
+++ Return: train parameters (must be freed with free_param_train)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
par_train_t *allocate_param_train(){
par_train_t *train = NULL;


  alloc((void**)&train, 1, sizeof(par_train_t));

  return train;
}


/** This function frees the training parameters
--- train:  tain parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_param_train(par_train_t *train){

  if (train == NULL) return;
  free((void*)train); train = NULL;

  return;
}


/** This function parses the training parameters
--- train:  train parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_param_train(par_train_t *train){
FILE *fpar = NULL;
char  buffer[NPOW_10] = "\0";


  train->params = allocate_params();


  // open parameter file
  if ((fpar = fopen(train->f_par, "r")) == NULL){
    printf("Unable to open parameter file!\n"); return FAILURE;}

  if (fscanf(fpar, "%s", buffer) < 0){
    printf("No valid parameter file!\n"); return FAILURE;}


  if (strcmp(buffer, "++PARAM_TRAIN_START++") != 0){
    printf("Not a training parameter file!\n"); return FAILURE;}


   register_train(train->params, train);


  // process line by line
  while (fgets(buffer, NPOW_10, fpar) != NULL) parse_parameter(train->params, buffer);
  fclose(fpar);


  #ifdef DEBUG
  print_parameter(train->params);
  #endif

  if (check_parameter(train->params) == FAILURE) return FAILURE;

  log_parameter(train->params);
  
  
  // some more checks

  if (train->sv.nCgrid != 3){
    printf("SVM_C_GRID needs to be a list of 3 floats: minVal maxVal logStep.\n"); return FAILURE;}
  if (train->sv.nGammagrid != 3){
    printf("SVM_GAMMA_GRID needs to be a list of 3 floats: minVal maxVal logStep.\n"); return FAILURE;}
  if (train->sv.Cgrid[_MIN_] > train->sv.Cgrid[_MAX_]){
    printf("SVM_C_GRID looks odd It needs to be a list of 3 floats: minVal maxVal logStep.\n"); return FAILURE;}
  if (train->sv.Gammagrid[_MIN_] > train->sv.Gammagrid[_MAX_]){
    printf("SVM_GAMMA_GRID looks odd It needs to be a list of 3 floats: minVal maxVal logStep.\n"); return FAILURE;}



  return SUCCESS;
}

