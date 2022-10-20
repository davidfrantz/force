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
Training paramater header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef PARAM_TRAIN_H
#define PARAM_TRAIN_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/param-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

// support vector parameters
typedef struct {
  int max_iter;
  float accuracy;
  int kfold;
  float *Cgrid;
  float *Gammagrid;
  int nCgrid, nGammagrid;
  float P;
} par_sv_t;

// decision tree parameters
typedef struct {
  int min_sample;
  int max_depth;
  float reg_accuracy;
} par_dt_t;

// random forest parameters
typedef struct {
  int ntree;
  float oob_accuracy;
  int feature_subset;
  int feature_importance;
  par_dt_t dt;
} par_rf_t;

// training parameters
typedef struct {
  params_t *params;
  char f_par[NPOW_10];
  char *f_feature;
  char *f_response;
  char *f_model;
  char *f_log;
  par_sv_t sv;
  par_rf_t rf;
  int method;
  float per_train;
  int random_split;
  int response_var;
  char **class_weights;
  int nclass_weights;
  float *priors;
  int npriors;
  char log[NPOW_14];
} par_train_t;

par_train_t *allocate_param_train();
void free_param_train(par_train_t *train);
int parse_param_train(par_train_t *train);

#ifdef __cplusplus
}
#endif

#endif

