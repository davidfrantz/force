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
This program trains (and validates) machine learning models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/aux-level/param-train-aux.h"
#include "../../modules/aux-level/train-aux.h"


typedef struct {
  int n;
  char fprm[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] parameter-file\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'parameter-file': ML parameter file\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;


  opterr = 0;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvi")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Train (and validate) Machine Learning models\n");
        exit(SUCCESS);
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }

  // non-optional parameters
  args->n = 1;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->fprm, NPOW_10, argv[optind++]);
    } else if (argc-optind < args->n){
      fprintf(stderr, "some non-optional arguments are missing.\n");
      usage(argv[0], FAILURE);
    } else if (argc-optind > args->n){
      fprintf(stderr, "too many non-optional arguments.\n");
      usage(argv[0], FAILURE);
    }
  } else {
    fprintf(stderr, "non-optional arguments are missing.\n");
    usage(argv[0], FAILURE);
  }

  return;
}


int main ( int argc, char *argv[] ){
args_t args;
par_train_t *train = NULL;
int f = 0, s, k, j, n_feature, n_sample, n_sample2;
int n_sample_train, n_sample_val;
int n_response_vars;
double **t_features  = NULL;
double **t_response  = NULL;
float   *r_response  = NULL;
int     *c_response  = NULL;
float **features_train  = NULL;
float  *r_response_train  = NULL;
int    *c_response_train  = NULL;
float **features_val  = NULL;
float  *r_response_val  = NULL;
int    *c_response_val  = NULL;
bool    *is_train  = NULL;
FILE   *flog = NULL;

Ptr<StatModel> model;
Ptr<TrainData> TrainData;

time_t TIME;


  time(&TIME);

  parse_args(argc, argv, &args);
  
  train = allocate_param_train();
  copy_string(train->f_par, NPOW_10, args.fprm);

  // parse parameter file
  if (parse_param_train(train) == FAILURE){
    printf("Reading parameter file failed!\n"); return FAILURE;}

    
  if ((flog = fopen(train->f_log, "w")) == NULL){
    printf("Unable to open logfile!\n"); 
    return FAILURE;}




  // read response variable
  if ((t_response = read_table_deprecated(train->f_response, 
      &n_sample, &n_response_vars)) == NULL){
    printf("unable to read response file. "); return FAILURE;}

  if (n_sample < 1){
    printf("no sample in response file. "); return FAILURE;}
    
  if (n_response_vars < train->response_var){
    printf("requested response variable is "
           "larger than columns in response file. ");
    return FAILURE;}

  alloc((void**)&c_response, n_sample, sizeof(int));
  alloc((void**)&r_response, n_sample, sizeof(float));

  for (s=0; s<n_sample; s++){

    c_response[s] = (int)t_response[s][train->response_var-1];
    r_response[s] = (float)t_response[s][train->response_var-1];

  }

  n_sample_train = n_sample*train->per_train/100;
  n_sample_val   = n_sample-n_sample_train;


  // read features
  if ((t_features = read_table_deprecated(train->f_feature, 
    &n_sample2, &n_feature)) == NULL){
  printf("unable to read feature file. "); return FAILURE;}


  if (n_sample2 != n_sample){
    printf("number of samples in feature (%d) and response (%d) files are different..\n",
      n_sample2, n_sample); return FAILURE;}



  alloc_2DC((void***)&features_train, n_sample_train, n_feature, sizeof(float));
  alloc((void**)&c_response_train, n_sample_train, sizeof(int));
  alloc((void**)&r_response_train, n_sample_train, sizeof(float));

  alloc_2DC((void***)&features_val, n_sample_val, n_feature, sizeof(float));
  alloc((void**)&c_response_val, n_sample_val, sizeof(int));
  alloc((void**)&r_response_val, n_sample_val, sizeof(float));

  alloc((void**)&is_train, n_sample, sizeof(bool));

  srand(time(NULL));
  for (k=0; k<n_sample_train; k++){
    if (train->random_split){
      do {
        s = rand() % n_sample;
      } while (is_train[s]);
    } else s = k;
    is_train[s] = true;
  }

  for (s=0, k=0, j=0; s<n_sample; s++){
    fprintf(flog, "sample: %d, train: %d\n", s, is_train[s]);
    if (is_train[s]){
      for (f=0; f<n_feature; f++) features_train[k][f] = t_features[s][f]/10000.0;
      r_response_train[k] = r_response[s];
      c_response_train[k] = c_response[s];
      k++;
    } else {
      for (f=0; f<n_feature; f++) features_val[j][f] = t_features[s][f]/10000.0;
      r_response_val[j] = r_response[s];
      c_response_val[j] = c_response[s];
      j++;
    }
  }


  fprintf(flog, "\n");
  fprintf(flog, "Loaded %d samples and %d features\n", n_sample, n_feature);
  fprintf(flog, "Training model with %d samples\n", n_sample_train);
  fprintf(flog, "Validating model with %d samples\n", n_sample_val);


  // get class priors
  class_priors(c_response, n_sample, train);


  Mat trainingDataMat(n_sample_train, n_feature, CV_32F, features_train[0]);
  Mat r_labelsMat(n_sample_train, 1, CV_32F, r_response_train);
  Mat c_labelsMat(n_sample_train, 1, CV_32S, c_response_train);
  
  switch (train->method){
    case _ML_SVR_:
      TrainData = TrainData::create(trainingDataMat, ROW_SAMPLE, r_labelsMat);
      model = train_svm(TrainData, train, flog);
      break;
    case _ML_SVC_:
      TrainData = TrainData::create(trainingDataMat, ROW_SAMPLE, c_labelsMat);
      model = train_svm(TrainData, train, flog);
      break;
    case _ML_RFR_:
      TrainData = TrainData::create(trainingDataMat, ROW_SAMPLE, r_labelsMat);
      model = train_rf(TrainData, train, flog);
      break;
    case _ML_RFC_:
      TrainData = TrainData::create(trainingDataMat, ROW_SAMPLE, c_labelsMat);
      model = train_rf(TrainData, train, flog);
      break;
    default:
      printf("unknown method\n");
      exit(1);
  }


  if (train->method == _ML_SVR_ || train->method == _ML_RFR_){
    predict_regression(features_train, r_response_train, model, n_sample_train, n_feature, flog);
    fprintf(flog, "Self-prediction [THIS IS NOT A VALIDATION!]\n");
    if (train->per_train < 100){
      predict_regression(features_val, r_response_val, model, n_sample_val, n_feature, flog);
      fprintf(flog, "Validation with %.2f%% of sample\n", 100-train->per_train);
    }
  } else if (train->method == _ML_SVC_ || train->method == _ML_RFC_){
    predict_classification(features_train, c_response_train, model, n_sample_train, n_feature, flog);
    fprintf(flog, "Self-prediction [THIS IS NOT A VALIDATION!]\n");
    if (train->per_train < 100){
      predict_classification(features_val, c_response_val, model, n_sample_val, n_feature, flog);
      fprintf(flog, "Validation with %.2f%% of sample\n", 100-train->per_train);
    }
  }
  fprintf(flog, "____________________________________________________________________\n");


  free_2D((void**)t_features, n_sample);
  free_2D((void**)t_response, n_sample);
  free((void*)c_response);
  free((void*)r_response);
  free_2DC((void**)features_train);
  free((void*)c_response_train);
  free((void*)r_response_train);
  free_2DC((void**)features_val);
  free((void*)c_response_val);
  free((void*)r_response_val);
  free((void*)is_train);
  free_param_train(train);

  fproctime_print(flog, "\nTraining", TIME);
  
  fclose(flog);


  return SUCCESS;
}

