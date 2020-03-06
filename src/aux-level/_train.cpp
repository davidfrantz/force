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
This program trains (and validates) machine learning models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/utils-cl.h"
#include "../aux-level/param-train-aux.h"
#include "../aux-level/train-aux.h"


int main ( int argc, char *argv[] ){
par_train_t *train = NULL;
int f = 0, s, k, j, n_feature, n_sample;
int n_sample_train, n_sample_val;
float **features  = NULL;
float  *r_response  = NULL;
int    *c_response  = NULL;
float **features_train  = NULL;
float  *r_response_train  = NULL;
int    *c_response_train  = NULL;
float **features_val  = NULL;
float  *r_response_val  = NULL;
int    *c_response_val  = NULL;
bool    *is_train  = NULL;
FILE   *fp = NULL;
FILE   *flog = NULL;
char buffer[NPOW_14] = "\0";
char *ptr = NULL;
const char *separator = " ";
size_t sampsize = NPOW_10;
size_t featsize = NPOW_05;

Ptr<StatModel> model;
Ptr<TrainData> TrainData;

time_t TIME;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 2){
    printf("usage: %s parameter-file\n\n", argv[0]);
    return FAILURE;
  }

  time(&TIME);
  
  train = allocate_param_train();
  
  train->f_par = argv[1];
  check_arg(argv[1]);

  // parse parameter file
  if (parse_param_train(train) == FAILURE){
    printf("Reading parameter file failed!\n"); return FAILURE;}

    
  if ((flog = fopen(train->f_log, "w")) == NULL){
    printf("Unable to open logfile!\n"); 
    return FAILURE;}


  alloc((void**)&c_response, sampsize, sizeof(int));
  alloc((void**)&r_response, sampsize, sizeof(float));

  // read response variable
  if ((fp = fopen(train->f_response, "r")) == NULL){
    printf("Unable to open response-file. "); exit(1);}

  s = 0;
  while (fgets(buffer, NPOW_14, fp) != NULL){

    c_response[s] = atoi(buffer);
    r_response[s] = atof(buffer);
    s++;

    // if extremely large size, attempt to increase buffer size
    if (s >= (int)sampsize){
      //printf("reallocate.. %lu %lu\n", s, sampsize);
      re_alloc((void**)&c_response, sampsize, sampsize*2, sizeof(int));
      re_alloc((void**)&r_response, sampsize, sampsize*2, sizeof(float));
      sampsize *= 2;
    }

  }

  fclose(fp);

  n_sample = s;
  n_sample_train = n_sample*train->per_train/100;
  n_sample_val   = n_sample-n_sample_train;


  alloc_2DC((void***)&features, n_sample, featsize, sizeof(float));

  // read features
  if ((fp = fopen(train->f_feature, "r")) == NULL){
    printf("Unable to open feature-file. "); exit(1);}

  s = 0;
  while (fgets(buffer, NPOW_14, fp) != NULL){

    ptr = strtok(buffer, separator);
    f = 0;

    while (ptr != NULL){
      features[s][f] = atof(ptr)/10000.0;
      ptr = strtok(NULL, separator);
      f++;

      // if many features, attempt to increase buffer size
      if (f >= (int)featsize){
        //printf("reallocate.. %lu %lu\n", s, sampsize);
        re_alloc_2DC((void***)&features, n_sample, featsize, n_sample, featsize*2, sizeof(float));
        featsize *= 2;
      }

    }

    s++;

  }

  fclose(fp);

  if (s != n_sample){
    printf("number of samples %d and responses %d are different..\n",
      n_sample, s); exit(FAILURE);
  }

  n_feature = f;


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
      for (f=0; f<n_feature; f++) features_train[k][f] = features[s][f];
      r_response_train[k] = r_response[s];
      c_response_train[k] = c_response[s];
      k++;
    } else {
      for (f=0; f<n_feature; f++) features_val[j][f] = features[s][f];
      r_response_val[j] = r_response[s];
      c_response_val[j] = c_response[s];
      j++;
    }
  }


  fprintf(flog, "\n");
  fprintf(flog, "Loaded %d samples and %d features\n", n_sample, n_feature);
  fprintf(flog, "Training model with %d samples\n", n_sample_train);
  fprintf(flog, "Validating model with %d samples\n", n_sample_val);


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


  free_2DC((void**)features);
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

