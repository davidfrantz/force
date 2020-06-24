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
This file contains functions for training machine learning models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "train-aux.h"


/** This function trains a Support Vector Machine model
--- TrainData: training data
--- train:     train parameters
+++ Return:    model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
Ptr<StatModel> train_svm(Ptr<TrainData> TrainData, par_train_t *train, FILE *fp){


  Ptr<SVM> svm = SVM::create();

  if (train->method == _ML_SVR_){
    svm->setType(SVM::EPS_SVR);
  } else if (train->method == _ML_SVC_){
    svm->setType(SVM::C_SVC);
  } else {
    printf("unknown SVM method\n");
    exit(FAILURE);
  }

  svm->setKernel(SVM::RBF);

  if (train->sv.max_iter > 0 && train->sv.accuracy > 0){
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, train->sv.max_iter, train->sv.accuracy));
  } else if (train->sv.max_iter > 0){
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, train->sv.max_iter, 0));
  } else if (train->sv.accuracy > 0){
    printf("warning: support vectors are constructed until SVM_ACCURACY is met..\n");
    printf("This might never happen. Many many many support vectors might be generated. Let's go.\n");
    svm->setTermCriteria(TermCriteria(TermCriteria::EPS, 0, train->sv.accuracy));
  } else {
    printf("RF_NTREE AND/OR RF_OOB_ACCURACY need to be given...\n"); exit(FAILURE);
  }

  if (train->sv.P > 0){
    svm->setP(train->sv.P);
  } else {
    svm->setP(FLT_EPSILON);
  }
  
  ParamGrid Cgrid = SVM::getDefaultGrid(SVM::C);
  ParamGrid Gammagrid = SVM::getDefaultGrid(SVM::GAMMA);

  Cgrid.minVal = train->sv.Cgrid[0]; 
  Cgrid.maxVal = train->sv.Cgrid[1];
  Cgrid.logStep = train->sv.Cgrid[2]; 
  Gammagrid.minVal = train->sv.Gammagrid[0]; 
  Gammagrid.maxVal = train->sv.Gammagrid[1];
  Gammagrid.logStep = train->sv.Gammagrid[2]; 

  ParamGrid Pgrid = SVM::getDefaultGrid(SVM::P); Pgrid.logStep = 0; Pgrid.minVal = 1e3; Pgrid.maxVal = 1e3;
  ParamGrid Nugrid = SVM::getDefaultGrid(SVM::NU); Nugrid.logStep = 0; Nugrid.minVal = 1e3; Nugrid.maxVal = 1e3;
  ParamGrid Coefgrid = SVM::getDefaultGrid(SVM::COEF); Coefgrid.logStep = 0; Coefgrid.minVal = 1e3; Coefgrid.maxVal = 1e3;
  ParamGrid Degreegrid = SVM::getDefaultGrid(SVM::DEGREE); Degreegrid.logStep = 0; Degreegrid.minVal = 1e3; Degreegrid.maxVal = 1e3;

  svm->trainAuto(TrainData, train->sv.kfold, Cgrid, Gammagrid, Pgrid, Nugrid, Coefgrid, Degreegrid, false);

  fprintf(fp, "\nSupport Vector Machine parameters\n");
  fprintf(fp, "--------------------------------------------------------------------\n");
  fprintf(fp, "Termination type: %d\n", svm->getTermCriteria().type);
  fprintf(fp, "max iterations: %d\n", svm->getTermCriteria().maxCount);
  fprintf(fp, "accuracy: %f\n", svm->getTermCriteria().epsilon);
  fprintf(fp, "k-fold CV: %d\n", train->sv.kfold);
  fprintf(fp, "P: %f\n", svm->getP());
  fprintf(fp, "C: %f\n", svm->getC());
  fprintf(fp, "Gamma: %f\n", svm->getGamma());
  fprintf(fp, "____________________________________________________________________\n");

  svm->save(train->f_model);

  return svm;
}


/** This function trains a Random Forest model
--- TrainData: training data
--- train:     train parameters
+++ Return:    model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
Ptr<StatModel> train_rf(Ptr<TrainData> TrainData, par_train_t *train, FILE *fp){
int v, i;
int n_feature;


  Ptr<RTrees> rf = RTrees::create();
  n_feature = TrainData->getNVars();


  // parameterize forest

  if (train->rf.ntree > 0 && train->rf.oob_accuracy > 0){
    rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS, train->rf.ntree, train->rf.oob_accuracy));
  } else if (train->rf.ntree > 0){
    rf->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, train->rf.ntree, 0));
  } else if (train->rf.oob_accuracy > 0){
    printf("warning: random forest is constructed until RF_OOB_ACCURACY is met..\n");
    printf("This might never happen. Many many many trees might be grown. Let's go.\n");
    rf->setTermCriteria(TermCriteria(TermCriteria::EPS, 0, train->rf.oob_accuracy));
  } else {
    printf("RF_NTREE AND/OR RF_OOB_ACCURACY need to be given...\n"); exit(FAILURE);
  }


  if (train->rf.feature_subset  > 0){
    rf->setActiveVarCount(train->rf.feature_subset);
  } else if (train->method == _ML_RFR_){
    rf->setActiveVarCount(n_feature/3);
  } else if (train->method == _ML_RFC_){
    rf->setActiveVarCount(sqrt(n_feature));
  }
  rf->setCalculateVarImportance(train->rf.feature_importance);

  
  // parameterize trees

  if (train->rf.dt.min_sample   > 0){
    rf->setMinSampleCount(train->rf.dt.min_sample);
  } else if (train->method == _ML_RFR_){
    rf->setMinSampleCount(5);
  } else if (train->method == _ML_RFC_){
    rf->setMinSampleCount(1);
  }
  if (train->rf.dt.max_depth  > 0){
    rf->setMaxDepth(train->rf.dt.max_depth);
  } else {
    rf->setMaxDepth(INT_MAX);
  }
  if (train->rf.dt.reg_accuracy > 0){
    rf->setRegressionAccuracy(train->rf.dt.reg_accuracy);
  } else {
    rf->setRegressionAccuracy(0.01);
  }


  fprintf(fp, "\nRandom Forest parameters\n");
  fprintf(fp, "--------------------------------------------------------------------\n");
  fprintf(fp, "Forest variables\n");
  fprintf(fp, "--------------------------------------------------------------------\n");
  fprintf(fp, "Termination type: %d\n", rf->getTermCriteria().type);
  fprintf(fp, "max trees: %d\n", rf->getTermCriteria().maxCount);
  fprintf(fp, "OOB accuracy: %f\n", rf->getTermCriteria().epsilon);
  fprintf(fp, "# of vars at split: %d\n", rf->getActiveVarCount());
  fprintf(fp, "compute variable importance: %d\n", rf->getCalculateVarImportance());
  fprintf(fp, "--------------------------------------------------------------------\n");
  fprintf(fp, "Tree variables\n");
  fprintf(fp, "--------------------------------------------------------------------\n");
  fprintf(fp, "max depth of tree: %d\n", rf->getMaxDepth());
  fprintf(fp, "min # of samples for split: %d\n", rf->getMinSampleCount());
  fprintf(fp, "regression accuracy: %f\n", rf->getRegressionAccuracy());
  fprintf(fp, "____________________________________________________________________\n");

  rf->train(TrainData, 0);

  if (rf->getCalculateVarImportance()){
    
    Mat vi = rf->getVarImportance();
    Mat idx;
    sortIdx(vi, idx, SORT_EVERY_COLUMN + SORT_DESCENDING);

    fprintf(fp, "\nVariable Importance\n");
    fprintf(fp, "--------------------------------------------------------------------\n");
    for (i=0; i<vi.rows; i++){
      v = idx.at<int>(i,0);
      fprintf(fp, "Variable %5d: %f\n", v, vi.at<float>(v,0));
    }
    fprintf(fp, "____________________________________________________________________\n");

  }
  
  rf->save(train->f_model);

  return rf;
}


/** This function makes a regression prediction and performs a validation
--- features:  features, on which to make predict
--- response:  response (truth)
--- model:     ML model
--- n_sample:  number of samples
--- n_feature: number of features
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void predict_regression(float **features, float *response, Ptr<StatModel> model, int n_sample, int n_feature, FILE *fp){
int s, f;
float pred;
double sum = 0;
float rmse;
double slope, intercept, rsq;
double mx = 0, my = 0, vx = 0, vy = 0, cov = 0;


  fprintf(fp, "\nFeature-Response Array\n");
  fprintf(fp, "--------------------------------------------------------------------\n");

  for (s=0; s<n_sample; s++){

    Mat sampleMat(1, n_feature, CV_32F, features[s]);
    pred = model->predict(sampleMat);

    if (s == 0){
      mx = response[s]; my = pred;
    } else {
      covar_recurrence(response[s], pred, &mx, &my, &vx, &vy, &cov, s+1);
    }
    sum += (pred-response[s])*(pred-response[s]);
    for (f=0; f<n_feature; f++){
      if (f < 3 || f > n_feature-4){
        fprintf(fp, " %.2f ", features[s][f]);
      } else if (f < 13){
        fprintf(fp, ".");
      }
    }
    fprintf(fp, "::: %+.2f -> %+.2f\n", response[s], pred);
  }


  rmse = sqrt(sum/n_sample);
  linreg_coefs(mx, my, covariance(cov, n_sample), variance(vx, n_sample), &slope, &intercept);
  linreg_rsquared(covariance(cov, n_sample), variance(vx, n_sample), variance(vy, n_sample), &rsq);

  fprintf(fp, "____________________________________________________________________\n");
  fprintf(fp, "y = %.3f + %.3f x, Rsq:  %.3f\n", intercept, slope, rsq);
  fprintf(fp, "RMSE: %.3f\n", rmse);


  return;
}


/** This function makes a classification prediction and performs a 
+++ validation
--- features:  features, on which to make predict
--- response:  response (truth)
--- model:     ML model
--- n_sample:  number of samples
--- n_feature: number of features
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void predict_classification(float **features, int *response, Ptr<StatModel> model, int n_sample, int n_feature, FILE *fp){
int s, f;
int pred;
float oa;
size_t correct = 0;


  fprintf(fp, "\nFeature-Response Array\n");
  fprintf(fp, "--------------------------------------------------------------------\n");

  for (s=0; s<n_sample; s++){
    Mat sampleMat(1, n_feature, CV_32F, features[s]);
    pred = model->predict(sampleMat);
    correct += (pred == response[s]);
    for (f=0; f<n_feature; f++){
      if (f < 3 || f > n_feature-4){
        fprintf(fp, "%.2f ", features[s][f]);
      } else if (f < 13){
        fprintf(fp, ".");
      }
    }
    fprintf(fp, "::: %+03d -> %+03d\n", response[s], pred);
  }

  oa = (float)correct/n_sample;

  fprintf(fp, "____________________________________________________________________\n");
  fprintf(fp, "OA: %.2f%%\n", oa*100);

  return;
}

