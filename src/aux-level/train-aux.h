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
Ttraining machine learning models header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TRAIN_ML_H
#define TRAIN_ML_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

/** OpenCV **/
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;

#include "../cross-level/stats-cl.h"
#include "../aux-level/param-train-aux.h"


#ifdef __cplusplus
extern "C" {
#endif

Ptr<StatModel> train_svm(Ptr<TrainData> TrainData, par_train_t *train, FILE *fp);
Ptr<StatModel> train_rf(Ptr<TrainData> TrainData, par_train_t *train, FILE *fp);
void predict_regression(float **features, float *response, Ptr<StatModel> model, int n_sample, int n_feature, FILE *fp);
void predict_classification(float **features, int *response, Ptr<StatModel> model, int n_sample, int n_feature, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif

