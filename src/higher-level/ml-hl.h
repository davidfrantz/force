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
Machine learning header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef ML_HL_H
#define ML_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <opencv2/ml.hpp>

#include "../cross-level/const-cl.h"
#include "../cross-level/stack-cl.h"
#include "../cross-level/stats-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/read-ard-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  //std::vector<cv::Ptr<cv::ml::StatModel>> model;
  std::vector<cv::Ptr<cv::ml::RTrees>> rf_model;
  std::vector<cv::Ptr<cv::ml::SVM>> sv_model;
} aux_ml_t;

typedef struct {
  short **mlp_;
  short **mli_;
  short **mlu_;
  short **rfp_;
  short **rfm_;
} ml_t;

stack_t **machine_learning(ard_t *features, stack_t *mask, int nf, par_hl_t *phl, aux_ml_t *mod, cube_t *cube, int *nproduct);

#ifdef __cplusplus
}
#endif

#endif

