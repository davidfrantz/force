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
Reading aux files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef READAUX_HL_H
#define READAUX_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <opencv2/ml.hpp>

#include "../cross-level/const-cl.h"
#include "../higher-level/param-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float **endmember; // endmember
  short ***library;  // feature libraries
  std::vector<cv::Ptr<cv::ml::StatModel>> ml_model;
} aux_t;

aux_t *read_aux(par_hl_t *phl);
void free_aux(par_hl_t *phl, aux_t *aux);

#ifdef __cplusplus
}
#endif

#endif

