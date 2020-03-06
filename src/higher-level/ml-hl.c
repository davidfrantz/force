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
This file contains functions for machine learning
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "ml-hl.h"

/** OpenCV **/
#include <opencv2/ml.hpp>
using namespace cv;
using namespace cv::ml;


stack_t **compile_ml(ard_t *features, ml_t *ml, par_hl_t *phl, cube_t *cube, int *nproduct);
stack_t *compile_ml_stack(stack_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the stacks, in which ML results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- features: input features
--- ml:       pointer to instantly useable ML image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output stacks (returned)
+++ Return:   stacks for ML results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **compile_ml(ard_t *features, ml_t *ml, par_hl_t *phl, cube_t *cube, int *nproduct){
stack_t **ML = NULL;
int o, nprod = 3;
int error = 0;
enum { _mlp_, _mli_, _mlu_ };
int prodlen[3] = { phl->mcl.nmodelset, phl->mcl.nmodelset, phl->mcl.nmodelset };
char prodname[3][NPOW_02] = { "MLP", "MLI", "MLU" };
int prodtype[3] = { _mlp_, _mli_, _mlu_ };
bool enable[3] = { phl->mcl.omlp, phl->mcl.omli, phl->mcl.omlu };
bool write[3]  = { phl->mcl.omlp, phl->mcl.omli, phl->mcl.omlu };
short ***ptr[3] = { &ml->mlp_, &ml->mli_, &ml->mlu_ };


  alloc((void**)&ML, nprod, sizeof(stack_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((ML[o] = compile_ml_stack(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(ML[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      }
    } else {
      ML[o]  = NULL;
      *ptr[o] = NULL;
    }
  }


  if (error > 0){
    printf("%d compiling ML product errors.\n", error);
    for (o=0; o<nprod; o++) free_stack(ML[o]);
    free((void*)ML);
    return NULL;
  }

  *nproduct = nprod;
  return ML;
}


/** This function compiles a ML stack
--- from:      stack from which most attributes are copied
--- nb:        number of bands in stack
--- write:     should this stack be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    stack for ML result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *compile_ml_stack(stack_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
stack_t *stack = NULL;
char dname[NPOW_10];
char fname[NPOW_10];
int nchar;


  if ((stack = copy_stack(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_stack_name(stack, "FORCE Machine Learning");
  set_stack_product(stack, prodname);
  
  //printf("dirname should be assemlbed in write_stack, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_stack_tilex(stack), get_stack_tiley(stack));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_stack_dirname(stack, dname);

  nchar = snprintf(fname, NPOW_10, "%s_HL_ML_%s", phl->mcl.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_stack_filename(stack, fname);

  if (write){
    set_stack_open(stack, OPEN_BLOCK);
  } else {
    set_stack_open(stack, OPEN_FALSE);
  }
  set_stack_format(stack, phl->format);
  set_stack_par(stack, phl->params->log);
  
  for (b=0; b<nb; b++){
    set_stack_save(stack, b, true);
  }

  return stack;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the machine learning prediction
+++ module
--- features: input features
--- mask:     mask image
--- nf:       number of features
--- phl:      HL parameters
--- model:    machine learning model
--- cube:     datacube definition
--- nproduct: number of output stacks (returned)
+++ Return:   stacks with ML results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **machine_learning(ard_t *features, stack_t *mask, int nf, par_hl_t *phl, std::vector<cv::Ptr<cv::ml::StatModel>> model, cube_t *cube, int *nproduct){
ml_t ml;
stack_t **ML;
small *mask_ = NULL;
bool regression;
float fpred[NPOW_03][NPOW_06];
int   ipred[NPOW_03][NPOW_06];
int nprod = 0;
int f, s, m, k, p, nc;
double mean, mean_old, var, std, mn;
short nodata;
bool valid;


  // import stacks
  nc = get_stack_chunkncells(features[0].DAT);
  nodata = get_stack_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }
  

  // compile products + stacks
  if ((ML = compile_ml(features, &ml, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile ML products!\n"); 
    *nproduct = 0;
    return NULL;
  }

  
  if (phl->mcl.method == _ML_SVR_ || phl->mcl.method == _ML_RFR_) regression = true; else regression = false;


  #pragma omp parallel private(f,s,m,k,mean,mn,mean_old,var,std,fpred,ipred,valid) shared(features,model,regression,ml,nf,nc,mask_,phl,ML,nodata) default(none)
  {

    #pragma omp for schedule(dynamic,1)
    for (p=0; p<nc; p++){

      // skip pixels that are masked
      if (mask_ != NULL && !mask_[p]){
        for (s=0; s<phl->mcl.nmodelset; s++){
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = nodata;
          if (ml.mli_ != NULL) ml.mli_[s][p] = nodata;
          if (ml.mlu_ != NULL) ml.mlu_[s][p] = nodata;
        }
        continue;
      }

      Mat sample(1, nf, CV_32F);
      for (f=0, valid=true; f<nf; f++){
        sample.at<float>(0,f) = features[f].dat[0][p]/10000.0;
        if (!features[f].msk[p] && phl->ftr.exclude) valid=false;
      }

      if (!valid){
        for (s=0; s<phl->mcl.nmodelset; s++){
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = nodata;
          if (ml.mli_ != NULL) ml.mli_[s][p] = nodata;
          if (ml.mlu_ != NULL) ml.mlu_[s][p] = nodata;
        }
        continue;
      }

      for (s=0, k=0; s<phl->mcl.nmodelset; s++){
        
        mean = mean_old = var = 0;
        for (m=0; m<phl->mcl.nmodel[s]; m++, k++){

          fpred[s][m] = model[k]->predict(sample);
          ipred[s][m] = (int)fpred[s][m];

          if (regression){

            if (m == 0){
              mean = fpred[s][m];
            } else {
              var_recurrence(fpred[s][m], &mean, &var, m+1);
            }

            if (m > 1 && fabs(mean-mean_old) < phl->mcl.converge){ k += phl->mcl.nmodel[s]-m; m++; break;}
            mean_old = mean;
            
          }

        }

        if (regression){
          mn  = mean*phl->mcl.scale;
          if (mn > 32767)  mn = 32767;
          if (mn < -32767) mn = -32767;
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = (short)mn;  
          std = standdev(var, m)*10000.0;
          if (std > 32767) std = 32767;
          if (ml.mlu_ != NULL) ml.mlu_[s][p] = (short)(std);
        } else {
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = (short)mode(ipred[s], phl->mcl.nmodel[s]);
        }
        if (ml.mli_ != NULL) ml.mli_[s][p] = m;
        
      }


    }

  }
  
  *nproduct = nprod;
  return ML;
}

