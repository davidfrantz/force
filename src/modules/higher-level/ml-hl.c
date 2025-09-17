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
This file contains functions for machine learning
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "ml-hl.h"

/** OpenCV **/
using namespace cv;
using namespace cv::ml;


brick_t **compile_ml(ard_t *features, ml_t *ml, par_hl_t *phl, cube_t *cube, int *nproduct);
brick_t *compile_ml_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the bricks, in which ML results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- features: input features
--- ml:       pointer to instantly useable ML image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for ML results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_ml(ard_t *features, ml_t *ml, par_hl_t *phl, cube_t *cube, int *nproduct){
brick_t **ML = NULL;
int s, c, sc, o, nprod = 5;
int error = 0;
int nchar;
char bname[NPOW_10];
char domain[NPOW_10];
enum { _set_, _setclass_ };
int prodlen[2] = { phl->mcl.nmodelset, phl->mcl.nclass_all_sets };
char prodname[5][NPOW_02] = { "MLP", "MLI", "MLU", "RFP", "RFM" };
int prodtype[5] = { _set_, _set_, _set_, _setclass_, _set_ };
bool enable[5] = { phl->mcl.omlp, phl->mcl.omli, phl->mcl.omlu, phl->mcl.orfp, phl->mcl.orfm };
bool write[5]  = { phl->mcl.omlp, phl->mcl.omli, phl->mcl.omlu, phl->mcl.orfp, phl->mcl.orfm };
short ***ptr[5] = { &ml->mlp_, &ml->mli_, &ml->mlu_, &ml->rfp_, &ml->rfm_ };


  alloc((void**)&ML, nprod, sizeof(brick_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((ML[o] = compile_ml_brick(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(ML[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      } else {
        
        switch (prodtype[o]){
          case _set_:
            for (s=0; s<phl->mcl.nmodelset; s++){
              basename_without_ext(phl->mcl.f_model[s][0], bname, NPOW_10);
              if (strlen(bname) > NPOW_10-1){
                nchar = snprintf(domain, NPOW_10, "MODELSET-%02d", s+1);
                if (nchar < 0 || nchar >= NPOW_10){ 
                  printf("Buffer Overflow in assembling domain\n"); error++;}
              } else { 
                copy_string(domain, NPOW_10, bname);
              }
              set_brick_domain(ML[o],   s, domain);
              set_brick_bandname(ML[o], s, domain);
            }
            break;
          case _setclass_:

            for (s=0, sc=0; s<phl->mcl.nmodelset; s++){
              basename_without_ext(phl->mcl.f_model[s][0], bname, NPOW_10);
              for (c=0; c<phl->mcl.nclass[s]; c++, sc++){
                if (strlen(bname) > NPOW_10-1){
                  nchar = snprintf(domain, NPOW_10, "MODELSET-%02d_CLASS-%03d", s+1, c+1);
                  if (nchar < 0 || nchar >= NPOW_10){ 
                    printf("Buffer Overflow in assembling domain\n"); error++;}
                } else { 
                  copy_string(domain, NPOW_10, bname);
                }
                set_brick_domain(ML[o],   sc, domain);
                set_brick_bandname(ML[o], sc, domain);
              }
            }




            break;
          default:
            printf("unknown ml type.\n"); error++;
            break;
        }
      }
    } else {
      ML[o]  = NULL;
      *ptr[o] = NULL;
    }
  }


  if (error > 0){
    printf("%d compiling ML product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(ML[o]);
    free((void*)ML);
    return NULL;
  }

  *nproduct = nprod;
  return ML;
}


/** This function compiles a ML brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for ML result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_ml_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
char dname[NPOW_10];
char fname[NPOW_10];
char subname[NPOW_03];
int nchar;


  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Machine Learning");
  set_brick_product(brick, prodname);

  if (phl->subfolders){
    copy_string(subname, NPOW_03, prodname);
  } else {
    subname[0] = '\0';
  }

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick), subname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%s_HL_ML_%s", phl->mcl.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);

  if (write){
    set_brick_open(brick, OPEN_CHUNK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, &phl->gdalopt);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);
  
  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
  }

  return brick;
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
--- nproduct: number of output bricks (returned)
+++ Return:   bricks with ML results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **machine_learning(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, aux_ml_t *mod, cube_t *cube, int *nproduct){
ml_t ml;
brick_t **ML;
small *mask_ = NULL;
bool regression;
bool rf, rfprob;
float **fpred = NULL;
int   **ipred = NULL;
int nprod = 0;
int mmax = -1;
int f, s, m, k, c, sc, p, nc;
double mean, mean_old, var, std, mn;
double *mean_prob;
int max_prob, max2_prob, win_class, ntree;
short nodata;
bool valid;


  // import bricks
  nc = get_brick_chunkncells(features[0].DAT);
  nodata = get_brick_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }
  

  // compile products + bricks
  if ((ML = compile_ml(features, &ml, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile ML products!\n"); 
    *nproduct = 0;
    return NULL;
  }

  
  if (phl->mcl.method == _ML_SVR_ || phl->mcl.method == _ML_RFR_) regression = true; else regression = false;
  if (phl->mcl.method == _ML_RFR_ || phl->mcl.method == _ML_RFC_) rf = true; else rf = false;
  if (phl->mcl.orfp || phl->mcl.orfm) rfprob = true; else rfprob = false;


  // maximum number of models
  for (s=0; s<phl->mcl.nmodelset; s++){
    if (phl->mcl.nmodel[s] > mmax) mmax = phl->mcl.nmodel[s];
  }

  if (mmax < 0){
    printf("number of models is invalid\n");
    *nproduct = 0;
    return NULL;
  }



  #pragma omp parallel private(f,s,m,k,c,sc,mean,mn,mean_old,var,std,fpred,ipred,mean_prob,max_prob,max2_prob,win_class,ntree,valid) shared(features,mod,regression,rf,rfprob,ml,nf,nc,mask_,phl,ML,nodata,mmax) default(none)
  {

    alloc_2D((void***)&fpred, phl->mcl.nmodelset, mmax, sizeof(float));
    alloc_2D((void***)&ipred, phl->mcl.nmodelset, mmax, sizeof(int));
    if (rfprob) alloc((void**)&mean_prob, phl->mcl.nclass_all_sets, sizeof(double));

    #pragma omp for schedule(dynamic,1)
    for (p=0; p<nc; p++){

      // skip pixels that are masked
      if (mask_ != NULL && !mask_[p]){
        for (s=0, sc=0; s<phl->mcl.nmodelset; s++){
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = nodata;
          if (ml.mli_ != NULL) ml.mli_[s][p] = nodata;
          if (ml.mlu_ != NULL) ml.mlu_[s][p] = nodata;
          if (ml.rfm_ != NULL) ml.rfm_[s][p] = nodata;
          for (c=0; c<phl->mcl.nclass[s]; c++, sc++){
            if (ml.rfp_ != NULL) ml.rfp_[sc][p] = nodata;
          }
        }
        continue;
      }

      Mat sample(1, nf, CV_32F);
      for (f=0, valid=true; f<nf; f++){
        sample.at<float>(0,f) = features[f].dat[0][p]/10000.0;
        if (!features[f].msk[p] && phl->ftr.exclude) valid=false;
      }

      if (!valid){
        for (s=0, sc=0; s<phl->mcl.nmodelset; s++){
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = nodata;
          if (ml.mli_ != NULL) ml.mli_[s][p] = nodata;
          if (ml.mlu_ != NULL) ml.mlu_[s][p] = nodata;
          if (ml.rfm_ != NULL) ml.rfm_[s][p] = nodata;
          for (c=0; c<phl->mcl.nclass[s]; c++, sc++){
            if (ml.rfp_ != NULL) ml.rfp_[sc][p] = nodata;
          }
        }
        continue;
      }

      for (s=0, k=0, sc=0; s<phl->mcl.nmodelset; s++){
        
        mean = mean_old = var = 0;
        ntree = 0;
        if (rfprob) memset(mean_prob, 0, phl->mcl.nclass_all_sets*sizeof(double));

        for (m=0; m<phl->mcl.nmodel[s]; m++, k++){

          if (rfprob){

            Mat vote;
            mod->rf_model[k]->getVotes(sample, vote, 0);

            win_class = 0;
            max_prob  = 0;

            for (c=0; c<phl->mcl.nclass[s]; c++){
              mean_prob[c] += vote.at<int>(1,c);
              ntree += vote.at<int>(1,c);
              if (vote.at<int>(1,c) > max_prob){
                win_class = vote.at<int>(0,c);
                max_prob  = vote.at<int>(1,c);
              }
            }


            ipred[s][m] = win_class;

          } else {

            if (rf){
              fpred[s][m] = mod->rf_model[k]->predict(sample);
            } else {
              fpred[s][m] = mod->sv_model[k]->predict(sample);
            }
            ipred[s][m] = (int)fpred[s][m];

          }

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
          if (mn > SHRT_MAX) mn = SHRT_MAX;
          if (mn < SHRT_MIN) mn = SHRT_MIN;
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = (short)mn;  
          std = standdev(var, m)*10000.0;
          if (std > SHRT_MAX) std = SHRT_MAX;
          if (ml.mlu_ != NULL) ml.mlu_[s][p] = (short)(std);
        } else {
          if (ml.mlp_ != NULL) ml.mlp_[s][p] = (short)mode(ipred[s], phl->mcl.nmodel[s]);
        }

        if (ml.mli_ != NULL) ml.mli_[s][p] = m;

        if (rfprob){

          for (c=0; c<phl->mcl.nclass[s]; c++) mean_prob[c] /= (ntree*0.0001);

          if (ml.rfp_ != NULL){
            for (c=0; c<phl->mcl.nclass[s]; c++) ml.rfp_[sc++][p] = (short)mean_prob[c];
          }
          
          if (ml.rfm_ != NULL){
            max_prob = max2_prob  = 0;
            for (c=0; c<phl->mcl.nclass[s]; c++){
              if (mean_prob[c] > max_prob) max_prob = mean_prob[c];
            }
            for (c=0; c<phl->mcl.nclass[s]; c++){
              if (mean_prob[c] > max2_prob && mean_prob[c] < max_prob) max2_prob = mean_prob[c];
            }
            ml.rfm_[s][p] = max_prob-max2_prob;
          }

        }

      }


    }
    
    
    free_2D((void**)fpred, phl->mcl.nmodelset);
    free_2D((void**)ipred, phl->mcl.nmodelset);
    if (rfprob) free((void*)mean_prob);

  }

  *nproduct = nprod;
  return ML;
}

