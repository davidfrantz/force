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
This file contains functions for reading aux files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-aux-hl.h"

#include <opencv2/ml.hpp>


int read_endmember(par_hl_t *phl, aux_t *aux);
int read_machine_learner(par_hl_t *phl, aux_t *aux);
int read_libraries(par_hl_t *phl, aux_t *aux);
int read_samples(par_hl_t *phl, aux_t *aux);


/** This function reads endmembers from a text file. Put each endmember in
+++ a separate column, separated by space. Do not use a header. Put prima-
+++ ry endmember into the first column, and if applicable, shade endmember
+++ into the last column. Values are integers, scaled by 10000.
--- phl:    HL parameters
--- aux:    auxilliary data
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_endmember(par_hl_t *phl, aux_t *aux){
int b, e = 0;
float scale = 10000.0;


  if ((aux->endmember.tab = read_table_deprecated(phl->tsa.sma.f_emb, 
      &aux->endmember.nb, &aux->endmember.ne)) == NULL){
    printf("unable to read endmembers. "); return FAILURE;}

  for (b=0; b<aux->endmember.nb; b++){
  for (e=0; e<aux->endmember.ne; e++){
    aux->endmember.tab[b][e] /= scale;
    if (aux->endmember.tab[b][e] < 0 || aux->endmember.tab[b][e] > 1){
      printf("\nendmember file is invalid.\n");
      printf("valid range of endmembers: [0,10000].\n");
      return FAILURE;
    }
  }
  }

  return SUCCESS;
}


/** This function reads a machine learning model in OpenCV xml format. 
--- phl:    HL parameters
--- aux:    auxilliary data
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_machine_learner(par_hl_t *phl, aux_t *aux){
int s, m;
char fname[NPOW_10];
int nchar;


  phl->mcl.nclass_all_sets = 0;


  for (s=0; s<phl->mcl.nmodelset; s++){
    
    phl->mcl.nclass[s] = 0;

    for (m=0; m<phl->mcl.nmodel[s]; m++){

      nchar = snprintf(fname, NPOW_10, "%s/%s", phl->mcl.d_model, phl->mcl.f_model[s][m]);
      if (nchar < 0 || nchar >= NPOW_10){
        printf("Buffer Overflow in assembling filename\n"); return FAILURE;}
        
      if (!fileexist(fname)){
        printf("Model %s does not exist.\n", fname); return FAILURE;}

      if (phl->mcl.method == _ML_SVR_ || 
          phl->mcl.method == _ML_SVC_){
            
        cv::Ptr<cv::ml::SVM> newmodel = cv::ml::SVM::create();
        newmodel = cv::ml::SVM::load(fname);
        aux->ml.sv_model.push_back(newmodel);
        
      } else if (phl->mcl.method == _ML_RFR_ || 
                 phl->mcl.method == _ML_RFC_){

        cv::Ptr<cv::ml::RTrees> newmodel = cv::ml::RTrees::create();
        newmodel = cv::ml::RTrees::load(fname);
        aux->ml.rf_model.push_back(newmodel);

        if (phl->mcl.method == _ML_RFC_){

          // number of classes need to be known to compute RF probability
          // ... I hate this piece of code, but didn't find another way to 
          //     catch number of classes in the classifier...
          cv::Mat sample(1, phl->ftr.nfeature, CV_32F);
          cv::Mat vote;
          newmodel->getVotes(sample, vote, 0);

          phl->mcl.nclass[s] = vote.cols;
          
          sample.release();
          vote.release();

        }

      }
    }

    phl->mcl.nclass_all_sets += phl->mcl.nclass[s];

  }

  return SUCCESS;
}


/** This function reads libraries from text files. Put each feature in
+++ a separate column, separated by space. Samples go to the lines. Do not
+++ use a header. For each library, the number of samples is allowed to 
+++ differ, but number of features need to be consistent.
+++ Mean and standard deviation are computed for ach table and feature;
+++ the tables may be rescaled.
--- phl:    HL parameters
--- aux:    auxilliary data
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_libraries(par_hl_t *phl, aux_t *aux){
int i, s, f, nsample, nfeatures, nfeatures_first;
char fname[NPOW_10];
int nchar;
double mx, vx, k;


  aux->library.n = phl->lib.n_lib;
  alloc((void**)&aux->library.tab, aux->library.n, sizeof(double**));
  alloc((void**)&aux->library.ns,  aux->library.n, sizeof(int));
  

  for (i=0; i<phl->lib.n_lib; i++){
    
    nchar = snprintf(fname, NPOW_10, "%s/%s", phl->lib.d_lib, phl->lib.f_lib[i]);
    if (nchar < 0 || nchar >= NPOW_10){
      printf("Buffer Overflow in assembling filename\n"); return FAILURE;}
    
    if ((aux->library.tab[i] = read_table_deprecated(fname, &nsample, &nfeatures)) == NULL){
      printf("unable to read library. "); return FAILURE;}
      
    aux->library.ns[i] = nsample;
    aux->library.nf    = nfeatures;
    
    if (i == 0) nfeatures_first = nfeatures;

    if (aux->library.nf != nfeatures_first){
      printf("number of features is different between libraries. "); return FAILURE;}

  }

  
  alloc_2D((void***)&aux->library.mean, aux->library.n, aux->library.nf, sizeof(double));
  alloc_2D((void***)&aux->library.sd,   aux->library.n, aux->library.nf, sizeof(double));
  
   
  for (i=0; i<aux->library.n;  i++){
  for (f=0; f<aux->library.nf; f++){

    mx = vx = k = 0;

    for (s=0; s<aux->library.ns[i]; s++){

      k++;
      if (k == 1){
        mx = aux->library.tab[i][s][f];
      } else {
        var_recurrence(aux->library.tab[i][s][f], &mx, &vx, k);
      }

    }
    
    aux->library.mean[i][f] = mx;
    aux->library.sd[i][f]   = standdev(vx, k);
    
    #ifdef FORCE_DEBUG
    printf("library %d, feature %d, mean: %f, sd: %f\n", i, f, aux->library.mean[i][f], aux->library.sd[i][f]);
    #endif
    
    if (phl->lib.rescale){
      for (s=0; s<aux->library.ns[i]; s++) aux->library.tab[i][s][f] = 
        (aux->library.tab[i][s][f]-aux->library.mean[i][f])/aux->library.sd[i][f];
    }

  }
  }

  if (phl->lib.rescale) aux->library.scaled = true;


  return SUCCESS;
}


/** This function reads sample locations from text files. Samples go to 
+++ the lines. Do not use a header. Put X-coords in 1st column and Y-
+++ coords in 2nd column. Coordinates must be in geographic decimal 
+++ degree notation (South and West coordinates are negative). There needs
+++ to be at least one additional column with a response variable (class 
+++ ID or some quantitative value). Multiple variables are supported-
--- phl:    HL parameters
--- aux:    auxilliary data
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_samples(par_hl_t *phl, aux_t *aux){
int ns, nr;


  if ((aux->sample.tab = read_table_deprecated(phl->smp.f_coord, &ns, &nr)) == NULL){
    printf("unable to read samples. "); return FAILURE;}

  if (nr < 3){
    printf("Less than 3 columns. "); return FAILURE;}

  alloc((void**)&aux->sample.visited, ns, sizeof(bool));

  aux->sample.ns = ns;
  aux->sample.nr = nr;
  aux->sample.nleft = ns;

  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function reads all applicable auxilliary data that should be
+++ read once on the main level (not within processing units).
--- phl:    HL parameters
+++ Return: auxilliary data
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
aux_t *read_aux(par_hl_t *phl){
aux_t *aux;


  alloc((void**)&aux, 1, sizeof(aux_t));
  
  if (phl->type == _HL_TSA_ && phl->tsa.sma.v){
    if (read_endmember(phl, aux) == FAILURE){
      printf("reading endmember file failed. "); 
      free_aux(phl, aux);
      return NULL;
    }
  } else aux->endmember.tab = NULL;

  if (phl->type == _HL_ML_){
    if (read_machine_learner(phl, aux) == FAILURE){
      printf("reading ML model file(s) failed. ");
      free_aux(phl, aux);
      return NULL;
    }
  }

  if (phl->type == _HL_LIB_){
    if (read_libraries(phl, aux) == FAILURE){
      printf("reading library file(s) failed. ");
      free_aux(phl, aux);
      return NULL;
    }
  }

  if (phl->type == _HL_SMP_){
    if (read_samples(phl, aux) == FAILURE){
      printf("reading sample file failed. ");
      free_aux(phl, aux);
      return NULL;
    }
  }
  

  return aux;
}


/** This function frees all auxilliary data.
--- phl:    HL parameters
--- aux:    auxilliary data
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_aux(par_hl_t *phl, aux_t *aux){
int i;
  
  if (aux != NULL){
  
    if (phl->type == _HL_TSA_ && phl->tsa.sma.v && aux->endmember.tab != NULL){
      free_2D((void**)aux->endmember.tab, aux->endmember.nb);
    }
    
    if (phl->type == _HL_LIB_&& 
        aux->library.tab  != NULL && aux->library.ns != NULL &&
        aux->library.mean != NULL && aux->library.sd != NULL){
      for (i=0; i<aux->library.n; i++) free_2D((void**)aux->library.tab[i], aux->library.ns[i]);
      free((void*)aux->library.tab);
      free((void*)aux->library.ns);
      free_2D((void**)aux->library.mean, aux->library.n);
      free_2D((void**)aux->library.sd,   aux->library.n);
    }
    
    if (phl->type == _HL_SMP_){
      free_2D((void**)aux->sample.tab, aux->sample.ns);
      free((void*)aux->sample.visited);
    }

    free((void*)aux); aux = NULL;

  }

  return;
}

