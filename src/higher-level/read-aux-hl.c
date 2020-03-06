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
This file contains functions for reading aux files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-aux-hl.h"


int read_endmember(par_hl_t *phl, aux_t *aux);
int read_machine_learner(par_hl_t *phl, aux_t *aux);


/** This function reads endmembers from a text file. Put each endmember in
+++ a separate column, separated by space. Do not use a header. Put prima-
+++ ry endmember into the first column, and if applicable, shade endmember
+++ into the last column. Values are integers, scaled by 10000.
--- phl:    HL parameters
--- aux:    auxilliary data
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_endmember(par_hl_t *phl, aux_t *aux){
FILE *fp;
char  buffer[NPOW_10] = "\0";
char *ptr = NULL;
const char *separator = " ";
int b, e = 0;
float scale = 10000.0;


  // open file
  if (!(fp = fopen(phl->tsa.sma.f_emb, "r"))){
    printf("unable to open endmember file. "); return FAILURE;}
    
  b = 0;
  while (fgets(buffer, NPOW_10, fp) != NULL){

    ptr = strtok(buffer, separator);
    e = 0;

    while (ptr != NULL){
      ptr = strtok(NULL, separator);
      e++;
    }

    b++;

  }
  
  fseek(fp, 0, SEEK_SET);

  phl->tsa.sma.nb = b;
  phl->tsa.sma.ne = e;
  alloc_2D((void***)&aux->endmember, phl->tsa.sma.nb, phl->tsa.sma.ne, sizeof(float));

  b = 0;
  while (fgets(buffer, NPOW_10, fp) != NULL){

    ptr = strtok(buffer, separator);
    e = 0;

    while (ptr != NULL){
      aux->endmember[b][e] = atof(ptr)/scale;
      if (aux->endmember[b][e] < 0 || aux->endmember[b][e] > 1){
        printf("\nendmember file is invalid.\n");
        printf("valid range: [0,10000].\n");
        fclose(fp);
        return FAILURE;
      }
      ptr = strtok(NULL, separator);
      e++;
    }

    b++;

  }
  
  fclose(fp);
  
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


  for (s=0; s<phl->mcl.nmodelset; s++){

    for (m=0; m<phl->mcl.nmodel[s]; m++){

      nchar = snprintf(fname, NPOW_10, "%s/%s", phl->mcl.d_model, phl->mcl.f_model[s][m]);
      if (nchar < 0 || nchar >= NPOW_10){
        printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

      if (phl->mcl.method == _ML_SVR_ || 
          phl->mcl.method == _ML_SVC_){
        cv::Ptr<cv::ml::SVM> newmodel = cv::ml::SVM::create();
        newmodel = cv::ml::SVM::load(fname);
        aux->ml_model.push_back(newmodel);
      } else if (phl->mcl.method == _ML_RFR_ || 
                 phl->mcl.method == _ML_RFC_){
        cv::Ptr<cv::ml::RTrees> newmodel = cv::ml::RTrees::create();
        newmodel = cv::ml::RTrees::load(fname);
        aux->ml_model.push_back(newmodel);
      }
    }

  }

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
  } else aux->endmember = NULL;

  if (phl->type == _HL_ML_){
    if (read_machine_learner(phl, aux) == FAILURE){
      printf("reading ML model file(s) failed. ");
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
  
  
  if (aux != NULL){
  
    if (phl->type == _HL_TSA_ && phl->tsa.sma.v && aux->endmember != NULL){
      free_2D((void**)aux->endmember, phl->tsa.sma.nb);
    }
    
    free((void*)aux); aux = NULL;

  }

  return;
}

