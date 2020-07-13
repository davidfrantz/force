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
This program is for testing small things. Needs to be compiled on demand
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include <float.h>   // macro constants of the floating-point library
#include <limits.h>  // macro constants of the integer types
#include <math.h>    // common mathematical functions
#include <stdbool.h> // boolean data type
#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions
#include <ctype.h>   // transform individual characters
#include <time.h>    // date and time handling functions

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing

#include "cross-level/param-cl.h"
#include "cross-level/alloc-cl.h"
#include "cross-level/dir-cl.h"
#include "cross-level/const-cl.h"
#include "cross-level/stats-cl.h"
#include "cross-level/stack-cl.h"

#include "higher-level/read-ard-hl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdalwarper.h"     // GDAL warper related entry points and defs
#include "ogr_spatialref.h" // coordinate systems services


unsigned crc8_slow(bool *data, int len)
{
unsigned int crc = 0xff;
int i, k;

    for (i=0; i<len; i++){
        crc ^= data[i];
        for (k = 0; k < 8; k++)
            crc = crc & 1 ? (crc >> 1) ^ 0xb2 : crc >> 1;
    }

    return crc ^ 0xff;
}


stack_t *read_this_block(){
stack_t *stack  = NULL;
int nb;
int nx, ny, nc;

  nx = 3000;
  ny = 300;
  nc = nx*ny;
  nb = 20;

  stack = allocate_stack(nb, nc, _DT_SHORT_);

  set_stack_ncols(stack, nx);
  set_stack_nrows(stack, ny);
  set_stack_chunkncols(stack, nx);
  set_stack_chunknrows(stack, ny);
  set_stack_nchunks(stack, 10);
  set_stack_chunk(stack, 5);
   
  return stack;
}



ard_t *read_data(int *nt){
int t;
ard_t *ard = NULL;
int error = 0;

int n = 250;




  alloc((void**)&ard, n, sizeof(ard_t));


  #pragma omp parallel shared(ard,n) reduction(+: error) default(none)
  {

    #pragma omp for
    for (t=0; t<n; t++){

      if ((ard[t].DAT = read_this_block()) == NULL ||
          (ard[t].dat = get_bands_short(ard[t].DAT)) == NULL){
        printf("Error reading ard\n"); error++;}

      
      // compile a 0-filled QAI stack, processing must continue..
      if ((ard[t].QAI = copy_stack(ard[t].DAT, 1, _DT_SHORT_)) == NULL || 
          (ard[t].qai = get_band_short(ard[t].QAI, 0)) == NULL){
        printf("Error compiling ard.\n"); error++;}
        

      ard[t].DST = NULL; ard[t].dst = NULL;
      ard[t].AOD = NULL; ard[t].aod = NULL;
      ard[t].HOT = NULL; ard[t].hot = NULL;
      ard[t].VZN = NULL; ard[t].vzn = NULL;
      ard[t].WVP = NULL; ard[t].wvp = NULL;

    }

  }
  
  
  if (error > 0){
    printf("%d reading errors. ", error); 
    free_ard(ard, n);
    *nt = -1;
    return NULL;
  }

  *nt = n;
  return ard;
}


void read_this(ard_t **ARD1, int *nt1, int pu, int npu, int thread){


  if (pu < 0 || pu >= npu) return;

  ARD1[pu] = NULL;
  nt1[pu]  = 0;

  omp_set_num_threads(thread);


  ARD1[pu] = read_data(&nt1[pu]);


  return;
}


int screen_this(ard_t *ard, int nt){
int t;
int error = 0;


  #pragma omp parallel shared(ard,nt) reduction(+: error) default(none)
  {

    #pragma omp for
    for (t=0; t<nt; t++){
      if ((ard[t].MSK = copy_stack(ard[t].QAI, 1, _DT_SMALL_)) == NULL || 
          (ard[t].msk = get_band_small(ard[t].MSK, 0)) == NULL){
        printf("Error compiling screened QAI stack."); error++;}
    }

  }

  if (error > 0){
    printf("%d screening QAI errors. ", error); 
    return FAILURE;
  }




  return SUCCESS;
}




typedef struct {
  short **cso_[NPOW_08];
} _cso_t;

stack_t *comp_cso_stack(stack_t *from, int nb){
stack_t *stack = NULL;


  if ((stack = copy_stack(from, nb, _DT_SHORT_)) == NULL) return NULL;

  return stack;
}


stack_t **comp_cso(ard_t *ard, _cso_t *cs, int nw, int *nproduct){
stack_t **CSO = NULL;
int o, nprod = 15;
int error = 0;
short ***ptr[NPOW_08];



  for (o=0; o<nprod; o++) ptr[o] = &cs->cso_[o];


  alloc((void**)&CSO, nprod, sizeof(stack_t*));


  for (o=0; o<nprod; o++){

    if ((CSO[o] = comp_cso_stack(ard[0].QAI, nw)) == NULL || (  *ptr[o] = get_bands_short(CSO[o])) == NULL){
      printf("Error compiling product.\n"); error++;}
  }


  if (error > 0){
    printf("%d compiling CSO product errors.\n", error);
    for (o=0; o<nprod; o++) free_stack(CSO[o]);
    free((void*)CSO);
    return NULL;
  }

  *nproduct = nprod;
  return CSO;
}



stack_t **cso(ard_t *ard, int nt, int *nproduct){
_cso_t cs;
stack_t **CSO;

int o, w, p, nprod = 0;

int nc;
int nw = 5;



  // import stacks
  nc = get_stack_chunkncells(ard[0].QAI);



  // compile products + stacks
  if ((CSO = comp_cso(ard, &cs, nw, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile CSO products!\n"); 
    free((void*)CSO);
    *nproduct = 0;
    return NULL;
  }

  
  #pragma omp parallel private(o,w) shared(cs,nc,nw,nprod) default(none)
  {


    #pragma omp for
    for (p=0; p<nc; p++){


      for (w=0; w<nw; w++){

        for (o=0; o<nprod; o++){
          cs.cso_[o][w][p] = o*w;
        }

      }

    }

  }



  *nproduct = nprod;
  return CSO;
}


void compute_this(ard_t **ARD1, int *nt1, stack_t ***OUTPUT, int *nprod, int pu, int npu, int thread){
  
  
  if (pu < 0 || pu >= npu) return;
  
  OUTPUT[pu] = NULL;
  nprod[pu]  = 0;
  


  omp_set_num_threads(thread);

  if (screen_this(ARD1[pu],   nt1[pu]) != SUCCESS) printf("screen error\n");


  OUTPUT[pu] = cso(ARD1[pu], nt1[pu], &nprod[pu]);
  

  
  free_ard(ARD1[pu], nt1[pu]);


  return;
}



void output_this(stack_t ***OUTPUT, int *nprod, int pu, int npu, int thread){
int o;


  if (pu < 0 || pu >= npu) return;
 
  if (nprod[pu] == 0) return;
  if (OUTPUT[pu] == NULL) return;
  
  omp_set_num_threads(thread);

  // would write output here

  for (o=0; o<nprod[pu]; o++) free_stack(OUTPUT[pu][o]);
  free((void*)OUTPUT[pu]);
  OUTPUT[pu] = NULL;

  return;
}



int main ( int argc, char *argv[] ){
ard_t   **ARD1    = NULL;
int     *nt1      = NULL;
stack_t ***OUTPUT = NULL;
int     *nprod    = NULL;
int ithread;
int cthread;
int othread;
int pu, pu_prev, pu_next, npu = 300;



unsigned crc1;;
unsigned crc2;;
unsigned crc3;;

bool buf1[12] = { 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1 };
bool buf2[12] = { 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 };
bool buf3[12] = { 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0 };

char *CRED = NULL;


  if ((CRED = getenv("FORCE_CREDENTIALS")) == NULL){
    printf("Environment variable not set.\n");
  } else {
    printf("Environment variable: %s\n", CRED);
  }

  exit(1);


crc1 = crc8_slow(buf1, 12);
crc2 = crc8_slow(buf2, 12);
crc3 = crc8_slow(buf3, 12);

printf("%#02x\n", crc1);
printf("%#02x\n", crc2);
printf("%#02x\n", crc3);



  if (argc != 4){ 
    printf("usage: %s\n", argv[0]);
    return FAILURE;
  }

  ithread = atoi(argv[1]);
  cthread = atoi(argv[2]);
  othread = atoi(argv[3]);


  alloc((void**)&ARD1,   npu, sizeof(ard_t*));
  alloc((void**)&OUTPUT, npu, sizeof(stack_t**));
  alloc((void**)&nprod,  npu, sizeof(int));
  alloc((void**)&nt1,    npu, sizeof(int));
  
  
  // enable nested threading
  //if (omp_get_thread_limit() < (ithread+othread+cthread)){
  //  printf("Number of threads exceeds system limit\n"); return FAILURE;}
  //omp_set_nested(true);
  //omp_set_max_active_levels(2);
  omp_set_max_active_levels(2);
  
  
  for (pu=-1; pu<=npu; pu++){
    
    pu_prev = pu-1;
    pu_next = pu+1;

    #pragma omp parallel num_threads(3) shared(ARD1,nt1,OUTPUT,nprod,pu,pu_next,pu_prev,npu,ithread,othread,cthread) default(none)
    {

      if (omp_get_thread_num() == 0){
        read_this(ARD1, nt1, pu_next, npu, ithread);
      } else if (omp_get_thread_num() == 1){
        compute_this(ARD1, nt1, OUTPUT, nprod, pu, npu, cthread);
      } else {
        output_this(OUTPUT, nprod, pu_prev, npu, othread);
      }

    }
    
  }
  
  

  free((void*)ARD1);
  free((void*)OUTPUT);
  free((void*)nt1);
  free((void*)nprod);


  return 0; 
}


