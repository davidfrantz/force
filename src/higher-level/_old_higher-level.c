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
This program is the FORCE Higher Level Processing System 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/stack-cl.h"
#include "../cross-level/tile-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/cite-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/quality-hl.h"
#include "../higher-level/level3-hl.h"
#include "../higher-level/tsa-hl.h"
#include "../higher-level/cso-hl.h"
#include "../higher-level/ml-hl.h"
#include "../higher-level/texture-hl.h"
#include "../higher-level/sample-hl.h"
#include "../higher-level/cf-improphe-hl.h"
#include "../higher-level/l2-improphe-hl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_multiproc.h"  // CPL Multi-Threading

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing


int main ( int argc, char *argv[] ){
par_hl_t *phl = NULL; // can be renamed to par, once par is not global anymore...
aux_t *aux = NULL;
ard_t **ARD = NULL;
ard_t **ARD_SECONDARY = NULL;

stack_t **MASK = NULL;
int mask_status;
int *nt = NULL;
int *nt_secondary = NULL;
//bool *MASK = NULL;
cube_t *cube = NULL;

stack_t ***OUTPUT;
int o, *nprod = NULL;

char *lock = NULL;
char dname[NPOW_10];
int tx = -1, tx_prev = -1, tx_next = -1;
int ty = -1, ty_prev = -1, ty_next = -1;



int read_error = 0;
int write_error = 0;
int compute_error = 0;
int nthread;
GDALDriverH driver;
//double geotran[6]   = { 0, 0, 0, 0, 0, 0 };

int line;

progress_t pro;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 2){ printf("usage: %s parameter-file\n", argv[0]); return FAILURE;}



  /** INITIALIZING
  +** *******************************************************************/

  phl = allocate_param_higher();

  phl->f_par = argv[1];

  // parse parameter file
  if (parse_param_higher(phl) == FAILURE){
    printf("Reading parameter file failed!\n"); return FAILURE;}

  // initialize citation file
  if (cite_open(phl->d_higher) == FAILURE){
    printf("Opening citation file failed!\n"); return FAILURE;}

  // parse auxiliary files
  if ((aux = read_aux(phl)) == NULL){
    printf("Reading aux file failed!\n"); return FAILURE;}

  // copy and read datacube definition
  if ((cube = copy_datacube_def(phl->d_lower, phl->d_higher, phl->blocksize)) == NULL){
    printf("Copying datacube definition failed.\n"); return FAILURE;}

  // update datacube with parameters
  update_datacube_extent(cube, phl->tx[_MIN_], phl->tx[_MAX_], phl->ty[_MIN_], phl->ty[_MAX_]);
  update_datacube_res(cube, phl->res);

  // compile active tiles
  if (tile_active(phl->f_tile, cube) == FAILURE){
    printf("Compiling active tiles failed!\n"); return FAILURE;}


  init_progess(&pro);


  printf("number of processing units: %d\n", pro->npu);
  printf(" (active tiles: %d, chunks per tile: %d)\n", cube->tn, cube->cn);

  // allocate array of Level 2 structs
  alloc((void**)&ARD,           pro->npu, sizeof(ard_t*));
  alloc((void**)&ARD_SECONDARY, pro->npu, sizeof(ard_t*));
  alloc((void**)&MASK,          pro->npu, sizeof(stack_t*));
  alloc((void**)&OUTPUT,        pro->npu, sizeof(stack_t**));
  alloc((void**)&nprod,         pro->npu, sizeof(int));
  alloc((void**)&nt,            pro->npu, sizeof(int));
  alloc((void**)&nt_secondary,  pro->npu, sizeof(int));


  // enable nested threading
  nthread = phl->ithread + phl->othread + phl->cthread;
  if (omp_get_thread_limit() < nthread){
    printf("Number of threads exceeds system limit\n"); return FAILURE;}
  omp_set_nested(true);
  omp_set_max_active_levels(2);


  GDALAllRegister();
  if ((driver = GDALGetDriverByName("JP2ECW")) != NULL) GDALDeregisterDriver(driver);
  

  /** LOOP OVER ALL CHUNKS
  +** *******************************************************************/

  for (pro->pu=-1; pro->pu<=pro->npu; pro->pu++){

    progress(&pro);


    #pragma omp parallel num_threads(3) private(mask_status,dname,lock,o,TIME_I,TIME_C,TIME_O) shared(ARD,ARD_SECONDARY,MASK,OUTPUT,aux,nprod,nt,nt_secondary,pu,pu_prev,pu_next,npu,tile,tile_prev,tile_next,chunk,chunk_prev,chunk_next,tx,tx_prev,tx_next,ty,ty_prev,ty_next,cube,phl,secs_i,secs_c,secs_o) reduction(+: read_error, write_error, compute_error) default(none)
    {

      /** PRELOAD NEXT CHUNK in reading team
      +** *****************************************************************/

      if (omp_get_thread_num() == 0){

        if (read_this_chunk(&pro)){

          time(&TIME_I);

          //printf("reading ARD tile X%04d_Y%04d chunk %d with %d threads.\n", tx_next, ty_next, chunk_next, phl->ithread); 

          omp_set_num_threads(phl->ithread);

          if ((MASK[pro->pu_next] = read_mask(&mask_status, tx_next, ty_next, chunk_next, cube, phl)) == NULL && mask_status != SUCCESS){
            
            if (mask_status == FAILURE){
              printf("error reading mask tile X%04d_Y%04d chunk %d.\n", tx_next, ty_next, chunk_next); 
              read_error++;
            } else if (mask_status == CANCEL){
              printf("no mask data. skip block.\n");
            }
            
            ARD[pro->pu_next] = ARD_SECONDARY[pro->pu_next] = NULL;
            nt[pro->pu_next]  = nt_secondary[pro->pu_next]  = 0;

          } else {

            ARD[pro->pu_next] = NULL;
            nt[pro->pu_next]  = 0;

            if (phl->input_level == _INP_FTR_){
              ARD[pro->pu_next] = read_features(&nt[pro->pu_next], tx_next, ty_next, chunk_next, cube, phl);
            } else if (phl->input_level == _INP_CON_){
              ARD[pro->pu_next] = read_confield(&nt[pro->pu_next], tx_next, ty_next, chunk_next, cube, phl);
            } else if (phl->input_level == _INP_ARD_ || phl->input_level == _INP_QAI_){
              ARD[pro->pu_next] = read_ard(&nt[pro->pu_next], tx_next, ty_next, chunk_next, cube, &phl->sen, phl);
            } else {
              printf("unknown input level\n"); read_error++;
            }

            if (ARD[pro->pu_next] == NULL && nt[pro->pu_next] < 0){
              printf("error reading data from tile X%04d_Y%04d chunk %d.\n", tx_next, ty_next, chunk_next); read_error++;}


            ARD_SECONDARY[pro->pu_next] = NULL;
            nt_secondary[pro->pu_next]  = 0;

            if (phl->input_level2 == _INP_FTR_){
              ARD_SECONDARY[pro->pu_next] = read_features(&nt_secondary[pro->pu_next], tx_next, ty_next, chunk_next, cube, phl);
            } else if (phl->input_level2 == _INP_CON_){
              ARD_SECONDARY[pro->pu_next] = read_confield(&nt_secondary[pro->pu_next], tx_next, ty_next, chunk_next, cube, phl);
            } else if (phl->input_level2 == _INP_ARD_ || phl->input_level2 == _INP_QAI_){
              ARD_SECONDARY[pro->pu_next] = read_ard(&nt_secondary[pro->pu_next], tx_next, ty_next, chunk_next, cube, &phl->sen2, phl);
            } else {
              printf("unknown input level\n"); read_error++;
            }

            if (ARD_SECONDARY[pro->pu_next] == NULL && nt_secondary[pro->pu_next] < 0){
              printf("error reading secondary data from tile X%04d_Y%04d chunk %d.\n", tx_next, ty_next, chunk_next); read_error++;}

          }

          secs_i = proctime(TIME_I);

        }

      /** COMPUTE THIS CHUNK in processing team
      +** *****************************************************************/

      } else if (omp_get_thread_num() == 1){

        // if data was loaded, do higher-level computing
        if (compute_this_chunk(&pro)){

          if (nt[pro->pu] > 0){

            time(&TIME_C);

            //printf("computing ARD tile X%04d_Y%04d chunk %d with %d threads.\n", tx, ty, chunk, phl->cthread); 

            omp_set_num_threads(phl->cthread);

            if (screen_qai(ARD[pro->pu], nt[pro->pu], &phl->qai)   != SUCCESS) compute_error++;
            if (screen_noise(ARD[pro->pu], nt[pro->pu], &phl->qai) == FAILURE) compute_error++;
            if (nt_secondary[pro->pu] > 0){
              if (screen_qai(ARD_SECONDARY[pro->pu], nt_secondary[pro->pu], &phl->qai)   != SUCCESS) compute_error++;
              if (screen_noise(ARD_SECONDARY[pro->pu], nt_secondary[pro->pu], &phl->qai) == FAILURE) compute_error++;
            }


            switch (phl->type){
              case _HL_BAP_:
                OUTPUT[pro->pu] = level3(ARD[pro->pu], ARD_SECONDARY[pro->pu], MASK[pro->pu], nt[pro->pu], nt_secondary[pro->pu], phl, cube, &nprod[pro->pu]);
                break;
              case _HL_TSA_:
                OUTPUT[pro->pu] = time_series_analysis(ARD[pro->pu], MASK[pro->pu], nt[pro->pu], phl, aux->endmember, cube, &nprod[pro->pu]);
                break;
              case _HL_CSO_:
                OUTPUT[pro->pu] = clear_sky_observations(ARD[pro->pu], MASK[pro->pu], nt[pro->pu], phl, cube, &nprod[pro->pu]);
                break;
              case _HL_ML_:
                OUTPUT[pro->pu] = machine_learning(ARD[pro->pu], MASK[pro->pu], nt[pro->pu], phl, aux->ml_model, cube, &nprod[pro->pu]);
                break;
              case _HL_SMP_:
                OUTPUT[pro->pu] = sample_points(ARD[pro->pu], MASK[pro->pu], nt[pro->pu], phl, cube, &nprod[pro->pu]);
                break;
              case _HL_TXT_:
                OUTPUT[pro->pu] = texture(ARD[pro->pu], MASK[pro->pu], nt[pro->pu], phl, cube, &nprod[pro->pu]);
                break;
              case _HL_LSM_:
                //OUTPUT[pro->pu] = landscape_metrics(ARD[pro->pu], MASK[pro->pu], nt[pro->pu], phl, cube, &nprod[pro->pu]);
                OUTPUT[pro->pu] = NULL;
                nprod[pro->pu]  = 0;
                break;
              case _HL_L2I_:
                OUTPUT[pro->pu] = level2_improphe(ARD[pro->pu], ARD_SECONDARY[pro->pu], MASK[pro->pu], nt[pro->pu], nt_secondary[pro->pu], phl, cube, &nprod[pro->pu]);
                break;
              case _HL_CFI_:
                OUTPUT[pro->pu] = confield_improphe(ARD[pro->pu], ARD_SECONDARY[pro->pu], MASK[pro->pu], nt[pro->pu], nt_secondary[pro->pu], phl, cube, &nprod[pro->pu]);
                break;
              default:
                OUTPUT[pro->pu] = NULL;
                nprod[pro->pu]  = 0;
                break;
            }

            secs_c = proctime(TIME_C);

          } else {
            
            OUTPUT[pro->pu] = NULL;
            nprod[pro->pu]  = 0;

          }

          free_ard(ARD[pro->pu], nt[pro->pu]);
          free_ard(ARD_SECONDARY[pro->pu], nt_secondary[pro->pu]);
          free_stack(MASK[pro->pu]);

        }
        
      /** OUTPUT PREVIOUS CHUNK in writing team
      +** *****************************************************************/
        
      } else {


        time(&TIME_O);

        if (write_this_chunk(&pro, nprod)){
          
          //printf("writing ARD tile X%04d_Y%04d chunk %d with %d threads.\n", tx_prev, ty_prev, chunk_prev, phl->othread); 


          sprintf(dname, "%s/X%04d_Y%04d", phl->d_higher, tx_prev, ty_prev); 
          
          // output path
          if ((lock = (char*)CPLLockFile(dname, 60)) == NULL){
            
            printf("Unable to lock directory %s (timeout: %ds). ", dname, 60);
            write_error++;

          } else {

            createdir(dname);
            CPLUnlockFile(lock);
            lock = NULL;

            omp_set_num_threads(phl->othread);
          
            if (OUTPUT[pro->pu_prev] != NULL){

              #pragma omp parallel shared(OUTPUT,pu_prev,nprod,phl) default(none)
              {

                #pragma omp for schedule(dynamic,1)
                for (o=0; o<nprod[pro->pu_prev]; o++){
                  if (phl->radius > 0) OUTPUT[pro->pu_prev][o] = crop_stack(OUTPUT[pro->pu_prev][o], phl->radius);
                  write_stack(OUTPUT[pro->pu_prev][o]);
                }

              }
              
            }
            
          }

          if (OUTPUT[pro->pu_prev] != NULL){
            for (o=0; o<nprod[pro->pu_prev]; o++) free_stack(OUTPUT[pro->pu_prev][o]);
            free((void*)OUTPUT[pro->pu_prev]);
            OUTPUT[pro->pu_prev] = NULL;
          }

        }
        
        secs_o = proctime(TIME_O);
        
      }


    }

    
    /** PROGRESS
    +********************************************************************/

    printf("Time (sec):      %7.0f %7.0f %7.0f\n", secs_i, secs_c, secs_o);

  }


  /** CLEAN
  +**********************************************************************/
  free((void*)ARD);
  free((void*)ARD_SECONDARY);
  free((void*)MASK);
  free((void*)OUTPUT);
  free((void*)nt);
  free((void*)nt_secondary);
  free((void*)nprod);
  free_datacube(cube);
  free_aux(phl, aux);
  free_param_higher(phl);
  
  cite_close();


  /** PROGRESS and EXECUTION TIME
  +**********************************************************************/
  secs_total_i += secs_i;
  secs_total_c += secs_c;
  secs_total_o += secs_o;
  secs_bound_i += MAX_(0, secs_i - MAX_(secs_c, secs_o));
  secs_bound_c += MAX_(0, secs_c - MAX_(secs_i, secs_o));
  secs_bound_o += MAX_(0, secs_o - MAX_(secs_i, secs_c));
  secs_total_a = secs_total_i + secs_total_c + secs_total_o;
  secs_total_t = proctime(TIME);

  printf("\n________________________________________\n");  
  printf("Progress:                       100.00%%\n");
  if (secs_total_a > 0){
    printf("Time for I/C/O:           %03.0f%%/%03.0f%%/%03.0f%%\n", 
      round(secs_total_i/secs_total_a*100), round(secs_total_c/secs_total_a*100), round(secs_total_o/secs_total_a*100));
  } else {
    printf("Time for I/C/O:        not available yet\n");
  }


  printf("\n________________________________________\n");
  set_secs(&total,     secs_total_t);        // total real time
  set_secs(&virtual_a, secs_total_a);        // total time spent for I/C/O when streaming wouldn't take place
  set_secs(&save,      secs_total_a-secs_total_t); // time saved through streaming relative to when streaming wouldn't take place
  printf("Real time:       %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    total.year, total.month, total.day, total.hh, total.mm, total.ss);  
  printf("Virtual time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    virtual_a.year, virtual_a.month, virtual_a.day, virtual_a.hh, virtual_a.mm, virtual_a.ss);
  printf("Saved time:      %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    save.year, save.month, save.day, save.hh, save.mm, save.ss);

  printf("\n________________________________________\n");
  set_secs(&virtual_i, secs_total_i);
  set_secs(&virtual_c, secs_total_c);
  set_secs(&virtual_o, secs_total_o);
  printf("Virtual I-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    virtual_i.year, virtual_i.month, virtual_i.day, virtual_i.hh, virtual_i.mm, virtual_i.ss);
  printf("Virtual C-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    virtual_c.year, virtual_c.month, virtual_c.day, virtual_c.hh, virtual_c.mm, virtual_c.ss);
  printf("Virtual O-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    virtual_o.year, virtual_o.month, virtual_o.day, virtual_o.hh, virtual_o.mm, virtual_o.ss);

  printf("\n________________________________________\n");
  set_secs(&bound_i, secs_bound_i);
  set_secs(&bound_c, secs_bound_c);
  set_secs(&bound_o, secs_bound_o);
  printf("I-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    bound_i.year, bound_i.month, bound_i.day, bound_i.hh, bound_i.mm, bound_i.ss);
  printf("C-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    bound_c.year, bound_c.month, bound_c.day, bound_c.hh, bound_c.mm, bound_c.ss);
  printf("O-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    bound_o.year, bound_o.month, bound_o.day, bound_o.hh, bound_o.mm, bound_o.ss);

  printf("\n________________________________________\n");
  printf("%d read, %d write, %d compute errors\n", 
    read_error, write_error, compute_error);
  printf("\n");

  return SUCCESS;
}

