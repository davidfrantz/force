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
This file contains functions for higher level tasks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "tasks-hl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"      // various convenience functions for CPL
#include "cpl_multiproc.h" // CPL Multi-Threading

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing

/** This function handles the reading tasks
--- pro:      progress handle
--- ibytes:   number of bytes read
--- MASK:     mask image
--- ARD1:     primary   ARD
--- ARD2:     secondary ARD
--- nt1:      number of primary   ARD products
--- nt2:      number of secondary ARD products
--- cube:     datacube definition
--- phl:      HL parameters
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void read_higher_level (progress_t *pro, off_t *ibytes, brick_t **MASK, ard_t **ARD1, ard_t **ARD2, int *nt1, int *nt2, cube_t *cube, par_hl_t *phl){
int mask_status;
off_t bytes = 0;

  if (!read_this_chunk(pro)) return;

  int pu = pro->next.processing_unit;
  int tile[2] = { pro->next.tile[_X_], pro->next.tile[_Y_] };
  int chunk[2] = { pro->next.chunk[_X_], pro->next.chunk[_Y_] };

  MASK[pu] = NULL;
  ARD1[pu] = NULL;
  ARD2[pu] = NULL;
  nt1[pu]  = 0;
  nt2[pu]  = 0;

  measure_progress(pro, _TASK_INPUT_, _CLOCK_TICK_);

  omp_set_num_threads(phl->ithread);

  MASK[pu] = read_mask(&mask_status, &bytes, tile, chunk, cube, phl);

  if (MASK[pu] == NULL && mask_status != SUCCESS){
    if (mask_status == FAILURE){
      printf("error reading mask tile X%04d_Y%04d chunk X:%d Y:%d.\n", 
        tile[_X_], tile[_Y_], chunk[_X_], chunk[_Y_]);
    } else if (mask_status == CANCEL){
      //printf("no mask data. skip chunk.\n");
    }
    measure_progress(pro, _TASK_INPUT_, _CLOCK_TOCK_);
    return;
  }


  if (phl->input_level1 == _INP_FTR_){
    ARD1[pu] = read_features(&bytes, &nt1[pu], tile, chunk, cube, phl);
  } else if (phl->input_level1 == _INP_CON_){
    ARD1[pu] = read_confield(&bytes, &nt1[pu], tile, chunk, cube, phl);
  } else if (phl->input_level1 == _INP_ARD_ || phl->input_level1 == _INP_QAI_){
    ARD1[pu] = read_ard(&bytes, &nt1[pu], tile, chunk, cube, &phl->sen, phl);
  } else if (phl->input_level1 != _INP_NONE_) {
    printf("unknown input level\n");
  }

  if (ARD1[pu] == NULL && nt1[pu] < 0){
    printf("error reading data from tile X%04d_Y%04d chunk X:%d Y:%d.\n", 
      tile[_X_], tile[_Y_], chunk[_X_], chunk[_Y_]);
    measure_progress(pro, _TASK_INPUT_, _CLOCK_TOCK_);
    return;
  }


  if (phl->input_level2 == _INP_FTR_){
    ARD2[pu] = read_features(&bytes, &nt2[pu], tile, chunk, cube, phl);
  } else if (phl->input_level2 == _INP_CON_){
    ARD2[pu] = read_confield(&bytes, &nt2[pu], tile, chunk, cube, phl);
  } else if (phl->input_level2 == _INP_ARD_ || phl->input_level2 == _INP_QAI_){
    ARD2[pu] = read_ard(&bytes, &nt2[pu], tile, chunk, cube, &phl->sen2, phl);
  } else if (phl->input_level2 != _INP_NONE_){
    printf("unknown input level\n");
  }

  if (ARD2[pu] == NULL && nt2[pu] < 0){
    printf("error reading secondary data from tile X%04d_Y%04d chunk X:%d Y:%d.\n", 
      tile[_X_], tile[_Y_], chunk[_X_], chunk[_Y_]);
    measure_progress(pro, _TASK_INPUT_, _CLOCK_TOCK_);
    return;
  }


  *ibytes = bytes;

  measure_progress(pro, _TASK_INPUT_, _CLOCK_TOCK_);

  return;
}


/** This function handles the computing tasks
--- pro:      progress handle
--- MASK:     mask image
--- ARD1:     primary   ARD
--- ARD2:     secondary ARD
--- nt1:      number of primary   ARD products
--- nt2:      number of secondary ARD products
--- cube:     datacube definition
--- phl:      HL parameters
--- aux:      auxilliary data
--- OUTPUT:   OUTPUT bricks
--- nproduct: number of output bricks
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compute_higher_level (progress_t *pro, brick_t **MASK, ard_t **ARD1, ard_t **ARD2, int *nt1, int *nt2, cube_t *cube, par_hl_t *phl, aux_t *aux, brick_t ***OUTPUT, int *nprod){
bool error = false;


  if (!compute_this_chunk(pro)) return;

  int pu = pro->now.processing_unit;


  OUTPUT[pu] = NULL;
  nprod[pu]  = 0;


  measure_progress(pro, _TASK_COMPUTE_, _CLOCK_TICK_);

  omp_set_num_threads(phl->cthread);


  if (nt1[pu] > 0){
    if (screen_qai(ARD1[pu], nt1[pu], MASK[pu], &phl->qai, phl->input_level1) != SUCCESS) error = true;
    if (phl->input_level1 == _INP_ARD_ || phl->input_level1 == _INP_QAI_){
      if (screen_noise(ARD1[pu], nt1[pu], MASK[pu], &phl->qai) == FAILURE) error = true;
    }
  } else {
    error = true;
  }

  if (nt2[pu] > 0){
    if (screen_qai(ARD2[pu], nt2[pu], MASK[pu], &phl->qai, phl->input_level2) != SUCCESS) error = true;
    if (phl->input_level2 == _INP_ARD_ || phl->input_level2 == _INP_QAI_){
      if (screen_noise(ARD2[pu], nt2[pu], MASK[pu], &phl->qai) == FAILURE) error = true;
    }
  }


  if (!error && phl->input_level1 == _INP_ARD_){
    if (spectral_adjust(ARD1[pu], MASK[pu], nt1[pu], phl) == FAILURE) error = true;
  }


  if (!error){

    switch (phl->type){
      case _HL_BAP_:
        OUTPUT[pu] = level3(ARD1[pu], ARD2[pu], MASK[pu], 
          nt1[pu], nt2[pu], phl, cube, &nprod[pu]);
        break;
      case _HL_TSA_:
        OUTPUT[pu] = time_series_analysis(ARD1[pu], MASK[pu], 
          nt1[pu], phl, &aux->endmember, cube, &nprod[pu]);
        break;
      case _HL_CSO_:
        OUTPUT[pu] = clear_sky_observations(ARD1[pu], MASK[pu], 
          nt1[pu], phl, cube, &nprod[pu]);
        break;
      case _HL_ML_:
        OUTPUT[pu] = machine_learning(ARD1[pu], MASK[pu], 
          nt1[pu], phl, &aux->ml, cube, &nprod[pu]);
        break;
      case _HL_SMP_:
        OUTPUT[pu] = sample_points(ARD1[pu], MASK[pu], 
          nt1[pu], phl, &aux->sample, cube, &nprod[pu]);
        break;
      case _HL_TXT_:
        OUTPUT[pu] = texture(ARD1[pu], MASK[pu], 
          nt1[pu], phl, cube, &nprod[pu]);
        break;
      case _HL_LSM_:
        OUTPUT[pu] = landscape_metrics(ARD1[pu], MASK[pu], 
          nt1[pu], phl, cube, &nprod[pu]);
        break;
      case _HL_L2I_:
        OUTPUT[pu] = level2_improphe(ARD1[pu], ARD2[pu], MASK[pu], 
          nt1[pu], nt2[pu], phl, cube, &nprod[pu]);
        break;
      case _HL_CFI_:
        OUTPUT[pu] = confield_improphe(ARD1[pu], ARD2[pu], MASK[pu], 
          nt1[pu], nt2[pu], phl, cube, &nprod[pu]);
        break;
      case _HL_LIB_:
        OUTPUT[pu] = library_completeness(ARD1[pu], MASK[pu], 
          nt1[pu], phl, aux->libraries, aux->n_libraries, cube, &nprod[pu]);
        break;
      case _HL_UDF_:
        OUTPUT[pu] = udf_plugin(ARD1[pu], MASK[pu], 
          nt1[pu], phl, cube, &nprod[pu]);
        break;
      default:
        printf("unknown processing module\n");
        break;
    }

  }

  
  free_ard(ARD1[pu], nt1[pu]);
  free_ard(ARD2[pu], nt2[pu]);
  free_brick(MASK[pu]);

  measure_progress(pro, _TASK_COMPUTE_, _CLOCK_TOCK_);

  return;
}


/** This function handles the output tasks
--- pro:      progress handle
--- obytes:   number of bytes written
--- OUTPUT:   OUTPUT bricks
--- nproduct: number of output bricks
--- phl:      HL parameters
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void output_higher_level (progress_t *pro, off_t *obytes, brick_t ***OUTPUT, int *nprod, par_hl_t *phl){
char dname[NPOW_10]; 
int nchar;
char *lock = NULL;
bool error = false;
off_t bytes = 0;
int o;


  if (!write_this_chunk(pro, nprod)) return;

  int pu = pro->last.processing_unit;
  int tile[2] = { pro->last.tile[_X_], pro->last.tile[_Y_] };


  if (OUTPUT[pu] == NULL) return;


  measure_progress(pro, _TASK_OUTPUT_, _CLOCK_TICK_);

  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, tile[_X_], tile[_Y_]);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); exit(1);}

  if ((lock = (char*)CPLLockFile(dname, 60)) == NULL){
    printf("Unable to lock directory %s (timeout: %ds). ", dname, 60);
    error = true;
  }

  if (!error){

    createdir(dname);
    CPLUnlockFile(lock);
    lock = NULL;

    omp_set_num_threads(phl->othread);
  
    #pragma omp parallel shared(OUTPUT,pu,nprod,phl) reduction(+: bytes) default(none)
    {

      CPLPushErrorHandler(CPLQuietErrorHandler);
      CPLSetConfigOption("GDAL_PAM_ENABLED", "YES");

      #pragma omp for schedule(dynamic,1)
      for (o=0; o<nprod[pu]; o++){
        if (phl->radius > 0) OUTPUT[pu][o] = crop_brick(
          OUTPUT[pu][o], phl->radius);
        write_brick(OUTPUT[pu][o]);
        if (OUTPUT[pu][o] != NULL && 
            get_brick_open(OUTPUT[pu][o]) != OPEN_FALSE){
            bytes += get_brick_size(OUTPUT[pu][o]);
        }
      }

      CPLPopErrorHandler();

    }
    


  }

  for (o=0; o<nprod[pu]; o++) free_brick(OUTPUT[pu][o]);
  free((void*)OUTPUT[pu]);
  OUTPUT[pu] = NULL;

  *obytes = bytes;

  measure_progress(pro, _TASK_OUTPUT_, _CLOCK_TOCK_);

  return;
}


/** This function prints a message if no input or output was detected
+++ and possibly instructs the caller to return an error if the
+++ strict flag is set to true
+++ Special behaviour when sampling module is used
--- ibytes:   number of bytes read
--- obytes:   number of bytes written
--- phl:      HL parameters
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int handle_no_io(off_t ibytes, off_t obytes, par_hl_t *phl){
bool warn_i = true;
bool warn_o = true;
int exit_code = SUCCESS;


  if (phl->type == _HL_SMP_){
    
    if (fileexist(phl->smp.f_sample)   &&
        fileexist(phl->smp.f_response) &&
        fileexist(phl->smp.f_coord)){

      warn_o = false;

    }
  }


  printf("________________________________________\n");
  if (warn_i){
    printf("data read    (uncompressed): "); print_humanreadable_bytes(ibytes);
  }
  if (warn_o){
    printf("data written (uncompressed): "); print_humanreadable_bytes(obytes);
  }

  if ((warn_i && ibytes == 0) || (warn_o && obytes == 0)){
    printf("________________________________________\n");
    if (phl->fail_if_empty){
      printf("Error: no input or output detected.\n"
             "Triple-check for mismatching entries in\n");
    } else {
      printf("Warning: no input or output detected. If\n"
             "unintentional, triple-check for mis-\n"
             "matching entries in\n");
    }
    printf("  DIR_MASK\n"
           "  BASE_MASK\n"
           "  X_TILE_RANGE\n"
           "  Y_TILE_RANGE\n"
           "  FILE_TILE\n"
           "  SENSORS\n"
           "  DATE_RANGE\n"
           "  DOY_RANGE\n"
           "  OUTPUT_***\n"
           "and make sure that your input file type\n"
           "  is one of .dat .bsq .bil .tif .vrt\n");
    printf("________________________________________\n");

    if (phl->fail_if_empty) exit_code = FAILURE;
  }


  return exit_code;
}

