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


  MASK[pro->pu_next] = NULL;
  ARD1[pro->pu_next] = NULL;
  ARD2[pro->pu_next] = NULL;
  nt1[pro->pu_next]  = 0;
  nt2[pro->pu_next]  = 0;

  measure_progress(pro, _TASK_INPUT_, _CLOCK_TICK_);

  omp_set_num_threads(phl->ithread);

  MASK[pro->pu_next] = read_mask(&mask_status, &bytes,
    pro->tx_next, pro->ty_next, pro->chunk_next, cube, phl);

  if (MASK[pro->pu_next] == NULL && mask_status != SUCCESS){
    if (mask_status == FAILURE){
      printf("error reading mask tile X%04d_Y%04d chunk %d.\n", 
        pro->tx_next, pro->ty_next, pro->chunk_next);
    } else if (mask_status == CANCEL){
      //printf("no mask data. skip block.\n");
    }
    measure_progress(pro, _TASK_INPUT_, _CLOCK_TOCK_);
    return;
  }


  if (phl->input_level1 == _INP_FTR_){
    ARD1[pro->pu_next] = read_features(&bytes, &nt1[pro->pu_next], 
      pro->tx_next, pro->ty_next, pro->chunk_next, cube, phl);
  } else if (phl->input_level1 == _INP_CON_){
    ARD1[pro->pu_next] = read_confield(&bytes, &nt1[pro->pu_next], 
      pro->tx_next, pro->ty_next, pro->chunk_next, cube, phl);
  } else if (phl->input_level1 == _INP_ARD_ || phl->input_level1 == _INP_QAI_){
    ARD1[pro->pu_next] = read_ard(&bytes, &nt1[pro->pu_next], 
      pro->tx_next, pro->ty_next, pro->chunk_next, cube, &phl->sen, phl);
  } else if (phl->input_level1 != _INP_NONE_) {
    printf("unknown input level\n");
  }

  if (ARD1[pro->pu_next] == NULL && nt1[pro->pu_next] < 0){
    printf("error reading data from tile X%04d_Y%04d chunk %d.\n", 
      pro->tx_next, pro->ty_next, pro->chunk_next);
    measure_progress(pro, _TASK_INPUT_, _CLOCK_TOCK_);
    return;
  }


  if (phl->input_level2 == _INP_FTR_){
    ARD2[pro->pu_next] = read_features(&bytes, &nt2[pro->pu_next], 
      pro->tx_next, pro->ty_next, pro->chunk_next, cube, phl);
  } else if (phl->input_level2 == _INP_CON_){
    ARD2[pro->pu_next] = read_confield(&bytes, &nt2[pro->pu_next], 
      pro->tx_next, pro->ty_next, pro->chunk_next, cube, phl);
  } else if (phl->input_level2 == _INP_ARD_ || phl->input_level2 == _INP_QAI_){
    ARD2[pro->pu_next] = read_ard(&bytes, &nt2[pro->pu_next], 
      pro->tx_next, pro->ty_next, pro->chunk_next, cube, &phl->sen2, phl);
  } else if (phl->input_level2 != _INP_NONE_){
    printf("unknown input level\n");
  }

  if (ARD2[pro->pu_next] == NULL && nt2[pro->pu_next] < 0){
    printf("error reading secondary data from tile X%04d_Y%04d chunk %d.\n", 
      pro->tx_next, pro->ty_next, pro->chunk_next);
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

  OUTPUT[pro->pu] = NULL;
  nprod[pro->pu]  = 0;


  measure_progress(pro, _TASK_COMPUTE_, _CLOCK_TICK_);

  omp_set_num_threads(phl->cthread);


  if (nt1[pro->pu] > 0){
    if (screen_qai(ARD1[pro->pu], nt1[pro->pu], MASK[pro->pu], &phl->qai, phl->input_level1) != SUCCESS) error = true;
    if (phl->input_level1 == _INP_ARD_ || phl->input_level1 == _INP_QAI_){
      if (screen_noise(ARD1[pro->pu], nt1[pro->pu], MASK[pro->pu], &phl->qai) == FAILURE) error = true;
    }
  } else {
    error = true;
  }

  if (nt2[pro->pu] > 0){
    if (screen_qai(ARD2[pro->pu], nt2[pro->pu], MASK[pro->pu], &phl->qai, phl->input_level2) != SUCCESS) error = true;
    if (phl->input_level2 == _INP_ARD_ || phl->input_level2 == _INP_QAI_){
      if (screen_noise(ARD2[pro->pu], nt2[pro->pu], MASK[pro->pu], &phl->qai) == FAILURE) error = true;
    }
  }


  if (!error && phl->input_level1 == _INP_ARD_){
    if (spectral_adjust(ARD1[pro->pu], MASK[pro->pu], nt1[pro->pu], phl) == FAILURE) error = true;
  }


  if (!error){

    switch (phl->type){
      case _HL_BAP_:
        OUTPUT[pro->pu] = level3(ARD1[pro->pu], ARD2[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], nt2[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      case _HL_TSA_:
        OUTPUT[pro->pu] = time_series_analysis(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, &aux->endmember, cube, &nprod[pro->pu]);
        break;
      case _HL_CSO_:
        OUTPUT[pro->pu] = clear_sky_observations(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      case _HL_ML_:
        OUTPUT[pro->pu] = machine_learning(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, &aux->ml, cube, &nprod[pro->pu]);
        break;
      case _HL_SMP_:
        OUTPUT[pro->pu] = sample_points(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, &aux->sample, cube, &nprod[pro->pu]);
        break;
      case _HL_TXT_:
        OUTPUT[pro->pu] = texture(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      case _HL_LSM_:
        OUTPUT[pro->pu] = landscape_metrics(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      case _HL_L2I_:
        OUTPUT[pro->pu] = level2_improphe(ARD1[pro->pu], ARD2[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], nt2[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      case _HL_CFI_:
        OUTPUT[pro->pu] = confield_improphe(ARD1[pro->pu], ARD2[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], nt2[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      case _HL_LIB_:
        OUTPUT[pro->pu] = library_completeness(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, &aux->library, cube, &nprod[pro->pu]);
        break;
      case _HL_UDF_:
        OUTPUT[pro->pu] = udf_plugin(ARD1[pro->pu], MASK[pro->pu], 
          nt1[pro->pu], phl, cube, &nprod[pro->pu]);
        break;
      default:
        printf("unknown processing module\n");
        break;
    }

  }

  
  free_ard(ARD1[pro->pu], nt1[pro->pu]);
  free_ard(ARD2[pro->pu], nt2[pro->pu]);
  free_brick(MASK[pro->pu]);

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
  if (OUTPUT[pro->pu_prev] == NULL) return;


  measure_progress(pro, _TASK_OUTPUT_, _CLOCK_TICK_);

  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, pro->tx_prev, pro->ty_prev);
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
  
    #pragma omp parallel shared(OUTPUT,pro,nprod,phl) reduction(+: bytes) default(none)
    {

      CPLPushErrorHandler(CPLQuietErrorHandler);
      CPLSetConfigOption("GDAL_PAM_ENABLED", "YES");

      #pragma omp for schedule(dynamic,1)
      for (o=0; o<nprod[pro->pu_prev]; o++){
        if (phl->radius > 0) OUTPUT[pro->pu_prev][o] = crop_brick(
          OUTPUT[pro->pu_prev][o], phl->radius);
        write_brick(OUTPUT[pro->pu_prev][o]);
        if (OUTPUT[pro->pu_prev][o] != NULL && 
            get_brick_open(OUTPUT[pro->pu_prev][o]) != OPEN_FALSE){
            bytes += get_brick_size(OUTPUT[pro->pu_prev][o]);
        }
      }

      CPLPopErrorHandler();

    }
    


  }

  for (o=0; o<nprod[pro->pu_prev]; o++) free_brick(OUTPUT[pro->pu_prev][o]);
  free((void*)OUTPUT[pro->pu_prev]);
  OUTPUT[pro->pu_prev] = NULL;

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
    if (phl->strict_io){
      printf("Error: no input or output deteccted.\n"
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

    if (phl->strict_io) exit_code = FAILURE;
  }


  return exit_code;
}

