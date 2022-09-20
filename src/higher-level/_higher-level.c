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

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../cross-level/const-cl.h"
#include "../cross-level/brick-cl.h"
#include "../cross-level/tile-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/cite-cl.h"
#include "../higher-level/progress-hl.h"
#include "../higher-level/tasks-hl.h"
#include "../higher-level/param-hl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing


typedef struct {
  int n;
  char fprm[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] parameter-file\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'parameter-file': parameter file for any higher level submodule\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;


  opterr = 0;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvi")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        printf("FORCE version: %s\n", _VERSION_);
        exit(SUCCESS);
      case 'i':
        printf("Higher level processing (compositing, time series analysis, ...)\n");
        exit(SUCCESS);
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }

  // non-optional parameters
  args->n = 1;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->fprm, NPOW_10, argv[optind++]);
    } else if (argc-optind < args->n){
      fprintf(stderr, "some non-optional arguments are missing.\n");
      usage(argv[0], FAILURE);
    } else if (argc-optind > args->n){
      fprintf(stderr, "too many non-optional arguments.\n");
      usage(argv[0], FAILURE);
    }
  } else {
    fprintf(stderr, "non-optional arguments are missing.\n");
    usage(argv[0], FAILURE);
  }

  return;
}


int main ( int argc, char *argv[] ){
args_t args;
par_hl_t    *phl      = NULL;
cube_t      *cube     = NULL;
aux_t       *aux      = NULL;
ard_t       **ARD1    = NULL;
ard_t       **ARD2    = NULL;
int         *nt1      = NULL;
int         *nt2      = NULL;
brick_t     **MASK    = NULL;
brick_t     ***OUTPUT = NULL;
int         *nprod    = NULL;
GDALDriverH driver;
progress_t  pro;
off_t ibytes = 0, obytes = 0;


  /** INITIALIZING
  +** *******************************************************************/
  
  parse_args(argc, argv, &args);

  phl = allocate_param_higher();
  copy_string(phl->f_par, NPOW_10, args.fprm);

  // parse parameter file
  if (parse_param_higher(phl) == FAILURE){
    printf("Reading parameter file failed!\n"); return FAILURE;}

  cite_me(_CITE_FORCE_);

  // parse auxiliary files
  if ((aux = read_aux(phl)) == NULL){
    printf("Reading aux file failed!\n"); return FAILURE;}

  // register python UDF plug-in
  register_python(phl);

  // copy and read datacube definition
  if ((cube = copy_datacube_def(phl->d_lower, phl->d_higher, phl->blocksize)) == NULL){
    printf("Copying datacube definition failed.\n"); return FAILURE;}

  // update datacube with parameters
  update_datacube_extent(cube, phl->tx[_MIN_], phl->tx[_MAX_], phl->ty[_MIN_], phl->ty[_MAX_]);
  update_datacube_res(cube, phl->res);
  update_gdaloptions_blocksize(phl->format, &phl->gdalopt, cube->cx, cube->cy);

  // compile active tiles
  if (tile_active(phl->f_tile, cube) == FAILURE){
    printf("Compiling active tiles failed!\n"); return FAILURE;}

  // initialize progress handle
  init_progess(&pro, cube, phl);


  // allocate array of Level 2 structs
  alloc((void**)&ARD1,   pro.npu, sizeof(ard_t*));
  alloc((void**)&ARD2,   pro.npu, sizeof(ard_t*));
  alloc((void**)&MASK,   pro.npu, sizeof(brick_t*));
  alloc((void**)&OUTPUT, pro.npu, sizeof(brick_t**));
  alloc((void**)&nprod,  pro.npu, sizeof(int));
  alloc((void**)&nt1,    pro.npu, sizeof(int));
  alloc((void**)&nt2,    pro.npu, sizeof(int));


  // enable nested threading
  if (omp_get_thread_limit() < (phl->ithread+phl->othread+phl->cthread)){
    printf("Number of threads exceeds system limit\n"); return FAILURE;}
  omp_set_nested(true);
  omp_set_max_active_levels(2);


  // make GDAL less verbose
  CPLPushErrorHandler(CPLQuietErrorHandler);

  // register GDAL drivers
  GDALAllRegister();
  if ((driver = GDALGetDriverByName("JP2ECW")) != NULL) GDALDeregisterDriver(driver);
  

  /** LOOP OVER ALL CHUNKS
  +** *******************************************************************/

  while (progress(&pro)){

    #pragma omp parallel num_threads(3) shared(ARD1,ARD2,MASK,OUTPUT,aux,nprod,nt1,nt2,pro,cube,phl) reduction(+: ibytes, obytes) default(none)
    {

      if (omp_get_thread_num() == 0){
        read_higher_level(&pro, &ibytes, MASK, ARD1, ARD2, nt1, nt2, cube, phl);
      } else if (omp_get_thread_num() == 1){
        compute_higher_level(&pro, MASK, ARD1, ARD2, nt1, nt2, cube, phl, aux, OUTPUT, nprod);
      } else {
        output_higher_level(&pro, &obytes, OUTPUT, nprod, phl);
      }

    }

  }

  printf("________________________________________\n");
  printf("bytes read    (uncompressed): "); print_humanreadable_bytes(ibytes);
  printf("bytes written (uncompressed): "); print_humanreadable_bytes(obytes);

  if (ibytes == 0 || obytes == 0){
    printf("________________________________________\n");
    printf("Warning: no input or output detected. If\n"
           "unintentional, triple-check for mis-\n"
           "matching entries in\n"
           "  DIR_MASK\n"
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
  }

  cite_push(phl->d_higher);


  /** CLEAN
  +**********************************************************************/
  free((void*)ARD1);
  free((void*)ARD2);
  free((void*)MASK);
  free((void*)OUTPUT);
  free((void*)nt1);
  free((void*)nt2);
  free((void*)nprod);
  free_datacube(cube);
  free_aux(phl, aux);
  free_param_higher(phl);

  deregister_python(phl);

  CPLPopErrorHandler();


  return SUCCESS;
}

