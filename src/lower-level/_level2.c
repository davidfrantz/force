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
This program is the FORCE Level-2 Processing System (single image)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/cite-cl.h"
#include "../cross-level/brick-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/quality-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/meta-ll.h"
#include "../lower-level/cube-ll.h"
#include "../lower-level/atc-ll.h"
#include "../lower-level/read-ll.h"
#include "../lower-level/sunview-ll.h"
#include "../lower-level/topo-ll.h"
#include "../lower-level/cloud-ll.h"
#include "../lower-level/atmo-ll.h"
#include "../lower-level/resmerge-ll.h"
#include "../lower-level/coreg-ll.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points

/** OpenMP **/
#include <omp.h> // multi-platform shared memory multiprocessing


int main( int argc, char *argv[] ){
int mission, c;
par_ll_t *pl2  = NULL; // can be renamed to par, once par is not global anymore...
meta_t   *meta = NULL;
multicube_t   *multicube = NULL;
atc_t    *atc  = NULL;

top_t   *TOP = NULL;
brick_t *DN  = NULL;
brick_t *TOA = NULL;
brick_t *QAI = NULL;

brick_t **LEVEL2 = NULL;
int nprod;
int err;
GDALDriverH driver;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 3){
    printf("usage: %s image-dir parameter-file\n\n", argv[0]);
    return FAILURE;
  }


  /** initialization + read metadata, parameter and tile file
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  time_t TIME; time(&TIME);


  #ifdef FORCE_DEBUG
  //db_check();
  #endif

  // allow a test, if l2ps is compiled in DEBUG mode
  if (strcmp(argv[1], "?") == 0){
    #ifdef FORCE_DEBUG
    return 1;
    #endif
    return 0;
  }


  pl2 = allocate_param_lower();

  // get command line parameters
  copy_string(pl2->d_level1, NPOW_10, argv[1]);
  copy_string(pl2->f_par,    NPOW_10, argv[2]);
  check_arg(argv[2]);


  // make GDAL less verbose
  CPLPushErrorHandler(CPLQuietErrorHandler);

  // register GDAL drivers  
  GDALAllRegister();
  if ((driver = GDALGetDriverByName("JP2ECW")) != NULL) GDALDeregisterDriver(driver);

  // parse parameter file
  if (parse_param_lower(pl2) != SUCCESS){
    printf("Parsing parameter file failed.\n"); return FAILURE;}

  cite_me(_CITE_FORCE_);
  cite_me(_CITE_L2PS_);

  // parse metadata
  if ((mission = parse_metadata(pl2, &meta, &DN)) == FAILURE){
    printf("Parsing metadata failed.\n"); return FAILURE;}

  // write and init a new datacube
  if ((multicube = start_multicube(pl2, DN)) == NULL){
    printf("Starting datacube(s) failed.\n"); return FAILURE;}


  // open threads
  if (omp_get_thread_limit() < (pl2->nthread)){
    printf("not enough threads allowed. Reduce NTHREAD.\n"); return FAILURE;}
  omp_set_num_threads(pl2->nthread);
  omp_set_nested(true);
  omp_set_max_active_levels(2);


  // do processing for every datacube [not the best solution, but more easy to implenet atm]
  for (c=0; c<multicube->n; c++){

    // skip inactive cubes
    if (!multicube->cover[c]) continue;


    /** read Digital Numbers + projection
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (read_level1(meta, mission, DN, pl2) != SUCCESS){
      printf("reading DNs failed.\n"); return FAILURE;}

    if ((atc = allocate_atc(pl2, meta, DN)) == NULL){
      printf("Allocating atc failed.\n"); return FAILURE;}


    /** initialize Quality Assurance Information
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (bounds_level1(meta, DN, &QAI, pl2) == FAILURE){
      printf("Compiling nodata / saturation masks failed.\n"); return FAILURE;}


    /** sun-target-view geometry
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (sun_target_view(pl2, meta, mission, atc, QAI) == FAILURE){
      printf("computing sun/view geometry failed.\n"); return FAILURE;}


    /** TOA reflectance + brightness temperature
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (convert_level1(meta, mission, atc, DN, &TOA, QAI) != SUCCESS){
      printf("DN to TOA conversion failed.\n"); return FAILURE;}
    free_brick_bands(DN);


    /** read/reproject/ckeck DEM and compute slope/aspect
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (compile_topography(pl2, atc, &TOP, QAI) != SUCCESS){
      printf("unable to compile topography.\n"); return FAILURE;}


   /** cloud and cloud shadow detection
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    err = detect_clouds(pl2, mission, atc, TOA, TOP->dem, TOP->exp, QAI);
    if (err == FAILURE){
      printf("error in cloud module.\n"); return FAILURE;
    } else if (err == CANCEL){
      proctime_print("Processing time", TIME);
      return SUCCESS;
    }


   /** coregistration
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (coregister(mission, pl2, TOA, QAI) != SUCCESS){
      printf("coregistration failed.\n"); return FAILURE;}


  /** resolution merge
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (resolution_merge(mission, pl2->resmerge, TOA, QAI) != SUCCESS){
      printf("unable to merge resolutions.\n"); return FAILURE;}


    /** radiometric correction
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if ((LEVEL2 = radiometric_correction(pl2, meta, mission, atc, multicube->cube[c], TOA, QAI, TOP, &nprod)) == NULL){
      printf("Error in radiometric module.\n"); return FAILURE;}
    free_atc(atc);


    /** reprojection, tiling and output
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (cube_level2(pl2, meta, multicube->cube[c], LEVEL2, nprod) != SUCCESS){
      printf("Error in geometric module.\n"); return FAILURE;}

  }

  free_param_lower(pl2); free_metadata(meta); free_multicube(multicube);
  free_brick(DN);

  cite_push(pl2->d_level2);
  
  CPLPopErrorHandler();


  printf("Success! "); proctime_print("Processing time", TIME);

  return SUCCESS;
}

