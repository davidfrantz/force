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
This file contains functions for parsing parameter files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "param-ll.h"


void register_lower(params_t *params, par_ll_t *pl2);
void parse_proj(par_ll_t *pl2);


/** This function registers all the lower level parameters that are parsed
+++ from the parameter file.
--- params: registered parameters
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_lower(params_t *params, par_ll_t *pl2){


  register_char_par(params,    "DIR_LEVEL2",            _CHAR_TEST_EXIST_, &pl2->d_level2);
  register_char_par(params,    "DIR_TEMP",              _CHAR_TEST_EXIST_, &pl2->d_temp);
  register_char_par(params,    "DIR_LOG",               _CHAR_TEST_EXIST_, &pl2->d_log);
  register_char_par(params,    "DIR_PROVENANCE",        _CHAR_TEST_EXIST_, &pl2->d_prov);
  register_char_par(params,    "FILE_AOI",              _CHAR_TEST_NULL_OR_EXIST_, &pl2->f_aoi);
  register_char_par(params,    "FILE_QUEUE",            _CHAR_TEST_NULL_OR_EXIST_, &pl2->f_queue);
  register_char_par(params,    "DIR_WVPLUT",            _CHAR_TEST_NULL_OR_EXIST_, &pl2->d_wvp);
  register_char_par(params,    "DIR_AOD",               _CHAR_TEST_NULL_OR_EXIST_, &pl2->d_aod);
  register_char_par(params,    "FILE_TILE",             _CHAR_TEST_NULL_OR_EXIST_, &pl2->f_tile);
  register_char_par(params,    "FILE_DEM",              _CHAR_TEST_NULL_OR_EXIST_, &pl2->fdem);
  register_char_par(params,    "DIR_COREG_BASE",        _CHAR_TEST_NULL_OR_EXIST_, &pl2->d_coreg);
  register_double_par(params,  "TILE_SIZE",             0, INT_MAX, &pl2->tilesize);
  register_double_par(params,  "BLOCK_SIZE",            0, INT_MAX, &pl2->chunksize);
  register_double_par(params,  "RESOLUTION_LANDSAT",    0, INT_MAX, &pl2->res_landsat);
  register_double_par(params,  "RESOLUTION_SENTINEL2",  0, INT_MAX, &pl2->res_sentinel2);
  register_bool_par(params,    "DO_REPROJ",             &pl2->doreproj);
  register_bool_par(params,    "DO_TILE",               &pl2->dotile);
  register_double_par(params,  "ORIGIN_LAT",            -90, 90, &pl2->orig_lat);
  register_double_par(params,  "ORIGIN_LON",            -180, 180, &pl2->orig_lon);
  register_charvec_par(params, "PROJECTION",            _CHAR_TEST_NONE_, &pl2->proj_, &pl2->nproj_);
  register_enum_par(params,    "RESAMPLING",            _TAGGED_ENUM_RESAMPLE_, _RESAMPLE_LENGTH_, &pl2->resample);
  register_enum_par(params,    "RES_MERGE",             _TAGGED_ENUM_RES_MERGE_, _RES_MERGE_LENGTH_, &pl2->resmerge);
  register_int_par(params,     "TIER",                  1, 3, &pl2->tier);
  register_bool_par(params,    "DO_TOPO",               &pl2->dotopo);
  register_bool_par(params,    "DO_ATMO",               &pl2->doatmo);
  register_bool_par(params,    "DO_AOD",                &pl2->doaod);
  register_bool_par(params,    "DO_BRDF",               &pl2->dobrdf);
  register_bool_par(params,    "MULTI_SCATTERING",      &pl2->domulti);
  register_bool_par(params,    "ADJACENCY_EFFECT",      &pl2->doenv);
  register_float_par(params,   "WATER_VAPOR",           0, 15, &pl2->wvp);
  register_bool_par(params,    "STRICT_WATER_VAPOR",    &pl2->wvp_strict);
  register_bool_par(params,    "IMPULSE_NOISE",         &pl2->impulse);
  register_bool_par(params,    "BUFFER_NODATA",         &pl2->bufnodata);
  register_bool_par(params,    "USE_DEM_DATABASE",      &pl2->use_dem_database);
  register_int_par(params,     "DEM_NODATA",            SHRT_MIN, SHRT_MAX, &pl2->dem_nodata);
  register_int_par(params,     "COREG_BASE_NODATA",     SHRT_MIN, SHRT_MAX, &pl2->coreg_nodata);
  register_bool_par(params,    "ERASE_CLOUDS",          &pl2->erase_cloud);
  register_float_par(params,   "MAX_CLOUD_COVER_FRAME", 1, 100, &pl2->maxcc);
  register_float_par(params,   "MAX_CLOUD_COVER_TILE",  1, 100, &pl2->maxtc);
  register_float_par(params,   "CLOUD_BUFFER",          0, 10000, &pl2->cldbuf);
  register_float_par(params,   "CIRRUS_BUFFER",         0, 10000, &pl2->cirbuf);
  register_float_par(params,   "SHADOW_BUFFER",         0, 10000, &pl2->shdbuf);
  register_float_par(params,   "SNOW_BUFFER",           0, 10000, &pl2->snwbuf);
  register_float_par(params,   "CLOUD_THRESHOLD",       0, 1, &pl2->cldprob);
  register_float_par(params,   "SHADOW_THRESHOLD",      0, 1, &pl2->shdprob);
  register_int_par(params,     "NPROC",                 1, INT_MAX, &pl2->nproc);
  register_int_par(params,     "NTHREAD",               1, INT_MAX, &pl2->nthread);
  register_bool_par(params,    "PARALLEL_READS",        &pl2->ithread);
  register_int_par(params,     "DELAY",                 0, INT_MAX, &pl2->delay);
  register_int_par(params,     "TIMEOUT_ZIP",           0, INT_MAX, &pl2->timeout);
  register_char_par(params,    "FILE_OUTPUT_OPTIONS",   _CHAR_TEST_NULL_OR_EXIST_, &pl2->f_gdalopt);
  register_enum_par(params,    "OUTPUT_FORMAT",         _TAGGED_ENUM_FMT_, _FMT_LENGTH_, &pl2->format);
  register_bool_par(params,    "OUTPUT_DST",            &pl2->odst);
  register_bool_par(params,    "OUTPUT_AOD",            &pl2->oaod);
  register_bool_par(params,    "OUTPUT_WVP",            &pl2->owvp);
  register_bool_par(params,    "OUTPUT_VZN",            &pl2->ovzn);
  register_bool_par(params,    "OUTPUT_HOT",            &pl2->ohot);
  register_bool_par(params,    "OUTPUT_OVV",            &pl2->oovv);

  return;
}


/** This function reparses the target projection parameter (special para-
+++ meter that cannot be parsed with the general parser).
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void parse_proj(par_ll_t *pl2){
int i;


  copy_string(pl2->proj, NPOW_10, pl2->proj_[0]);
  
  for (i=1; i<pl2->nproj_; i++){
    strncat(pl2->proj, " ",           NPOW_10-strlen(pl2->proj)-1);
    strncat(pl2->proj, pl2->proj_[i], NPOW_10-strlen(pl2->proj)-1);
  }

  return;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function allocates the L2 parameters
+++ Return: L2 parameters (must be freed with free_param_lower)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
par_ll_t *allocate_param_lower(){
par_ll_t *pl2 = NULL;


  alloc((void**)&pl2, 1, sizeof(par_ll_t));

  return pl2;
}


/** This function frees the L2 parameters
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_param_lower(par_ll_t *pl2){


  if (pl2 == NULL) return;
  free((void*)pl2); pl2 = NULL;

  return;
}


/** This function parses the Level 2 parameters
--- pl2:    L2 parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_param_lower(par_ll_t *pl2){
FILE *fpar = NULL;
char  buffer[NPOW_16] = "\0";
char  ext[NPOW_10] = "\0";
char  bname[NPOW_10] = "\0";


  pl2->params = allocate_params();
  
  if (pl2->d_level1 != NULL){
    
    // if .SAFE directory (S2) was given, use 1st granule
    extension(pl2->d_level1, ext, NPOW_10);

    if (strcmp(ext, ".SAFE") == 0){
      
      strncat(pl2->d_level1, "/GRANULE", NPOW_10-strlen(pl2->d_level1)-1);

      if (findfile_pattern(pl2->d_level1, "L1C", NULL, bname, NPOW_10) != SUCCESS){
         printf("Unable to dive down .SAFE file!\n"); return FAILURE;}

      copy_string(pl2->d_level1, NPOW_10, bname);

    }

    basename_without_ext(pl2->d_level1, pl2->b_level1, NPOW_10);
    //printf("%s: ", pl2->b_level1);
  } else {
    printf("No input image given!\n"); return FAILURE;}


  // open parameter file
  if ((fpar = fopen(pl2->f_par, "r")) == NULL){
    printf("Unable to open parameter file!\n"); return FAILURE;}

  // check
  if (fscanf(fpar, "%s", buffer) < 0){
    printf("Cannot scan parameter file!\n"); return FAILURE;}
    
  // check
  if (strcmp(buffer, "++PARAM_LEVEL2_START++") != 0){
    printf("No valid parameter file! '++PARAM_LEVEL2_START++' is missing.\n"); return FAILURE;}

  // register parameters
  register_lower(pl2->params, pl2);

  // parse line by line
  while (fgets(buffer, NPOW_16, fpar) != NULL) parse_parameter(pl2->params, buffer);
  
  // close
  fclose(fpar);


  #ifdef DEBUG
  print_parameter(pl2->params);
  #endif


  // check the parameters
  if (check_parameter(pl2->params) == FAILURE) return FAILURE;

  // log the parameters
  log_parameter(pl2->params);
  
  
  // re-parse special cases
  parse_proj(pl2);


  // some more special checks

  if (pl2->dem_nodata == 0){
    printf("DEM_NODATA is 0. Check if this is correct. 0 is a bad choice of DEM nodata. ");}
  if (pl2->coreg_nodata == 0){
    printf("COREG_BASE_NODATA is 0. Check if this is correct. 0 is a bad choice of reflectance nodata. ");}


  if (!pl2->doaod){
    if (strcmp(pl2->d_aod, "NULL") == 0 || !fileexist(pl2->d_aod)){
      printf("If DO_AOD = FALSE, DIR_AOD needs to be given. "); return FAILURE;}
  }

  if (!pl2->doatmo) pl2->doaod = pl2->dobrdf = pl2->dotopo = pl2->doenv = false;

  if (pl2->dotopo && !fileexist(pl2->fdem)){
    printf("FILE_DEM does not exist in the file system. Give a DEM, or use DOTOPO = FALSE + FILE_DEM = NULL (surface will be assumed flat, z = 0m). "); return FAILURE;}

  if (pl2->use_dem_database && (strcmp(pl2->fdem, "NULL") == 0)) {
    printf("If USE_DEM_DATABASE = TRUE, FILE_DEM cannot be NULL.\n");
    return FAILURE;
  }

  if (pl2->use_dem_database && (!fileexist(pl2->fdem))) {
    printf("If USE_DEM_DATABASE = TRUE, FILE_DEM needs to exist.\n");
    return FAILURE;
  }

  if (pl2->format != _FMT_CUSTOM_){
    default_gdaloptions(pl2->format, &pl2->gdalopt);
  } else {
    if (strcmp(pl2->f_gdalopt, "NULL") == 0 || !fileexist(pl2->f_gdalopt)){
      printf("If OUTPUT_FORMAT = CUSTOM, FILE_OUTPUT_OPTIONS needs to be given. "); 
      return FAILURE;
    } else {
      parse_gdaloptions(pl2->f_gdalopt, &pl2->gdalopt);
    }
  }

  return SUCCESS;
}

