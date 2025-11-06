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
This file contains functions for organizing bricks in memory, and output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "gdalopt-cl.h"


static const char *GTIFF_OPTIONS[][2] = {
    {"COMPRESS",   "ZSTD"},
    {"PREDICTOR",  "2"},
    {"INTERLEAVE", "BAND"},
    {"BIGTIFF",    "YES"},
    {"TILED",      "YES"},
    {"BLOCKXSIZE", "256"},
    {"BLOCKYSIZE", "256"}
};
#define GTIFF_OPTION_COUNT (sizeof(GTIFF_OPTIONS)/sizeof(GTIFF_OPTIONS[0]))

static const char *COG_OPTIONS[][2] = {
    {"COMPRESS",            "ZSTD"},
    {"PREDICTOR",           "YES"},
    {"INTERLEAVE",          "TILE"},
    {"BLOCKSIZE",           "256"},
    {"BIGTIFF",             "YES"},
    {"OVERVIEW_RESAMPLING", "AVERAGE"}
};
#define COG_OPTION_COUNT (sizeof(COG_OPTIONS)/sizeof(COG_OPTIONS[0]))


/** This function sets default GDAL output options
--- format:  Default format
--- gdalopt: GDAL options (returned)
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void default_gdaloptions(int format, gdalopt_t *gdalopt){
int nchar[_TV_LENGTH_] = {0};


  if (_TV_LENGTH_ != 2){
    printf("Error: _TV_LENGTH_ is expected to be 2.\n");
    exit(1);
  }

  switch (format){
    case _FMT_ENVI_:
    fill_string(&gdalopt->extension, "dat");
    fill_string(&gdalopt->driver, "ENVI");
    break;
    case _FMT_GTIFF_:
      fill_string(&gdalopt->extension, "tif");
      fill_string(&gdalopt->driver, "GTiff");
      for (int j=0; j<_TV_LENGTH_; j++){
        for (size_t i=0; i<GTIFF_OPTION_COUNT; i++){
          nchar[j] = (strlen(GTIFF_OPTIONS[i][j]) > nchar[j]) ? strlen(GTIFF_OPTIONS[i][j]) : nchar[j];
        }
        alloc_string_vector(&gdalopt->options[j], GTIFF_OPTION_COUNT, nchar[j]);
        for (size_t i=0; i<GTIFF_OPTION_COUNT; i++){
          fill_string_vector(&gdalopt->options[j], i, GTIFF_OPTIONS[i][j]);
        }
      }
      break;
    case _FMT_COG_:
      fill_string(&gdalopt->extension, "tif");
      fill_string(&gdalopt->driver, "COG");
      for (int j=0; j<_TV_LENGTH_; j++){
        for (size_t i=0; i<COG_OPTION_COUNT; i++){
          nchar[j] = (strlen(COG_OPTIONS[i][j]) > nchar[j]) ? strlen(COG_OPTIONS[i][j]) : nchar[j];
        }
        alloc_string_vector(&gdalopt->options[j], COG_OPTION_COUNT, nchar[j]);
        for (size_t i=0; i<COG_OPTION_COUNT; i++){
          fill_string_vector(&gdalopt->options[j], i, COG_OPTIONS[i][j]);
        }
      }
      break;
    case _FMT_JPEG_:
      fill_string(&gdalopt->extension, "jpg");
      fill_string(&gdalopt->driver, "JPEG");
      break;
    case _FMT_CUSTOM_:
      break;
    default:
      printf("unknown format.\n");
      exit(FAILURE);
  }

  return;
}


/** This function reads GDAL output options
--- fname:   text file
--- gdalopt: GDAL options (returned)
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void parse_gdaloptions(char *fname, gdalopt_t *gdalopt){
int nrows;
int nchar[_TV_LENGTH_] = {0};
char ***tagval = NULL;
bool b_driver = false;
bool b_ext = false;


  if (_TV_LENGTH_ != 2){
    printf("Error: _TV_LENGTH_ is expected to be 2.\n");
    exit(1);
  }

  tagval = read_tagvalue(fname, &nrows);
  if (tagval == NULL){
    printf("Reading GDAL options failed.\n");
    exit(FAILURE);
  }


  for (int i=0; i<nrows; i++){
    nchar[0] = (strlen(tagval[i][_TV_TAG_]) > nchar[_TV_TAG_]) ? strlen(tagval[i][_TV_TAG_]) : nchar[_TV_TAG_];
    nchar[1] = (strlen(tagval[i][_TV_VAL_]) > nchar[_TV_VAL_]) ? strlen(tagval[i][_TV_VAL_]) : nchar[_TV_VAL_];
  }

  alloc_string_vector(&gdalopt->options[_TV_TAG_],  nrows - 2, nchar[_TV_TAG_]);
  alloc_string_vector(&gdalopt->options[_TV_VAL_],  nrows - 2, nchar[_TV_VAL_]);

  for (int i=0, o=0; i<nrows; i++){

    if (strcmp(tagval[i][_TV_TAG_], "DRIVER") == 0){
      fill_string(&gdalopt->driver, tagval[i][_TV_VAL_]);
      b_driver = true;
    } else if (strcmp(tagval[i][_TV_TAG_], "EXTENSION") == 0){
      fill_string(&gdalopt->extension, tagval[i][_TV_VAL_]);
      b_ext = true;
    } else {
      fill_string_vector(&gdalopt->options[_TV_TAG_], o, tagval[i][_TV_TAG_]);
      fill_string_vector(&gdalopt->options[_TV_VAL_], o, tagval[i][_TV_VAL_]);
      o++;
    }

  }


  if (!b_driver){
    printf("Driver not found in GDAL options file (e.g. DRIVER = COG)\n");
    exit(FAILURE);
  }

  if (!b_ext){
    printf("Extension not found in GDAL options file (e.g. EXTENSION = tif)\n");
    exit(FAILURE);
  }
 
  #ifdef FORCE_DEBUG
  print_gdaloptions(gdalopt);
  #endif

  free_3D((void***)tagval, nrows, _TV_LENGTH_);

  return;
}


/** This function prints the GDAL output options
--- gdalopt: GDAL options
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_gdaloptions(gdalopt_t *gdalopt){


  printf("GDAL output options :::\n");
  printf("Driver:    %s\n", gdalopt->driver.string);
  printf("Extension: %s\n", gdalopt->extension.string);

  if (gdalopt->options[_TV_TAG_].number != gdalopt->options[_TV_VAL_].number){
    printf("Error: Number of GDAL option tags and values do not match.\n");
    exit(1);
  }

  for (int o=0; o<gdalopt->options[_TV_TAG_].number; o++){
    printf("Option %2d: %s = %s\n", 
      o, 
      gdalopt->options[_TV_TAG_].string[o], 
      gdalopt->options[_TV_VAL_].string[o]);
  }

  return;
}


/** This function copies GDAL output options
--- dst:    destination GDAL options
--- src:    source GDAL options
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void copy_gdaloptions(gdalopt_t *dst, gdalopt_t *src){

  free_gdaloptions(dst); // to not leak memory

  fill_string(&dst->driver, src->driver.string);
  fill_string(&dst->extension, src->extension.string);

  if (src->options[_TV_TAG_].number != src->options[_TV_VAL_].number){
    printf("Error: Number of GDAL option tags and values do not match.\n");
    exit(1);
  }

  if (src->options[_TV_TAG_].number > 0){
    alloc_string_vector(&dst->options[_TV_TAG_], src->options[_TV_TAG_].number, src->options[_TV_TAG_].length);
    alloc_string_vector(&dst->options[_TV_VAL_], src->options[_TV_VAL_].number, src->options[_TV_VAL_].length);
    for (int o=0; o<src->options[_TV_TAG_].number; o++){
      fill_string_vector(&dst->options[_TV_TAG_], o, src->options[_TV_TAG_].string[o]);
      fill_string_vector(&dst->options[_TV_VAL_], o, src->options[_TV_VAL_].string[o]);
    }
  } else {
    dst->options[_TV_TAG_].string = NULL;
    dst->options[_TV_VAL_].string = NULL;
  }

  return;
}

/** This function frees GDAL output options
--- gdalopt: GDAL options
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_gdaloptions(gdalopt_t *gdalopt){

  free_string(&gdalopt->driver);
  free_string(&gdalopt->extension);
  if (gdalopt->options[_TV_TAG_].number > 0) free_string_vector(&gdalopt->options[_TV_TAG_]);
  if (gdalopt->options[_TV_VAL_].number > 0) free_string_vector(&gdalopt->options[_TV_VAL_]);

  return;
}
