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
This file contains functions for organizing bricks in memory, and output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "gdalopt-cl.h"


/** This function sets default GDAL output options
--- format:  Default format
--- gdalopt: GDAL options (returned)
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void default_gdaloptions(int format, gdalopt_t *gdalopt){
int o = 0;


  switch (format){
    case _FMT_ENVI_:
      copy_string(gdalopt->extension,   NPOW_04, "dat");
      copy_string(gdalopt->driver,      NPOW_04, "ENVI");
      break;
    case _FMT_GTIFF_:
      copy_string(gdalopt->extension,   NPOW_04, "tif");
      copy_string(gdalopt->driver,      NPOW_04, "GTiff");
      copy_string(gdalopt->option[o++], NPOW_10, "COMPRESS");
      copy_string(gdalopt->option[o++], NPOW_10, "LZW");
      copy_string(gdalopt->option[o++], NPOW_10, "PREDICTOR");
      copy_string(gdalopt->option[o++], NPOW_10, "2");
      copy_string(gdalopt->option[o++], NPOW_10, "INTERLEAVE");
      copy_string(gdalopt->option[o++], NPOW_10, "BAND");
      copy_string(gdalopt->option[o++], NPOW_10, "BIGTIFF");
      copy_string(gdalopt->option[o++], NPOW_10, "YES");
      break;
    case _FMT_COG_:
      copy_string(gdalopt->extension,   NPOW_04, "tif");
      copy_string(gdalopt->driver,      NPOW_04, "COG");
      copy_string(gdalopt->option[o++], NPOW_10, "COMPRESS");
      copy_string(gdalopt->option[o++], NPOW_10, "LZW");
      copy_string(gdalopt->option[o++], NPOW_10, "PREDICTOR");
      copy_string(gdalopt->option[o++], NPOW_10, "YES");
      copy_string(gdalopt->option[o++], NPOW_10, "INTERLEAVE");
      copy_string(gdalopt->option[o++], NPOW_10, "PIXEL");
      copy_string(gdalopt->option[o++], NPOW_10, "BIGTIFF");
      copy_string(gdalopt->option[o++], NPOW_10, "YES");
      copy_string(gdalopt->option[o++], NPOW_10, "TILED");
      copy_string(gdalopt->option[o++], NPOW_10, "YES");
      break;
    case _FMT_JPEG_:
      copy_string(gdalopt->extension,   NPOW_04, "jpg");
      copy_string(gdalopt->driver,      NPOW_04, "JPEG");
      break;
    case _FMT_CUSTOM_:
      break;
    default:
      printf("unknown format.\n");
      exit(FAILURE);
  }

  gdalopt->n = o;


  return;
}


/** This function updates the blocksize in GDAL output options
--- format:  Default format
--- gdalopt: GDAL options (returned)
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void update_gdaloptions_blocksize(int format, gdalopt_t *gdalopt, int cx, int cy){
int nchar;
int o = gdalopt->n;
char blockxsize[NPOW_10];
char blockysize[NPOW_10];


  if (format != _FMT_GTIFF_) return;

  if (cx > 0){
    nchar = snprintf(blockxsize, NPOW_10, "%d", cx);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling BLOCKXSIZE\n"); exit(FAILURE);}
    copy_string(gdalopt->option[o++], NPOW_10, "BLOCKXSIZE");
    copy_string(gdalopt->option[o++], NPOW_10, blockxsize);
  }
  if (cy > 0){
    nchar = snprintf(blockysize, NPOW_10, "%d", cy);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling BLOCKYSIZE\n"); exit(FAILURE);}
    copy_string(gdalopt->option[o++], NPOW_10, "BLOCKYSIZE");
    copy_string(gdalopt->option[o++], NPOW_10, blockysize);
  }

  gdalopt->n = o;


  return;
}


/** This function reads GDAL output options
--- fname:   text file
--- gdalopt: GDAL options (returned)
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void parse_gdaloptions(char *fname, gdalopt_t *gdalopt){
int i, o = 0;
int nrows;
char ***tagval = NULL;
bool b_driver = false;
bool b_ext = false;


  tagval = read_tagvalue(fname, &nrows);

  for (i=0; i<nrows; i++){

    if (strcmp(tagval[i][_TV_TAG_], "DRIVER") == 0){
      copy_string(gdalopt->driver, NPOW_04, tagval[i][_TV_VAL_]);
      b_driver = true;
    } else if (strcmp(tagval[i][_TV_TAG_], "EXTENSION") == 0){
      copy_string(gdalopt->extension, NPOW_04, tagval[i][_TV_VAL_]);
      b_ext = true;
    } else {
      if (o >= (NPOW_06-1)){
        printf("too many GDAL output options."); 
        exit(FAILURE);
      }
      if (tagval[i][_TV_TAG_][0] != '#'){
        copy_string(gdalopt->option[o++], NPOW_10, tagval[i][_TV_TAG_]);
        copy_string(gdalopt->option[o++], NPOW_10, tagval[i][_TV_VAL_]);
      }
    }

  }

  gdalopt->n = o;

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
int o;


  printf("GDAL output options :::\n");
  printf("Driver:    %s\n", gdalopt->driver);
  printf("Extension: %s\n", gdalopt->extension);
  for (o=0; o<gdalopt->n; o+=2){
    printf("Option %2d: %s = %s\n", o, gdalopt->option[o], gdalopt->option[o+1]);
  }

  return;
}
