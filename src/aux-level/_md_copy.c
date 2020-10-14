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
This program copies FORCE metadata from one file to another
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_conv.h"       // various convenience functions for CPL
#include "cpl_string.h"       // various convenience functions for strings


int main ( int argc, char *argv[] ){
int b, nb;
GDALDatasetH src, dst;
GDALRasterBandH bsrc, bdst;
char  *fsrc  = NULL;
char  *fdst  = NULL;
char **meta  = NULL;
char **bmeta = NULL;
const char *bname = NULL;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 3){ printf("Usage: %s src dst\n\n", argv[0]); return FAILURE;}


  fsrc = argv[1];
  fdst = argv[2];


  GDALAllRegister();

  if ((src = GDALOpenEx(fsrc, GDAL_OF_READONLY, NULL, NULL, NULL)) == NULL){
    printf("unable to open %s\n\n", fsrc); return FAILURE;}

  if ((dst = GDALOpenEx(fdst, GDAL_OF_UPDATE,   NULL, NULL, NULL)) == NULL){
    printf("unable to open %s\n\n", fdst); return FAILURE;}

  if ((nb = GDALGetRasterCount(src)) != GDALGetRasterCount(dst)){
    printf("src and dst images have different number of bands\n\n"); 
    return FAILURE;}


  // copy FORCE domain
  meta = GDALGetMetadata(src, "FORCE");
  //printf("Number of metadata items: %d\n", CSLCount(meta));
  //CSLPrint(meta, NULL);
  //CSLDestroy(meta);
  GDALSetMetadata(dst, meta, "FORCE");

  for (b=0; b<nb; b++){

    bsrc = GDALGetRasterBand(src, b+1);
    bdst = GDALGetRasterBand(dst, b+1);

    // copy FORCE domain
    bmeta = GDALGetMetadata(bsrc, "FORCE");
    //printf("Number of metadata items in band %d: %d\n", b+1, CSLCount(bmeta));
    //CSLPrint(bmeta, NULL);
    //CSLDestroy(bmeta);
    GDALSetMetadata(bdst, bmeta, "FORCE");

    // copy bandname
    bname = GDALGetDescription(bsrc);
    GDALSetDescription(bdst, bname);

  }

  GDALClose(src);
  GDALClose(dst);


  return SUCCESS; 
}

