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
This program copies FORCE metadata from one file to another
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/string-cl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"       // public (C callable) GDAL entry points
#include "cpl_conv.h"   // various convenience functions for CPL
#include "cpl_string.h" // various convenience functions for strings


typedef struct {
  int n;
  char fsrc[NPOW_10];
  char fdst[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] src-file dst-file\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'src-file': source of metadata\n");
  printf("  - 'dst-file': destination of metadata\n");
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
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Copy FORCE metadata from one file to another\n");
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
  args->n = 2;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->fsrc, NPOW_10, argv[optind++]);
      copy_string(args->fdst, NPOW_10, argv[optind++]);
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


int main (int argc, char *argv[]){
args_t args;
int b, nb;
GDALDatasetH src, dst;
GDALRasterBandH bsrc, bdst;
char **meta  = NULL;
char **bmeta = NULL;
const char *bname = NULL;


  parse_args(argc, argv, &args);

  GDALAllRegister();

  if ((src = GDALOpenEx(args.fsrc, GDAL_OF_READONLY, NULL, NULL, NULL)) == NULL){
    printf("unable to open %s\n\n", args.fsrc); return FAILURE;}

  if ((dst = GDALOpenEx(args.fdst, GDAL_OF_UPDATE,   NULL, NULL, NULL)) == NULL){
    printf("unable to open %s\n\n", args.fdst); return FAILURE;}

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

  GDALDestroy();

  return SUCCESS; 
}

