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
This program computes a histogram of the given image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/string-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points


typedef struct {
  int n;
  int band;
  char file_input[NPOW_10];
  char file_output[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-b band] [-o output-file] input-image\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -b band = band to use,\n");
  printf("     defaults to 1\n");
  printf("\n");
  printf("  -o output-file  = output file path with extension,\n");
  printf("     defaults to './histogram.csv'\n");
  printf("\n");  
  printf("  Positional arguments:\n");
  printf("  - 'input-image': image for computing the histogram\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;


  opterr = 0;

  // default parameters
  copy_string(args->file_output,  NPOW_10, "histogram.csv");
  args->band = 1;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvio:b:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        printf("FORCE version: %s\n", _VERSION_);
        exit(SUCCESS);
      case 'i':
        printf("Compute image histogram\n");
        exit(SUCCESS);
      case 'o':
        copy_string(args->file_output, NPOW_10, optarg);
        break;
      case 'b':
        args->band = atoi(optarg);
        if (args->band < 0){
          fprintf(stderr, "Band must be >= 1\n");
          usage(argv[0], FAILURE);  
        }
        break;
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
      copy_string(args->file_input, NPOW_10, argv[optind++]);
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


int main(int argc, char *argv[]){
args_t args;
GDALDatasetH  fp;
GDALRasterBandH band;
int i, j, nx, ny;
short *line = NULL;
short nodata;
int has_nodata;

int offset = SHRT_MAX+1;
int length = USHRT_MAX+1;
off_t counts[length];
FILE *fout = NULL;


  parse_args(argc, argv, &args);

  GDALAllRegister();
  if ((fp = GDALOpen(args.file_input, GA_ReadOnly)) == NULL){
    fprintf(stderr, "could not open %s.\n", args.file_input); exit(1);}

  nx  = GDALGetRasterXSize(fp);
  ny  = GDALGetRasterYSize(fp);
  
  alloc((void**)&line, nx, sizeof(short));

  band = GDALGetRasterBand(fp, args.band);

  nodata = (short)GDALGetRasterNoDataValue(band, &has_nodata);
  if (!has_nodata){
    fprintf(stderr, "input image has no nodata value.\n"); 
    exit(1);
  }



  memset(counts, 0, sizeof(off_t)*length);

  for (i=0; i<ny; i++){

    if (GDALRasterIO(band, GF_Read, 0, i, nx, 1, 
      line, nx, 1, GDT_Int16, 0, 0) == CE_Failure){
      fprintf(stderr, "could not read line %d.\n", i+1); exit(1);}

    for (j=0; j<nx; j++){

      if (line[j] == nodata) continue;

      counts[line[j] + offset]++;

    }

  }

  GDALClose(fp);

  free((void*)line);



    
  if ((fout = fopen(args.file_output, "w")) == NULL){
    fprintf(stderr, "Unable to open output file %s\n", args.file_output); 
    return FAILURE;}

  fprintf(fout, "value,count\n");

  for (i=0; i<length; i++){

    if (counts[i] == 0) continue;

    fprintf(fout, "%d,%lu\n", i - offset , counts[i]);

  }

  fclose(fout);

  return SUCCESS;
}

