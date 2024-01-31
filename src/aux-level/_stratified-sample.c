/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2024 David Frantz

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

#include <time.h>

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/table-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points


typedef struct {
  int n;
  int band;
  char file_input_image[NPOW_10];
  char file_input_sample_size[NPOW_10];
  char file_output[NPOW_10];
  char column_sample_size[NPOW_10];
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
  args->n = 3;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->file_input_image, NPOW_10, argv[optind++]);
      copy_string(args->file_input_sample_size, NPOW_10, argv[optind++]);
      copy_string(args->column_sample_size, NPOW_10, argv[optind++]);
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
table_t sample_size;
int col_size, col_class, col_count, row;

int offset = SHRT_MAX+1;
int length = USHRT_MAX+1;
double **dictionary = NULL;
enum { _CLASS_, _SIZE_, _PROBABILITY_, _COUNT_, _DICT_LENGTH_ };

  parse_args(argc, argv, &args);

  sample_size = read_table(args.file_input_sample_size, false, true);
  if ((col_size = find_table_col(&sample_size, args.column_sample_size)) < 0){
    printf("could not find column name %s in file-sample-size\n", args.column_sample_size); exit(FAILURE);}
  if ((col_class = find_table_col(&sample_size, "class")) < 0){
    printf("could not find column name %s in file-sample-size\n", "class"); exit(FAILURE);}
  if ((col_count = find_table_col(&sample_size, "count")) < 0){
    printf("could not find column name %s in file-sample-size\n", "count"); exit(FAILURE);}

  print_table(&sample_size, false);
  printf("column %s in column %d\n", "class", col_class);
  printf("column %s in column %d\n", "count", col_count);
  printf("column %s in column %d\n", args.column_sample_size, col_size);

  printf("min/max class: %d/%d\n", (int)sample_size.min[col_class], (int)sample_size.max[col_class]);



  alloc_2D((void***)&dictionary, length, _DICT_LENGTH_, sizeof(double));

  for (row=0; row<sample_size.nrow; row++){
    dictionary[(int)sample_size.data[row][col_class] + offset][_CLASS_] = sample_size.data[row][col_class];
    dictionary[(int)sample_size.data[row][col_class] + offset][_SIZE_] = sample_size.data[row][col_size];
    dictionary[(int)sample_size.data[row][col_class] + offset][_PROBABILITY_] = sample_size.data[row][col_size] / sample_size.data[row][col_count];

    printf("%.2f %.2f %.12f\n", 
      dictionary[(int)sample_size.data[row][col_class] + offset][_CLASS_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_SIZE_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_PROBABILITY_]);
  }


  srand(time(NULL)); 
  printf("random number: %f\n", (double) (rand() % (size_t) sample_size.sum[col_count])/sample_size.sum[col_count]);
  printf("random number: %f\n", (double) (rand() % (size_t) sample_size.sum[col_count])/sample_size.sum[col_count]);
  printf("random number: %f\n", (double) (rand() % (size_t) sample_size.sum[col_count])/sample_size.sum[col_count]);
  printf("random number: %f\n", (double) (rand() % (size_t) sample_size.sum[col_count])/sample_size.sum[col_count]);
  printf("random number: %f\n", (double) (rand() % (size_t) sample_size.sum[col_count])/sample_size.sum[col_count]);


  GDALAllRegister();
  if ((fp = GDALOpen(args.file_input_image, GA_ReadOnly)) == NULL){
    fprintf(stderr, "could not open %s.\n", args.file_input_image); exit(1);}

  nx  = GDALGetRasterXSize(fp);
  ny  = GDALGetRasterYSize(fp);
  
  alloc((void**)&line, nx, sizeof(short));

  band = GDALGetRasterBand(fp, args.band);

  nodata = (short)GDALGetRasterNoDataValue(band, &has_nodata);
  if (!has_nodata){
    fprintf(stderr, "input image has no nodata value.\n"); 
    exit(1);
  }

 int cnt = 0;
double probability;
 /** FILE *fp_ = NULL;
  fp_ = fopen("test.txt", "w");
  for (i=0; i<346589861; i++){
    probability = (double)rand()/(double)RAND_MAX;
    fprintf(fp_, "%f\n", probability);
  }
  fclose(fp_);
  exit(0);
 **/
  for (i=0; i<ny; i++){

    if (GDALRasterIO(band, GF_Read, 0, i, nx, 1, 
      line, nx, 1, GDT_Int16, 0, 0) == CE_Failure){
      fprintf(stderr, "could not read line %d.\n", i+1); exit(1);}

    for (j=0; j<nx; j++){

      if (line[j] == nodata) continue;

      //probability = (rand() % (size_t) sample_size.sum[col_count])/sample_size.sum[col_count]*100;
      probability = (double)rand()/(double)RAND_MAX;

      //printf("probability: %f\n", probability);

      if (dictionary[line[j] + offset][_PROBABILITY_] >= probability){
        dictionary[line[j] + offset][_COUNT_]++;
        printf("%d samples selected (row %d, col %d)\n", cnt++, i, j);
      }
        

    }

  }


 for (row=0; row<sample_size.nrow; row++){
    printf("%.2f, %.2f, %.12f, %.0f,\n", 
      dictionary[(int)sample_size.data[row][col_class] + offset][_CLASS_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_SIZE_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_PROBABILITY_],
      dictionary[(int)sample_size.data[row][col_class] + offset][_COUNT_]);
  }


  GDALClose(fp);

  free((void*)line);





  free_table(&sample_size);
  free_2D((void**)dictionary, length);

  return SUCCESS;
}

