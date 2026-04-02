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
This program computes map accuracy and area statistics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/cross-level/cite-cl.h"
#include "../../modules/aux-level/validation-aux.h"

typedef struct{
  char file_input_count[NPOW_10];
  char file_input_sample[NPOW_10];
  char file_output[NPOW_10];
  float pixel_area;
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-o output-file] -c count-file -s sample-file -a pixel-area\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -o output-file  = output file path with extension,\n");
  printf("     defaults to './accuracy-assessment.md'\n");
  printf("  -c count-file  = csv table with pixel counts per class\n");
  printf("     2 columns named class and count\n");
  printf("  -s sample-file = vector file (or csv table) with predicted and reference class labels\n");
  printf("     2 columns named label_map and label_reference\n");
  printf("  -a pixel-area  = area of one pixel in desired reporting unit, e.g.\n");
  printf("      100 for a Sentinel-2 based map to be reported in m², or\n");
  printf("     0.01 for a Sentinel-2 based map to be reported in ha\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;
bool given_c = false;
bool given_s = false;
bool given_a = false;


  opterr = 0;

  // default parameters
  copy_string(args->file_output,  NPOW_10, "accuracy-assessment.md");
 
  // optional parameters
  while ((opt = getopt(argc, argv, "hvio:c:s:a:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Compute map accuracy and area statistics\n");
        exit(SUCCESS);
      case 'o':
        copy_string(args->file_output, NPOW_10, optarg);
        if (fileexist(args->file_output)){
          fprintf(stderr, "output report already exists: %s.\n", args->file_output); 
          usage(argv[0], FAILURE);
        }
        break;
      case 'c':
        copy_string(args->file_input_count, NPOW_10, optarg);
        if (!fileexist(args->file_input_count)){
          fprintf(stderr, "count file does not exist: %s.\n", args->file_input_count); 
          usage(argv[0], FAILURE);
        }
        given_c = true;
        break;
      case 's':
        copy_string(args->file_input_sample, NPOW_10, optarg);
        if (!fileexist(args->file_input_sample)){
          fprintf(stderr, "sample file does not exist: %s.\n", args->file_input_sample); 
          usage(argv[0], FAILURE);
        }
        given_s = true;
        break;
      case 'a':
        args->pixel_area = atof(optarg);
        given_a = true;
        if (args->pixel_area <= 0.0){
          fprintf(stderr, "Pixel area must be > 0.0\n");
          usage(argv[0], FAILURE);  
        }
        break;
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else{
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }

  if (!given_c){
    fprintf(stderr, "count file argument is missing.\n");
    usage(argv[0], FAILURE);
  }

  if (!given_s){
    fprintf(stderr, "sample file argument is missing.\n");
    usage(argv[0], FAILURE);
  }

  if (!given_a){
    fprintf(stderr, "pixel area argument is missing.\n");
    usage(argv[0], FAILURE);
  }


  return;
}


int main(int argc, char *argv[]){
args_t args;


  parse_args(argc, argv, &args);

  cite_me(_CITE_ACCURACY_1_);
  cite_me(_CITE_ACCURACY_2_);

  // read count file
  class_t classes = {0};
  compile_validation_classes(args.file_input_count, args.pixel_area, &classes);
  
  // read and compile labels file
  label_t labels = {0};
  compile_validation_labels(args.file_input_sample, &labels);

  // build confusion matrix
  confusion_t classical_confusion = {0};
  compile_confusion_matrix(&labels, &classes, &classical_confusion);
  
  // Olofsson et al. 2013, eq. 1
  confusion_t adjusted_confusion = {0};
  compute_estimated_area_proportions(&classical_confusion, &classes, &adjusted_confusion);

  // Olofsson et al. 2013, eq. 2
  compute_unbiased_area(&classes, &adjusted_confusion);
  
  // Olofsson et al. 2013, eq. 3-5
  compute_confidence_of_adjusted_area(&classical_confusion, &classes, &adjusted_confusion);

  // Olofsson et al. 2013, eq. 6-8
  accuracy_t classical_accuracy = {0};
  accuracy_metrics(&classical_confusion, &classical_accuracy);
  
  // Olofsson et al. 2013, eq. 6-8
  accuracy_t adjusted_accuracy = {0};
  accuracy_metrics(&adjusted_confusion, &adjusted_accuracy);

  accuracy_t standard_error = {0};
  alloc_2D((void***)&standard_error.class, classes.n, _ACC_LENGTH_, sizeof(double));

  // Olofsson et al. 2014, eq. 5
  compute_confidence_of_overall_accuracy(&classical_confusion, &classes, &adjusted_accuracy, &standard_error);

  // Olofsson et al. 2014, eq. 6
  compute_confidence_of_users_accuracy(&classical_confusion, &classes, &adjusted_accuracy, &standard_error);

  // Olofsson et al. 2014, eq. 7
  compute_confidence_of_producers_accuracy(&classical_confusion, &classes, &adjusted_accuracy, &standard_error);

  // compile the report and output
  generate_validation_report(args.file_output, &classes, &classical_confusion, &adjusted_confusion, &classical_accuracy, &adjusted_accuracy, &standard_error);

  // write citations
  char output_path[NPOW_10];
  directoryname(args.file_output, output_path, NPOW_10);
  cite_push(output_path);
  
  // clean up
  free_validation_memory(&classical_confusion, &adjusted_confusion, &classes, &labels, &classical_accuracy, &adjusted_accuracy, &standard_error);

  return SUCCESS;
}
