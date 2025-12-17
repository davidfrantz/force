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
This program is the general entry point to FORCE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
//#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/higher-level/index-parse-hl.h"
#include "../../modules/higher-level/sensor-hl.h"


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-p] [-s] [-x]\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -p  = print path of FORCE runtime-data\n");
  printf("  -s  = print sensor definitions\n");
  printf("  -x  = print index definitions\n");
  printf("\n");

  exit(exit_code);
  return;
}


typedef struct {
  bool print_path;
  bool print_sensors;
  bool print_indices;
} args_t;

void parse_args(int argc, char *argv[], args_t *args){
int opt;

  opterr = 0;
  
  if (argc < 2){
    usage(argv[0], FAILURE);
  }

  // optional parameters
  while ((opt = getopt(argc, argv, "hvipsx")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Print runtime data information\n");
        exit(SUCCESS);
      case 'p':
        args->print_path = true;
        break;
      case 's':
        args->print_sensors = true;
        break;
      case 'x':
        args->print_indices = true;
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
  if (optind < argc){
    konami_args(argv[optind]);
    fprintf(stderr, "Unknown non-optional parameter.\n");
    usage(argv[0], FAILURE);
  }

  return;
}


int main (int argc, char *argv[]){
args_t args = {0};

  parse_args(argc, argv, &args);

  if (args.print_path){

    char d_exe[NPOW_10];
    get_install_directory(d_exe, NPOW_10);

    char d_runtime_data[NPOW_10];
    concat_string_2(d_runtime_data, NPOW_10, d_exe, "force-misc/runtime-data", "/");

    if (!fileexist(d_runtime_data)){
      fprintf(stderr, "Error: Runtime data directory does not exist: %s\n", d_runtime_data);
      exit(FAILURE);
    }

    printf("Runtime data directory: %s\n", d_runtime_data);

  }

  if (args.print_sensors){
    print_all_sensor_definitions();
  }

  if (args.print_indices){
    print_index_definitions();
  }

  return SUCCESS;
}

