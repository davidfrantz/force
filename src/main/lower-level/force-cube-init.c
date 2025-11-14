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
This program initializes a datacube-definition.prj
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/cube-cl.h"
#include "../../modules/lower-level/param-ll.h"
#include "../../modules/lower-level/cube-ll.h"


typedef struct {
  int n;
  double geo[2]; // lon/lat
  double tile_size[2];
  char dcube[NPOW_10];
  char proj[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-d datacube-dir] [-o lon/lat] \n", exe);
  printf("          [-t tile-size] projection\n");
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -d datacube-dir = output directory for datacube definition\n");
  printf("     default: current working directory\n");
  printf("\n");
  printf("  -o lon,lat = origin coordinates of the grid\n");
  printf("     use geographic coordinates!\n");
  printf("     longitude is X!\n");
  printf("     latitude  is Y!\n");
  printf("     default: -25,60 (is ignored for pre-defined projections!)\n");
  printf("\n");
  printf("  -t tile-size\n");
  printf("     default: 30000,30000 (is ignored for pre-defined projections!)\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - Projection (custom WKT string or built-in projection\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;
char buffer[NPOW_10];
char *ptr = NULL;
char *saveptr = NULL;
const char *separator = ",";
int i;


  opterr = 0;

  // default parameters
  args->tile_size[_X_] = 30000;
  args->tile_size[_Y_] = 30000;
  args->geo[_X_] = -25;
  args->geo[_Y_] =  60;
  copy_string(args->dcube, 1024, ".");

  // optional parameters
  while ((opt = getopt(argc, argv, "hvid:o:t:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Initialize a datacube definition\n");
        exit(SUCCESS);
      case 'd':
        copy_string(args->dcube, NPOW_10, optarg);
        break;
      case 'o':
        copy_string(buffer, NPOW_10, optarg);
        ptr = strtok_r(buffer, separator, &saveptr);
        i = 0;
        while (ptr != NULL){
          if (i < 2) args->geo[i] = atof(ptr);
          ptr = strtok_r(NULL, separator, &saveptr);
          i++;
        }
        if (i != 2){
          fprintf(stderr, "Coordinate must have 2 numbers.\n");
          usage(argv[0], FAILURE);  
        }
        break;
      case 't':
        copy_string(buffer, NPOW_10, optarg);
        ptr = strtok_r(buffer, separator, &saveptr);
        i = 0;
        while (ptr != NULL){
          if (i < 2) args->tile_size[i] = atof(ptr);
          if (args->tile_size[i] <= 0){
            fprintf(stderr, "Tile size must be > 0.\n");
            usage(argv[0], FAILURE);  
          }
          ptr = strtok_r(NULL, separator, &saveptr);
          i++;
        }
        if (i != 2){
          fprintf(stderr, "Tile size must have 2 numbers.\n");
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
      copy_string(args->proj, NPOW_10, argv[optind++]);
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
par_ll_t pl2;
multicube_t *multicube = NULL;


  parse_args(argc, argv, &args);

  pl2.res = 1;
  pl2.doreproj = true;
  pl2.dotile   = true;

  pl2.d_level2 = args.dcube;
  copy_string(pl2.proj, NPOW_10, args.proj);
  alloc((void**)&pl2.tile_size, (pl2.n_tile_size = 2), sizeof(double));
  pl2.tile_size[_X_] = args.tile_size[_X_];
  pl2.tile_size[_Y_] = args.tile_size[_Y_];
  pl2.orig_lon = args.geo[_X_];
  pl2.orig_lat = args.geo[_Y_];

  if ((multicube = start_multicube(&pl2, NULL)) == NULL){
    printf("Starting datacube(s) failed.\n"); return FAILURE;}

  free((void*)pl2.tile_size);
  free_multicube(multicube);

  return SUCCESS;
}

