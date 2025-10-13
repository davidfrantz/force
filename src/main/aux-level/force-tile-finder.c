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
This program returns the tile ID and pixel that contains the requested 
input coordinate.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/cube-cl.h"
#include "../../modules/cross-level/warp-cl.h"


typedef struct {
  int n;
  double geo[2]; // lon/lat
  double res;
  char dcube[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-p lon,lat] [-r resolution] datacube-dir\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -p lon,lat  = point of interest\n");
  printf("     use geographic coordinates!\n");
  printf("     longitude is X!\n");
  printf("     latitude  is Y!\n");
  printf("\n");
  printf("  -r resolution  = target resolution\n");
  printf("     this is needed to compute the pixel number\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'datacube-dir': directory of existing datacube\n");
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
  args->geo[_X_] =  6.675589; // where FORCE was "born"
  args->geo[_Y_] = 49.748134;
  args->res =  10;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvip:r:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Find the tile, pixel, and chunk of a given coordinate\n");
        exit(SUCCESS);
      case 'p':
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
      case 'r':
        args->res = atof(optarg);
        if (args->res <= 0){
          fprintf(stderr, "Resolution must be > 0.\n");
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
      copy_string(args->dcube, NPOW_10, argv[optind++]);
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
coord_t map, tile;
int t_ulx, t_uly, ti, tj;
cube_t *cube = NULL;


   parse_args(argc, argv, &args);

  // read datacube definition
  if ((cube = read_datacube_def(args.dcube)) == NULL){
    printf("Reading datacube definition failed.\n"); return FAILURE;}
  update_datacube_res(cube, args.res);


  // get target coordinates in target css coordinates
  if ((warp_geo_to_any(args.geo[_X_],  args.geo[_Y_], &map.x, &map.y, cube->projection)) == FAILURE){
    printf("Computing target coordinates in dst_srs failed!\n"); return FAILURE;}


  // find the tile the target coordinates fall into
  tile_find(map.x, map.y, &tile.x, &tile.y, &t_ulx, &t_uly, cube);

  // find pixel in tile
  tj = (int)((map.x-tile.x)/cube->resolution);
  ti = (int)((tile.y-map.y)/cube->resolution);


  // Print to stdout
  printf("Point { LON/LAT (%.2f,%.2f) | X/Y (%.2f,%.2f) }\n"
          "       is in tile X%04d_Y%04d at pixel J/I %d/%d\n"
          "       considering a resolution of %.2f\n",
    args.geo[_X_], args.geo[_Y_], map.x, map.y, 
    t_ulx, t_uly, tj, ti, cube->resolution);

          
  free_datacube(cube);

  return SUCCESS;
}

