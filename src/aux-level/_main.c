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
This program is the general entry point to FORCE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/konami-cl.h"


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i]\n", exe);

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[]){
int opt;


  opterr = 0;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvi")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        printf("FORCE version: %s\n", _VERSION_);
        exit(SUCCESS);
      case 'i':
        printf("Entrypoint, print short disclaimer, available modules etc.\n");
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
  if (optind < argc){
    konami_args(argv[optind]);
    fprintf(stderr, "Unknown non-optional parameter.\n");
    usage(argv[0], FAILURE);
  }

  return;
}


int main (int argc, char *argv[]){
char user[NPOW_10];


  parse_args(argc, argv);

  if (getlogin_r(user, NPOW_10) != 0) copy_string(user, NPOW_10, "user");

  printf("\n##########################################################################\n");

  printf("\nHello %s! You are currently running FORCE v. %s\n", 
         user, _VERSION_);
  printf("Framework for Operational Radiometric Correction for "
         "Environmental monitoring\n");
  printf("Copyright (C) 2013-2021 David Frantz, "
         "david.frantz@geo.hu-berlin.de\n");
  printf("+ many community contributions.\n");

  printf("\nFORCE is free software under the terms of the "
         "GNU General Public License as published by the "
         "Free Software Foundation, see "
         "<http://www.gnu.org/licenses/>.\n");

  printf("\nThank you for using FORCE! This software is "
         "being developed in the hope that it will be "
         "helpful for you and your work.\n");

  printf("\nHowever, it is requested that you to use the "
         "software in accordance with academic standards "
         "and fair usage. Without this, software like FORCE "
         "will not survive. This includes citation of the "
         "software and the scientific publications, proper "
         "acknowledgement in any public presentation, or an "
         "offer of co-authorship of scientific articles in "
         "case substantial help in setting up, modifying or "
         "running the software is provided by the author(s).\n");

  printf("\nAt minimum, the citation of following paper is requested:\n"
         "Frantz, D. (2019). FORCE—Landsat + Sentinel-2 Analysis "
         "Ready Data and Beyond. Remote Sensing, 11, 1124\n");

  printf("\nEach FORCE module will generate a \"CITEME\" file with "
         "suggestions for references to be cited. "
         "This list is based on the specific parameterization "
         "you are using.\n");
  
  printf("\nThe documentation and tutorials are available at force-eo.readthedocs.io\n");

  printf("\nFORCE consists of several components:\n"
         "\nLevel 1 Archiving System (L1AS)\n"
         "+ force-level1-csd       Download from cloud storage + maintenance of Level 1 "
         "Landsat and Sentinel-2 data pool\n"
         "+ force-level1-landsat   Maintenance of Level 1 Landsat "
         "data pool (deprecated)\n"
         "+ force-level1-sentinel2 Download from ESA + maintenance of Level 1 "
         "Sentinel-2 data pool (deprecated)\n"
         "\nLevel 2 Processing System (L2PS)\n"
         "+ force-level2           Level 2 processing of image archive\n"
         "+ force-l2ps             Level 2 processing of single image\n"
         "\nWater Vapor Database (WVDB)\n"
         "+ force-lut-modis        Generation and maintenance of water vapor database "
         "using MODIS products\n"
         "\nHigher Level Processing System (HLPS)\n"
         "+ force-higher-level     Higher level processing (compositing, "
         "time series analysis, ...)\n"
         "\nAuxiliary (AUX)\n"
         "+ force-parameter        Generation of parameter files\n"
         "+ force-magic-parameters Replace variables in parameter file with vectors\n"
         "+ force-train            Training (and validation) of Machine "
         "Learning models\n"
         "+ force-synthmix         Synthetic Mixing of training data\n"
         "+ force-qai-inflate      Inflate QAI bit layers\n"
         "+ force-tile-extent      Compute suggested processing extent and "
         "tile allow-list based on a vector layer\n"
         "+ force-tile-finder      Find the tile, pixel, and chunk of a "
         "given coordinate\n"
         "+ force-tabulate-grid    Extract the processing grid as shapefile\n"
         "+ force-cube             Ingestion of auxiliary data into data"
         "cube format\n"
         "+ force-procmask         Processing masks from raster images\n"
         "+ force-pyramid          Generation of image pyramids\n"
         "+ force-mosaic           Mosaicking of image chips\n"
         "+ force-stack            Stack images, works with 4D data model\n"
         "+ force-mdcp             Copy FORCE metadata from one file to another\n");

  printf("\n##########################################################################\n\n");


  return SUCCESS;
}

