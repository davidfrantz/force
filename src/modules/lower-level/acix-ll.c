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
This file contains functions for handling ACIX-specific changes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "acix-ll.h"


#ifdef ACIX


/** in ACIX, images are cropped to subregions, thus image boundaries can't
+++ be obtained from the image data. This function reads the information 
+++ from angle metadata
--- pl2:    L2 parameters
--- ulx:    column ul
--- uly:    row    ul
--- urx:    column ur
--- ury:    row    ur
--- lrx:    column lr
--- lry:    row    lr
--- llx:    column ll
--- lly:    row    ll
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_angledata_landsat(par_ll_t *pl2, int *ulx, int *uly, int *urx, int *ury, int *lrx, int *lry, int *llx, int *lly){
FILE *fp = NULL;
char metaname[NPOW_10];
char  buffer[NPOW_10] = "\0";
char *tag = NULL;
char *tokenptr = NULL;
char *separator = " =,()\":";
bool x_found = false;
bool y_found = false;


  // scan directory for ANG.txt file
  if (findfile_pattern(pl2->d_level1, "ANG", ".txt", metaname, NPOW_10) != SUCCESS){
    printf("Unable to find Landsat angle metadata (ANG file)! "); return FAILURE;}

  // open ANG file
  if ((fp = fopen(metaname, "r")) == NULL){
    printf("Unable to open Landsat angle metadata (ANG file)! "); return FAILURE;}

 
  // process line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){

    // get tag
    tokenptr = strtok(buffer, separator);
    tag=tokenptr;

    // extract parameters by comparing tag
    while (tokenptr != NULL){

      tokenptr = strtok(NULL, separator);
      // Landsat sensor
      if (strcmp(tag, "BAND01_L1T_IMAGE_CORNER_LINES") == 0){
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *uly = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *ury = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *lry = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *lly = atoi(tokenptr);
        y_found = true;
      } else if (strcmp(tag, "BAND01_L1T_IMAGE_CORNER_SAMPS") == 0){
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *ulx = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *urx = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *lrx = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        //printf("tag: %s, ptr: %s\n", tag, tokenptr);
        *llx = atoi(tokenptr);
        x_found = true;
      }
      
      if (x_found && y_found) break;

      // in case tag (key words) is not the first word in a line
      tag = tokenptr;

    }

  }

  fclose(fp);
  

  if (!x_found || !y_found){
    printf("could not parse corner coordinates from ANG file.\n");
    return FAILURE;
  }

  return SUCCESS;
}

#endif

