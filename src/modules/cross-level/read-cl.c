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
This file contains functions for reading all-purpose files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-cl.h"


/** This function reads a tag and value file
--- fname:  text file
--- nrows: number of rows (returned)
+++ Return: tag and values
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
char ***read_tagvalue(char *fname, int *nrows){
FILE *fp;
char  buffer[NPOW_16] = "\0";
char *ptr = NULL;
const char *separator = " =";
char ***tagval = NULL;
int n = 0;
int n_buf = 1;



  alloc_3D((void****)&tagval, n_buf, _TV_LENGTH_, NPOW_10, sizeof(char));

  // open file
  if (!(fp = fopen(fname, "r"))){
    printf("unable to open tag and value file %s. ", fname); 
    free_3D((void***)tagval, n_buf, _TV_LENGTH_);
    return NULL;
  }


  // read line by line
  while (fgets(buffer, NPOW_16, fp) != NULL){

    buffer[strcspn(buffer, "\r\n")] = 0;

    char *saveptr = NULL;
    if ((ptr = strtok_r(buffer, separator, &saveptr)) == NULL){
      printf("could not read tag/value pair from:\n  %s\n", fname);
      free_3D((void***)tagval, n_buf, _TV_LENGTH_);
      fclose(fp);
      return NULL;
    } else {
      copy_string(tagval[n][_TV_TAG_], NPOW_10, ptr);
    }

    if ((ptr = strtok_r(NULL, separator, &saveptr)) == NULL){
      printf("could not read tag/value pair from:\n  %s\n", fname);
      free_3D((void***)tagval, n_buf, _TV_LENGTH_);
      fclose(fp);
      return NULL;
    } else {
      copy_string(tagval[n][_TV_VAL_], NPOW_10, ptr);
    }

    n++;

    // if too many rows, add twice of previous rows to buffer
    if (n >= n_buf){
      re_alloc_3D((void****)&tagval, n_buf, _TV_LENGTH_, NPOW_10, n_buf*2, _TV_LENGTH_, NPOW_10, sizeof(double));
      n_buf *= 2;
    }

  }

  fclose(fp);


  // re-shape buffer to actual dimensions, 1st rows, then cols
  if (n != n_buf){
    re_alloc_3D((void****)&tagval, n_buf, _TV_LENGTH_, NPOW_10, n, _TV_LENGTH_, NPOW_10, sizeof(double));
  }


  *nrows = n;
  return tagval;
}
