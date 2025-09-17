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


/** This function reads a table.
--- fname:  text file
--- nrows: number of rows (returned)
--- ncols: number of cols (returned)
+++ Return: table
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double **read_table_deprecated(char *fname, int *nrows, int *ncols){
FILE *fp;
char  buffer[NPOW_16] = "\0";
char *ptr = NULL;
const char *separator = " ,\t";
double **tab = NULL;
int ni = 0;
int nj = 0;
int nj_first = 0;
int ni_buf = NPOW_00;
int nj_buf = NPOW_00;


  alloc_2D((void***)&tab, ni_buf, nj_buf, sizeof(double));

  // open file
  if (!(fp = fopen(fname, "r"))){
    printf("unable to open table %s. ", fname); return NULL;}

  ni       = 0;
  nj_first = 0;
    
  // read line by line
  while (fgets(buffer, NPOW_16, fp) != NULL){
    
    buffer[strcspn(buffer, "\r\n")] = 0;

    ptr = strtok(buffer, separator);
    nj = 0;

    // parse one line
    while (ptr != NULL){
      tab[ni][nj] = atof(ptr); ptr = strtok(NULL, separator);
      nj++;

      // if too many cols, add twice of previous cols to buffer
      if (nj >= nj_buf){
        re_alloc_2D((void***)&tab, ni_buf, nj_buf, ni_buf, nj_buf*2, sizeof(double));
        nj_buf *= 2;
      }

    }

    // number of cols in 1st row
    if (ni == 0) nj_first = nj;
    ni++;

    // table is regular?
    if (ni > 0 && nj != nj_first){
      printf("unable to read table %s. Different number of cols found in line %d", fname, ni); 
      return NULL;
    }

    // if too many rows, add twice of previous rows to buffer
    if (ni >= (int)ni_buf){
      re_alloc_2D((void***)&tab, ni_buf, nj_buf, ni_buf*2, nj_buf, sizeof(double));
      ni_buf *= 2;
    }

  }

  fclose(fp);


  // re-shape buffer to actual dimensions, 1st rows, then cols
  if (ni != ni_buf || nj != nj_buf){
    re_alloc_2D((void***)&tab, ni_buf, nj_buf, ni, nj_buf, sizeof(double));
    re_alloc_2D((void***)&tab, ni,     nj_buf, ni, nj,     sizeof(double));
  }


  *nrows = ni;
  *ncols = nj;
  return tab;
}


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
int n_buf = NPOW_00;



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

    if ((ptr = strtok(buffer, separator)) == NULL){
      printf("could not read tag/value pair from:\n  %s\n", fname);
      free_3D((void***)tagval, n_buf, _TV_LENGTH_);
      return NULL;
    } else {
      copy_string(tagval[n][_TV_TAG_], NPOW_10, ptr);
    }

    if ((ptr = strtok(NULL, separator)) == NULL){
      printf("could not read tag/value pair from:\n  %s\n", fname);
      free_3D((void***)tagval, n_buf, _TV_LENGTH_);
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
