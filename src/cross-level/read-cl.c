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
This file contains functions for reading all-purpose files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-cl.h"


/** This function reads a table.
--- fname:  text file
--- nrows: number of rows (returned)
--- ncols: number of cols (returned)
+++ Return: table
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double **read_table(char *fname, size_t *nrows, size_t *ncols){
FILE *fp;
char  buffer[NPOW_16] = "\0";
char *ptr = NULL;
const char *separator = " \t";
double **tab = NULL;
size_t ni = 0;
size_t nj = 0;
size_t nj_first = 0;
size_t ni_buf = NPOW_00;
size_t nj_buf = NPOW_00;


  alloc_2D((void***)&tab, ni_buf, nj_buf, sizeof(double));

  // open file
  if (!(fp = fopen(fname, "r"))){
    printf("unable to open table %s. ", fname); return NULL;}

  ni       = 0;
  nj_first = 0;
    
  // read line by line
  while (fgets(buffer, NPOW_16, fp) != NULL){
    
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
      printf("unable to read table %s. Different number of cols found in line %lu", fname, ni); 
      return NULL;
    }

    // if too many rows, add twice of previous rows to buffer
    if (ni >= (int)ni_buf){
      re_alloc_2D((void***)&tab, ni_buf, nj_buf, ni_buf*2, nj_buf, sizeof(double));
      ni_buf *= 2;
    }

  }

  fclose(fp);


  // re-shape buffer to actual dimensions
  if (ni != ni_buf || nj != nj_buf){
    re_alloc_2D((void***)&tab, ni_buf, nj_buf, ni, nj, sizeof(double));
  }


  *nrows = ni;
  *ncols = nj;
  return tab;
}

