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
--- fname:         text file
--- has_row_names: Has the table row    names? true/false
--- has_col_names: Has the table column names? true/false
+++ Return:        table
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
table_t read_table(char *fname, bool has_row_names, bool has_col_names){
FILE *fp;
table_t table;
char  buffer[NPOW_16] = "\0";
char *ptr = NULL;
const char *separator = " ,\t";
int line = 0;
int row  = 0;
int col  = 0;
int nrow_buf = NPOW_00;
int ncol_buf = NPOW_00;
double mx, vx, k;


  // allocate table data
  alloc_2D((void***)&table.data, nrow_buf, ncol_buf, sizeof(double));

  // allocate row names
  if ((table.has_row_names = has_row_names)){
    alloc_2D((void***)&table.row_names, nrow_buf, NPOW_10, sizeof(char));
  } else {
    table.row_names = NULL;
  }

  // allocate column names
  if ((table.has_col_names = has_col_names)){
    alloc_2D((void***)&table.col_names, ncol_buf, NPOW_10, sizeof(char));
  } else {
    table.col_names = NULL;
  }


  // open file
  if (!(fp = fopen(fname, "r"))){
    printf("unable to open table %s\n", fname); 
    exit(FAILURE);
  }

    
  // read line by line
  while (fgets(buffer, NPOW_16, fp) != NULL){
    
    buffer[strcspn(buffer, "\r\n")] = 0;

    ptr = strtok(buffer, separator);
    col = 0;

    // parse one line
    while (ptr != NULL){

      // parse column names
      if (row == 0 && has_col_names && table.ncol == 0){

        // skip 1st item if there are row names
        // if there is no other item after, exit with error
        if (has_row_names && col == 0){
          if ((ptr = strtok(NULL, separator)) == NULL){
            printf("unable to read table %s. malformed col_names.\n", fname);
            exit(FAILURE);
          }
        }

        copy_string(table.col_names[col], NPOW_10, ptr);
        ptr = strtok(NULL, separator);
        col++;

        // if too many cols, add twice of previous cols to buffer
        if (col >= ncol_buf){
          re_alloc_2D((void***)&table.col_names, ncol_buf, NPOW_10, ncol_buf*2, NPOW_10, sizeof(char));
          re_alloc_2D((void***)&table.data, nrow_buf, ncol_buf, nrow_buf, ncol_buf*2, sizeof(double));
          ncol_buf *= 2;
        }

      } else {

        // parse row name
        // if there is no other item after, exit with error
        if (has_row_names && col == 0){
          copy_string(table.row_names[row], NPOW_10, ptr);
          if ((ptr = strtok(NULL, separator)) == NULL){
            printf("unable to read table %s. malformed row_names.\n", fname);
            exit(FAILURE);
          }
        }

        table.data[row][col] = atof(ptr); 
        ptr = strtok(NULL, separator);
        col++;

        // if too many cols, add twice of previous cols to buffer
        if (col >= ncol_buf){
          re_alloc_2D((void***)&table.data, nrow_buf, ncol_buf, nrow_buf, ncol_buf*2, sizeof(double));
          ncol_buf *= 2;
        }

      }

    }

    // number of cols in 1st row
    if (row == 0) table.ncol = col;
    if (!has_col_names || line > 0) row++;

    // table is regular?
    if (row > 0 && col != table.ncol){
      printf("unable to read table %s. Different number of cols found in line %d\n", fname, row); 
      exit(FAILURE);
    }

    // if too many rows, add twice of previous rows to buffer
    if (row >= (int)nrow_buf){
      re_alloc_2D((void***)&table.data, nrow_buf, ncol_buf, nrow_buf*2, ncol_buf, sizeof(double));
      if (has_row_names) re_alloc_2D((void***)&table.row_names, nrow_buf, NPOW_10, nrow_buf*2, NPOW_10, sizeof(char));
      nrow_buf *= 2;
    }

    line++;

  }

  table.nrow = row;
  fclose(fp);

  
  // re-shape buffer to actual dimensions, 1st rows, then cols
  if (table.nrow != nrow_buf || table.ncol != ncol_buf){
    re_alloc_2D((void***)&table.data, nrow_buf,   ncol_buf, table.nrow, ncol_buf,   sizeof(double));
    re_alloc_2D((void***)&table.data, table.nrow, ncol_buf, table.nrow, table.ncol, sizeof(double));
    if (has_row_names) re_alloc_2D((void***)&table.row_names, nrow_buf, NPOW_10, table.nrow, NPOW_10, sizeof(char));
    if (has_col_names) re_alloc_2D((void***)&table.col_names, ncol_buf, NPOW_10, table.ncol, NPOW_10, sizeof(char));
  }

  alloc((void**)&table.row_mask, table.nrow, sizeof(bool));
  memset(table.row_mask, 1, table.nrow);
  table.n_active_rows = table.nrow;

  alloc((void**)&table.col_mask, table.ncol, sizeof(bool));
  memset(table.col_mask, 1, table.ncol);
  table.n_active_cols = table.ncol;


  alloc((void**)&table.mean, table.ncol, sizeof(double));
  alloc((void**)&table.sd,   table.ncol, sizeof(double));

  for (col=0; col<table.ncol; col++){

    mx = vx = k = 0;

    for (row=0; row<table.nrow; row++){

      k++;
      if (k == 1){
        mx = table.data[row][col];
      } else {
        var_recurrence(table.data[row][col], &mx, &vx, k);
      }

    }
 
    table.mean[col] = mx;
    table.sd[col]   = standdev(vx, k);

  }


  return table;
}


/** This function allocates an empty table.
--- nrow:          number of rows
--- ncol:          number of columns
--- has_row_names: Has the table row    names? true/false
--- has_col_names: Has the table column names? true/false
+++ Return:        table
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
table_t allocate_table(int nrow, int ncol, bool has_row_names, bool has_col_names){
table_t table;


  // allocate table data
  alloc_2D((void***)&table.data, nrow, ncol, sizeof(double));

  // allocate row names
  if ((table.has_row_names = has_row_names)){
    alloc_2D((void***)&table.row_names, nrow, NPOW_10, sizeof(char));
  } else {
    table.row_names = NULL;
  }

  // allocate column names
  if ((table.has_col_names = has_col_names)){
    alloc_2D((void***)&table.col_names, ncol, NPOW_10, sizeof(char));
  } else {
    table.col_names = NULL;
  }


  alloc((void**)&table.row_mask, table.nrow, sizeof(bool));
  memset(table.row_mask, 1, table.nrow);
  table.n_active_rows = table.nrow;

  alloc((void**)&table.col_mask, table.ncol, sizeof(bool));
  memset(table.col_mask, 1, table.ncol);
  table.n_active_cols = table.ncol;


  alloc((void**)&table.mean, table.ncol, sizeof(double));
  alloc((void**)&table.sd,   table.ncol, sizeof(double));

  return table;
}


/** This function prints a table.
--- table:  table
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_table(table_t *table, bool truncate){
int row, col;
int nrow_print = 6, ncol_print = 6;
int width, *max_width = NULL;


  if (!truncate || nrow_print > table->nrow) nrow_print = table->nrow;
  if (!truncate || ncol_print > table->ncol) ncol_print = table->ncol;
  
  alloc((void**)&max_width, table->ncol + 1, sizeof(int));

  if (table->has_row_names){
    for (row=0; row<table->nrow; row++){
      width = strlen(table->row_names[row]);
      if (width > max_width[0]) max_width[0] = width;
    }
  }

  for (col=0; col<table->ncol; col++){

    if (table->has_col_names){
        width = strlen(table->col_names[col]);
        if (width > max_width[col+1]) max_width[col+1] = width;
    }

    for (row=0; row<table->nrow; row++){
        width = num_decimal_places((int)table->data[row][col]) + 4; // + 2 decimal digits, + decimal point, + sign
        if (width > max_width[col+1]) max_width[col+1] = width;
    }

  }


  if (table->has_col_names){

    if (table->has_row_names) printf("%*s ", max_width[0], "");

    for (col=0; col<ncol_print; col++) printf("%*s ", max_width[col+1], table->col_names[col]);
    if (truncate && col < table->ncol) printf("...");
    printf("\n");

  }

  for (row=0; row<nrow_print; row++){

    if (table->has_row_names) printf("%*s ", max_width[0], table->row_names[row]);
    for (col=0; col<ncol_print; col++) printf("%+*.2f ", max_width[col+1], table->data[row][col]);
    if (truncate && col < table->ncol) printf("...");
    printf("\n");

  }
  if (truncate && row < table->nrow) printf("...\n");
  printf("\n");


  free((void*)max_width);

  return;
}


/** This function frees a table.
--- table:  table
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_table(table_t *table){


  if (table->nrow < 1 || table->ncol < 1){
    printf("wrong dimensions (%d, %d). cannot free table.\n", table->nrow, table->ncol);
    exit(FAILURE);
  }

  if (table->has_row_names && table->row_names != NULL){
    free_2D((void**)table->row_names, table->nrow);
  }

  if (table->has_col_names && table->col_names != NULL){
    free_2D((void**)table->col_names, table->ncol);
  }

  if (table->data != NULL){
    free_2D((void**)table->data, table->nrow);
    table->data = NULL;
  }

  if (table->row_mask != NULL){
    free((void*)table->row_mask);
    table->row_mask = NULL;
  }

  if (table->col_mask != NULL){
    free((void*)table->col_mask);
    table->col_mask = NULL;
  }

  if (table->mean != NULL){
    free((void*)table->mean);
    table->mean = NULL;
  }

  if (table->sd != NULL){
    free((void*)table->sd);
    table->sd = NULL;
  }

  return;
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
    printf("unable to open tag and value file %s. ", fname); return NULL;}


  // read line by line
  while (fgets(buffer, NPOW_16, fp) != NULL){

    buffer[strcspn(buffer, "\r\n")] = 0;

    if ((ptr = strtok(buffer, separator)) == NULL){
      printf("could not read tag/value pair from:\n  %s\n", fname);
      exit(FAILURE);
    } else {
      copy_string(tagval[n][_TV_TAG_], NPOW_10, ptr);
    }

    if ((ptr = strtok(NULL, separator)) == NULL){
      printf("could not read tag/value pair from:\n  %s\n", fname);
      exit(FAILURE);
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
