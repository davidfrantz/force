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
This program builds and maintains a water vapor database that can be
used for atmospheric correction in the FORCE Level-2 Processing System
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** C Standard library **/
#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/datesys-cl.h"
#include "../cross-level/stats-cl.h"
#include "../lower-level/modwvp-ll.h"


int main( int argc, char *argv[] ){
char *fcoords = NULL;
char *dir_wvp = NULL;
char *dir_geo = NULL;
char *dir_hdf = NULL;
char    **SEN = NULL;
float   **COO = NULL;
float    *WVP = NULL;
double ***AVG = NULL;
int m, c, nc;
date_t d_now, d_end;
char tablename[NPOW_10];
char  key[NPOW_10];
int nchar;


  /** usage **/
  if (argc >= 2) konami_args(argv[1]);
  if (argc < 5 || (argc > 5 && argc != 11)){
    printf("usage: %s coords dir-wvp dir-geometa dir-eoshdf\n", argv[0]);
    printf("           [start-year start-month start-day\n");
    printf("            end-year   end-month   end-day]\n\n");
    exit(1);
  }
  

  /** parse arguments **/
  fcoords = argv[1]; dir_wvp = argv[2];
  dir_geo = argv[3]; dir_hdf = argv[4];

  if (argc > 5){
    d_now.year  = atoi(argv[5]);
    d_now.month = atoi(argv[6]);
    d_now.day   = atoi(argv[7]);
    d_end.year  = atoi(argv[8]);
    d_end.month = atoi(argv[9]);
    d_end.day   = atoi(argv[10]);
  } else {
    d_now.year  = 2000;
    d_now.month =  2;
    d_now.day   =  24;
    current_date(&d_end);
  }
  

  // get app key
  get_laads_key(key, NPOW_08);


  /** initialize date **/
  //date_set(d_now, timeinfo);

  /** parse coordinates **/
  if ((nc = parse_coord_list(fcoords, &COO)) < 1){
    printf("error parsing coordinates.\n"); exit(1);}

  alloc((void**)&WVP, nc, sizeof(float));
  alloc_2D((void***)&SEN, nc, NPOW_02, sizeof(char));



  /** Step 1: daily Look-up-tables
  +++ do for each day between start and end **/

  while (0 < 1){

    printf("%4d/%02d/%02d. ", d_now.year, d_now.month, d_now.day); fflush(stdout);

    // LUT name
    nchar = snprintf(tablename, NPOW_10, "%s/WVP_%04d-%02d-%02d.txt", dir_wvp, 
      d_now.year, d_now.month, d_now.day);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

    // create LUT only if it doesn't exist
    if (!fileexist(tablename)){
      printf("do. ");
      create_wvp_lut(dir_geo, dir_hdf, tablename, d_now, nc, COO, WVP, SEN, key);
    } else {
      printf("LUT exists.\n");
    }

    // if iterated date is end date, stop.
    if (d_now.year  == d_end.year && 
        d_now.month == d_end.month && 
        d_now.day   == d_end.day) break;

    // go to next day
    //date_plus(&d_now, timeinfo);
    date_plus(&d_now);

  }

  // monthly averages
  printf("build climatology!\n");

  //get_startdate(sy, sm, sd);
  d_now.year = 2000, d_now.month =  2, d_now.day =  24;
  current_date(&d_end);
  //date_set(d_now, timeinfo);


  alloc_3D((void****)&AVG, 3, 12, nc, sizeof(double));


  /** Step 2: Look-up-table climatology
  +++ do for each day between MODIS start and today **/

  while (0 < 1){

    // LUT name
    nchar = snprintf(tablename, NPOW_10, "%s/WVP_%04d-%02d-%02d.txt", dir_wvp, 
      d_now.year, d_now.month, d_now.day);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

    // sum LUT values
    if (fileexist(tablename)){

      // read the table
      read_wvp_lut(tablename, nc, COO, WVP);

      for (c=0; c<nc; c++){
        if (WVP[c] == 9999) continue;
        
        AVG[2][d_now.month-1][c]++;

        if (AVG[2][d_now.month-1][c] == 1){
          AVG[0][d_now.month-1][c] = WVP[c];
        } else {
          var_recurrence(WVP[c], &AVG[0][d_now.month-1][c], &AVG[1][d_now.month-1][c], AVG[2][d_now.month-1][c]);
        }
      }

    }

    // if iterated date is today, stop.
    if (d_now.year  == d_end.year && 
        d_now.month == d_end.month && 
        d_now.day   == d_end.day) break;

    // go to next day
    date_plus(&d_now);

  }

  // finalize stats
  for (m=0; m<12; m++){
  for (c=0; c<nc; c++){

    if (AVG[2][m][c] < 1){
      AVG[0][m][c] = 9999;
    }

    if (AVG[2][m][c] < 2){
      AVG[1][m][c] = 9999;
    } else {
      AVG[1][m][c] = standdev(AVG[1][m][c], AVG[2][m][c]);
    }

  }
  }


  // write climatology
  write_avg_table(dir_wvp, nc, COO, AVG);

  // free memory
  free((void*)WVP);
  free_2D((void**)COO, 2);
  free_3D((void***)AVG, 3, 12);
  free_2D((void**)SEN, nc);

  return SUCCESS;
}

