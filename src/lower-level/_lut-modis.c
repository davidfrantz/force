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

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/datesys-cl.h"
#include "../cross-level/stats-cl.h"
#include "../lower-level/modwvp-ll.h"


typedef struct {
  int n;
  char fcoord[NPOW_10];
  char dwvp[NPOW_10];
  char dgeo[NPOW_10];
  char dhdf[NPOW_10];
  date_t date_start;
  date_t date_end;
  bool daily;
  bool climatology;
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-d] [-t] [-c] coords-file wvp-dir geometa-dir download-dir\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -d  = date range as YYYYMMDD,YYYYMMDD\n");
  printf("        default: 20000224,today\n");
  printf("  -t  = build daily tables? Default: true\n");
  printf("  -c  = build climatology? Default: true\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'coords-file':  text file with coordinates\n");
  printf("  - 'wvp-dir':      directory for water vapor database\n");
  printf("  - 'geometa-dir':  download directory for geometa data\n");
  printf("  - 'download-dir': download directory for HDF images\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
char cy[5], cm[3], cd[3];
int opt;


  opterr = 0;

  // default parameters
  args->date_start.year  = 2000;
  args->date_start.month =  2;
  args->date_start.day   =  24;
  current_date(&args->date_end);
  args->daily       = true;
  args->climatology = true;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvid:t:c:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        printf("FORCE version: %s\n", _VERSION_);
        exit(SUCCESS);
      case 'i':
        printf("Generation and maintenance of water vapor database using MODIS products\n");
        exit(SUCCESS);
      case 'd':
        if (strlen(optarg) != 17){
          fprintf(stderr, "date is not in format 'YYYYMMDD,YYYYMMDD'\n");
          usage(argv[0], FAILURE);
        }
        strncpy(cy, optarg,   4); cy[4] = '\0';
        strncpy(cm, optarg+4, 2); cm[2] = '\0';
        strncpy(cd, optarg+6, 2); cd[2] = '\0';
        set_date(&args->date_start, atoi(cy), atoi(cm), atoi(cd));
        strncpy(cy, optarg+9,   4); cy[4] = '\0';
        strncpy(cm, optarg+13, 2); cm[2] = '\0';
        strncpy(cd, optarg+15, 2); cd[2] = '\0';
        set_date(&args->date_end, atoi(cy), atoi(cm), atoi(cd));
        break;
      case 't':
        if (strcmp(optarg, "true") == 0){
          args->daily = true;
        } else if (strcmp(optarg, "false") == 0){
          args->daily = false;
        } else {
          fprintf(stderr, "unknown -t option (is: %s, valid: true or false)\n", optarg);
          usage(argv[0], FAILURE);
        }
        break;
      case 'c':
        if (strcmp(optarg, "true") == 0){
          args->climatology = true;
        } else if (strcmp(optarg, "false") == 0){
          args->climatology = false;
        } else {
          fprintf(stderr, "unknown -c option (is: %s, valid: true or false)\n", optarg);
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

  #ifdef FORCE_DEBUG
  print_date(&args->date_start);
  print_date(&args->date_end);
  #endif

  // non-optional parameters
  args->n = 4;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->fcoord, NPOW_10, argv[optind++]);
      copy_string(args->dwvp,   NPOW_10, argv[optind++]);
      copy_string(args->dgeo,   NPOW_10, argv[optind++]);
      copy_string(args->dhdf,   NPOW_10, argv[optind++]);
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


int main( int argc, char *argv[] ){
args_t args;
char    **SEN = NULL;
float   **COO = NULL;
float    *WVP = NULL;
double ***AVG = NULL;
int m, c, nc;
char tablename[NPOW_10];
char  key[NPOW_10];
int nchar;


  parse_args(argc, argv, &args);

  // get app key / token
  get_laads_key(key, NPOW_08);


  /** parse coordinates **/
  if ((nc = parse_coord_list(args.fcoord, &COO)) < 1){
    printf("error parsing coordinates.\n"); exit(1);}

  alloc((void**)&WVP, nc, sizeof(float));
  alloc_2D((void***)&SEN, nc, NPOW_02, sizeof(char));


  /** Step 1: daily Look-up-tables
  +++ do for each day between start and end **/

  if (args.daily){

    while (0 < 1){

      printf("%4d/%02d/%02d. ", args.date_start.year, args.date_start.month, args.date_start.day); fflush(stdout);

      // LUT name
      nchar = snprintf(tablename, NPOW_10, "%s/WVP_%04d-%02d-%02d.txt", args.dwvp, 
        args.date_start.year, args.date_start.month, args.date_start.day);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

      // create LUT only if it doesn't exist
      if (!fileexist(tablename)){
        printf("do. ");
        create_wvp_lut(args.dgeo, args.dhdf, tablename, args.date_start, nc, COO, WVP, SEN, key);
      } else {
        printf("LUT exists.\n");
      }

      // if iterated date is end date, stop.
      if (args.date_start.year  == args.date_end.year && 
          args.date_start.month == args.date_end.month && 
          args.date_start.day   == args.date_end.day) break;

      // go to next day
      //date_plus(&args.date_start, timeinfo);
      date_plus(&args.date_start);

    }

  }


  /** Step 2: Look-up-table climatology
  +++ do for each day between MODIS start and today **/
  if (args.climatology){

    // monthly averages
    printf("build climatology!\n");

    //get_startdate(sy, sm, sd);
    args.date_start.year = 2000, args.date_start.month =  2, args.date_start.day =  24;
    current_date(&args.date_end);
    //date_set(args.date_start, timeinfo);

    alloc_3D((void****)&AVG, 3, 12, nc, sizeof(double));

    while (0 < 1){

      // LUT name
      nchar = snprintf(tablename, NPOW_10, "%s/WVP_%04d-%02d-%02d.txt", args.dwvp, 
        args.date_start.year, args.date_start.month, args.date_start.day);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

      // sum LUT values
      if (fileexist(tablename)){

        // read the table
        read_wvp_lut(tablename, nc, COO, WVP);

        for (c=0; c<nc; c++){
          if (WVP[c] == 9999) continue;
          
          AVG[2][args.date_start.month-1][c]++;

          if (AVG[2][args.date_start.month-1][c] == 1){
            AVG[0][args.date_start.month-1][c] = WVP[c];
          } else {
            var_recurrence(WVP[c], &AVG[0][args.date_start.month-1][c], &AVG[1][args.date_start.month-1][c], AVG[2][args.date_start.month-1][c]);
          }
        }

      }

      // if iterated date is today, stop.
      if (args.date_start.year  == args.date_end.year && 
          args.date_start.month == args.date_end.month && 
          args.date_start.day   == args.date_end.day) break;

      // go to next day
      date_plus(&args.date_start);

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
    write_avg_table(args.dwvp, nc, COO, AVG);

    // free memory
    free_3D((void***)AVG, 3, 12);

  }


  // free memory
  free((void*)WVP);
  free_2D((void**)COO, 2);
  free_2D((void**)SEN, nc);

  return SUCCESS;
}

