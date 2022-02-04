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
This file contains some utility functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "utils-cl.h"


void print_process_info(){
date_t date;
char **sys_meta = NULL;
int n_sys_meta = 0;
int i = 0;


  printf("Process Start (GMT):\n");
  current_date_gmt(&date);
  print_date(&date);

  printf("\nSystem Info:\n");
  sys_meta = system_info(&n_sys_meta);
  for (i=0; i<n_sys_meta; i+=2) printf("%s: %s\n", sys_meta[i], sys_meta[i+1]);
  if (sys_meta  != NULL){ free_2DC((void**)sys_meta); sys_meta  = NULL;}

  return;
}


/** Print integer vector to stdout
--- v:      vector
--- name:   string that indicates what is printed (printed to stdout) 
--- n:      number of elements
--- big:    number of digits
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_ivector(int *v, const char *name, int n, int big){
int i;

  printf("%s:", name);
  for (i=0; i<n; i++) printf(" %0*d", big, v[i]);
  printf("\n");

  return;
}


/** Print float vector to stdout
--- v:      vector
--- name:   string that indicates what is printed (printed to stdout) 
--- n:      number of elements
--- big:    number of digits before decimal point
--- small:  number of digits after decimal point
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_fvector(float *v, const char *name, int n, int big, int small){
int i;

  printf("%s:", name);
  for (i=0; i<n; i++) printf(" %0*.*f", big+small+1, small, v[i]);
  printf("\n");

  return;
}


/** Print double vector to stdout
--- v:      vector
--- name:   string that indicates what is printed (printed to stdout) 
--- n:      number of elements
--- big:    number of digits before decimal point
--- small:  number of digits after decimal point
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_dvector(double *v, const char *name, int n, int big, int small){
int i;

  printf("%s:", name);
  for (i=0; i<n; i++) printf(" %0*.*f", big+small+1, small, v[i]);
  printf("\n");

  return;
}


/** Measure time
+++ This function measures the processing time and prints to stdout
--- start:  start time
+++ Return: time in seconds
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double proctime(time_t start){
time_t now;
double secs;

  time(&now); secs = difftime(now, start);

  return secs;
}


/** Measure time and print
+++ This function measures the processing time and prints to stdout
--- string: string that indicates what was measured (printed to stdout) 
--- start:  start time
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void proctime_print(const char *string, time_t start){
time_t now;
double secs;
int mins;

  time(&now); secs = difftime(now, start);
  if (secs >= 60){
    mins = floor(secs/60); secs = secs-mins*60;
  } else mins = 0;
  printf("%s: %02d mins %02.0f secs\n", string, mins, secs);

  return;
}


/** Measure time and write to file
+++ This function measures the processing time and prints to stdout
--- string: string that indicates what was measured (printed to stdout) 
--- start:  start time
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void fproctime_print(FILE *fp, const char *string, time_t start){
time_t now;
double secs;
int mins;

  time(&now); secs = difftime(now, start);
  if (secs >= 60){
    mins = floor(secs/60); secs = secs-mins*60;
  } else mins = 0;
  fprintf(fp, "%s: %02d mins %02.0f secs\n", string, mins, secs);

  return;
}


/** Equality test for floats
+++ This function tests for quasi equality of floats
--- a:      number 1
--- b:      number 2
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool fequal(float a, float b){
float diff, max, A, B;

  diff = fabs(a-b);
  A = fabs(a);
  B = fabs(b);

  max = (B > A) ? B : A;

  if (diff <= max * FLT_EPSILON) return true;

  return false;
}

