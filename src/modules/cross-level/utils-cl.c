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
This file contains some utility functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "utils-cl.h"


/** Get software version
+++ This function gets the Software version. If dst is NULL, the version
+++ is simply printed to stdout.
--- dst:    destination buffer
--- size:   size of destination buffer
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_version(char *dst, size_t size){
char dname_exe[NPOW_10];
char fname_version[NPOW_10];
char buffer[NPOW_16] = "\0";
FILE *fp = NULL;


  get_install_directory(dname_exe, NPOW_10);
  concat_string_3(fname_version,  NPOW_10, dname_exe, "force-misc", "force-version.txt", "/");

  if (!(fp = fopen(fname_version, "r"))){
    printf("unable to open version file %s\n", fname_version); exit(FAILURE);}

  if (fgets(buffer, NPOW_16, fp) == NULL){
    printf("unable to read from version file %s\n", fname_version); exit(FAILURE);}

  buffer[strcspn(buffer, "\r\n")] = 0;

  if (dst == NULL){
    puts(buffer);
  } else {
    copy_string(dst, size, buffer);
  }

  fclose(fp);

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


/** Number of decimal places in integer
--- i:      integer
+++ Return: # decimal places
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int num_decimal_places(int i){

  if (i < 0) return num_decimal_places((i == INT_MIN) ? INT_MAX: -i);
  if (i < 10) return 1;

  return 1 + num_decimal_places(i/10);
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


/** Equality test for doubles
+++ This function tests for quasi equality of doubles
--- a:      number 1
--- b:      number 2
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool dequal(double a, double b){
double diff, max, A, B;

  diff = fabs(a-b);
  A = fabs(a);
  B = fabs(b);

  max = (B > A) ? B : A;

  if (diff <= max * DBL_EPSILON) return true;

  return false;
}


/** Print bytes as human-readable string
--- bytes:  bytes
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_humanreadable_bytes(off_t bytes){
double dbytes = (double)bytes;
char unit[9][NPOW_10] = { "B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" };
int i = 0;

  while (dbytes >= 1024 && i < 8){
      dbytes /= 1024;
      i++;
  }

  printf("%.2f %s\n", dbytes, unit[i]);

  return;
}
