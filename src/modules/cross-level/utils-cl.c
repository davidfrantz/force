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
  concat_string_2(fname_version,  NPOW_10, dname_exe, _FORCE_VERSION_FILE_, "/");

  if (!(fp = fopen(fname_version, "r"))){
    EXIT_ERROR("unable to open version file %s", fname_version);
  }

  if (fgets(buffer, NPOW_16, fp) == NULL){
    EXIT_ERROR("unable to read from version file %s", fname_version);
  }

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


/** Equality test for floats against 0
+++ This function tests for quasi equality of a float against 0. fequal is
+++ unsuitable for this as its relative tolerance collapses to 0 when one
+++ of the numbers is 0. Pass tol = NULL to use the default tolerance
+++ (FLT_EPSILON)
--- a:      number
--- tol:    tolerance (NULL for default)
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool fequal0(float a, float *tol){
float t = (tol != NULL) ? *tol : FLT_EPSILON;

  return (fabsf(a) <= t);
}


/** Equality test for doubles against 0
+++ This function tests for quasi equality of a double against 0. dequal is
+++ unsuitable for this as its relative tolerance collapses to 0 when one
+++ of the numbers is 0. Pass tol = NULL to use the default tolerance
+++ (DBL_EPSILON)
--- a:      number
--- tol:    tolerance (NULL for default)
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool dequal0(double a, double *tol){
double t = (tol != NULL) ? *tol : DBL_EPSILON;

  return (fabs(a) <= t);
}


/** Divisibility test for floats
+++ This function tests whether a is (quasi) evenly divisible by b
--- a:      dividend
--- b:      divisor
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool fdivisible(float a, float b){
float remainder, tol;

  if (fequal0(b, NULL)) return false;

  tol = fabsf(b) * FLT_EPSILON;
  remainder = fmodf(fabsf(a), fabsf(b));

  return (remainder <= tol || fabsf(b) - remainder <= tol);
}


/** Divisibility test for doubles
+++ This function tests whether a is (quasi) evenly divisible by b
--- a:      dividend
--- b:      divisor
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool ddivisible(double a, double b){
double remainder, tol;

  if (dequal0(b, NULL)) return false;

  tol = fabs(b) * DBL_EPSILON;
  remainder = fmod(fabs(a), fabs(b));

  return (remainder <= tol || fabs(b) - remainder <= tol);
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


/** Calculate the weighted average of a sequence
--- values:   value sequence
--- weights:  weight sequence
+++ average:  pointer to the calculated average
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int weighted_average_of_seq(seq_t *values, seq_t *weights, float *average){

  if (values == NULL || weights == NULL || average == NULL ||
      values->n <= 0 || weights->n <= 0){
    RETURN_ERROR("Invalid input.");
  }

  // use tolerant comparisons, exact bounds may be off by float rounding
  if ((weights->start < values->start && !fequal(weights->start, values->start)) ||
      (weights->end   > values->end   && !fequal(weights->end,   values->end))){
    RETURN_ERROR("Weight sequence (%.2f-%.2f) is not within value sequence range (%.2f-%.2f).\n", 
      weights->start, weights->end, values->start, values->end);
  }

  // if we need support for different step sizes, interpolation will be needed
  if (!fequal(weights->step, values->step)){
    RETURN_ERROR("Weight sequence step (%.2f) is not equal to value sequence step (%.2f).\n", 
      weights->step, values->step);
  }

  if (weights->values == NULL || values->values == NULL){
    RETURN_ERROR("Weight or value sequence is not initialized.");
  }

  // if we need support for sequences that start off the step-grid, interpolation will be needed
  if (!fdivisible(weights->start - values->start, values->step)){
    RETURN_ERROR("Weight sequence does not start on the step-grid of the value sequence.");
  }

  int offset = (int)lround((weights->start - values->start) / values->step);

  if (offset < 0 || offset + weights->n > values->n){
    RETURN_ERROR("Weight sequence is out of bounds of the value sequence.");
  }

  float sum_weighted_values = 0.0;
  float sum_weights = 0.0;


  for (int i=0; i<weights->n; i++){

    if (weights->values[i] < 0.0){
      RETURN_ERROR("Weight sequence contains negative values.");
    }

    sum_weighted_values += weights->values[i] * values->values[i+offset];
    sum_weights += weights->values[i];

  }

  if (fequal0(sum_weights, NULL)){
    RETURN_ERROR("Weight sequence sums to zero.");
  }

  *average = sum_weighted_values / sum_weights;

  return SUCCESS;
}


/** Calculate the weighted centroid of a sequence
--- weights:  weight sequence
+++ average:  pointer to the calculated average
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int weighted_centroid_of_seq(seq_t *weights, float *average){

  if (weights == NULL || average == NULL || weights->n <= 0){
    RETURN_ERROR("Invalid input.");
  }


  if (weights->values == NULL){
    RETURN_ERROR("Weight or value sequence is not initialized.");
  }

  float sum_weighted_values = 0.0;
  float sum_weights = 0.0;


  for (int i=0; i<weights->n; i++){

    if (weights->values[i] < 0.0){
      RETURN_ERROR("Weight sequence contains negative values.");
    }

    sum_weighted_values += weights->values[i] * (weights->start + i * weights->step);
    sum_weights += weights->values[i];

  }

  if (fequal0(sum_weights, NULL)){
    RETURN_ERROR("Weight sequence sums to zero.");
  }

  *average = sum_weighted_values / sum_weights;

  return SUCCESS;
}
