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
This file contains functions for string handling
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "string-cl.h"


/** Copy string
+++ This function copies a source string into a destination buffer. 
+++ strncpy copies as many characters from src to dst as there is space
+++ in src. The string is padded with zeros. This way, buffer overflow 
+++ won't happen. If dst is longer than src, dst will be truncated. The 
+++ truncation will be detected and the program will interrupt.
--- dst:    destination buffer
--- size:   size of destination buffer
--- src:    source string
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void copy_string(char *dst, size_t size, const char *src){

  strncpy(dst, src, size);
  if (dst[size-1] != '\0'){
    printf("cannot copy, string too long:\n%s\n", src);
    exit(1);
  }

  return;
}


int char_to_int(const char *src, int *val){
long int temp_val;
char *temp;
errno = 0;


  temp_val = strtol(src, &temp, 0);

  if (temp == src || *temp != '\0' || errno == ERANGE){
    return FAILURE;}

  if (temp_val < INT_MIN ||
      temp_val > INT_MAX){
    return FAILURE;}

  *val = (int)temp_val;
  return SUCCESS;
}


int char_to_float(const char *src, float *val){
float temp_val;
char *temp;
errno = 0;


  temp_val = strtof(src, &temp);

  if (temp == src || *temp != '\0' || errno == ERANGE){
    return FAILURE;}

  *val = (float)temp_val;
  return SUCCESS;
}

