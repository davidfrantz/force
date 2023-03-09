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


/** Concatenate strings (2)
+++ This function concatenates several strings into a destination buffer.
+++ It is checked that the buffer doesn't overflow; error if so.
--- dst:    destination buffer
--- size:   size of destination buffer
--- src1:   source string 1
--- src2:   source string 2
--- delim:  deliminator (string)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void concat_string_2(char *dst, size_t size, const char *src1, const char *src2, const char *delim){
int nchar;

  nchar = snprintf(dst, NPOW_10, "%s%s%s", src1, delim, src2);
  if (nchar < 0 || nchar >= size){ 
    printf("Buffer Overflow in assembling string\n"); 
    exit(1);
  }

  return;
}


/** Concatenate strings (3)
+++ This function concatenates several strings into a destination buffer.
+++ It is checked that the buffer doesn't overflow; error if so.
--- dst:    destination buffer
--- size:   size of destination buffer
--- src1:    source string 1
--- src2:    source string 2
--- src3:    source string 3
--- delim:  deliminator (string)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void concat_string_3(char *dst, size_t size, const char *src1, const char *src2, const char *src3, const char *delim){
int nchar;

  nchar = snprintf(dst, NPOW_10, "%s%s%s%s%s", src1, delim, src2, delim, src3);
  if (nchar < 0 || nchar >= size){ 
    printf("Buffer Overflow in assembling string\n"); 
    exit(1);
  }

  return;
}


/** Search/Replace string
+++ This function searches for a pattern and replaces its first 
+++ occurence with the replacement string. If no match is found, 
+++ nothing happens. Otherwise, the source string will be modified.
+++ It is checked that the buffer doesn't overflow; error if so.
--- source:     source string
--- search:     search pattern
--- replace:    replacement string
--- buffer_len: length of buffer holding source
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void replace_string(char *source, const char *search, const char *replace, size_t buffer_len){
char *buffer = NULL;
char *match = NULL;
char *suffix = NULL;
char *insert_point = NULL;
size_t search_len;
size_t source_len;
size_t replace_len;
size_t prefix_len;
size_t suffix_len;


  source_len  = strlen(source);
  search_len  = strlen(search);
  replace_len = strlen(replace);

  //printf("length buffer:  %lu\n", buffer_len);
  //printf("length source:  %lu\n", source_len);
  //printf("length search:  %lu\n", search_len);
  //printf("length replace: %lu\n", replace_len);

  if (source_len - search_len + replace_len >= buffer_len){
    printf("Error: Insufficient buffer size for replacing.\n");
    exit(1);
  }

  alloc((void**)&buffer, buffer_len, sizeof(char**));
  insert_point = buffer;

  // search for pattern
  match = strstr(source, search);

  // no match, stop
  if (match == NULL) return;

  // length before match
  prefix_len = match - source;

  // length after match
  suffix = source + prefix_len + search_len;
  suffix_len = source_len - prefix_len - search_len;

  //printf("length prefix: %lu\n", prefix_len);
  //printf("length suffix: %lu\n", suffix_len);

  // copy prefix
  memcpy(insert_point, source, prefix_len);
  insert_point += prefix_len;

  // copy replacement
  memcpy(insert_point, replace, replace_len);
  insert_point += replace_len;

  // copy suffix
  memcpy(insert_point, suffix, suffix_len);
  insert_point += suffix_len;

  // terminate
  *insert_point = '\0';

  // copy back and free memory
  strcpy(source, buffer);
  free((void*)buffer);

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

