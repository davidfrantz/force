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
    printf("cannot copy, string too long (%ld -> %ld):\n%s\n", strlen(src), size, src);
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

  nchar = snprintf(dst, size, "%s%s%s", src1, delim, src2);
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

  nchar = snprintf(dst, size, "%s%s%s%s%s", src1, delim, src2, delim, src3);
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



/** Overwrites part of a string with substring
+++ This function overwrites part of a string with a given replacement string.
+++ It is checked that the buffer doesn't overflow; error if so.
--- source:      source string
--- offset:      offset where to insert the replacement string
--- replace:     replacement string
--- replace_len: length of the replacement string
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void overwrite_string_part(char *source, size_t offset, const char *replace, size_t replace_len){


  if (replace_len > strlen(replace)){
    printf("Error: Replacement string length (%lu) exceeds length of replacement string (%lu).\n", replace_len, strlen(replace));
    exit(1);
  }

  if (offset + replace_len > strlen(source)){
    printf("Error: Offset (%lu) plus replacement length (%lu) exceeds source length (%lu).\n", offset, replace_len, strlen(source));
    exit(1);
  }
  
  memcpy(source + offset, replace, replace_len);

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


/** Check if two strings are equal
+++ This function checks if two strings are equal.
+++ It returns true if they are equal, false otherwise.
--- str1:    first string
--- str2:    second string
+++ Return:  true (1) if equal, false (0) otherwise
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool strings_equal(const char *str1, const char *str2) {

  if (strcmp(str1, str2) == 0) {
    return true;
  }

  return false;
}


/** Check if a vector of strings contains a given string
+++ This function checks if a vector of strings contains a given string.
+++ It returns true if the string is found, false otherwise.
--- vector:  array of strings
--- size:    size of the array
--- target:  string to search for
+++ Return:  true (1) if found, false (0) otherwise
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool vector_contains(const char **vector, size_t size, const char *target) {
  
  for (size_t i = 0; i < size; i++) {
    if (strings_equal(vector[i], target)) {
      return true;
    }
  }

  return false;
}


/** Check if a vector of strings contains a given string
+++ This function checks if a vector of strings contains a given string.
+++ It returns the index of the string if found, -1 otherwise.
--- vector:  array of strings
--- size:    size of the array
--- target:  string to search for
+++ Return:  index of the string if found, -1 otherwise
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int vector_contains_pos(const char **vector, size_t size, const char *target) {
  
  for (size_t i = 0; i < size; i++) {
    if (strings_equal(vector[i], target)) {
      return i;
    }
  }

  return -1;
}



// functions for dynamic string and string vector handling

/** Allocate string
+++ This function allocates memory for a string structure.
+++ If the string structure already holds a string, it is freed first
+++ to prevent memory leaks.
--- str:    string structure (modified)
--- length: length of the string to allocate
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_string(string_t *str, size_t length){


  if (str == NULL) {
    printf("Error: Non-NULL pointer passed to alloc_string.\n");
    exit(1);
  }

  free_string(str); // prevent memory leak

  if (length <= 0){
    printf("Error: Cannot allocate string of length <= 0.\n");
    exit(1);
  }

  str->length = length + 1; // +1 for null terminator
  alloc((void**)&str->string, length + 1, sizeof(char));

  return;
}


/** Free string
+++ This function frees the memory allocated for a string structure.
--- str:    string structure (modified, freed)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_string(string_t *str){

  if (str == NULL) return;

  if (str->string != NULL){
    free((void*)str->string);
    str->string = NULL;
  }
  str->length = 0;

  return;
}


/** Fill string
+++ This function fills a string structure with a given source string.
+++ If the string structure already holds a string, it is freed first
+++ to prevent memory leaks. Memory is allocated to hold the new string.
--- str:    string structure (modified)
--- src:    source string
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void fill_string(string_t *str, const char *src){

  if (str == NULL || src == NULL) {
    printf("Error: NULL pointer passed to fill_string.\n");
    exit(1);
  }

  //printf("Filling string with source: %s (length: %zu)\n", src, strlen(src));

  alloc_string(str, strlen(src));
  copy_string(str->string, str->length, src);

  return;
}


/** Allocate string vector
+++ This function allocates memory for a string vector structure.
+++ All strings in the vector will have the same length.
--- str_vec:  string vector structure (modified)
--- number:   number of strings
--- length:   length of each string
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_string_vector(string_vector_t *str_vec, size_t number, size_t length){

  if (str_vec == NULL) {
    printf("Error: NULL pointer passed to alloc_string_vector.\n");
    exit(1);
  }

  if (number <= 0){
    printf("Error: Cannot allocate string vector of number <= 0.\n");
    exit(1);
  }

  if (length <= 0){
    printf("Error: Cannot allocate string vector of length <= 0.\n");
    exit(1);
  }

  str_vec->number = number;
  str_vec->length = length + 1; // +1 for null terminator
  alloc_2D((void***)&str_vec->string, number, length + 1, sizeof(char));

  return;
}


/** Re-allocate string vector
+++ This function re-allocates memory for a string vector structure.
+++ All strings in the vector will have the same length.
--- str_vec:    string vector structure (modified)
--- new_number: new number of strings
--- new_length: new length of each string
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void re_alloc_string_vector(string_vector_t *str_vec, size_t new_number, size_t new_length){

  if (str_vec == NULL) {
    printf("Error: NULL pointer passed to realloc_string_vector.\n");
    exit(1);
  }

  if (new_number <= 0){
    printf("Error: Cannot reallocate string vector of number <= 0.\n");
    exit(1);
  }

  if (new_length <= 0){
    printf("Error: Cannot reallocate string vector of length <= 0.\n");
    exit(1);
  }

  re_alloc_2D((void***)&str_vec->string, str_vec->number, str_vec->length, new_number, new_length + 1, sizeof(char));
  str_vec->number = new_number;
  str_vec->length = new_length + 1; // +1 for null terminator

  return;
}


/** Free string vector
+++ This function frees the memory allocated for a string vector structure.
--- str_vec:  string vector structure (modified, freed)
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_string_vector(string_vector_t *str_vec){

    if (str_vec == NULL) {
    printf("Error: NULL pointer passed to free_string_vector.\n");
    exit(1);
  }

  if (str_vec->number <= 0){
    printf("Error: No string items to free.\n");
    exit(1);
  }

  if (str_vec->length <= 0){
    printf("Error: No strings to free.\n");
    exit(1);
  }

  free_2D((void**)str_vec->string, str_vec->number);
  str_vec->string = NULL;
  str_vec->number = 0;
  str_vec->length = 0;

  return;
}


/** Fill string vector
+++ This function fills a string vector structure at a given position
+++ with a given source string. If the position is out of bounds, the
+++ string vector is re-allocated to accommodate the new position. If the
+++ source string is longer than the current length, the string vector
+++ is re-allocated to accommodate the new length.
--- str_vec:  string vector structure (modified)
--- pos:      position to fill
--- new_str:  source string
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void fill_string_vector(string_vector_t *str_vec, size_t pos, const char *new_str){

  if (str_vec == NULL || new_str == NULL) {
    printf("Error: NULL pointer passed to add_string_to_vector.\n");
    exit(1);
  }

  if (pos < 0){
    printf("Error: Position (%ld) out of bounds in add_string_to_vector.\n", pos);
    exit(1);
  }

  if (pos >= str_vec->number || strlen(new_str) >= str_vec->length){
    int new_number = (pos >= str_vec->number) ? pos + 1 : str_vec->number;
    int new_length = (strlen(new_str) >= str_vec->length) ? strlen(new_str) : str_vec->length;
    re_alloc_string_vector(str_vec, new_number, new_length);
  }

  copy_string(str_vec->string[pos], str_vec->length, new_str);

  return;
}
