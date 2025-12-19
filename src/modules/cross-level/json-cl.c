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
This file contains functions for parsing JSON files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "json-cl.h"


/** Load JSON file into a Jansson json_t struct.
+++ The returned struct must be freed with json_decref after use.
--- json:   Pointer to json_t* to receive the loaded JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_json(json_t **json, char *path_json){


  json_error_t error;
  json_t *js;
  js = json_load_file(path_json, 0, &error);
  if (!js){
    fprintf(stderr, "Error: %s\n", error.text);
    return FAILURE;
  }

  *json = js;

  return SUCCESS;
}




/** Extract a JSON item from a parent JSON object, independent of its type.
--- item: Pointer to json_t* to receive the extracted JSON item
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_item(json_t **item, char *key, json_t *json){

  json_t *it = json_object_get(json, key);

  if (it == NULL){
      fprintf(stderr, "Error: Item with key %s not found in JSON.\n", key);
      return FAILURE;
  }

  *item = it;

  return SUCCESS;
}


/** Extract a JSON object from a parent JSON object.
--- object: Pointer to json_t* to receive the extracted JSON object
--- key: Key of the JSON item to extract
--- json: JSON object
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_object(json_t **object, char *key, json_t *json){

  json_t *obj = json_object_get(json, key);

  if (obj == NULL){
      fprintf(stderr, "Error: Item with key %s not found in JSON.\n", key);
      return FAILURE;
  }
  if (json_is_object(obj)){
      *object = obj;
  } else {
      fprintf(stderr, "Error: Item with key %s is not an object, type is: %d.\n", key, json_typeof(obj));
      return FAILURE;
  }

  return SUCCESS;
}


/** Extract a string from a JSON item.
--- string: Buffer to store the extracted string
--- size: Size of the buffer
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_string(char *string, size_t size, char *key, json_t *json){

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_string(item)){
      copy_string(string, size, json_string_value(item));
      #ifdef FORCE_DEBUG
      printf("%s: %s\n", key, string);
      #endif
  } else {
      fprintf(stderr, "Error: Item with key %s is not a string, type is: %d.\n", key, json_typeof(item));
      return FAILURE;
  }

  return SUCCESS;
}


/** Extract an integer from a JSON item.
--- integer: Pointer to int to store the extracted integer
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_integer(int *integer, char *key, json_t *json){

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_integer(item)){
      *integer = (int)json_integer_value(item);
      #ifdef FORCE_DEBUG
      printf("%s: %d\n", key, *integer);
      #endif
  } else {
      fprintf(stderr, "Error: Item with key %s is not an integer, type is: %d.\n", key, json_typeof(item));
      return FAILURE;
  }

  return SUCCESS;
}



/** Extract a float from a JSON item.
--- floating: Pointer to floating to store the extracted float
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_float(float *floating, char *key, json_t *json){

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_real(item)){
      *floating = (float)json_real_value(item);
      #ifdef FORCE_DEBUG
      printf("%s: %f\n", key, *floating);
      #endif
  } else {
      fprintf(stderr, "Error: Item with key %s is not a float, type is: %d.\n", key, json_typeof(item));
      return FAILURE;
  }

  return SUCCESS;
}


/** Extract a boolean from a JSON item.
--- boolean: Pointer to bool to store the extracted boolean
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_boolean(bool *boolean, char *key, json_t *json){

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_boolean(item)){
      *boolean = json_is_true(item);
      #ifdef FORCE_DEBUG
      printf("%s: %s\n", key, (*boolean) ? "true" : "false");
      #endif
  } else {
      fprintf(stderr, "Error: Item with key %s is not a boolean, type is: %d.\n", key, json_typeof(item));
      return FAILURE;
  }

  return SUCCESS;
}


/** Extract an array of strings from a JSON item.
+++ Allocates a 2D array of strings.
--- strings: Pointer to char** to receive the array of strings (must be freed with free_2D)
--- n_strings: Pointer to int to receive the number of strings
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_string_array(char ***strings, int *n_strings, char *key, json_t *json){
char **buffer = NULL;
int n_buffer = 0;

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_array(item)){

    n_buffer = json_array_size(item);

    alloc_2D((void***)&buffer, n_buffer, NPOW_10, sizeof(char));

    for (int i=0; i<n_buffer; i++){
        json_t *value = json_array_get(item, i);
        if (json_is_string(value)){
          copy_string(buffer[i], NPOW_10, json_string_value(value));
          #ifdef FORCE_DEBUG
          printf("%s[%d]: %s\n", key, i, buffer[i]);
          #endif
        } else {
          fprintf(stderr, "Error: Element %d in %s array is not a string, type is: %d.\n", i, key, json_typeof(item));
          return FAILURE;
        }
    }
  } else {
    fprintf(stderr, "Error: Item %s is not an array, type is: %d.\n", key, json_typeof(item));
    return FAILURE;
  }

  *strings = buffer;
  *n_strings = n_buffer;
 
  return SUCCESS;
}


/** Extract an array of integers from a JSON item.
+++ Allocates an array of integers.
--- integers: Pointer to int* to receive the array of integers (must be freed)
--- n_integers: Pointer to int to receive the number of integers
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_integer_array(int **integers, int *n_integers, char *key, json_t *json){
int *buffer = NULL;
int n_buffer = 0;

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_array(item)){

    n_buffer = json_array_size(item);

    alloc((void**)&buffer, n_buffer, sizeof(int));

    for (int i=0; i<n_buffer; i++){
        json_t *value = json_array_get(item, i);
        if (json_is_integer(value)){
          buffer[i] = (int)json_integer_value(value);
          #ifdef FORCE_DEBUG
          printf("%s[%d]: %d\n", key, i, buffer[i]);
          #endif
        } else {
          fprintf(stderr, "Error: Element %d in %s array is not an integer, type is: %d.\n", i, key, json_typeof(item));
          return FAILURE;
        }
    }
  } else {
    fprintf(stderr, "Error: Item %s is not an array, type is: %d.\n", key, json_typeof(item));
    return FAILURE;
  }

  *integers = buffer;
  *n_integers = n_buffer;
 
  return SUCCESS;
}


/** Extract an array of floats from a JSON item.
+++ Allocates an array of floats.
--- floats: Pointer to float* to receive the array of floats (must be freed)
--- n_floats: Pointer to int to receive the number of floats
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_float_array(float **floats, int *n_floats, char *key, json_t *json){
float *buffer = NULL;
int n_buffer = 0;

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_array(item)){

    n_buffer = json_array_size(item);

    alloc((void**)&buffer, n_buffer, sizeof(float));

    for (int i=0; i<n_buffer; i++){
        json_t *value = json_array_get(item, i);
        if (json_is_real(value)){
          buffer[i] = (float)json_real_value(value);
          #ifdef FORCE_DEBUG
          printf("%s[%d]: %f\n", key, i, buffer[i]);
          #endif
        } else {
          fprintf(stderr, "Error: Element %d in %s array is not a float, type is: %d.\n", i, key, json_typeof(item));
          return FAILURE;
        }
    }
  } else {
    fprintf(stderr, "Error: Item %s is not an array, type is: %d.\n", key, json_typeof(item));
    return FAILURE;
  }

  *floats = buffer;
  *n_floats = n_buffer;
 
  return SUCCESS;
}


/** Extract an array of booleans from a JSON item.
+++ Allocates an array of booleans.
--- booleans: Pointer to bool* to receive the array of booleans (must be freed)
--- n_booleans: Pointer to int to receive the number of booleans
--- key: Key of the JSON item to extract
--- json: JSON item
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_json_boolean_array(bool **booleans, int *n_booleans, char *key, json_t *json){
bool *buffer = NULL;
int n_buffer = 0;

  json_t *item = NULL;
  if (get_json_item(&item, key, json) != SUCCESS){
      return FAILURE;
  }

  if (json_is_array(item)){

    n_buffer = json_array_size(item);

    alloc((void**)&buffer, n_buffer, sizeof(bool));

    for (int i=0; i<n_buffer; i++){
        json_t *value = json_array_get(item, i);
        if (json_is_boolean(value)){
          buffer[i] = json_is_true(value);
          #ifdef FORCE_DEBUG
          printf("%s[%d]: %s\n", key, i, (buffer[i]) ? "true" : "false");
          #endif
        } else {
          fprintf(stderr, "Error: Element %d in %s array is not a boolean, type is: %d.\n", i, key, json_typeof(item));
          return FAILURE;
        }
    }
  } else {
    fprintf(stderr, "Error: Item %s is not an array, type is: %d.\n", key, json_typeof(item));
    return FAILURE;
  }

  *booleans = buffer;
  *n_booleans = n_buffer;
 
  return SUCCESS;
}

