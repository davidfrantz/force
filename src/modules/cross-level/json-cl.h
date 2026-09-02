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
JSON parsing functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef JSON_CL_H
#define JSON_CL_H

#include <stdio.h>   // core input and output functions
#include <stdbool.h>  // boolean data type
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/string-cl.h"

#include <jansson.h> // JSON library


#ifdef __cplusplus
extern "C" {
#endif

int load_json(json_t **json, char *path_json);
int get_json_object(json_t **object, char *key, json_t *json);
int get_json_string(char *string, size_t size, char *key, json_t *json);
int get_json_integer(int *integer, char *key, json_t *json);
int get_json_float(float *floating, char *key, json_t *json);
int get_json_boolean(bool *boolean, char *key, json_t *json);
int get_json_string_array(char ***strings, int *n_strings, char *key, json_t *json);
int get_json_integer_array(int **integers, int *n_integers, char *key, json_t *json);
int get_json_float_array(float **floats, int *n_floats, char *key, json_t *json);
int get_json_boolean_array(bool **booleans, int *n_booleans, char *key, json_t *json);

#ifdef __cplusplus
}
#endif

#endif

