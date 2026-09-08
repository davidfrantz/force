/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2026 David Frantz

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


#include "runtime_data-cl.h"


/** Load runtime data into a Jansson json_t struct.
+++ The returned struct must be freed with json_decref after use.
--- runtime_file: relative path to the JSON file containing runtime data
--- runtime_data: Pointer to json_t* to receive the loaded JSON object
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_runtime_data(char *runtime_file, json_t **runtime_data){

  char d_exe[NPOW_10];
  get_install_directory(d_exe, NPOW_10);

  char path_json[NPOW_10];
  concat_string_2(path_json, NPOW_10, d_exe, runtime_file, "/");

  if (load_json(runtime_data, path_json) != SUCCESS){
    fprintf(stderr, "Error loading JSON file %s\n", path_json);
    return FAILURE;
  }

  return SUCCESS;
}

