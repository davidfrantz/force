/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2025 David Frantz

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


#include "sensor-hl.h"


int load_sensor_runtime_data(json_t **def_sensors);
int get_band_intersection(int *n_intersect, char ***intersect_bands, int n_sensors, int *nbands, char ***band_names);
int get_band_union(int *n_union, char ***union_bands, int n_sensors, int *nbands, char ***band_names);
int get_band_numbers_to_read(sen_t *sen, int *nbands, char ***band_names);
int check_target_sensor(sen_t *sen, json_t *def_all_sensors);


/** Load sensor definitions from the JSON runtime data into a Jansson json_t struct.
+++ The returned struct must be freed with json_decref after use.
--- def_sensors: Pointer to json_t* to receive the loaded JSON object
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_sensor_runtime_data(json_t **def_sensors){

  char d_exe[NPOW_10];
  get_install_directory(d_exe, NPOW_10);

  char path_json[NPOW_10];
  concat_string_2(path_json, NPOW_10, d_exe, _FORCE_SENSOR_FILE_, "/");

  json_t *def;

  if (load_json(&def, path_json) != SUCCESS){
    fprintf(stderr, "Error loading JSON file %s\n", path_json);
    return FAILURE;
  }

  *def_sensors = def;

  return SUCCESS;
}


/** Print all sensor definitions from the runtime data to stdout.
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_sensor_runtime_data(){

  json_t *def_all_sensors = NULL;
  if (load_sensor_runtime_data(&def_all_sensors) != SUCCESS){
    fprintf(stderr, "Error: Could not parse sensor definitions.\n");
    exit(FAILURE);
  }

  int error = 0;

  void *iter = json_object_iter(def_all_sensors);
  while (iter){

    const char *request_name = json_object_iter_key(iter);
    printf("Sensor request name: %s\n", request_name);

    char name[NPOW_10];
    if (get_json_string(name, NPOW_10, "name", json_object_iter_value(iter)) != SUCCESS) {
      fprintf(stderr, "Error: Could not get sensor name for %s.\n", request_name);
      error++;
      iter = json_object_iter_next(def_all_sensors, iter);
      continue;
    }
    printf("  Sensor ID in file names: %s\n", name);

    char description[NPOW_10];
    if (get_json_string(description, NPOW_10, "description", json_object_iter_value(iter)) != SUCCESS) {
      fprintf(stderr, "Error: Could not get sensor description for %s.\n", request_name);
      error++;
      iter = json_object_iter_next(def_all_sensors, iter);
      continue;
    }
    printf("  Description: %s\n", description);

    int n_bands = 0;
    if (get_json_integer(&n_bands, "bands", json_object_iter_value(iter)) != SUCCESS) {
      fprintf(stderr, "Error: Could not get number of bands for %s.\n", request_name);
      error++;
      iter = json_object_iter_next(def_all_sensors, iter);
      continue;
    }
    printf("  Number of bands: %d\n", n_bands);

    int n_band_names = 0;
    char **band_names = NULL;
    if (get_json_string_array(&band_names, &n_band_names, "band_names", json_object_iter_value(iter)) != SUCCESS) {
      fprintf(stderr, "Error: Could not get band names for %s.\n", request_name);
      error++;
      iter = json_object_iter_next(def_all_sensors, iter);
      continue;
    }
    printf("  Number of band names: %d\n", n_band_names);
    for (int b=0; b<n_band_names; b++){
      printf("  %02d: %s\n", b+1, band_names[b]);
    }
    printf("\n");

    free_2D((void**)band_names, n_band_names); 
    band_names = NULL;

    iter = json_object_iter_next(def_all_sensors, iter);

  }
  
  json_decref(def_all_sensors);

  if (error > 0){
    fprintf(stderr, "Encountered %d error(s) while parsing sensor definitions.\n", error);
    exit(FAILURE);
  }

  return;
}


/** Determine overlapping bands among multiple sensors.
--- n_intersect: how many bands are overlapping (returned)
--- intersect_bands: intersecting band names (returned, must be freed with free_2D)
--- n_sensors: Number of sensors
--- nbands: Array of band counts per sensor
--- band_names: 2D array of band names per sensor
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_band_intersection(int *n_intersect, char ***intersect_bands, int n_sensors, int *nbands, char ***band_names){

  int s_first = 0;
  char **buffer = NULL;
  int ctr = 0;
  alloc_2D((void***)&buffer, nbands[s_first], NPOW_10, sizeof(char));

  for (int b_first=0; b_first<nbands[s_first]; b_first++){

    bool ignore = false;

    for (int s_next=1; s_next<n_sensors; s_next++){

      if (!vector_contains((const char **)band_names[s_next], nbands[s_next], band_names[s_first][b_first])){
        ignore = true;
        break;
      }

    }

    if (!ignore){
      copy_string(buffer[ctr], NPOW_10, band_names[s_first][b_first]);
      ctr++;
    }

  }

  if (ctr < 1){
    printf("No overlapping bands found. Check SENSORS.\n");
    return FAILURE;
  }

  re_alloc_2D((void***)&buffer, nbands[s_first], NPOW_10, ctr, NPOW_10, sizeof(char));

  #ifdef FORCE_DEBUG
  printf("Number of overlapping bands: %d\n", ctr);
  #endif

  *n_intersect = ctr;
  *intersect_bands = buffer;

  return SUCCESS;
}


/** Determine all bands among multiple sensors.
--- n_union: how many bands are there overall (returned)
--- union_bands: unioned band names (returned, must be freed with free_2D)
--- n_sensors: Number of sensors
--- nbands: Array of band counts per sensor
--- band_names: 2D array of band names per sensor
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_band_union(int *n_union, char ***union_bands, int n_sensors, int *nbands, char ***band_names){

  char **buffer = NULL;
  int n_all = 0;
  for (int s=0; s<n_sensors; s++) n_all += nbands[s];
  alloc_2D((void***)&buffer, n_all, NPOW_10, sizeof(char));

  int ctr = 0;

  for (int s=0; s<n_sensors; s++){
    for (int b=0; b<nbands[s]; b++){

      if (!vector_contains((const char **)buffer, ctr, band_names[s][b])){
        copy_string(buffer[ctr], NPOW_10, band_names[s][b]);
        ctr++;
      }

    }
  }

  if (ctr < 1){
    printf("No unioned bands found. Check SENSORS.\n");
    return FAILURE;
  }

  re_alloc_2D((void***)&buffer, n_all, NPOW_10, ctr, NPOW_10, sizeof(char));

  #ifdef FORCE_DEBUG
  printf("Number of unioned bands: %d\n", ctr);
  #endif

  *n_union = ctr;
  *union_bands = buffer;

  return SUCCESS;
}


/** Find the band numbers to read for each sensor based on the selected bands.
--- sen: Pointer to par_sen_t struct (sen->band_number will be allocated)
--- nbands: Array of band counts per sensor
--- band_names: 2D array of band names per sensor
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_band_numbers_to_read(sen_t *sen, int *nbands, char ***band_names){


  alloc_2D((void***)&sen->band_number, sen->n,  sen->n_bands, sizeof(int));

  for (int s=0; s<sen->n; s++){
    for (int b=0; b<sen->n_bands; b++){
      sen->band_number[s][b] = vector_contains_pos((const char **)band_names[s], nbands[s], sen->band_names[b]);
      sen->band_number[s][b]++; // from 0-based to 1-based
    }
  }

  return SUCCESS;
}




/** Check whether input and output sensors are compatible.
--- sen: Pointer to par_sen_t struct
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_target_sensor(sen_t *sen, json_t *def_all_sensors){
int n_bands = 0;
int n_band_names = 0;
char **band_names = NULL;


  json_t *def_target = NULL;
  if (get_json_object(&def_target, sen->target, def_all_sensors) != SUCCESS){
    fprintf(stderr, "Error: Could not parse sensor definition for target sensor %s.\n", sen->target);
    return FAILURE;
  }

  // get sensor name
  if (get_json_string(sen->target, NPOW_10, "name", def_target) != SUCCESS){
    fprintf(stderr, "Error: Could not parse target sensor name for %s.\n", sen->target);
    return FAILURE;
  }
  
  if (get_json_integer(&n_bands, "bands", def_target) != SUCCESS){
    fprintf(stderr, "Error: Could not parse number of bands for target sensor %s.\n", sen->target);
    return FAILURE;
  }

  // get band names and number
  if (get_json_string_array(&band_names, &n_band_names, "band_names", def_target) != SUCCESS){
    fprintf(stderr, "Error: Could not parse band names for %s.\n", sen->target);
    return FAILURE;
  }

  if (n_bands != n_band_names){
    fprintf(stderr, "Error: Target sensor %s has inconsistent band definition: %d bands, but %d band names.\n", sen->target, n_bands, n_band_names);
    return FAILURE;
  }

  if (n_band_names != sen->n_bands){
    fprintf(stderr, "Error: Target sensor %s has %d bands, but %d bands were determined from input sensor combination.\n", sen->target, n_band_names, sen->n_bands);
    return FAILURE;
  }

  for (int b=0; b<n_band_names; b++){
    if (!vector_contains((const char **)band_names, n_band_names, sen->band_names[b])){
      fprintf(stderr, "Error: Band %s in target sensor %s is not part of the determined band set from input sensors.\n", sen->band_names[b], sen->target);
      return FAILURE;
    }
  }

  free_2D((void**)band_names, n_band_names); band_names = NULL;

  return SUCCESS;
}



/** Parse all sensor definitions and determine overlapping bands.
+++ Populates par_sen_t with band mapping information.
--- sen: Pointer to par_sen_t struct
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int retrieve_sensor(sen_t *sen){

  int *nbands = NULL;
  alloc((void**)&nbands, sen->n, sizeof(int));

  char ***band_names = NULL;
  alloc((void**)&band_names, sen->n, sizeof(char**));

  // ugly hack to make sure spectral adjustment works when no target sensor is included in SENSORS
  if (sen->spec_adjust){
    re_alloc_2D((void***)&sen->sensor, sen->n, NPOW_10, sen->n+1, NPOW_10, sizeof(char));
    copy_string(sen->sensor[sen->n], NPOW_10, "SEN2A");
    sen->n++;
  }

  int error = 0;

  // get all sensor definitions
  json_t *def_all_sensors = NULL;
  if (load_sensor_runtime_data(&def_all_sensors) != SUCCESS){
    fprintf(stderr, "Error: Could not parse sensor definitions.\n");
    return FAILURE;
  }

  for (int s=0; s<sen->n; s++){

    json_t *def_sensor = NULL;
    if (get_json_object(&def_sensor, sen->sensor[s], def_all_sensors) != SUCCESS){
      fprintf(stderr, "Error: Could not parse sensor definition for sensor %s.\n", sen->sensor[s]);
      return FAILURE;
    }

    // get sensor name
    if (get_json_string(sen->sensor[s], NPOW_10, "name", def_sensor) != SUCCESS){
      fprintf(stderr, "Error: Could not parse sensor name for %s.\n", sen->sensor[s]);
      error++;
      continue;
    }
    
    // get band names and number
    if (get_json_string_array(&band_names[s], &nbands[s], "band_names", def_sensor) != SUCCESS){
      fprintf(stderr, "Error: Could not parse band names for %s.\n", sen->sensor[s]);
      error++;
      continue;
    }
 
  }

  if (error > 0){
    fprintf(stderr, "Error: Could not parse sensor definition(s).\n");
    json_decref(def_all_sensors);
    return FAILURE;
  }
  
  // determine overlapping bands
  if (!sen->spec_adjust){
    if (get_band_intersection(&sen->n_bands, &sen->band_names, sen->n, nbands, band_names) != SUCCESS){
      fprintf(stderr, "Error: Could not determine intersected bands.\n");
      return FAILURE;
    }
  } else {
    if (get_band_union(&sen->n_bands, &sen->band_names, sen->n, nbands, band_names) != SUCCESS){
      fprintf(stderr, "Error: Could not determine unioned bands.\n");
      return FAILURE;
    }
  }
  
  // ugly hack to make sure spectral adjustment works when no target sensor is included in SENSORS
  if (sen->spec_adjust){
    re_alloc_2D((void***)&sen->sensor, sen->n, NPOW_10, sen->n-1, NPOW_10, sizeof(char));
    sen->n--;
  }
  
  // determine bands to read
  if (get_band_numbers_to_read(sen, nbands, band_names) != SUCCESS){
    fprintf(stderr, "Error: Could not determine bands to read.\n");
    return FAILURE;
  }
  
  // compare with target sensor if combination is sensible
  if (check_target_sensor(sen, def_all_sensors) != SUCCESS){
    fprintf(stderr, "Error: Target sensor is not compatible with selected input sensors.\n");
    return FAILURE;
  }
  
  // clean up
  json_decref(def_all_sensors);
  for (int s=0; s<sen->n; s++) free_2D((void**)band_names[s], nbands[s]);
  free((void*)band_names); band_names = NULL;
  free((void*)nbands); nbands = NULL;

  
  #ifdef FORCE_DEBUG
  printf("Waveband mapping:\n");
  for (int s=0; s<sen->n; s++){
    printf("Sensor # %02d: %s with %d retained bands:\n", s, sen->sensor[s], sen->n_bands);
    for (int b=0; b<sen->n_bands; b++){
      printf("  %s (# %02d)", sen->band_names[b], sen->band_number[s][b]);
    }
    printf("\n");
  }
  #endif

  return SUCCESS;
}
