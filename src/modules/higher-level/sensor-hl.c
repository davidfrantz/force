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


/** Load a sensor definition from a JSON file into a Jansson json_t struct.
+++ The returned struct must be freed with json_decref after use.
--- def_sensor: Pointer to json_t* to receive the loaded JSON object
--- sensor_name: Name of the sensor (e.g. "SEN2A")
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_sensor_definition(json_t **def_sensor, char *sensor_name){

  char d_exe[NPOW_10];
  get_install_directory(d_exe, NPOW_10);

  char file_sensor[NPOW_10];
  concat_string_2(file_sensor, NPOW_10, sensor_name, ".json", "");

  char path_sensor[NPOW_10];
  concat_string_3(path_sensor, NPOW_10, d_exe, "force-misc/runtime-data/sensors", file_sensor, "/");

  json_error_t error;
  json_t *def;
  def = json_load_file(path_sensor, 0, &error);
  if (!def){
    fprintf(stderr, "Error: %s\n", error.text);
    return FAILURE;
  }

  *def_sensor = def;

  return SUCCESS;
}


/** Extract the sensor name from a JSON sensor definition.
--- name: Buffer to store the sensor name
--- size: Size of the buffer
--- def_sensor: JSON object with sensor definition
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_sensor_name(char *name, size_t size, json_t *def_sensor){

  json_t *def_name = json_object_get(def_sensor, "name");
  if (def_name == NULL) {
      fprintf(stderr, "Error: Item `Name` not found.\n");
      return FAILURE;
  }
  if (json_is_string(def_name)) {
      copy_string(name, size, json_string_value(def_name));
      printf("Sensor: %s\n", name);
  } else {
      fprintf(stderr, "Error: Item `Name` is not a string.\n");
      return FAILURE;
  }

  return SUCCESS;
}


/** Extract the number of bands from a JSON sensor definition.
--- nbands: Pointer to int to receive the number of bands
--- def_sensor: JSON object with sensor definition
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_sensor_band_number(int *nbands, json_t *def_sensor){

  json_t *def_bands = json_object_get(def_sensor, "bands");
  if (def_bands == NULL) {
      fprintf(stderr, "Error: Item `Bands` not found.\n");
      return FAILURE;
  }
  if (json_is_integer(def_bands)) {
      *nbands = (int)json_integer_value(def_bands);
      if (*nbands < 1){
        fprintf(stderr, "Error: Item `Bands` is less than 1.\n");
        return FAILURE;
      }
  } else {
      fprintf(stderr, "Error: Item `Bands` is not an integer.\n");
      return FAILURE;
  }

  return SUCCESS;
}


/** Extract the band names from a JSON sensor definition.
+++ Allocates a 2D array of strings for band names.
--- names: Pointer to char** to receive band names (must be freed with free_2D)
--- nbands: Number of bands
--- def_sensor: JSON object with sensor definition
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_sensor_bandnames(char ***names, int nbands, json_t *def_sensor){
char **band_names = NULL;


  alloc_2D((void***)&band_names, nbands, NPOW_10, sizeof(char));
  
  json_t *def_band_names = json_object_get(def_sensor, "band_names");

  if (def_band_names == NULL) {
      fprintf(stderr, "Error: Item `band_names` not found\n");
      return FAILURE;
  }
  if (json_is_array(def_band_names)) {
    if (json_array_size(def_band_names) != nbands) {
        fprintf(stderr, "Error: Size of `band_names` array  does not match `Bands` value.\n");
        return FAILURE;
    }
    for (int b=0; b<nbands; b++){
        json_t *def_band_name = json_array_get(def_band_names, b);
        if (json_is_string(def_band_name)){
          copy_string(band_names[b], NPOW_10, json_string_value(def_band_name));
          printf("  %02d: %s\n", b+1, band_names[b]);
        } else {
          fprintf(stderr, "Error: Element %d in `band_names` array is not a string.\n", b+1);
          return FAILURE;
        }
    }
  } else {
    fprintf(stderr, "Error: Item `band_names` is not an array.\n");
    return FAILURE;
  }

  *names = band_names;

  return SUCCESS;
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

  printf("Number of overlapping bands: %d\n", ctr);

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

  printf("Number of unioned bands: %d\n", ctr);

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
int check_target_sensor(sen_t *sen){
int nbands = 0;
char **band_names = NULL;


  // get full sensor definition
  json_t *def_sensor = NULL;
  if (load_sensor_definition(&def_sensor, sen->target) != SUCCESS){
    fprintf(stderr, "Error: Could not parse target sensor definition for %s.\n", sen->target);
    return FAILURE;
  }

  // get sensor name
  if (get_sensor_name(sen->target, NPOW_10, def_sensor) != SUCCESS){
    fprintf(stderr, "Error: Could not parse target sensor name for %s.\n", sen->target);
    return FAILURE;
  }
  
  // get number of bands
  if (get_sensor_band_number(&nbands, def_sensor) != SUCCESS){
    fprintf(stderr, "Error: Could not parse number of bands for %s.\n", sen->target);
    return FAILURE;
  }

  // get band names
  if (get_sensor_bandnames(&band_names, nbands, def_sensor) != SUCCESS){
    fprintf(stderr, "Error: Could not parse band names for %s.\n", sen->target);
    return FAILURE;
  }

  // clean up
  json_decref(def_sensor);


  if (nbands != sen->n_bands){
    fprintf(stderr, "Error: Target sensor %s has %d bands, but %d bands were determined from input sensor combination.\n", sen->target, nbands, sen->n_bands);
    return FAILURE;
  }

  for (int b=0; b<nbands; b++){
    if (!vector_contains((const char **)band_names, nbands, sen->band_names[b])){
      fprintf(stderr, "Error: Band %s in target sensor %s is not part of the determined band set from input sensors.\n", sen->band_names[b], sen->target);
      return FAILURE;
    }
  }

  free_2D((void**)band_names, nbands); band_names = NULL;

  return SUCCESS;
}




/** Parse all sensor definitions and determine overlapping bands.
+++ Populates par_sen_t with band mapping information.
--- sen: Pointer to par_sen_t struct
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int retrieve_sensor(sen_t *sen){

printf("\nDEVELOP ALERT: band synthesizing logic for spectral adjustment needs to be re-implemented!!! Currently non-functional, might even crash!\n\n");
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

  for (int s=0; s<sen->n; s++){

    // get full sensor definition
    json_t *def_sensor = NULL;
    if (load_sensor_definition(&def_sensor, sen->sensor[s]) != SUCCESS){
      fprintf(stderr, "Error: Could not parse sensor definition for %s.\n", sen->sensor[s]);
      error++;
      continue;
    }

    // get sensor name
    if (get_sensor_name(sen->sensor[s], NPOW_10, def_sensor) != SUCCESS){
      fprintf(stderr, "Error: Could not parse sensor name for %s.\n", sen->sensor[s]);
      error++;
    }
    
    // get number of bands
    if (get_sensor_band_number(&nbands[s], def_sensor) != SUCCESS){
      fprintf(stderr, "Error: Could not parse number of bands for %s.\n", sen->sensor[s]);
      error++;
    }

    // get band names
    if (get_sensor_bandnames(&band_names[s], nbands[s], def_sensor) != SUCCESS){
      fprintf(stderr, "Error: Could not parse band names for %s.\n", sen->sensor[s]);
      error++;
    }

    // clean up
    json_decref(def_sensor);

  }

  if (error > 0){
    fprintf(stderr, "Error: Could not parse sensor definition(s).\n");
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
  if (check_target_sensor(sen) != SUCCESS){
    fprintf(stderr, "Error: Target sensor is not compatible with selected input sensors.\n");
    return FAILURE;
  }

  // clean up
  for (int s=0; s<sen->n; s++) free_2D((void**)band_names[s], nbands[s]);
  free((void*)band_names); band_names = NULL;
  free((void*)nbands); nbands = NULL;

  
  #ifdef FORCE_DEBUG
  #endif
  printf("Waveband mapping:\n");
  for (int s=0; s<sen->n; s++){
    printf("Sensor # %02d: %s with %d retained bands:\n", s, sen->sensor[s], sen->n_bands);
    for (int b=0; b<sen->n_bands; b++){
      printf("  %s (# %02d)", sen->band_names[b], sen->band_number[s][b]);
    }
    printf("\n");
  }

  return SUCCESS;
}
