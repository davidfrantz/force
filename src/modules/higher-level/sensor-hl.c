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
int get_sensor_definition(json_t **def_sensor, char *sensor_name){

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
      copy_string(name, NPOW_10, json_string_value(def_name));
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
+++ Sets ignore_bands for bands not present in all sensors.
--- n_overlapping_bands: Pointer to int for number of overlapping bands
--- ignore_bands: Pointer to bool* array for ignored bands (must be freed)
--- n_sensors: Number of sensors
--- nbands: Array of band counts per sensor
--- band_names: 2D array of band names per sensor
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_overlapping_bands(int *n_overlapping_bands, bool **ignore_bands, int n_sensors, int *nbands, char ***band_names){

  int s_first = 0;
  bool *ignore = NULL;
  int n_overlap = nbands[s_first];
  alloc((void**)&ignore, nbands[s_first], sizeof(bool));

  for (int b_first=0; b_first<nbands[s_first]; b_first++){

    for (int s_next=1; s_next<n_sensors; s_next++){

      bool found = false;
      for (int b_next=0; b_next<nbands[s_next]; b_next++){
        if (strings_equal(band_names[s_first][b_first], band_names[s_next][b_next])){
          found = true;
          break;
        }
      }

      if (!found){
        ignore[b_first] = true;
        n_overlap--;
        break;
      }

    }

  }

  if (n_overlap < 1){
    printf("No overlapping bands found. Check SENSORS.\n");
    return FAILURE;
  }

  printf("Number of overlapping bands: %d\n", n_overlap);

  *n_overlapping_bands = n_overlap;
  *ignore_bands = ignore;

  return SUCCESS;
}


/** Map overlapping bands to their indices for each sensor.
+++ Populates sen->band and sen->domain arrays.
--- sen: Pointer to par_sen_t struct (sen->band and sen->domain will be allocated)
--- ignore_bands: Array of bools indicating ignored bands
--- nbands: Array of band counts per sensor
--- band_names: 2D array of band names per sensor
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_bands_to_read(par_sen_t *sen, bool *ignore_bands, int *nbands, char ***band_names){


  alloc_2D((void***)&sen->band,   sen->n,  sen->nb, sizeof(int));
  alloc_2D((void***)&sen->domain, sen->nb, NPOW_10, sizeof(char));

  for (int b_first=0, s_first=0, b_target=0; b_first<nbands[s_first]; b_first++){
    
    if (!ignore_bands[b_first]){
      
      if (b_target >= sen->nb){
        fprintf(stderr, "Error: Target band is out of bounds. This should not have happened.\n");
        return FAILURE;
      }

      copy_string(sen->domain[b_target], NPOW_10, band_names[s_first][b_first]);

      for (int s_next=0; s_next<sen->n; s_next++){
        for (int b_next=0; b_next<nbands[s_next]; b_next++){

          if (strings_equal(sen->domain[b_target], band_names[s_next][b_next])){

            sen->band[s_next][b_target] = b_next + 1;
            break;

          }

        }
      }

      b_target++;

    }

  }

  return SUCCESS;
}


/** Parse all sensor definitions and determine overlapping bands.
+++ Populates par_sen_t with band mapping information.
--- sen: Pointer to par_sen_t struct
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_sensor(par_sen_t *sen){

printf("\nDEVELOP ALERT: band synthesizing logic for spectral adjustment needs to be re-implemented!!! Currently non-functional, might even crash!\n\n");
  int *nbands = NULL;
  alloc((void**)&nbands, sen->n, sizeof(int));

  char ***band_names = NULL;
  alloc((void**)&band_names, sen->n, sizeof(char**));

  int error = 0;

  for (int s=0; s<sen->n; s++){

    // get full sensor definition
    json_t *def_sensor = NULL;
    if (get_sensor_definition(&def_sensor, sen->sensor[s]) != SUCCESS){
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
  bool *ignore_bands = NULL;
  if (get_overlapping_bands(&sen->nb, &ignore_bands, sen->n, nbands, band_names) != SUCCESS){
    fprintf(stderr, "Error: Could not determine overlapping bands.\n");
    return FAILURE;
  }

  // determine bands to read
  if (get_bands_to_read(sen, ignore_bands, nbands, band_names) != SUCCESS){
    fprintf(stderr, "Error: Could not determine bands to read.\n");
    return FAILURE;
  }

  // clean up
  for (int s=0; s<sen->n; s++) free_2D((void**)band_names[s], nbands[s]);
  free((void*)band_names); band_names = NULL;
  free((void*)nbands); nbands = NULL;
  free((void*)ignore_bands); ignore_bands = NULL;

  
  #ifdef FORCE_DEBUG
  #endif
  printf("Waveband mapping:\n");
  for (int s=0; s<sen->n; s++){
    printf("Sensor # %02d: %s with %d retained bands:\n", s, sen->sensor[s], sen->nb);
    for (int b=0; b<sen->nb; b++){
      printf("  %s (# %02d)", sen->domain[b], sen->band[s][b]);
    }
    printf("\n");
  }

  return SUCCESS;
}
