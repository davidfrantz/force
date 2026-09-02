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


#include "index-parse-hl.h"

int load_index_runtime_data(json_t **def_indices);
int get_index_bandnames(char ***bandnames, int *n_band_names, char *index_name, json_t *def_indices);
int get_required_bands(char ***required_band_names, int *n_required, int *index_type, char *index_name, sen_t *sen, json_t *def_indices);
int check_available_bands(char **required_band_names, int n_required, bool *use_band, sen_t *sen);
int remove_unused_bands(bool *use_band, sen_t *sen);


/** Load index definitions from the JSON runtime data into a Jansson json_t struct.
+++ The returned struct must be freed with json_decref after use.
--- def_indices: Pointer to json_t* to receive the loaded JSON object
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_index_runtime_data(json_t **def_indices){

  char d_exe[NPOW_10];
  get_install_directory(d_exe, NPOW_10);

  char path_json[NPOW_10];
  concat_string_2(path_json, NPOW_10, d_exe, _FORCE_INDEX_FILE_, "/");

  json_t *def;

  if (load_json(&def, path_json) != SUCCESS){
    fprintf(stderr, "Error loading JSON file %s\n", path_json);
    return FAILURE;
  }

  *def_indices = def;

  return SUCCESS;
}


/** Print all index definitions from the runtime data to stdout.
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_index_runtime_data(){

  json_t *def_indices = NULL;
  if (load_index_runtime_data(&def_indices) != SUCCESS){
    fprintf(stderr, "Error: Could not parse index definitions.\n");
    exit(FAILURE);
  }

  int error = 0;

  void *iter = json_object_iter(def_indices);
  while (iter){

    const char *index_name = json_object_iter_key(iter);
    printf("Index: %s\n", index_name);

    char **band_names = NULL;
    int n_band_names = 0;
    if (get_json_string_array(&band_names, &n_band_names, "band_names", json_object_iter_value(iter)) != SUCCESS) {
      fprintf(stderr, "Error: Could not get band names for %s.\n", index_name);
      error++;
      iter = json_object_iter_next(def_indices, iter);
      continue;
    }
    printf("  Required bands: %d\n", n_band_names);
    for (int b=0; b<n_band_names; b++){
      printf("  %02d: %s\n", b+1, band_names[b]);
    }
    printf("\n");

    free_2D((void**)band_names, n_band_names); band_names = NULL;

    iter = json_object_iter_next(def_indices, iter);

  }

  json_decref(def_indices);

  if (error > 0){
    fprintf(stderr, "Encountered %d error(s) while parsing index definitions.\n", error);
    exit(FAILURE);
  }

  return;
}


/** Extract the band names from a JSON index definition.
+++ Allocates a 2D array of strings for band names.
--- names: Pointer to char** to receive band names (must be freed with free_2D)
--- n_band_names: Number of bands
--- index_name: Name of the index (e.g. "NDVI"), needs to be present in the JSON file
--- def_index: JSON object with index definition
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_index_bandnames(char ***bandnames, int *n_band_names, char *index_name, json_t *def_all_indices){
  
  json_t *def_index = NULL;
  if (get_json_object(&def_index, index_name, def_all_indices) != SUCCESS){
    fprintf(stderr, "Error: Could not parse index definition for %s.\n", index_name);
    return FAILURE;
  }
  
  char **names = NULL;
  int n_names = 0;
  if (get_json_string_array(&names, &n_names, "band_names", def_index) != SUCCESS) {
    fprintf(stderr, "Error: Could not get band names for %s.\n", index_name);
    return FAILURE;
  }

  #ifdef FORCE_DEBUG
  for (int b=0; b<n_names; b++) printf("  %02d: %s\n", b+1, names[b]);
  #endif

  *n_band_names = n_names;
  *bandnames = names;

  return SUCCESS;
}

int get_required_bands(char ***required_band_names, int *n_required, int *index_type, char *index_name, sen_t *sen, json_t *def_indices){
int error = 0;
char **names = NULL;
int n_names = 0;


// special case: band name as index: use that band directly
if (vector_contains((const char **)sen->band_names, sen->n_bands, index_name)){
  
  *index_type = _INDEX_TYPE_BAND_;
    #ifdef FORCE_DEBUG
    printf("Use band name as index directly.\n");
    #endif
    
    n_names = 1;
    alloc_2D((void***)&names, n_names, NPOW_10, sizeof(char));
    copy_string(names[0], NPOW_10, index_name);
    
  // special case: SMA index: use all bands
  } else if (strings_equal(index_name, "SMA")){

    *index_type = _INDEX_TYPE_SMA_;
    #ifdef FORCE_DEBUG
    printf("SMA index requires all bands.\n");
    #endif
    
    n_names = sen->n_bands;
    alloc_2D((void***)&names, n_names, NPOW_10, sizeof(char));
    for (int b=0; b<n_names; b++) copy_string(names[b], NPOW_10, sen->band_names[b]);

   // common case: index defined by equation, get required bands from JSON definition
  } else {

    *index_type = _INDEX_TYPE_EQUATION_;
    #ifdef FORCE_DEBUG
    printf("Index %s requires specific bands.\n", index_name);
    #endif

    if (get_index_bandnames(&names, &n_names, index_name, def_indices) != SUCCESS){
      fprintf(stderr, "Error: Could not load index definition for %s.\n", index_name);
      fprintf(stderr, "       If INDEX is supposed to be a band, check sensor configuration.\n");
      error++;
    }
    
  }


  *required_band_names = names;
  *n_required = n_names;

  return error;
}

int check_available_bands(char **required_band_names, int n_required, bool *use_band, sen_t *sen){
int error = 0;


  for (int b_required=0; b_required<n_required; b_required++){

    bool found = false;

    for (int b_available=0; b_available<sen->n_bands; b_available++){

      if (strings_equal(sen->band_names[b_available], required_band_names[b_required])){
        #ifdef FORCE_DEBUG
        printf("  Required band %s is available.\n", required_band_names[b_required]);
        #endif
        use_band[b_available] = true;
        found = true;
        break;
      }

    }

    if (!found){
      fprintf(stderr, "Error: Required band %s is not available given the requested sensors and their configuration.\n", required_band_names[b_required]);
      error++;
      continue;
    }

  }

  return error;
}


int remove_unused_bands(bool *use_band, sen_t *sen){

  
  // go through all available bands and check if they are needed
  int n_used = sen->n_bands;
  int b_available = 0;

  while (b_available < n_used){

    if (use_band[b_available]) {

      #ifdef FORCE_DEBUG
      printf("  Keeping used band %s.\n", sen->band_names[b_available]);
      #endif
      b_available++;

    } else {

      #ifdef FORCE_DEBUG
      printf("  Removing unused band %s.\n", sen->band_names[b_available]);
      #endif

      // Shift left
      for (int i=b_available; i<n_used - 1; i++) {
        use_band[i] = use_band[i+1];
        copy_string(sen->band_names[i], NPOW_10, sen->band_names[i+1]);
      }
      for (int s=0; s<sen->n; s++) {
        for (int i=b_available; i<n_used - 1; i++) {
          sen->band_number[s][i] = sen->band_number[s][i+1];
        }
      }
      n_used--; // update local count

      // do not increment b_available, check new value at this index
    }
  }

  re_alloc_2D((void***)&sen->band_number, sen->n,  sen->n_bands, sen->n, n_used, sizeof(int));
  re_alloc_2D((void***)&sen->band_names, sen->n_bands, NPOW_10, n_used, NPOW_10, sizeof(char));
  sen->n_bands = n_used;

  return SUCCESS;
}


/** This function frees the index parameters
--- index:  index parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_indices(index_t *index){

  if (index->type != NULL) free((void*)index->type); 
  index->type = NULL;

  if (index->band_names != NULL){
    for (int i=0; i<index->n; i++){
      if (index->band_names[i] != NULL){
        free_2D((void**)index->band_names[i], index->n_bands[i]);
        index->band_names[i] = NULL;
     }
    }
    free((void*)index->band_names);
  }
  index->band_names = NULL;

  if (index->n_bands != NULL) free((void*)index->n_bands); 
  index->n_bands = NULL;

  return;
}

/** This function checks that each index can be computed with the given
+++ set of sensors. It also kicks out unused bands to remove I/O
--- index:  index parameters
--- sen:    sensor parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int retrieve_indices(index_t *index, sen_t *sen){


  // load index definitions
  json_t *def_indices = NULL;
  if (load_index_runtime_data(&def_indices) != SUCCESS){
    fprintf(stderr, "Error: Could not parse index definitions.\n");
    return FAILURE;
  }

  int error = 0;
  bool *use_bands = NULL;
  alloc((void**)&use_bands, sen->n_bands, sizeof(bool));

  alloc((void**)&index->type, index->n, sizeof(int));
  alloc((void**)&index->n_bands, index->n, sizeof(int));
  alloc((void**)&index->band_names, index->n, sizeof(char**));

  for (int i=0; i<index->n; i++){

    if (get_required_bands(&index->band_names[i], &index->n_bands[i], &index->type[i], index->names[i], sen, def_indices) > 0){
      fprintf(stderr, "Error: Could not determine required bands for index %s.\n", index->names[i]);
      error++;
      continue;
    }

    if (check_available_bands(index->band_names[i], index->n_bands[i], use_bands, sen) > 0){
      fprintf(stderr, "Error: Not all required bands for index %s are available with the selected sensors.\n", index->names[i]);
      error++;
      continue;
    }

  }

  if (error > 0){
    fprintf(stderr, "Error: Failed to parse index definitions for %d indices.\n", error);
    exit(FAILURE);
  }

  // remove unused bands if not all bands are needed
  if (!sen->spec_adjust){
    if (remove_unused_bands(use_bands, sen) != SUCCESS){
      fprintf(stderr, "Error: Could not remove unused bands.\n");
      return FAILURE;
    }
  }


  json_decref(def_indices);
  free((void**)use_bands); use_bands = NULL;


  #ifdef FORCE_DEBUG
  printf("Waveband mapping after index parsing:\nIndices: ");
  for (int i=0; i<index->n; i++) printf(" %s", index->names[i]);
  printf("\n");
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
