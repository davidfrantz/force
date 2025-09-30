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


/** Load index definitions from a JSON file into a Jansson json_t struct.
+++ The returned struct must be freed with json_decref after use.
--- def_indices: Pointer to json_t* to receive the loaded JSON object
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_index_definitions(json_t **def_indices){

  char d_exe[NPOW_10];
  get_install_directory(d_exe, NPOW_10);

  char path_sensor[NPOW_10];
  concat_string_3(path_sensor, NPOW_10, d_exe, "force-misc/runtime-data", "indices.json", "/");

  json_error_t error;
  json_t *def;
  def = json_load_file(path_sensor, 0, &error);
  if (!def){
    fprintf(stderr, "Error: %s\n", error.text);
    return FAILURE;
  }

  *def_indices = def;

  return SUCCESS;
}


/** Extract the band names from a JSON index definition.
+++ Allocates a 2D array of strings for band names.
--- names: Pointer to char** to receive band names (must be freed with free_2D)
--- nbands: Number of bands
--- index_name: Name of the index (e.g. "NDVI"), needs to be present in the JSON file
--- def_index: JSON object with index definition
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_index_bandnames(char ***bandnames, int *nbands, char *index_name, json_t *def_indices){
  
  json_t *def_index = json_object_get(def_indices, index_name);
  
  if (def_index == NULL) {
    fprintf(stderr, "Error: Item %s not found\n", index_name);
    fprintf(stderr, "There is no definition for this index.\n");
    return FAILURE;
  }

  char **names = NULL;
  int n_names = 0;

  if (json_is_array(def_index)) {

    n_names = json_array_size(def_index);

    alloc_2D((void***)&names, n_names, NPOW_10, sizeof(char));


    for (int b=0; b<n_names; b++){
        json_t *def_band_name = json_array_get(def_index, b);
        if (json_is_string(def_band_name)){
          copy_string(names[b], NPOW_10, json_string_value(def_band_name));
          printf("  %02d: %s\n", b+1, names[b]);
        } else {
          fprintf(stderr, "Error: Element %d in %s array is not a string.\n", b+1, index_name);
          return FAILURE;
        }
    }
  } else {
    fprintf(stderr, "Error: Item %s is not an array.\n", index_name);
    return FAILURE;
  }

  *nbands = n_names;
  *bandnames = names;

  return SUCCESS;
}

/*
  for (int b_available=0; b_available<sen->n_bands; b_available++){
    use_bands[b_available] = true;
  }
  index->n_bands[i] = sen->n_bands;
  alloc_2D((void***)&index->band_names[i], sen->n_bands, NPOW_10, sizeof(char));
  memcpy(index->band_names[i], sen->band_names, sen->n_bands * NPOW_10 * sizeof(char));
*/  

/*
  for (int b_available=0; b_available<sen->n_bands; b_available++){
    if (strings_equal(sen->band_names[b_available], index->names[i])){
      use_bands[b_available] = true;
      break;
    }
  }
  index->n_bands[i] = 1;
  alloc_2D((void***)&index->band_names[i], 1, NPOW_10, sizeof(char));
  copy_string(index->band_names[i][0], NPOW_10, index->names[i]);
*/


int get_required_bands(char ***required_band_names, int *n_required, int *index_type, char *index_name, sen_t *sen, json_t *def_indices){
int error = 0;
char **names = NULL;
int n_names = 0;


// special case: band name as index: use that band directly
if (vector_contains((const char **)sen->band_names, sen->n_bands, index_name)){
  
    printf("Use band name as index directly.\n");
    *index_type = _INDEX_TYPE_BAND_;
    
    n_names = 1;
    alloc_2D((void***)&names, n_names, NPOW_10, sizeof(char));
    copy_string(names[0], NPOW_10, index_name);
    
  // special case: SMA index: use all bands
  } else if (strings_equal(index_name, "SMA")){

    printf("SMA index requires all bands.\n");
    *index_type = _INDEX_TYPE_SMA_;
    
    n_names = sen->n_bands;
    alloc_2D((void***)&names, n_names, NPOW_10, sizeof(char));
    for (int b=0; b<n_names; b++) copy_string(names[b], NPOW_10, sen->band_names[b]);

   // common case: index defined by equation, get required bands from JSON definition
  } else {

    printf("Index %s requires specific bands.\n", index_name);
    *index_type = _INDEX_TYPE_EQUATION_;

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
        printf("  Required band %s is available.\n", required_band_names[b_required]);
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

      printf("  Keeping used band %s.\n", sen->band_names[b_available]);
      b_available++;

    } else {

      printf("  Removing unused band %s.\n", sen->band_names[b_available]);

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

/** This function checks that each index can be computed with the given
+++ set of sensors. It also kicks out unused bands to remove I/O
--- index:  index parameters
--- sen:    sensor parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int retrieve_indices(index_t *index, sen_t *sen){


  // load index definitions
  json_t *def_indices = NULL;
  if (load_index_definitions(&def_indices) != SUCCESS){
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
  #endif
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

  return SUCCESS;
}
