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
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void load_runtime_data(char *runtime_file, json_t **runtime_data){

  char d_exe[NPOW_10];
  get_install_directory(d_exe, NPOW_10);

  #ifdef FORCE_DEBUG
  printf("Loading runtime data from: %s/%s\n", d_exe, runtime_file);
  #endif

  char path_json[NPOW_10];
  concat_string_2(path_json, NPOW_10, d_exe, runtime_file, "/");

  load_json(runtime_data, path_json);

  return;
}


/** Free the memory allocated for the runtime data.
--- rtd: Pointer to the runtime data structure
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_runtime_data(rtd_t *rtd){

  if (rtd == NULL) return;

  if (rtd->band_mapping.domains != NULL) free_2D((void**)rtd->band_mapping.domains, rtd->band_mapping.nbands);
  if (rtd->band_mapping.l1_bands != NULL) free_2D((void**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands);
  if (rtd->band_mapping.l2_output != NULL) free((void*)rtd->band_mapping.l2_output);
  memset(&rtd->band_mapping, 0, sizeof(rtd_band_mapping_t));


  memset(&rtd->sensor_mapping, 0, sizeof(rtd_sensor_mapping_t));


  if (rtd->rsr_mapping.rsr != NULL){
    for (int b=0; b<rtd->rsr_mapping.nbands; b++){
      if (rtd->rsr_mapping.rsr[b].values != NULL) free((void*)rtd->rsr_mapping.rsr[b].values);
    }
    free((void*)rtd->rsr_mapping.rsr);
  }
  memset(&rtd->rsr_mapping, 0, sizeof(rtd_rsr_mapping_t));

  memset(rtd, 0, sizeof(rtd_t));

  return;
}

/** Load the band mapping runtime data for a specific spacecraft.
--- spacecraft_name: Name of the spacecraft
--- rtd: Pointer to the runtime data structure
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_runtime_data_band_mapping(char *spacecraft_name, rtd_t *rtd){

  if (rtd == NULL){
    RETURN_ERROR("Runtime data structure is NULL in load_runtime_data_band_mapping.");
  }

  if (rtd->band_mapping.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("Band mapping runtime data is already loaded.");
  }

  json_t *band_mapping = NULL;
  load_runtime_data(_FORCE_L1_BAND_MAP_FILE_, &band_mapping);

  json_t *spacecraft_bandmap;
  if (get_json_item(&spacecraft_bandmap, spacecraft_name, band_mapping) != SUCCESS){
    free_runtime_data(rtd);
    json_decref(band_mapping);
    RETURN_ERROR("Could not find key %s in band mapping runtime data.", spacecraft_name);
  }

  if (!json_is_array(spacecraft_bandmap)){
    free_runtime_data(rtd);
    json_decref(band_mapping);
    RETURN_ERROR("Not a valid JSON array.");
  }

  copy_string(rtd->band_mapping.spacecraft_name, NPOW_10, spacecraft_name);
  rtd->band_mapping.nbands = json_array_size(spacecraft_bandmap);

  if (rtd->band_mapping.nbands <= 0){
    free_runtime_data(rtd);
    json_decref(band_mapping);
    RETURN_ERROR("No bands found for spacecraft %s in band mapping runtime data.", spacecraft_name);
  }

  if (rtd->band_mapping.domains != NULL || rtd->band_mapping.l1_bands != NULL || rtd->band_mapping.l2_output != NULL){
    free_runtime_data(rtd);
    json_decref(band_mapping);
    RETURN_ERROR("Runtime data structure is not empty.");
  }

  alloc_2D((void***)&rtd->band_mapping.domains, rtd->band_mapping.nbands, NPOW_10, sizeof(char));
  alloc_2D((void***)&rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, NPOW_10, sizeof(char));
  alloc((void**)&rtd->band_mapping.l2_output, rtd->band_mapping.nbands, sizeof(bool));

  for (int b=0; b<rtd->band_mapping.nbands; b++){

    json_t *band_bandmap = json_array_get(spacecraft_bandmap, b);
    if (!json_is_object(band_bandmap)){
      free_runtime_data(rtd);
      json_decref(band_mapping);
      RETURN_ERROR("Not a valid JSON object.");
    }

    if (get_json_string(rtd->band_mapping.l1_bands[b], NPOW_10, "band", band_bandmap) != SUCCESS){
      free_runtime_data(rtd);
      json_decref(band_mapping);
      RETURN_ERROR("Error: Could not find `band` for item %d in spacecraft mapping runtime data.", b);
    }

    if (get_json_string(rtd->band_mapping.domains[b], NPOW_10, "domain", band_bandmap) != SUCCESS){
      free_runtime_data(rtd);
      json_decref(band_mapping);
      RETURN_ERROR("Could not find `domain` for item %d in spacecraft mapping runtime data.", b);
    }

    if (get_json_bool(&rtd->band_mapping.l2_output[b], "output", band_bandmap) != SUCCESS){
      free_runtime_data(rtd);
      json_decref(band_mapping);
      RETURN_ERROR("Error: Could not find `output` for item %d in spacecraft mapping runtime data.", b);
    }

  }

  json_decref(band_mapping);

  rtd->band_mapping.loaded = true;

  return SUCCESS;
}



/** Load the sensor mapping runtime data for a specific spacecraft.
--- spacecraft_name: Name of the spacecraft
--- rtd: Pointer to the runtime data structure
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_runtime_data_sensor_mapping(char *spacecraft_name, rtd_t *rtd){

  if (rtd == NULL){
    RETURN_ERROR("Runtime data structure is NULL in load_runtime_data_sensor_mapping.");
  }

  if (rtd->sensor_mapping.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("Sensor mapping runtime data is already loaded.");
  }

  json_t *sensor_mapping = NULL;
  load_runtime_data(_FORCE_L1_SENSOR_MAP_FILE_, &sensor_mapping);

  if (get_json_string(rtd->sensor_mapping.l2_sensor, NPOW_10, spacecraft_name, sensor_mapping) != SUCCESS){
    free_runtime_data(rtd);
    json_decref(sensor_mapping);
    RETURN_ERROR("Error: Could not find sensor %s in sensor mapping runtime data.", spacecraft_name);
  }

  json_decref(sensor_mapping);
  
  copy_string(rtd->sensor_mapping.spacecraft_name, NPOW_10, spacecraft_name);
  rtd->sensor_mapping.loaded = true;

  return SUCCESS;
}


int load_runtime_data_seq(json_t *json_parent, char *key, seq_t *seq){


  json_t *seq_object;
  if (get_json_object(&seq_object, key, json_parent) != SUCCESS){
    RETURN_ERROR("Could not find key %s in seq runtime data.", key);
  }

  if (get_json_float(&seq->start, "min", seq_object) != SUCCESS){
    RETURN_ERROR("Could not find `min` in seq runtime data.");
    return FAILURE;
  }

  if (get_json_float(&seq->end, "max", seq_object) != SUCCESS){
    RETURN_ERROR("Could not find `max` in seq runtime data.");
  }

  if (get_json_float(&seq->step, "step", seq_object) != SUCCESS){
    RETURN_ERROR("Could not find `step` in seq runtime data.");
  }
  
  if (get_json_float_array(&seq->values, &seq->n, "values", seq_object) != SUCCESS){
    RETURN_ERROR("Could not find `values` in seq runtime data.");
  }

  if (seq->values == NULL || seq->n < 1 ||
      (fequal0(seq->start, NULL) && fequal0(seq->end, NULL)) || 
       fequal0(seq->step, NULL)){
    RETURN_ERROR("Could not retrieve seq runtime data.");
  }

  if (seq->n != (seq->end - seq->start) / seq->step + 1){
    RETURN_ERROR("Number of seq values (%d) does not match expected number (%d).", 
            seq->n, (int)((seq->end - seq->start) / seq->step + 1));
  }

  return SUCCESS;
}


/** Load the RSR mapping runtime data for a specific spacecraft.
--- spacecraft_name: Name of the spacecraft
--- rtd: Pointer to the runtime data structure
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int load_runtime_data_rsr_mapping(char *spacecraft_name, rtd_t *rtd){

  if (rtd == NULL){
    RETURN_ERROR("Runtime data structure is NULL.");
  }

  if (rtd->rsr_mapping.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("RSR runtime data is already loaded.");
  }

  if (!rtd->band_mapping.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("Band mapping runtime data must be loaded before loading RSR data.");
  }

  json_t *rsr_data = NULL;
  load_runtime_data(_FORCE_L1_RSR_FILE_, &rsr_data);

  json_t *spacecraft_rsr;
  if (get_json_object(&spacecraft_rsr, spacecraft_name, rsr_data) != SUCCESS){
    free_runtime_data(rtd);
    json_decref(rsr_data);
    RETURN_ERROR("Could not find key %s in RSR runtime data.", spacecraft_name);
  }

  rtd->rsr_mapping.nbands = rtd->band_mapping.nbands;
  alloc((void**)&rtd->rsr_mapping.rsr, rtd->rsr_mapping.nbands, sizeof(seq_t));

  for (int b=0; b<rtd->band_mapping.nbands; b++){

    char domain[NPOW_10];
    copy_string(domain, NPOW_10, rtd->band_mapping.domains[b]);

    if (load_runtime_data_seq(spacecraft_rsr, rtd->band_mapping.domains[b], &rtd->rsr_mapping.rsr[b]) != SUCCESS){
      free_runtime_data(rtd);
      json_decref(rsr_data);
      RETURN_ERROR("Could not load RSR data for %s band of spacecraft %s.", domain, spacecraft_name);
    }

  }

  json_decref(rsr_data);

  copy_string(rtd->rsr_mapping.spacecraft_name, NPOW_10, spacecraft_name);
  rtd->rsr_mapping.loaded = true;

  return SUCCESS;
}


int load_runtime_data_E0(rtd_t *rtd){

  if (rtd == NULL) RETURN_ERROR("Runtime data structure is NULL.");

  if (rtd->E0.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("E0 runtime data is already loaded.");
  }

  json_t *e0_data = NULL;
  load_runtime_data(_FORCE_L1_E0_FILE_, &e0_data);

  if (load_runtime_data_seq(e0_data, "E0", &rtd->E0.spectrum) != SUCCESS){
    free_runtime_data(rtd);
    json_decref(e0_data);
    RETURN_ERROR("Could not load E0 data.");
  }

  json_decref(e0_data);

  rtd->E0.loaded = true;

  return SUCCESS;
}


int load_runtime_data_absorption(rtd_t *rtd){

  if (rtd == NULL) RETURN_ERROR("Runtime data structure is NULL.");

  if (rtd->absorption.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("Absorption runtime data is already loaded.");
  }

  json_t *abs_data = NULL;
  load_runtime_data(_FORCE_L1_ABSORPTION_FILE_, &abs_data);

  rtd->absorption.number = _GAS_LENGTH_;
  alloc((void**)&rtd->absorption.spectrum, rtd->absorption.number, sizeof(seq_t));

  if (load_runtime_data_seq(abs_data, "Ozone", &rtd->absorption.spectrum[_GAS_OZONE_]) != SUCCESS){
    free_runtime_data(rtd);
    json_decref(abs_data);
    RETURN_ERROR("Could not load Ozone data.");
  }

  if (load_runtime_data_seq(abs_data, "Water vapor", &rtd->absorption.spectrum[_GAS_WATER_]) != SUCCESS){
    free_runtime_data(rtd);
    json_decref(abs_data);
    RETURN_ERROR("Could not load Water vapor data.");
  }

  json_decref(abs_data);

  rtd->absorption.loaded = true;

  return SUCCESS;
}

int load_runtime_data_water_library(rtd_t *rtd){

  if (rtd == NULL) RETURN_ERROR("Runtime data structure is NULL.");

  if (rtd->water_library.loaded){
    free_runtime_data(rtd);
    RETURN_ERROR("Water library runtime data is already loaded.");
  }

  json_t *lib_data = NULL;
  load_runtime_data(_FORCE_L1_WATER_LIBRARY_FILE_, &lib_data);

  // lib_data is a JSON object with one key per spectrum, e.g. "Water 000"
  rtd->water_library.number = (int)json_object_size(lib_data);

  if (rtd->water_library.number <= 0){
    free_runtime_data(rtd);
    json_decref(lib_data);
    RETURN_ERROR("Could not determine number of water library spectra.");
  }

  if (rtd->water_library.number > 999){
    free_runtime_data(rtd);
    json_decref(lib_data);
    RETURN_ERROR("Water library exceeds 1000 spectra.");
  }

  alloc((void**)&rtd->water_library.spectrum, rtd->water_library.number, sizeof(seq_t));

  for (int i=0; i<rtd->water_library.number; i++){

    char key[NPOW_10];
    int nchar = 0;
    if ((nchar = snprintf(key, NPOW_10, "Water %03d", i)) < 0 || nchar >= NPOW_10){
      free_runtime_data(rtd);
      json_decref(lib_data);
      RETURN_ERROR("Could not concatenate water library key.");
    }
    if (load_runtime_data_seq(lib_data, key, &rtd->water_library.spectrum[i]) != SUCCESS){
      free_runtime_data(rtd);
      json_decref(lib_data);
      RETURN_ERROR("Could not load water library data.");
    }

  }

  json_decref(lib_data);

  rtd->water_library.loaded = true;

  return SUCCESS;
}
