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

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Runtime data header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef RUNTIME_DATA_CL_H
#define RUNTIME_DATA_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/json-cl.h"
#include "../cross-level/sys-cl.h"
#include "../cross-level/utils-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

// Directory and file names for runtime data (as macros)
#define _FORCE_RUNTIME_DATA_DIR_      "force-misc/runtime-data"
#define _FORCE_SENSOR_FILE_           "force-misc/runtime-data/sensors.json"
#define _FORCE_INDEX_FILE_            "force-misc/runtime-data/indices.json"
#define _FORCE_L1_BAND_MAP_FILE_      "force-misc/runtime-data/level-1/band-mapping.json"
#define _FORCE_L1_SENSOR_MAP_FILE_    "force-misc/runtime-data/level-1/sensor-mapping.json"
#define _FORCE_L1_RSR_FILE_           "force-misc/runtime-data/level-1/relative-spectral-response.json"
#define _FORCE_L1_E0_FILE_            "force-misc/runtime-data/level-1/exoatmospheric-irradiance.json"
#define _FORCE_L1_ABSORPTION_FILE_    "force-misc/runtime-data/level-1/gaseous-absorption.json"
#define _FORCE_L1_WATER_LIBRARY_FILE_ "force-misc/runtime-data/level-1/water-library.json"

typedef struct {
  bool loaded;
  char spacecraft_name[NPOW_10];
  int nbands;
  char **domains;
  char **l1_bands;
  bool *l2_output;
} rtd_band_mapping_t;

typedef struct {
  bool loaded;
  char spacecraft_name[NPOW_10];
  char l2_sensor[NPOW_10];
} rtd_sensor_mapping_t;

typedef struct {
  bool loaded;
  char spacecraft_name[NPOW_10];
  int nbands;
  seq_t *rsr;
} rtd_rsr_mapping_t;

typedef struct {
  bool loaded;
  seq_t spectrum;
} rtd_E0_t;

typedef struct {
  bool loaded;
  int number;
  seq_t *spectrum;
} rtd_absorption_t;


typedef struct {
  bool loaded;
  int number;
  seq_t *spectrum;
} rtd_water_library_t;

typedef struct {
  rtd_band_mapping_t band_mapping;
  rtd_sensor_mapping_t sensor_mapping;
  rtd_rsr_mapping_t rsr_mapping;
  rtd_E0_t E0;
  rtd_absorption_t absorption;
  rtd_water_library_t water_library;
} rtd_t;

void load_runtime_data(char *runtime_file, json_t **runtime_data);
void free_runtime_data(rtd_t *rtd);
int load_runtime_data_band_mapping(char *spacecraft_name, rtd_t *rtd);
int load_runtime_data_sensor_mapping(char *spacecraft_name, rtd_t *rtd);
int load_runtime_data_rsr_mapping(char *spacecraft_name, rtd_t *rtd);
int load_runtime_data_E0(rtd_t *rtd);
int load_runtime_data_absorption(rtd_t *rtd);
int load_runtime_data_water_library(rtd_t *rtd);

#ifdef __cplusplus
}
#endif

#endif
