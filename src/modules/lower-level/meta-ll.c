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
This file contains functions for parsing metadata
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "meta-ll.h"

/** This function allocates the metadata
+++ Return: metadata (must be freed with free_metadata)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
meta_t *allocate_metadata(){
meta_t *meta = NULL;


  alloc((void**)&meta, 1, sizeof(meta_t));
  init_metadata(meta);

  return meta;
}


/** This function frees the metadata
--- meta:   metadata
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_metadata(meta_t *meta){

  if (meta == NULL) return;

  if (meta->cal != NULL) free((void*)meta->cal);
  if (meta->wavelength != NULL) free((void*)meta->wavelength);

  meta->cal = NULL;
  meta->wavelength = NULL;

  free_string_vector(&meta->image_path);

  if (meta->view_grid.zen != NULL) free((void*)meta->view_grid.zen);
  if (meta->view_grid.azi != NULL) free((void*)meta->view_grid.azi);

  meta->view_grid.zen = NULL;
  meta->view_grid.azi = NULL;

  free((void*)meta); 
  meta = NULL;

  return;
}


/** This function initializes metadata items that cannot be tested against NULL
--- meta:   metadata
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int init_metadata(meta_t *meta){

  meta->saturation = INT_MIN;
  meta->nodata = INT_MIN;

  meta->ulx = DBL_MIN;
  meta->uly = DBL_MIN;

  meta->col_offset = INT_MIN;
  meta->row_offset = INT_MIN;

  return SUCCESS;
}


int test_metadata(meta_t *meta){

  if (meta == NULL) RETURN_ERROR("Metadata is NULL.");
  if (meta->mission == _UNKNOWN_) RETURN_ERROR("Mission is not set.");
  if (meta->band_number <= 0) RETURN_ERROR("Number of bands is not set.");
  if (meta->image_path.length <= 0) RETURN_ERROR("Image path length is not set.");
  if (meta->image_path.number <= 0) RETURN_ERROR("Image path number is not set.");
  if (meta->image_path.string == NULL) RETURN_ERROR("Image path string is NULL.");
  if (meta->saturation == INT_MIN) RETURN_ERROR("Saturation is not set.");
  if (meta->nodata == INT_MIN) RETURN_ERROR("Nodata is not set.");
  if (meta->res <= 0.0) RETURN_ERROR("Resolution is not set.");
  if (dequal(meta->ulx, DBL_MIN) || dequal(meta->uly, DBL_MIN)) RETURN_ERROR("Upper-left coordinates are not set.");
  if (meta->nrow < 1 || meta->ncol < 1 || meta->ncell < 1) RETURN_ERROR("Image dimensions are not set.");
  if (meta->col_offset == INT_MIN || meta->row_offset == INT_MIN) RETURN_ERROR("Image offsets are not set.");
  if (meta->date.year < 1900 || meta->date.year > 2100) RETURN_ERROR("Acquisition date is not set.");
  if (meta->epsg == 0) RETURN_ERROR("EPSG code is not set.");
  if (meta->cal == NULL) RETURN_ERROR("Calibration is NULL.");
  if (meta->wavelength == NULL) RETURN_ERROR("Wavelength is NULL.");
  if (meta->refsys_type[0] == '\0' || meta->refsys_id[0] == '\0') RETURN_ERROR("Reference system is not set.");
  if (meta->mission == SENTINEL2){
    if (meta->view_grid.zen == NULL || meta->view_grid.azi == NULL) RETURN_ERROR("View grid is not set.");
    if (meta->view_grid.nrow < 1 || meta->view_grid.ncol < 1 || meta->view_grid.ncell < 1) RETURN_ERROR("View grid dimensions are not set.");
    if (meta->view_grid.res < 1) RETURN_ERROR("View grid resolution is not set.");
  }

  return SUCCESS;
}


/** This function prints the metadata
--- meta:   metadata
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_metadata(meta_t *meta){

  printf("Metadata readout:\n");
  printf("  Mission: %d\n", meta->mission);
  printf("  Number of bands: %d\n", meta->band_number);
  printf("  Image path number: %d\n", meta->image_path.number);
  printf("  Image path length: %d\n", meta->image_path.length);
  for (int b=0; b<meta->image_path.number; b++){
    printf("    Band %d: %s\n", b+1, meta->image_path.string[b]);
  }
  printf("  Saturation: %d\n", meta->saturation);
  printf("  Nodata: %d\n", meta->nodata);
  printf("  Resolution: %f\n", meta->res);
  printf("  Upper-left coordinates: (%f, %f)\n", meta->ulx, meta->uly);
  printf("  Image dimensions: (%d, %d, %d)\n", meta->ncol, meta->nrow, meta->ncell);
  printf("  Image offsets: (%d, %d)\n", meta->col_offset, meta->row_offset);
  printf("  Acquisition date:\n");
  print_date(&meta->date);
  printf("  EPSG code: %d\n", meta->epsg);
  printf("  Processing tier: %d\n", meta->tier);
  printf("  Reference system: %s %s\n", meta->refsys_type, meta->refsys_id);
  printf("  Calibration:\n");
  for (int b=0; b<meta->band_number; b++){
    printf("    Band %d:\n", b+1);
    printf("      Type: %d\n", meta->cal[b].type);
    printf("      Reflectance scale: %f\n", meta->cal[b].reflectance.scale);
    printf("      Reflectance offset: %f\n", meta->cal[b].reflectance.offset);
    printf("      Radiance min: %f\n", meta->cal[b].radiance.lmin);
    printf("      Radiance max: %f\n", meta->cal[b].radiance.lmax);
    printf("      Radiance quantize min: %f\n", meta->cal[b].radiance.qmin);
    printf("      Radiance quantize max: %f\n", meta->cal[b].radiance.qmax);
    printf("      Temperature K1: %f\n", meta->cal[b].temperature.k1);
    printf("      Temperature K2: %f\n", meta->cal[b].temperature.k2);
  }
  for (int b=0; b<meta->band_number; b++){
    printf("  Band %d wavelength: %f\n", b+1, meta->wavelength[b]);
  }
  if (meta->mission == SENTINEL2){
    printf("  View grid:\n");
    printf("    Dimensions: (%d, %d, %d)\n", meta->view_grid.nrow, meta->view_grid.ncol, meta->view_grid.ncell);
    printf("    Resolution: %f\n", meta->view_grid.res);
    printf("    Zenith angles:\n");
    for (int i=0, p=0; i<meta->view_grid.nrow; i++){
      printf("    ");
    for (int j=0; j<meta->view_grid.ncol; j++, p++){
      printf(" %5.2f", meta->view_grid.zen[p]);
    }
    printf("\n");
    }
    printf("    Azimuth angles:\n");
    for (int i=0, p=0; i<meta->view_grid.nrow; i++){
      printf("    ");
    for (int j=0; j<meta->view_grid.ncol; j++, p++){
      printf(" %5.2f", meta->view_grid.azi[p]);
    }
    printf("\n");
    }
  }

  return;
}


/** This function reads the Landsat metadata
--- pl2:    L2 parameters
--- rtd:     runtime data
--- meta:   metadata
--- dn:     Digital Number brick
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata_landsat(par_ll_t *pl2, rtd_t *rtd, meta_t *meta){


  #ifdef FORCE_DEBUG
  printf("reading Landsat metadata\n");
  #endif

 // get Satellite ID first
  lnd_mtd_t lnd_mtd = {0};
  if (parse_metadata_landsat_platform(pl2->d_level1, &lnd_mtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to parse Landsat platform metadata.\n");
    return FAILURE;
  }

  // load runtime data for band mapping
  if (load_runtime_data_band_mapping(lnd_mtd.spacecraft_name, rtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to load band mapping for %s.\n", lnd_mtd.spacecraft_name);
    return FAILURE;
  }

  // load runtime data for sensor mapping
  if (load_runtime_data_sensor_mapping(lnd_mtd.spacecraft_name, rtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to load sensor mapping for %s.\n", lnd_mtd.spacecraft_name);
    return FAILURE;
  }

  // load runtime data for RSR mapping
  if (load_runtime_data_rsr_mapping(lnd_mtd.spacecraft_name, rtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to load RSR mapping for %s.\n", lnd_mtd.spacecraft_name);
    return FAILURE;
  }
;
  if (parse_metadata_landsat_mtl(pl2->d_level1, rtd, &lnd_mtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to parse Landsat metadata.\n");
    return FAILURE;
  }


  /** fill general-purpose metadata
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  meta->band_number = lnd_mtd.nband;

  // full image paths for each band
  alloc_string_vector(&meta->image_path, lnd_mtd.nband, NPOW_00);
  for (int b=0; b<lnd_mtd.nband; b++){
    char full_path[NPOW_10];
    concat_string_2(full_path, NPOW_10, pl2->d_level1, lnd_mtd.image_files[b], "/");
    fill_string_vector(&meta->image_path, b, full_path);
  }

  int nchar = 0;
  nchar = snprintf(meta->refsys_type, NPOW_10, "WRS-%d", lnd_mtd.wrs_type);
  if (nchar < 0 || nchar >= NPOW_10) RETURN_ERROR("Failed to format reference system type.");
  copy_string(meta->refsys_id, NPOW_10, lnd_mtd.wrs_path_row);

  meta->saturation = lnd_mtd.saturation;
  meta->nodata = lnd_mtd.nodata;

  meta->tier = lnd_mtd.tier;

  meta->res = lnd_mtd.res;
  meta->ulx = lnd_mtd.ulx;
  meta->uly = lnd_mtd.uly;
  meta->ncol = lnd_mtd.ncol;
  meta->nrow = lnd_mtd.nrow;
  meta->ncell = lnd_mtd.ncell;
  meta->col_offset = 0.0;
  meta->row_offset = 0.0;
  meta->epsg = lnd_mtd.epsg;

  copy_date(&lnd_mtd.date, &meta->date);

  alloc((void**)&meta->cal, lnd_mtd.nband, sizeof(cal_t));

  for (int b=0; b<lnd_mtd.nband; b++){
    if (!strings_equal(rtd->band_mapping.domains[b], "TEMP")){
      meta->cal[b].type = _CAL_REF_;
    } else {
      meta->cal[b].type = _CAL_BT_;
    }
    meta->cal[b].reflectance.scale = lnd_mtd.reflectance_scale[b];
    meta->cal[b].reflectance.offset = lnd_mtd.reflectance_offset[b];
    meta->cal[b].radiance.lmin = lnd_mtd.radiance_min[b];
    meta->cal[b].radiance.lmax = lnd_mtd.radiance_max[b];
    meta->cal[b].radiance.qmin = lnd_mtd.quantize_min[b];
    meta->cal[b].radiance.qmax = lnd_mtd.quantize_max[b];
    meta->cal[b].temperature.k1 = lnd_mtd.k1[b];
    meta->cal[b].temperature.k2 = lnd_mtd.k2[b];
  }

  alloc((void**)&meta->wavelength, lnd_mtd.nband, sizeof(float));
  for (int b=0; b<lnd_mtd.nband; b++){
    weighted_centroid_of_seq(&rtd->rsr_mapping.rsr[b], &meta->wavelength[b]);
  }

  free_metadata_landsat(&lnd_mtd);

  return SUCCESS;
}


/** This function reads the Sentinel-2 metadata
--- pl2:    L2 parameters
--- rtd:     runtime data
--- meta:   metadata
--- dn:     Digital Number brick
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata_sentinel2(par_ll_t *pl2, rtd_t *rtd, meta_t *meta){


  #ifdef FORCE_DEBUG
  printf("reading Sentinel-2 metadata\n");
  #endif

  // magic numbers...
  int nd = 12; // number of detectors

  // get Satellite ID first
  s2_mtd_t s2_mtd = {0};
  if (parse_metadata_sentinel2_platform(pl2->d_level1, &s2_mtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to parse Sentinel-2 platform metadata.\n");
    return FAILURE;
  }

  // load runtime data for band mapping
  if (load_runtime_data_band_mapping(s2_mtd.spacecraft_name, rtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to load band mapping for %s.\n", s2_mtd.spacecraft_name);
    return FAILURE;
  }

  // load runtime data for sensor mapping
  if (load_runtime_data_sensor_mapping(s2_mtd.spacecraft_name, rtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to load sensor mapping for %s.\n", s2_mtd.spacecraft_name);
    return FAILURE;
  }

  // parse top-level (.SAFE) xml
  if (parse_metadata_sentinel2_safe(pl2->d_level1, rtd, &s2_mtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to parse Sentinel-2 top-level metadata.\n");
    return FAILURE;
  }

  // parse granule xml
  if (parse_metadata_sentinel2_granule(s2_mtd.granule_path, nd, rtd, &s2_mtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to parse Sentinel-2 granule metadata.\n");
    return FAILURE;
  }

  // parse MGRS tile ID
  char mgrs_tile[NPOW_10];
  if (parse_mgrs_tile(pl2->b_level1, mgrs_tile, NPOW_10) != SUCCESS){
    fprintf(stderr, "Error: Failed to parse Sentinel-2 MGRS tile ID.\n");
    return FAILURE;
  }

  // construct view grid and identify required image subset
  if (construct_sentinel2_view_grid(pl2, &s2_mtd) != SUCCESS){
    fprintf(stderr, "Error: Failed to construct Sentinel-2 view grid.\n");
    return FAILURE;
  }




  /** fill general-purpose metadata
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  // full image paths for each band
  alloc_string_vector(&meta->image_path, s2_mtd.nband, strlen(s2_mtd.image_files[0]) + 1);
  for (int b=0; b<s2_mtd.nband; b++){
    fill_string_vector(&meta->image_path, b, s2_mtd.image_files[b]);
  }

  copy_string(meta->refsys_type, NPOW_10, "MGRS");
  copy_string(meta->refsys_id, NPOW_10, mgrs_tile);
  
  meta->saturation = s2_mtd.saturation;
  meta->nodata = s2_mtd.nodata;
  meta->band_number = s2_mtd.nband;

  meta->tier = 1;

  meta->res = s2_mtd.res;
  meta->ulx = s2_mtd.ulx;
  meta->uly = s2_mtd.uly;
  meta->ncol = s2_mtd.ncol;
  meta->nrow = s2_mtd.nrow;
  meta->ncell = s2_mtd.ncell;
  meta->col_offset = s2_mtd.col_offset;
  meta->row_offset = s2_mtd.row_offset;
  meta->epsg = s2_mtd.epsg;

  copy_date(&s2_mtd.date, &meta->date);

  alloc((void**)&meta->cal, s2_mtd.nband, sizeof(cal_t));
  for (int b=0; b<s2_mtd.nband; b++){
    meta->cal[b].type = _CAL_REF_;
    meta->cal[b].reflectance.scale = s2_mtd.scale;
    meta->cal[b].reflectance.offset = s2_mtd.offset[b];
  }
  
  // transfer ownership of viewing angles from s2_mtd to meta
  meta->view_grid = s2_mtd.subset_view_grid;
  s2_mtd.subset_view_grid.zen = NULL;
  s2_mtd.subset_view_grid.azi = NULL;

  // transfer ownership of RSR arrays from s2_mtd to rtd
  rtd->rsr_mapping.rsr = s2_mtd.rsr;
  rtd->rsr_mapping.nbands = s2_mtd.nband;
  rtd->rsr_mapping.loaded = true;
  copy_string(rtd->rsr_mapping.spacecraft_name, NPOW_10, s2_mtd.spacecraft_name);
  s2_mtd.rsr = NULL;
  
  alloc((void**)&meta->wavelength, s2_mtd.nband, sizeof(float));
  for (int b=0; b<s2_mtd.nband; b++){
    weighted_centroid_of_seq(&rtd->rsr_mapping.rsr[b], &meta->wavelength[b]);
  }

  free_metadata_sentinel2(&s2_mtd);

  return SUCCESS;
}


/** This function identifies the satellite mission, i.e. Landsat or Senti-
+++ nel-2
--- pl2:    L2 parameters
--- meta:   metadata
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata_mission(par_ll_t *pl2, meta_t *meta){
char metaname[NPOW_10];

  if (findfile_pattern(pl2->d_level1, "MTL", ".txt", metaname, NPOW_10) == SUCCESS){
    pl2->res = pl2->res_landsat;
    meta->mission = LANDSAT;
  } else if (findfile_pattern(pl2->d_level1, "MTD", ".xml", metaname, NPOW_10) == SUCCESS){
    pl2->res = pl2->res_sentinel2;
    meta->mission = SENTINEL2;
  } else {
    meta->mission = _UNKNOWN_;
    RETURN_ERROR("unknown Satellite Mission.");
  }

  #ifdef FORCE_DEBUG
  printf("\nMission: %d\n", meta->mission);
  #endif

  return SUCCESS;
}


/** This function appends the reference system to the DEM name
--- pl2:    L2 parameters
--- meta:   metadata
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int append_reference_system_to_dem(par_ll_t *pl2, meta_t *meta) {
char dir_name[NPOW_10];
int nchar;

  if (pl2->use_dem_database) {

    copy_string(dir_name, NPOW_10, pl2->fdem);

    nchar = snprintf(pl2->fdem, NPOW_10, "%s/%s_%s.vrt", dir_name, meta->refsys_type, meta->refsys_id);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling string for DEM database\n"); 
      return FAILURE;
    }
  
    if (!fileexist(pl2->fdem)) {
      printf("DEM database file %s does not exist!\n", pl2->fdem);
      return FAILURE;
    }

    #ifdef FORCE_DEBUG
    printf("DEM database file: %s\n", pl2->fdem);
    #endif
  }

  return SUCCESS;
}


/** This function reads the metadata
--- pl2:      L2 parameters
--- rtd:      runtime data
--- metadata: metadata
--- dn:       Digital Number brick
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata(par_ll_t *pl2, rtd_t *rtd, meta_t **metadata){
meta_t *meta = NULL;

  meta = allocate_metadata();

  if (parse_metadata_mission(pl2, meta) != SUCCESS) RETURN_ERROR("Failed to parse metadata mission.");

  switch (meta->mission){
    case LANDSAT:
      if (parse_metadata_landsat(pl2, rtd, meta)  != SUCCESS) RETURN_ERROR("Failed to parse Landsat metadata.");
      break;
    case SENTINEL2:
      if (parse_metadata_sentinel2(pl2, rtd, meta) != SUCCESS) RETURN_ERROR("Failed to parse Sentinel-2 metadata.");
      break;
    default:
      RETURN_ERROR("Unknown mission.");
  }

  #ifdef FORCE_DEBUG
  print_metadata(meta);
  #endif

  if (test_metadata(meta) != SUCCESS) RETURN_ERROR("Metadata not valid.");

  if (append_reference_system_to_dem(pl2, meta) != SUCCESS){
    RETURN_ERROR("Failed to append reference system to DEM-Database.");
  }

  *metadata = meta;
  return SUCCESS;
}

