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
This file contains functions for parsing Landsat Level 1 metadata
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "meta_lnd-ll.h"


void free_metadata_landsat(lnd_mtd_t *mtd){

  if (mtd == NULL) return;

  if (mtd->image_files != NULL) free_2D((void**)mtd->image_files, mtd->nband);
  if (mtd->reflectance_scale  != NULL) free((void*)mtd->reflectance_scale);
  if (mtd->reflectance_offset != NULL) free((void*)mtd->reflectance_offset);
  if (mtd->radiance_min != NULL) free((void*)mtd->radiance_min);
  if (mtd->radiance_max != NULL) free((void*)mtd->radiance_max);
  if (mtd->quantize_min != NULL) free((void*)mtd->quantize_min);
  if (mtd->quantize_max != NULL) free((void*)mtd->quantize_max);
  if (mtd->k1 != NULL) free((void*)mtd->k1);
  if (mtd->k2 != NULL) free((void*)mtd->k2);

  mtd->image_files = NULL;
  mtd->reflectance_scale  = NULL;
  mtd->reflectance_offset = NULL;
  mtd->radiance_min = NULL;
  mtd->radiance_max = NULL;
  mtd->quantize_min = NULL;
  mtd->quantize_max = NULL;
  mtd->k1 = NULL;
  mtd->k2 = NULL;

  mtd->nband = 0;

  return;
}


int parse_metadata_landsat_platform(char *d_level1, lnd_mtd_t *mtd){

  if (d_level1 == NULL || mtd == NULL){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Invalid input.");
  }

  FILE *fp;
  char buffer[NPOW_10];
  char metaname[NPOW_10];

  // scan directory for MTL.txt file
  if (findfile_pattern(d_level1, "MTL", ".txt", metaname, NPOW_10) != SUCCESS){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Unable to find Landsat metadata (MTL file) in %s", d_level1); 
  }

  #ifdef FORCE_DEBUG
  printf("Found Landsat metadata file: %s\n", metaname);
  #endif

  // open MTL file
  if ((fp = fopen(metaname, "r")) == NULL){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Unable to open Landsat metadata %s ", metaname); 
  }


  // process line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){

    #ifdef FORCE_DEBUG
    printf("MTL: %s", buffer);
    #endif

    // get tag
    char *saveptr;
    const char *separator = " =\":\n";
    char *tokenptr = strtok_r(buffer, separator, &saveptr);
    if (tokenptr == NULL) continue; // skip empty lines

    char *tag = tokenptr;

    tokenptr = strtok_r(NULL, separator, &saveptr);
    if (tokenptr == NULL) continue; // no tag/value pairs in this line

    // Landsat sensor
    if (strings_equal(tag, "SPACECRAFT_ID")){
      copy_string(mtd->spacecraft_name, NPOW_10, tokenptr);
      break;
    }

  }

  fclose(fp);

  return SUCCESS;
}



int parse_metadata_landsat_mtl(char *d_level1, rtd_t *rtd, lnd_mtd_t *mtd){

  if (d_level1 == NULL || rtd == NULL || mtd == NULL ||
      rtd->band_mapping.l1_bands == NULL || rtd->band_mapping.domains == NULL || 
      rtd->band_mapping.nbands <= 0){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Invalid input.");
  }

  mtd->nband = rtd->band_mapping.nbands;

  // magic numbers for nodata and saturation values
  #ifdef FORCE_MAGIC
  fprintf(stderr, "Warning: Using magic numbers for nodata and saturation values\n");
  #endif
  
  mtd->nodata = 0;

  if (strings_equal(mtd->spacecraft_name, "LANDSAT_8") || 
      strings_equal(mtd->spacecraft_name, "LANDSAT_9")){
    mtd->saturation = USHRT_MAX;
  } else {
    mtd->saturation = UCHAR_MAX;
  }

  alloc_2D((void***)&mtd->image_files, mtd->nband, NPOW_10, sizeof(char));
  alloc((void**)&mtd->reflectance_scale,  mtd->nband, sizeof(int));
  alloc((void**)&mtd->reflectance_offset, mtd->nband, sizeof(int));
  alloc((void**)&mtd->radiance_min, mtd->nband, sizeof(float));
  alloc((void**)&mtd->radiance_max, mtd->nband, sizeof(float));
  alloc((void**)&mtd->quantize_min, mtd->nband, sizeof(float));
  alloc((void**)&mtd->quantize_max, mtd->nband, sizeof(float));
  alloc((void**)&mtd->k1, mtd->nband, sizeof(float));
  alloc((void**)&mtd->k2, mtd->nband, sizeof(float));

  for (int b=0; b<mtd->nband; b++){
    mtd->reflectance_scale[b]  = FLT_MIN;
    mtd->reflectance_offset[b] = FLT_MIN;
    mtd->radiance_min[b] = FLT_MAX;
    mtd->radiance_max[b] = FLT_MIN;
    mtd->quantize_min[b] = FLT_MAX;
    mtd->quantize_max[b] = FLT_MIN;
    mtd->k1[b] = FLT_MIN;
    mtd->k2[b] = FLT_MIN;
  }
  
  mtd->ulx   = DBL_MIN; 
  mtd->uly   = DBL_MIN;

  int temp_nrow = 0, temp_ncol = 0;
  double temp_res = 0.0;

  FILE *fp;
  char buffer[NPOW_10];
  char metaname[NPOW_10];

  // scan directory for MTL.txt file
  if (findfile_pattern(d_level1, "MTL", ".txt", metaname, NPOW_10) != SUCCESS){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Unable to find Landsat metadata (MTL file) in %s", d_level1); 
  }

  #ifdef FORCE_DEBUG
  printf("Found Landsat metadata file: %s\n", metaname);
  #endif

  // open MTL file
  if ((fp = fopen(metaname, "r")) == NULL){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Unable to open Landsat metadata %s ", metaname); 
  }


  // process line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){

    #ifdef FORCE_DEBUG
    printf("MTL: %s", buffer);
    #endif

    // get tag
    char *saveptr;
    const char *separator = " =\"\n";
    char *tokenptr = strtok_r(buffer, separator, &saveptr);
    if (tokenptr == NULL) continue; // skip empty lines

    char *tag = tokenptr;

    tokenptr = strtok_r(NULL, separator, &saveptr);
    if (tokenptr == NULL) continue; // no tag/value pairs in this line

    if (strings_equal(tag, "PROCESSING_LEVEL")){
      copy_string(mtd->processing_level, NPOW_10, tokenptr);
    } else if (strings_equal(tag, "COLLECTION_NUMBER")){
      char_to_int(tokenptr, &mtd->collection_number);
    } else if (strings_equal(tag, "COLLECTION_CATEGORY")){
      copy_string(mtd->collection_category, NPOW_10, tokenptr);
    } else if (strings_equal(tag, "LANDSAT_PRODUCT_ID")){
      copy_string(mtd->landsat_product_id, NPOW_10, tokenptr);
    } else if (strings_equal(tag, "WRS_TYPE")){
      char_to_int(tokenptr, &mtd->wrs_type);
    } else if (strings_equal(tag, "WRS_PATH")){
      char_to_int(tokenptr, &mtd->wrs_path);
    } else if (strings_equal(tag, "WRS_ROW")){
      char_to_int(tokenptr, &mtd->wrs_row);
    } else if (strings_equal(tag, "DATE_ACQUIRED")){
      copy_string(mtd->date_char, NPOW_10, tokenptr);
    } else if (strings_equal(tag, "SCENE_CENTER_TIME")){
      copy_string(mtd->time_char, NPOW_10, tokenptr);
    } else if (strings_equal(tag, "REFLECTIVE_SAMPLES")){
      char_to_int(tokenptr, &mtd->ncol);
    } else if (strings_equal(tag, "REFLECTIVE_LINES")){
      char_to_int(tokenptr, &mtd->nrow);
    } else if (strings_equal(tag, "GRID_CELL_SIZE_REFLECTIVE")){
      char_to_double(tokenptr, &mtd->res);
    } else if (strings_equal(tag, "THERMAL_SAMPLES")){
      char_to_int(tokenptr, &temp_ncol);
    } else if (strings_equal(tag, "THERMAL_LINES")){
      char_to_int(tokenptr, &temp_nrow);
    } else if (strings_equal(tag, "GRID_CELL_SIZE_THERMAL")){
      char_to_double(tokenptr, &temp_res);
    } else if (strings_equal(tag, "CORNER_UL_PROJECTION_X_PRODUCT")){
      char_to_double(tokenptr, &mtd->ulx);
      mtd->ulx -= 15.0; // adjust for pixel center
    } else if (strings_equal(tag, "CORNER_UL_PROJECTION_Y_PRODUCT")){
      char_to_double(tokenptr, &mtd->uly);
      mtd->uly += 15.0; // adjust for pixel center
    } else if (strings_equal(tag, "UTM_ZONE")){
      #ifdef FORCE_MAGIC
      fprintf(stderr, "Warning: Using magic number offset for UTM EPSG codes\n");
      #endif
      char_to_int(tokenptr, &mtd->epsg);
      mtd->epsg += 32600; // convert to EPSG code
    } else if (strings_equal(tag, "TRUE_SCALE_LAT")){
      float true_scale_lat;
      char_to_float(tokenptr, &true_scale_lat);
      #ifdef FORCE_MAGIC
      fprintf(stderr, "Warning: Using magic numbers for polar stereographic EPSG codes\n");
      #endif
      if (fequal(true_scale_lat, 71.0)){
        mtd->epsg = 3995; // Arctic Polar Stereographic
      } else if (fequal(true_scale_lat, -71.0)){
        mtd->epsg = 3031; // Antarctic Polar Stereographic
      }
    } else if (strstr(tag, "FILE_NAME_BAND_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        copy_string(mtd->image_files[b], NPOW_10, tokenptr);
      }
    } else if (strstr(tag, "RADIANCE_MINIMUM_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->radiance_min[b]);
      }
    } else if (strstr(tag, "RADIANCE_MAXIMUM_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->radiance_max[b]);
      }
    } else if (strstr(tag, "QUANTIZE_CAL_MIN_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->quantize_min[b]);
      }
    } else if (strstr(tag, "QUANTIZE_CAL_MAX_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->quantize_max[b]);
      }
    } else if (strstr(tag, "REFLECTANCE_MULT_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->reflectance_scale[b]);
      }
    } else if (strstr(tag, "REFLECTANCE_ADD_") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->reflectance_offset[b]);
      }
    } else if (strstr(tag, "K1_CONSTANT") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->k1[b]);
      }
    } else if (strstr(tag, "K2_CONSTANT") != NULL){
      delete_until_match(tag, "BAND_", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, tag)) >= 0){
        char_to_float(tokenptr, &mtd->k2[b]);
      }
    }

  }

  fclose(fp);

  // Test if we got everything
  if (strlen(mtd->processing_level) == 0){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not retrieve Landsat processing level.");
  }

  if (!strings_equal(mtd->processing_level, "L1TP") && 
      !strings_equal(mtd->processing_level, "L1GT") && 
      !strings_equal(mtd->processing_level, "L1GS")){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Expected L1TP, L1GT, or L1GS, got %s.", mtd->processing_level); 
  }

  if (!strings_equal(mtd->collection_category, "T1") && 
      !strings_equal(mtd->collection_category, "T2") && 
      !strings_equal(mtd->collection_category, "RT")){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Expected T1, T2, or RT, got %s.", mtd->collection_category); 
  }

  // this logic should be revised
  mtd->tier = 1;
  if (!strings_equal(mtd->processing_level, "L1TP")){
    mtd->tier = 2;
  }
  if (strings_equal(mtd->collection_category, "T2")){
    mtd->tier = 2;
  } else if (strings_equal(mtd->collection_category, "RT")){
    mtd->tier = 3;
  }

  if (mtd->collection_number != 2){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Expected Collection 2, got %d.", mtd->collection_number); 
  }

  if (strlen(mtd->landsat_product_id) == 0){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not retrieve Landsat product ID.");
  }

  if (mtd->wrs_type != 2){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Expected WRS type 2, got %d.", mtd->wrs_type);
  }

  if (mtd->wrs_path < 1 || mtd->wrs_path > 233){
    free_metadata_landsat(mtd);
    RETURN_ERROR("WRS path (%d) is out of bounds.", mtd->wrs_path);
  }

  if (mtd->wrs_row < 1 || mtd->wrs_row > 248){
    free_metadata_landsat(mtd);
    RETURN_ERROR("WRS row (%d) is out of bounds.", mtd->wrs_row);
  }

  int nchar = snprintf(mtd->wrs_path_row, NPOW_10, "%03d%03d", mtd->wrs_path, mtd->wrs_row);
  if (nchar < 0 || nchar >= NPOW_10){ 
    free_metadata_landsat(mtd);
    RETURN_ERROR("Buffer Overflow in assembling Path/Row string"); 
  }

  char utc_string[NPOW_10];
  concat_string_2(utc_string, NPOW_10, mtd->date_char, mtd->time_char, "T");
  if (date_from_utc(&mtd->date, utc_string) != SUCCESS){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not parse Landsat acquisition date/time.");
  }

  if (mtd->nrow < 1 || mtd->ncol < 1){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not retrieve Landsat image dimensions.");
  }
  mtd->ncell = mtd->nrow * mtd->ncol;

  if (mtd->nrow != temp_nrow || mtd->ncol != temp_ncol){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Reflective and thermal image dimensions do not match.");
  }

  if (mtd->res <= 0.0){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not retrieve Landsat image resolution.");
  }

  if (mtd->res != temp_res){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Reflective and thermal image resolutions do not match.");
  }

  if (dequal(mtd->ulx, DBL_MIN) || dequal(mtd->uly, DBL_MIN)){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not retrieve Landsat image upper-left coordinates.");
  }

  if (mtd->epsg == 0){
    free_metadata_landsat(mtd);
    RETURN_ERROR("Could not retrieve Landsat EPSG code.");
  }

  for (int b=0; b<mtd->nband; b++){
    if (mtd->image_files[b] == NULL || strlen(mtd->image_files[b]) == 0){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat image file for band %d.", b);
    }
    if (!strings_equal(rtd->band_mapping.domains[b], "TEMP") && 
        fequal(mtd->reflectance_scale[b], FLT_MIN)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat reflectance scale for band %d.", b);
    }
    if (!strings_equal(rtd->band_mapping.domains[b], "TEMP") && 
    fequal(mtd->reflectance_offset[b], FLT_MIN)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat reflectance offset for band %d.", b);
    }
    if (fequal(mtd->radiance_min[b], FLT_MAX)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat radiance minimum for band %d.", b);
    }
    if (fequal(mtd->radiance_max[b], FLT_MIN)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat radiance maximum for band %d.", b);
    }
    if (fequal(mtd->quantize_min[b], FLT_MAX)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat quantize minimum for band %d.", b);
    }
    if (fequal(mtd->quantize_max[b], FLT_MIN)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat quantize maximum for band %d.", b);
    }
    if (strings_equal(rtd->band_mapping.domains[b], "TEMP") && 
        fequal(mtd->k1[b], FLT_MIN)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat K1 constant for band %d.", b);
    }
    if (strings_equal(rtd->band_mapping.domains[b], "TEMP") && 
        fequal(mtd->k2[b], FLT_MIN)){
      free_metadata_landsat(mtd);
      RETURN_ERROR("Could not retrieve Landsat K2 constant for band %d.", b);
    }
  }

  #ifdef FORCE_DEBUG
  printf("metadata parsed from %s:\n", metaname);
  printf("  spacecraft = %s\n", mtd->spacecraft_name);
  printf("  processing level = %s\n", mtd->processing_level);
  printf("  collection number = %d\n", mtd->collection_number);
  printf("  collection category = %s\n", mtd->collection_category);
  printf("  tier = %d\n", mtd->tier);
  printf("  product ID = %s\n", mtd->landsat_product_id);
  printf("  WRS type = %d, path = %d, row = %d, path/row = %s\n", mtd->wrs_type, mtd->wrs_path, mtd->wrs_row, mtd->wrs_path_row);
  printf("  acquisition date = %s, time = %s\n",  mtd->date_char, mtd->time_char);
  print_date(&mtd->date);
  printf("  image dimensions = %d x %d, resolution = %.2f\n", mtd->ncol, mtd->nrow, mtd->res);
  printf("  upper-left coordinates = (%.2f, %.2f), EPSG = %d\n", mtd->ulx, mtd->uly, mtd->epsg);
  printf("  nodata = %d, saturation = %d\n", mtd->nodata, mtd->saturation);
  for (int b=0; b<mtd->nband; b++){
    printf("  band %d:\n", b+1);
    printf("    file = %s\n", mtd->image_files[b]);
    printf("    reflectance scale = %.6f\n", mtd->reflectance_scale[b]);
    printf("    reflectance offset = %.6f\n", mtd->reflectance_offset[b]);
    printf("    radiance min = %.6f\n", mtd->radiance_min[b]);
    printf("    radiance max = %.6f\n", mtd->radiance_max[b]);
    printf("    quantize min = %.6f\n", mtd->quantize_min[b]);
    printf("    quantize max = %.6f\n", mtd->quantize_max[b]);
    printf("    k1 = %.6f\n", mtd->k1[b]);
    printf("    k2 = %.6f\n", mtd->k2[b]);
  }
  #endif

  return SUCCESS;
}

