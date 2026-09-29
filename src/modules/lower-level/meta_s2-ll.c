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
This file contains functions for parsing Sentinel-2 Level 1 metadata
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "meta_s2-ll.h"


void free_metadata_sentinel2(s2_mtd_t *mtd){

  if (mtd == NULL) return;

  if (mtd->image_files != NULL) free_2D((void**)mtd->image_files, mtd->nband);
  if (mtd->offset != NULL) free((void*)mtd->offset);
  mtd->image_files = NULL;
  mtd->offset = NULL;

  if (mtd->rsr != NULL){
    for (int b=0; b< mtd->nband; b++){
      if (mtd->rsr[b].values != NULL) free((void*)mtd->rsr[b].values);
      mtd->rsr[b].values = NULL;
    }
    free((void*)mtd->rsr);
  }
  mtd->rsr = NULL;

  if (mtd->view_grid.zen != NULL) free_3D((void***)mtd->view_grid.zen, mtd->nband, mtd->ndetector);
  if (mtd->view_grid.azi != NULL) free_3D((void***)mtd->view_grid.azi, mtd->nband, mtd->ndetector);
  mtd->view_grid.zen = NULL;
  mtd->view_grid.azi = NULL;

  if (mtd->subset_view_grid.zen != NULL) free((void*)mtd->subset_view_grid.zen);
  if (mtd->subset_view_grid.azi != NULL) free((void*)mtd->subset_view_grid.azi);
  mtd->subset_view_grid.zen = NULL;
  mtd->subset_view_grid.azi = NULL;

  mtd->nband = 0;
  mtd->ndetector = 0;

  return;
}


int parse_mgrs_tile(char *b_level1, char *mgrs, size_t size){

  if (b_level1 == NULL || mgrs == NULL){
    RETURN_ERROR("Invalid input.");
  }

  if (size < 7){
    RETURN_ERROR("Buffer for MGRS tile ID is too small.");
  }

  char buffer[NPOW_10];
  copy_string(buffer, NPOW_10, b_level1);

  char *saveptr = NULL;
  char *tokenptr = strtok_r(buffer, "_", &saveptr);

  // get 6th token, which is the MGRS tile ID
  for (int i=0; i<5; i++){
    if (tokenptr == NULL){
      RETURN_ERROR("Could not retrieve Sentinel-2 MGRS tile ID from %s.", b_level1);
    }
    tokenptr = strtok_r(NULL, "_", &saveptr);
  }

  if (strlen(tokenptr) != 6){
    RETURN_ERROR("Sentinel-2 MGRS tile ID has wrong length. Got %s, expected 6 characters.", tokenptr);
  }
  copy_string(mgrs, size, tokenptr);

  if (mgrs[0] != 'T'){
    RETURN_ERROR("Could not retrieve Sentinel-2 MGRS tile ID. Got %s, expected Txxxxxx.", mgrs);
  }

  char utm_zone_str[3] = {0};
  strncpy(utm_zone_str, mgrs+1, 2);
  int utm_zone;
  char_to_int(utm_zone_str, &utm_zone);
  if (utm_zone < 1 || utm_zone > 60){
    RETURN_ERROR("Invalid UTM zone (%d) in MGRS tile ID.", utm_zone);
  }

  return SUCCESS;
}

int parse_metadata_sentinel2_platform(char *d_level1, s2_mtd_t *mtd){

  if (d_level1 == NULL || mtd == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid input.");
  }
  
  // scan directory for xml file
  char metaname[NPOW_10];
  if (findfile_pattern(d_level1, "MTD", ".xml", metaname, NPOW_10) == FAILURE){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Finding top-level S2 metadata file failed in %s.", d_level1);
  }

  #ifdef FORCE_DEBUG
  printf("top-level metadata: %s\n", metaname);
  #endif

  // open xml
  FILE *fp = NULL;
  if ((fp = fopen(metaname, "r")) == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Unable to open S2 metadata file %s", metaname);
  }

  // process line by line
  char buffer[NPOW_13];
  while (fgets(buffer, NPOW_13, fp) != NULL){

    #ifdef FORCE_DEBUG
    printf("XML: %s", buffer);
    #endif

    // Sentinel-2 spacecraft name [A-D]
    if (strstr(buffer, "<SPACECRAFT_NAME") != NULL){
      get_xml_string_value(buffer, mtd->spacecraft_name, NPOW_10);
      break;
    }

  }

  fclose(fp);

  if (strlen(mtd->spacecraft_name) == 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 spacecraft name.");
  }


  #ifdef FORCE_DEBUG
  printf("metadata parsed from %s:\n", metaname);
  printf("  spacecraft = %s\n", mtd->spacecraft_name);
  #endif

  return SUCCESS;
}

int parse_metadata_sentinel2_safe(char *d_level1, rtd_t *rtd, s2_mtd_t *mtd){

  if (d_level1 == NULL || rtd == NULL || mtd == NULL ||
      rtd->band_mapping.l1_bands == NULL || rtd->band_mapping.domains == NULL || 
      rtd->band_mapping.nbands <= 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid input.");
  }

  mtd->nband = rtd->band_mapping.nbands;

  mtd->nodata = INT_MIN;
  mtd->saturation = INT_MIN;

  // scan directory for xml file
  char metaname[NPOW_10];
  if (findfile_pattern(d_level1, "MTD", ".xml", metaname, NPOW_10) == FAILURE){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Finding top-level S2 metadata file failed in %s.", d_level1);
  }

  #ifdef FORCE_DEBUG
  printf("top-level metadata: %s\n", metaname);
  #endif

  // open xml
  FILE *fp = NULL;
  if ((fp = fopen(metaname, "r")) == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Unable to open S2 metadata file %s", metaname);
  }

  // allocate memory for image files, offsets, and response functions
  alloc_2D((void***)&mtd->image_files, mtd->nband, NPOW_10, sizeof(char));
  alloc((void**)&mtd->offset, mtd->nband, sizeof(float));
  alloc((void**)&mtd->rsr, mtd->nband, sizeof(seq_t));

  // initialize offsets to FLT_MIN to detect missing values (0 could be a valid offset)
  for (int b=0; b<mtd->nband; b++) mtd->offset[b] = FLT_MIN;
  
  // keep track of how many IMAGE_FILE tags we have seen to build a dictionary
  // because some metadata are not tagged with a band_id attribute, so we need to keep track of the order they appear in the XML
  // looks superfluous, but is important if runtime data definition order ever becomes different from metadata order
  int b_occurence = 0; 
  char **band_order = NULL;
  alloc_2D((void***)&band_order, mtd->nband, NPOW_10, sizeof(char));

  // process line by line
  char buffer[NPOW_13];
  string_t xml_open = {0};
  string_t xml_value = {0};
  string_t xml_close = {0};

  while (fgets(buffer, NPOW_13, fp) != NULL){

    #ifdef FORCE_DEBUG
    printf("XML: %s", buffer);
    #endif

    // processing baseline
    if (strstr(buffer, "<PROCESSING_BASELINE") != NULL){
      get_xml_float_value(buffer, &mtd->processing_baseline);
    } else if (strstr(buffer, "<PROCESSING_LEVEL") != NULL){
      get_xml_string_value(buffer, mtd->processing_level, NPOW_10);
    } else if (strstr(buffer, "<PRODUCT_START_TIME") != NULL){
      split_xml_line(buffer, &xml_open, &xml_value, &xml_close);
      date_from_utc(&mtd->date, xml_value.string);
    } else if (strstr(buffer, "<IMAGE_FILE") != NULL){
      char image_path[NPOW_10];
      char image_file[NPOW_10];
      char granule_name[NPOW_10];
      get_xml_string_value(buffer, image_path, NPOW_10);
      basename_without_ext(image_path, image_file, NPOW_10);
      copy_string(granule_name, NPOW_10, image_path);
      delete_after_match(granule_name, "/IMG_DATA", false);
      concat_string_2(mtd->granule_path, NPOW_10, d_level1, granule_name, "/");
      delete_until_match(image_file, "_B", false);
      int b;
      if ((b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, image_file)) >= 0){
        char image_path_ext[NPOW_10];
        concat_string_2(image_path_ext, NPOW_10, image_path, ".jp2", "");
        concat_string_2(mtd->image_files[b], NPOW_10, d_level1, image_path_ext, "/");
        copy_string(band_order[b_occurence], NPOW_10, image_file);
        b_occurence++;
      }
    } else if (strstr(buffer, "<SPECIAL_VALUE_TEXT") != NULL){
      char special_value_type[NPOW_10];
      get_xml_string_value(buffer, special_value_type, NPOW_10);
      if (fgets(buffer, NPOW_13, fp) != NULL){
        #ifdef FORCE_DEBUG
        printf("XML: %s", buffer);
        #endif
        if (strcmp(special_value_type, "SATURATED") == 0){
          get_xml_int_value(buffer, &mtd->saturation);
        } else if (strcmp(special_value_type, "NODATA") == 0){
          get_xml_int_value(buffer, &mtd->nodata);
        }
      }
    } else if (strstr(buffer, "<QUANTIFICATION_VALUE") != NULL){
      get_xml_float_value(buffer, &mtd->scale);
    } else if (strstr(buffer, "<RADIO_ADD_OFFSET") != NULL){
      split_xml_line(buffer, &xml_open, &xml_value, &xml_close);
      int band_id;
      get_xml_attribute_int_value(xml_open.string, "band_id", &band_id);
      if (band_id < 0 || band_id >= mtd->nband){
        free_metadata_sentinel2(mtd);
        free_2D((void**)band_order, mtd->nband);
        fclose(fp);
        RETURN_ERROR("Band ID (%s) is out of bounds.", xml_open.string);
      }
      int b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, band_order[band_id]);
      if (b < 0 || b>= mtd->nband){
        free_metadata_sentinel2(mtd);
        free_2D((void**)band_order, mtd->nband);
        fclose(fp);
        RETURN_ERROR("Band ID (%s) is out of bounds.", band_order[band_id]);
      }
      char_to_float(xml_value.string, &mtd->offset[b]);
    } else if (strstr(buffer, "<Spectral_Information") != NULL &&
               strstr(buffer, "<Spectral_Information_List>") == NULL){
      char band_id[NPOW_10];
      get_xml_attribute_string_value(buffer, "physicalBand", band_id, NPOW_10);
      replace_string(band_id, "B", "", NPOW_10);
      // weird hack to handle inconsistency in band_id being a single digit, e.g. "1" instead of "01"
      if (strlen(band_id) == 1){
        band_id[1] = band_id[0];
        band_id[0] = '0';
        band_id[2] = '\0';
      }
      int b = vector_contains_pos((const char**)rtd->band_mapping.l1_bands, rtd->band_mapping.nbands, band_id);
      if (b < 0 || b >= mtd->nband){
        free_metadata_sentinel2(mtd);
        free_2D((void**)band_order, mtd->nband);
        fclose(fp);
        RETURN_ERROR("Band ID (%s) is out of bounds.", band_id);
      }
      while (fgets(buffer, NPOW_13, fp) != NULL &&
             strstr(buffer, "</Spectral_Information") == NULL){
        #ifdef FORCE_DEBUG
        printf("XML: %s", buffer);
        #endif
        if (strstr(buffer, "<MIN") != NULL){
          get_xml_float_value(buffer, &mtd->rsr[b].start);
        } else if (strstr(buffer, "<MAX") != NULL){
          get_xml_float_value(buffer, &mtd->rsr[b].end);
        } else if (strstr(buffer, "<STEP") != NULL){
          get_xml_float_value(buffer, &mtd->rsr[b].step);
        } else if (strstr(buffer, "<VALUES") != NULL){
          get_xml_float_values(buffer, &mtd->rsr[b].values, &mtd->rsr[b].n);
        }
      }

    }

  }

  fclose(fp);

  free_string(&xml_open);
  free_string(&xml_value);
  free_string(&xml_close);
  free_2D((void**)band_order, mtd->nband);

  // Test if we got everything

  if (fequal(mtd->processing_baseline, 0.0)){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Processing baseline not found in metadata.");
  }

  if (mtd->processing_baseline < 4.0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Outdated Sentine-2 processing baseline detected: %.2f", mtd->processing_baseline);
  }

  if (strlen(mtd->processing_level) == 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 processing level.");
  }

  if (strcmp(mtd->processing_level, "Level-1C") != 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Unsupported processing level detected: %s.", mtd->processing_level);
  }

  if (b_occurence != mtd->nband){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Number of IMAGE_FILE tags (%d) does not match expected number of bands (%d).", 
            b_occurence, mtd->nband);
  }

  if (mtd->nodata == INT_MIN){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 NODATA value.");
  }

  if (mtd->saturation == INT_MIN){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 SATURATION value.");
  }

  if (!date_is_valid(&mtd->date, true)){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 acquisition date.");
  }

  if (fequal0(mtd->scale, NULL)){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 quantification value.");
  }


  for (int b=0; b<mtd->nband; b++){
    if (fequal(mtd->offset[b], FLT_MIN)){
      free_metadata_sentinel2(mtd);
      RETURN_ERROR("Could not retrieve additive scaling factor for band %d.", b);
    }
  }

  for (int b=0; b<mtd->nband; b++){
    if (mtd->rsr[b].values == NULL || mtd->rsr[b].n < 1 ||
       (fequal(mtd->rsr[b].start, 0.0) && fequal(mtd->rsr[b].end, 0.0)) || 
        fequal(mtd->rsr[b].step, 0.0)){
      free_metadata_sentinel2(mtd);
      RETURN_ERROR("Could not retrieve RSR information for band %d.", b);
    }
    if (mtd->rsr[b].n != (mtd->rsr[b].end - mtd->rsr[b].start) / mtd->rsr[b].step + 1){
      free_metadata_sentinel2(mtd);
      RETURN_ERROR("Number of RSR values (%d) does not match expected number (%d) for band %d.", 
              mtd->rsr[b].n, (int)((mtd->rsr[b].end - mtd->rsr[b].start) / mtd->rsr[b].step + 1), b);
    }
    if (mtd->rsr[b].values == NULL){
      free_metadata_sentinel2(mtd);
      RETURN_ERROR("RSR values array is NULL for band %d.", b);
    }
    for (int i=0; i<mtd->rsr[b].n; i++){
      if (mtd->rsr[b].values[i] < 0){
        free_metadata_sentinel2(mtd);
        RETURN_ERROR("RSR value at index %d is negative for band %d.", i, b);
      }
    }
  }

  #ifdef FORCE_DEBUG
  printf("metadata parsed from %s:\n", metaname);
  printf("  processing baseline = %.2f\n", mtd->processing_baseline);
  printf("  processing level = %s\n", mtd->processing_level);
  printf("  acquisition date = \n");
  print_date(&mtd->date);
  printf("  nodata value = %d\n", mtd->nodata);
  printf("  saturation value = %d\n", mtd->saturation);
  printf("  quantification value = %.2f\n", mtd->scale);
  printf("  additive scaling factors:\n");
  for (int b=0; b<mtd->nband; b++){
    printf("    band %d: %.2f\n", b, mtd->offset[b]);
  }
  printf("  RSR information:\n");
  for (int b=0; b<mtd->nband; b++){
    printf("    band %d: start = %f, end = %f, step = %f, n = %d\n", 
      b, mtd->rsr[b].start, mtd->rsr[b].end, mtd->rsr[b].step, mtd->rsr[b].n);
  }
  printf("  image files:\n");
  for (int b=0; b<mtd->nband; b++){
    printf("    band %d: %s\n", b, mtd->image_files[b]);
  }
  #endif


  return SUCCESS;
}

int parse_metadata_sentinel2_granule(char *d_granule, int detector_number, rtd_t *rtd, s2_mtd_t *mtd){

  if (d_granule == NULL || detector_number < 0 || mtd == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid input.");
  }

  mtd->nrow  = INT_MIN; 
  mtd->ncol  = INT_MIN;
  mtd->ncell = INT_MIN;
  mtd->ulx   = DBL_MIN; 
  mtd->uly   = DBL_MIN;
  mtd->res   = DBL_MAX;

  mtd->view_grid.res    = DBL_MAX;
  mtd->view_grid.nrow   = INT_MIN;
  mtd->view_grid.ncol   = INT_MIN;
  mtd->view_grid.ncell  = INT_MIN;
  mtd->view_grid.zen = NULL;
  mtd->view_grid.azi = NULL;
  mtd->view_grid.nodata = FLT_MAX;

  mtd->ndetector = detector_number;

  // scan directory for xml file
  char metaname[NPOW_10];
  if (findfile_pattern(d_granule, "MTD", ".xml", metaname, NPOW_10) == FAILURE){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Finding granule metadata file (`*MTD*.xml`) failed in %s", d_granule); 
  }

  #ifdef FORCE_DEBUG
  printf("granule-level metadata: %s\n", metaname);
  #endif

  // open xml
  FILE *fp = NULL;
  if ((fp = fopen(metaname, "r")) == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Unable to open S2 metadata file %s", metaname); 
  }

  // process line by line
  char buffer[NPOW_13];
  while (fgets(buffer, NPOW_13, fp) != NULL){

    #ifdef FORCE_DEBUG
    printf("XML: %s", buffer);
    #endif

    if (strstr(buffer, "<HORIZONTAL_CS_CODE") != NULL){
      char epsg_str[NPOW_10];
      get_xml_string_value(buffer, epsg_str, NPOW_10);
      if (strstr(epsg_str, "EPSG:") != NULL) char_to_int(epsg_str+5, &mtd->epsg);
    } else if (strstr(buffer, "<NROWS") != NULL){
      int temp;
      get_xml_int_value(buffer, &temp);
      if (temp > mtd->nrow) mtd->nrow = temp;
    } else if (strstr(buffer, "<NCOLS") != NULL){
      int temp;
      get_xml_int_value(buffer, &temp);
      if (temp > mtd->ncol) mtd->ncol = temp;
    } else if (strstr(buffer, "<ULX") != NULL){
      get_xml_double_value(buffer, &mtd->ulx); // all res are TL-aligned, so we can just take any
    } else if (strstr(buffer, "<ULY") != NULL){
      get_xml_double_value(buffer, &mtd->uly); // all res are TL-aligned, so we can just take any
    } else if (strstr(buffer, "<XDIM") != NULL){
      double temp;
      get_xml_double_value(buffer, &temp);
      if (temp < mtd->res) mtd->res = fabs(temp); // pixels are square, so we can just take x or y. fabs just in case
    } else if (strstr(buffer, "<Tile_Angles") != NULL){

      int b = -1, d = -1;
      bool v = false, z = false, a = false;
      int i = 0;

      while (fgets(buffer, NPOW_13, fp) != NULL &&
             strstr(buffer, "</Tile_Angles") == NULL){
        #ifdef FORCE_DEBUG
        printf("XML: %s", buffer);
        #endif
        if (strstr(buffer, "<COL_STEP") != NULL ||
            strstr(buffer, "<ROW_STEP") != NULL){
          double temp;
          get_xml_double_value(buffer, &temp);
          if (dequal(mtd->view_grid.res, DBL_MAX)){
            mtd->view_grid.res = temp;
          } else if (!dequal(mtd->view_grid.res, temp)){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("Inconsistent view grid size in metadata.");
          } else if (mtd->view_grid.res <= 0){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("Invalid view grid resolution (%.2f) in metadata.", mtd->view_grid.res);
          }
          if (mtd->ncol < 0 || mtd->nrow < 0){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("Invalid image dimensions (%d x %d) in metadata.", mtd->ncol, mtd->nrow);
          }
          mtd->view_grid.ncol = ceil(mtd->ncol*mtd->res/mtd->view_grid.res) + 1;
          mtd->view_grid.nrow = ceil(mtd->nrow*mtd->res/mtd->view_grid.res) + 1;
          mtd->view_grid.ncell = mtd->view_grid.nrow*mtd->view_grid.ncol;
          if (mtd->view_grid.zen == NULL){
            alloc_3D((void****)&mtd->view_grid.zen, mtd->nband, mtd->ndetector, mtd->view_grid.ncell, sizeof(float));
            for (int b=0; b<mtd->nband; b++){
              for (int d=0; d<mtd->ndetector; d++){
                for (int i=0; i<mtd->view_grid.ncell; i++){
                  mtd->view_grid.zen[b][d][i] = mtd->view_grid.nodata;
                }
              }
            } 
            if (mtd->view_grid.azi == NULL){
              alloc_3D((void****)&mtd->view_grid.azi, mtd->nband, mtd->ndetector, mtd->view_grid.ncell, sizeof(float));
              for (int b=0; b<mtd->nband; b++){
                for (int d=0; d<mtd->ndetector; d++){
                  for (int i=0; i<mtd->view_grid.ncell; i++){
                    mtd->view_grid.azi[b][d][i] = mtd->view_grid.nodata;
                  }
                }
              } 
            } 
          }
        } else if (strstr(buffer, "<Viewing_Incidence_Angles_Grids") != NULL){
          v = true;
          get_xml_attribute_int_value(buffer, "bandId", &b);
          get_xml_attribute_int_value(buffer, "detectorId", &d);
          d--; // detectorId is 1-based in the metadata, but we use 0-based indexing
        } else if (strstr(buffer, "</Viewing_Incidence_Angles_Grids") != NULL){
          v = false;
        } else if (strstr(buffer, "<Zenith") != NULL){
          z = true; i = 0;
        } else if (strstr(buffer, "</Zenith") != NULL){
          z = false;
        } else if (strstr(buffer, "<Azimuth") != NULL){
          a = true; i = 0;
        } else if (strstr(buffer, "</Azimuth") != NULL){
          a = false;
        } else if (strstr(buffer, "<VALUES") != NULL){
          
          if (!v) continue; // we only care about viewing angles

          if (i < 0){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("Negative row index (%d) while parsing sun/view angles in metadata.", i);
          }
          if (i >= mtd->view_grid.nrow){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("More rows of sun/view angles than expected (%d) in metadata.", mtd->view_grid.nrow);
          }
          if (b < 0 || b >= mtd->nband){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("Band ID (%d) is out of bounds.", b);
          }
          if (d < 0 || d >= mtd->ndetector){
            free_metadata_sentinel2(mtd);
            fclose(fp);
            RETURN_ERROR("Detector ID (%d) is out of bounds.", d);
          }

          char **value_list = NULL;
          int value_count = 0;
          get_xml_string_values(buffer, &value_list, &value_count);

          if (value_count != mtd->view_grid.ncol){
            free_metadata_sentinel2(mtd);
            free_2D((void**)value_list, value_count);
            fclose(fp);
            RETURN_ERROR("Number of values (%d) does not match expected number (%d) in metadata.", value_count, mtd->view_grid.ncol);
          }          

          for (int j=0; j<value_count; j++){
            if (z){
              if (strcmp(value_list[j], "NaN") != 0){
                mtd->view_grid.zen[b][d][i*mtd->view_grid.ncol+j] = atof(value_list[j]);
              } else {
                mtd->view_grid.zen[b][d][i*mtd->view_grid.ncol+j] = mtd->view_grid.nodata;
              }
            } else if (a){
              if (strcmp(value_list[j], "NaN") != 0){
                mtd->view_grid.azi[b][d][i*mtd->view_grid.ncol+j] = atof(value_list[j]);
              } else {
                mtd->view_grid.azi[b][d][i*mtd->view_grid.ncol+j] = mtd->view_grid.nodata;
              }
            } else {
              free_metadata_sentinel2(mtd);
              fclose(fp);
              RETURN_ERROR("Unexpected state while parsing sun/view angles in metadata.");
            }
          }

          free_2D((void**)value_list, value_count);

          i++;

       }
      }
    }
  }

  fclose(fp);

  // check if we got everything
  if (mtd->epsg == 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Could not retrieve Sentinel-2 EPSG code.");
  }

  if (mtd->nrow <= 0 || mtd->ncol <= 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid image dimensions (%d x %d) in metadata.", mtd->ncol, mtd->nrow);
  } else {
    mtd->ncell = mtd->nrow*mtd->ncol;
  }
  
  if (dequal(mtd->ulx, DBL_MIN) || dequal(mtd->uly, DBL_MIN)){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid upper-left coordinates (%f, %f) in metadata.", mtd->ulx, mtd->uly);
  }
    
  if (mtd->res <= 0 || dequal(mtd->res, DBL_MAX)){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid resolution (%.2f) in metadata.", mtd->res);
  }
  
  if (mtd->view_grid.res <= 0 || dequal(mtd->view_grid.res, DBL_MAX) || 
      mtd->view_grid.ncol <= 0 || mtd->view_grid.nrow <= 0){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid view grid resolution (%.2f) or dimensions (%d x %d) in metadata.", 
      mtd->view_grid.res, mtd->view_grid.ncol, mtd->view_grid.nrow);
  }

  if (mtd->view_grid.zen == NULL || mtd->view_grid.azi == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("View angle grids not properly initialized in metadata.");
  }

  #ifdef FORCE_DEBUG
  printf("metadata parsed from %s:\n", metaname);
  printf("  rows = %d, cols = %d, cells = %d, resolution = %.2f\n", 
    mtd->nrow, mtd->ncol, mtd->ncell, mtd->res);
  printf("  ulx = %.2f, uly = %.2f\n", mtd->ulx, mtd->uly);
  printf("  view grid: rows = %d, cols = %d, cells = %d, resolution = %.2f\n", 
    mtd->view_grid.nrow, mtd->view_grid.ncol, 
    mtd->view_grid.ncell, mtd->view_grid.res);
  #endif
  
  return SUCCESS;
}



int construct_sentinel2_view_grid(par_ll_t *pl2, s2_mtd_t *mtd){

  if (pl2 == NULL || mtd == NULL){
    free_metadata_sentinel2(mtd);
    RETURN_ERROR("Invalid input.");
  }

  // interpolate the grids
  // angles are given at grid intersections, we need one value per cell
  interpolate_sentinel2_view_grid(mtd);

  // collapse view grids
  collapse_sentinel2_view_grid(mtd);

  // subset the grids to the actual sensor coverage area
  subset_sentinel2_view_grid(pl2, mtd);

  return SUCCESS;
}



/** This function interpolates the sun and view angle grids given in the 
+++ Sentinel-2 metadata. The interpolation grid will be interpolated to
+++ a cell-based grid, and then clipped to the image extent.
+++ int_grid: interpolation grid (modified)
--- nx:       number of columns
--- ny:       number of rows
--- nodata:   nodata value
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void interpolate_sentinel2_view_grid(s2_mtd_t *mtd){

  if (mtd == NULL){
    EXIT_ERROR("Invalid input.");
  }

  for (int b=0; b<mtd->nband; b++){
  for (int d=0; d<mtd->ndetector; d++){
    
    int cell_nx = mtd->view_grid.ncol - 1;
    int cell_ny = mtd->view_grid.nrow - 1;
    int cell_nc = cell_nx * cell_ny;
    
    float *cell_grid_zen = NULL;
    alloc((void**)&cell_grid_zen, cell_ny*cell_nx, sizeof(float));

    float *cell_grid_azi = NULL;
    alloc((void**)&cell_grid_azi, cell_ny*cell_nx, sizeof(float));
    
    float sum_zen, num_zen;
    float sum_azi, num_azi;

    for (int i=0, cell_p=0; i<cell_ny; i++){
    for (int j=0; j<cell_nx; j++, cell_p++){

      sum_zen = num_zen = 0;
      sum_azi = num_azi = 0;

      for (int ii=0; ii<=1; ii++){
      for (int jj=0; jj<=1; jj++){

        int int_p = (i+ii) * mtd->view_grid.ncol + (j+jj);

        if (!fequal(mtd->view_grid.zen[b][d][int_p], mtd->view_grid.nodata)){
          sum_zen += mtd->view_grid.zen[b][d][int_p];
          num_zen++;
        }
        if (!fequal(mtd->view_grid.azi[b][d][int_p], mtd->view_grid.nodata)){
          sum_azi += mtd->view_grid.azi[b][d][int_p];
          num_azi++;
        }

      }
      }


      if (num_zen > 0){
        cell_grid_zen[cell_p] = sum_zen/num_zen;
      } else {
        cell_grid_zen[cell_p] = mtd->view_grid.nodata;
      }
      if (num_azi > 0){
        cell_grid_azi[cell_p] = sum_azi/num_azi;
      } else {
        cell_grid_azi[cell_p] = mtd->view_grid.nodata;
      }

    }
    }

    memmove(mtd->view_grid.zen[b][d], cell_grid_zen, cell_nc*sizeof(float));
    re_alloc((void**)&mtd->view_grid.zen[b][d], mtd->view_grid.ncell, cell_nc, sizeof(float));
    free((void*)cell_grid_zen);

    memmove(mtd->view_grid.azi[b][d], cell_grid_azi, cell_nc*sizeof(float));
    re_alloc((void**)&mtd->view_grid.azi[b][d], mtd->view_grid.ncell, cell_nc, sizeof(float));
    free((void*)cell_grid_azi);

  }
  }

  mtd->view_grid.ncol--;
  mtd->view_grid.nrow--;

  return;
}


/** This function collapses the Sentinel-2 view angle grids to a single
+++ grid. All band- and detector-based grids will be averaged. There is
+++ room for improvement at this point. The collapsed grid will be put
+++ into the first slot.
+++ grid:     3D grid (modified)
--- nb:       number of bands
--- nd:       number of detectors
--- nx:       number of columns
--- ny:       number of rows
--- nodata:   nodata value
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void collapse_sentinel2_view_grid(s2_mtd_t *mtd){
float sum_zen, num_zen, sum_azi, num_azi;

  if (mtd == NULL){
    EXIT_ERROR("Invalid input.");
  }

  for (int i=0, p=0; i<mtd->view_grid.nrow; i++){
  for (int j=0; j<mtd->view_grid.ncol; j++, p++){

    sum_zen = num_zen = 0;
    sum_azi = num_azi = 0;

    for (int b=0; b<mtd->nband; b++){
    for (int d=0; d<mtd->ndetector; d++){

      if (!fequal(mtd->view_grid.zen[b][d][p], mtd->view_grid.nodata)){
        sum_zen += mtd->view_grid.zen[b][d][p];
        num_zen++;
      }
      if (!fequal(mtd->view_grid.azi[b][d][p], mtd->view_grid.nodata)){
        sum_azi += mtd->view_grid.azi[b][d][p];
        num_azi++;
      }

    }
    }

    if (num_zen > 0){
      mtd->view_grid.zen[0][0][p] = sum_zen/num_zen;
    } else {
      mtd->view_grid.zen[0][0][p] = mtd->view_grid.nodata;
    }

    if (num_azi > 0){
      mtd->view_grid.azi[0][0][p] = sum_azi/num_azi;
    } else {
      mtd->view_grid.azi[0][0][p] = mtd->view_grid.nodata;
    }

  }
  }
  

  return;
}


void subset_sentinel2_view_grid(par_ll_t *pl2, s2_mtd_t *mtd){

  if (pl2 == NULL || mtd == NULL){
    free_metadata_sentinel2(mtd);
    EXIT_ERROR("Invalid input.");
  }

  int left, right, top, bottom;

  if (pl2->doreproj || pl2->dotile){

    // get image subset
    left = mtd->view_grid.ncol-1; right  = 0;
    top  = mtd->view_grid.nrow-1; bottom = 0;

    for (int i=0; i<mtd->view_grid.nrow; i++){
    for (int j=0; j<mtd->view_grid.ncol; j++){

      if (!fequal(mtd->view_grid.zen[0][0][i*mtd->view_grid.nrow+j], mtd->view_grid.nodata) && 
          !fequal(mtd->view_grid.azi[0][0][i*mtd->view_grid.nrow+j], mtd->view_grid.nodata)){
        if (j < left)   left   = j;
        if (j > right)  right  = j;
        if (i < top)    top    = i;
        if (i > bottom) bottom = i;
      }

    }
    }

    right++;  // lower-right corner of cell
    bottom++; // lower-right corner of cell

    if (left > 0) left--; // one to the left to fill the missing left edge

    while (fmod(left*mtd->view_grid.res, 60) != 0 && left > 0) left--;
    while (fmod(top*mtd->view_grid.res,  60) != 0 && top  > 0) top--;
    while (fmod(right*mtd->view_grid.res,  60) != 0 && right  < (mtd->view_grid.ncol-1)) right++;
    while (fmod(bottom*mtd->view_grid.res, 60) != 0 && bottom < (mtd->view_grid.nrow-1)) bottom++;

  } else {
  
    left = 0; right  = mtd->view_grid.ncol;
    top  = 0; bottom = mtd->view_grid.nrow;

  }

  mtd->subset_view_grid.ncol  = right-left;
  mtd->subset_view_grid.nrow  = bottom-top;
  mtd->subset_view_grid.ncell = mtd->subset_view_grid.ncol * mtd->subset_view_grid.nrow;
  mtd->subset_view_grid.res  = mtd->view_grid.res;
  mtd->subset_view_grid.nodata = mtd->view_grid.nodata;

  if (mtd->subset_view_grid.ncol <= 0 || mtd->subset_view_grid.nrow <= 0){
    EXIT_ERROR("no valid cell after subsetting. Abort.");
  }




  alloc((void**)&mtd->subset_view_grid.zen, mtd->subset_view_grid.ncell, sizeof(float));
  alloc((void**)&mtd->subset_view_grid.azi, mtd->subset_view_grid.ncell, sizeof(float));

  // copy values to final view grids
  for (int i=0, p=0; i<mtd->subset_view_grid.nrow; i++){
  for (int j=0; j<mtd->subset_view_grid.ncol; j++, p++){

    int ii = i+top;
    int jj = j+left;

    mtd->subset_view_grid.zen[p] = mtd->view_grid.zen[0][0][ii*mtd->view_grid.ncol+jj];
    mtd->subset_view_grid.azi[p] = mtd->view_grid.azi[0][0][ii*mtd->view_grid.ncol+jj];

  }
  }

  float res_ratio = mtd->view_grid.res / mtd->res;
  mtd->col_offset = left * res_ratio;
  mtd->row_offset = top  * res_ratio;
  mtd->ulx += mtd->col_offset * mtd->res;
  mtd->uly -= mtd->row_offset * mtd->res;

  if ((mtd->col_offset + mtd->subset_view_grid.ncol * res_ratio) > mtd->ncol){
    mtd->ncol -= mtd->col_offset;
  } else {
    mtd->ncol = mtd->subset_view_grid.ncol * res_ratio;
  }
  if ((mtd->row_offset + mtd->subset_view_grid.nrow * res_ratio) > mtd->nrow){
    mtd->nrow -= mtd->row_offset;
  } else {
    mtd->nrow = mtd->subset_view_grid.nrow * res_ratio;
  }
  mtd->ncell = mtd->nrow * mtd->ncol;

  free_3D((void***)mtd->view_grid.zen, mtd->nband, mtd->ndetector);
  mtd->view_grid.zen = NULL;
  free_3D((void***)mtd->view_grid.azi, mtd->nband, mtd->ndetector);
  mtd->view_grid.azi = NULL;

  #ifdef FORCE_DEBUG
  printf("active image subset: UL-X %d (%.2f) / UL-Y %d (%.2f) with width %d and height %d\n", 
    mtd->col_offset, mtd->ulx, mtd->row_offset, mtd->uly, mtd->ncol, mtd->nrow);
    printf("coarse cells: width/height %d/%d\n", mtd->subset_view_grid.ncol, mtd->subset_view_grid.nrow);
  #endif

  return;
}