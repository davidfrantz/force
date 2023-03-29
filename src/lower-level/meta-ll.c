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

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points


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

  free_calibration(meta->cal);

  if (meta->s2.szen != NULL) free_2D((void**)meta->s2.szen, meta->s2.ny);
  if (meta->s2.vzen != NULL) free_2D((void**)meta->s2.vzen, meta->s2.ny);
  if (meta->s2.sazi != NULL) free_2D((void**)meta->s2.sazi, meta->s2.ny);
  if (meta->s2.vazi != NULL) free_2D((void**)meta->s2.vazi, meta->s2.ny);

  free((void*)meta); meta = NULL;

  return;
}


/** This function initializes the metadata to enable value testing
--- meta:   metadata
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int init_metadata(meta_t *meta){

  #ifdef FORCE_DEBUG
  printf("check that all meta is initialized, brick as well?\n");
  #endif

  meta->fill = SHRT_MIN;

  meta->dtype      = meta->fill;
  meta->sat        = meta->fill;
  meta->tier       = meta->fill;

  meta->cal = NULL;

  meta->s2.szen   = NULL;
  meta->s2.sazi   = NULL;
  meta->s2.vzen   = NULL;
  meta->s2.vazi   = NULL;
  meta->s2.nx     = meta->fill;
  meta->s2.ny     = meta->fill;
  meta->s2.nodata = meta->fill;

  return SUCCESS;
}


/** This function allocates the calibration coefficients
+++ Return: calibration coefficients (must be freed with free_calibration)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
cal_t *allocate_calibration(int nb){
cal_t *cal = NULL;
int b;

  alloc((void**)&cal, nb, sizeof(cal_t));
  for (b=0; b<nb; b++) init_calibration(&cal[b]);

  return cal;
}


/** This function frees the calibration coefficients
--- cal:    calibration coefficients
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_calibration(cal_t *cal){

  if (cal == NULL) return;

  free((void*)cal); cal = NULL;

  return;
}


/** This function initializes the calibration coefficients
--- cal:    calibration coefficients
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int init_calibration(cal_t *cal){

  copy_string(cal->fname,     NPOW_10, "NULL");
  copy_string(cal->orig_band, NPOW_03, "NULL");

  cal->fill     = SHRT_MIN;

  cal->rsr_band = cal->fill;
  cal->lmax     = cal->fill;
  cal->lmin     = cal->fill;
  cal->qmax     = cal->fill;
  cal->qmin     = cal->fill;
  cal->rmul     = cal->fill;
  cal->radd     = cal->fill;
  cal->k1       = cal->fill;
  cal->k2       = cal->fill;

  return SUCCESS;
}


/** This function tests if all metadata are OK
--- pl2:    L2 parameters
--- meta:   metadata
--- DN:     Digital Number brick
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_metadata(par_ll_t *pl2, meta_t *meta, brick_t *DN){
int b, b_temp, nb;
#ifdef FORCE_DEBUG
printf("init and check for brick struct, too?\n");
#endif

  if ((nb = get_brick_nbands(DN)) < 0){
    printf("error in retrieving number of bands. "); return FAILURE;}

  if ((b_temp = find_domain(DN, "TEMP")) < 0){
    printf("error in retrieving temp band. "); return FAILURE;}

//  if(strcmp(meta->satellite, "NULL") == 0){
//    printf("error in satellite name. "); return FAILURE;}
   if (meta->dtype == meta->fill){
    printf("error in dtype. "); return FAILURE;}
  if (meta->sat == meta->fill){
    printf("error in sat. "); return FAILURE;}
 //  if (meta->scale == fill){
 //   printf("error in scale. "); return FAILURE;}
//  if (meta->res == fill){
//    printf("error in resolution. "); return FAILURE;}
//  if (meta->map_ul_x == fill || meta->map_ul_y == fill){
//    printf("error in map ul. "); return FAILURE;}
//  if (atof(meta->refsys) == 0){
//    printf("error in path/row. "); return FAILURE;}/  //if (meta->date.day == fill){
  //  printf("error in date.day. "); return FAILURE;}
  //if (meta->date.doy == fill){
  //  printf("error in date.doy. "); return FAILURE;}
  //if (meta->date.month == fill){
  //  printf("error in date.month. "); return FAILURE;}
  //if (meta->date.year == fill){
  //  printf("error in date.year. "); return FAILURE;}
  //if (meta->date.hh == fill){
  //  printf("error in time.hh. "); return FAILURE;}
  //if (meta->date.mm == fill){
  //  printf("error in time.mm. "); return FAILURE;}
  //if (meta->date.ss == fill){
  //  printf("error in time.ss. "); return FAILURE;}
  //if (meta->date.tz == fill){
  //  printf("error in time.tz. "); return FAILURE;}
  if (meta->tier > pl2->tier){
    printf("Unacceptable tier.\n"); exit(1);}
  for (b=0; b<nb; b++){
    if(strcmp(meta->cal[b].fname, "NULL") == 0){
      printf("error in fname. "); return FAILURE;}
    if (meta->cal[b].lmax == meta->cal[b].fill){
      printf("error in Lmax. "); return FAILURE;}
    if (meta->cal[b].lmax == meta->cal[b].fill){
      printf("error in Lmin. "); return FAILURE;}
    if (meta->cal[b].qmax == meta->cal[b].fill){
      printf("error in Qcalmax. "); return FAILURE;}
    if (meta->cal[b].qmin == meta->cal[b].fill){
      printf("error in Qcalmin. "); return FAILURE;}
    if (b != b_temp && meta->cal[b].rmul == meta->cal[b].fill){
      printf("error in reflectance scale factor (x). "); return FAILURE;}
    if (b != b_temp && meta->cal[b].radd == meta->cal[b].fill){
      printf("error in reflectance scale factor (+). "); return FAILURE;}
  }
  if (meta->cal[b_temp].k1 == meta->cal[b_temp].fill){
    printf("error in k1. "); return FAILURE;}
  if (meta->cal[b_temp].k2 == meta->cal[b_temp].fill){
    printf("error in k2. "); return FAILURE;}


  return SUCCESS;
}


/** This function prints the metadata
--- meta:   metadata
--- nb:     number of bands
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int print_metadata(meta_t *meta, int nb){
int b;
char basename[NPOW_10];

  for (b=0; b<nb; b++){
    basename_with_ext(meta->cal[b].fname, basename, NPOW_10);
    printf("DN: %s\n", basename);
    printf(" LMAX/LMIN %.2f/%.2f, QMAX/QMIN %.2f/%.2f, R*/R+ %.5f/%.2f, K1/K2 %.2f/%.2f\n",
      meta->cal[b].lmax, meta->cal[b].lmin,
      meta->cal[b].qmax, meta->cal[b].qmin,
      meta->cal[b].rmul, meta->cal[b].radd,
      meta->cal[b].k1,   meta->cal[b].k2);
  }

  printf("dtype: %d\n", meta->dtype);
  printf("sat: %d\n", meta->sat);
  printf("refsys   = %s\n", meta->refsys);
  //printf("res: %d\n", meta->res);
  //printf("yyyy/mm/dd + doy = %04d/%02d/%02d + %03d\n", meta->date.year, meta->date.month, meta->date.day, meta->date.doy);
  //printf("hh/mm/ss= %02d/%02d/%02d\n", meta->date.hh, meta->date.mm, meta->date.ss);
  //printf("ul: %.1f/%.1f\n", meta->map_ul_x, meta->map_ul_y);
  printf("tier: %d\n", meta->tier);
//  printf("satellite ID: %s\n", meta->satellite);
 // printf("scale factor: %d\n", meta->scale);

  return SUCCESS;
}


/** This function reads the Landsat metadata
--- pl2:    L2 parameters
--- meta:   metadata
--- dn:     Digital Number brick
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata_landsat(par_ll_t *pl2, meta_t *meta, brick_t **dn){
brick_t *DN = NULL;
FILE *fp = NULL;
char metaname[NPOW_10];
char  buffer[NPOW_10] = "\0";
char *tag = NULL;
char *tokenptr = NULL;
char *tokenptr2 = NULL;
char *separator = " =\":";
char *separator2 = "-";
char sensor[NPOW_04];
int nchar;
int b, b_temp = 0, nb = 0, lid = 0, path = 0, row = 0;
date_t date;
GDALDatasetH fp_;
char domain[NPOW_10];
const char domains_tm[7][NPOW_10] = {
  "BLUE", "GREEN", "RED", 
  "NIR", "SWIR1", "SWIR2", "TEMP"
};
const char domains_oli[9][NPOW_10] = {
  "ULTRABLUE", "BLUE", "GREEN", "RED", 
  "NIR", "CIRRUS", "SWIR1", "SWIR2", "TEMP"
};
const char band_id_tm[7][NPOW_03] = {
  "1", "2", "3", "4", "5", "7", "6" };
const char band_id_oli[9][NPOW_03] = {
  "1", "2", "3", "4", "5", "9", "6", "7", "10" };


  #ifdef FORCE_DEBUG
  printf("reading Landsat metadata\n");
  #endif

  // scan directory for MTL.txt file
  if (findfile(pl2->d_level1, "MTL", ".txt", metaname, NPOW_10) != SUCCESS){
    printf("Unable to find Landsat metadata (MTL file)! "); return FAILURE;}

  // open MTL file
  if ((fp = fopen(metaname, "r")) == NULL){
    printf("Unable to open Landsat metadata (MTL file)! "); return FAILURE;}


  // process line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){

    if (strstr(buffer, "END_GROUP = L1_METADATA_FILE")) break;

    // get tag
    tokenptr = strtok(buffer, separator);
    tag=tokenptr;

    // extract parameters by comparing tag
    while (tokenptr != NULL){

      tokenptr = strtok(NULL, separator);

      // Landsat sensor
      if (strcmp(tag, "SPACECRAFT_ID") == 0){

        lid = atoi(tokenptr+strlen(tokenptr)-1);

        // set datatype, saturation value, and number of bands
        if (lid >= 8){
          meta->dtype = 16;
          meta->sat = 65535;
          nb = 9;
        } else {
          meta->dtype = 8;
          meta->sat = 255;
          nb = 7;
        }

        DN = allocate_brick(nb, 0, _DT_NONE_);
        set_brick_nprovenance(DN, 1);
        set_brick_provenance(DN, 0, pl2->d_level1);

        nchar = snprintf(sensor, NPOW_04, "LND%02d", lid);
        if (nchar < 0 || nchar >= NPOW_04){
          printf("Buffer Overflow in assembling sensor\n"); return FAILURE;}


        meta->cal = allocate_calibration(nb);


        for (b=0; b<nb; b++){

          set_brick_sensor(DN, b, sensor);
          if (lid >= 8){
            copy_string(meta->cal[b].orig_band, NPOW_03, band_id_oli[b]);
            set_brick_domain(DN, b, domains_oli[b]);
          } else {
            copy_string(meta->cal[b].orig_band, NPOW_03, band_id_tm[b]);
            set_brick_domain(DN, b, domains_tm[b]);
          }

        }

        b_temp = find_domain(DN, "TEMP");


      // file names
      } else if (strstr(tag, "FILE_NAME") != NULL){
        for (b=0; b<nb; b++) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 0);

      // product type
      } else if (strcmp(tag, "DATA_TYPE") == 0 ||       // collection 1
                 strcmp(tag, "PROCESSING_LEVEL") == 0){ // collection 2
        if (strstr(tokenptr, "L1T")  != NULL ||
            strstr(tokenptr, "L1TP") != NULL){
          meta->tier = 1;
        } else meta->tier = 2;

      // tier level
      } else if (strcmp(tag, "COLLECTION_CATEGORY") == 0){
        if (strstr(tokenptr, "T2") != NULL) meta->tier = 2;
        if (strstr(tokenptr, "RT") != NULL) meta->tier = 3;

      // dimension variables
      } else if (strcmp(tag, "REFLECTIVE_SAMPLES") == 0){
        set_brick_ncols(DN, atoi(tokenptr));
      } else if (strcmp(tag, "REFLECTIVE_LINES") == 0){
        set_brick_nrows(DN, atoi(tokenptr));

      // resolution variables
      } else if (strcmp(tag, "GRID_CELL_SIZE_REFLECTIVE") == 0){
        set_brick_res(DN, atoi(tokenptr));

      // bounding box variables: map
      } else if (strcmp(tag, "CORNER_UL_PROJECTION_X_PRODUCT") == 0){
        set_brick_ulx(DN, atof(tokenptr)-15);
      } else if (strcmp(tag, "CORNER_UL_PROJECTION_Y_PRODUCT") == 0){
        set_brick_uly(DN, atof(tokenptr)+15);

      // acquisition variables
      } else if (strcmp(tag, "WRS_PATH") == 0){
        path = atoi(tokenptr);
      } else if (strcmp(tag, "WRS_ROW") == 0){
        row = atoi(tokenptr);
      } else if (strcmp(tag, "DATE_ACQUIRED") == 0){
        tokenptr2 = strtok(tokenptr, separator2);
        date.year = atoi(tokenptr2);
        tokenptr2 = strtok(NULL, separator2);
        date.month = atoi(tokenptr2);
        tokenptr2 = strtok(NULL, separator2);
        date.day = atoi(tokenptr2);
        date.doy = md2doy(date.month, date.day);
        date.week = doy2week(date.doy);
        date.ce  = doy2ce(date.doy, date.year);
      } else if (strcmp(tag, "SCENE_CENTER_TIME") == 0){
        date.hh = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        date.mm = atoi(tokenptr);
        tokenptr = strtok(NULL, separator);
        date.ss = atoi(tokenptr);
        date.tz = 0;

      // calibration variables
      } else if (strstr(tag, "RADIANCE_MAXIMUM_") != NULL){
        for (b=0; b<nb; b++) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 1);
      } else if (strstr(tag, "RADIANCE_MINIMUM_") != NULL){
        for (b=0; b<nb; b++) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 2);
      } else if (strstr(tag, "QUANTIZE_CAL_MAX_") != NULL){
        for (b=0; b<nb; b++) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 3);
      } else if (strstr(tag, "QUANTIZE_CAL_MIN_") != NULL){
        for (b=0; b<nb; b++) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 4);
      } else if (strstr(tag, "REFLECTANCE_MULT_") != NULL){
        for (b=0; b<nb; b++){
          if (b != b_temp) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 5);
        }
      } else if (strstr(tag, "REFLECTANCE_ADD_") != NULL){
        for (b=0; b<nb; b++){
          if (b != b_temp) parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b], lid, 6);
        }
      } else if (strstr(tag, "K1_CONSTANT_") != NULL){
        parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b_temp], lid, 7);
      } else if (strstr(tag, "K2_CONSTANT_") != NULL){
        parse_metadata_band(pl2->d_level1, tag, tokenptr, &meta->cal[b_temp], lid, 8);
      }

      // in case tag (key words) is not the first word in a line
      tag = tokenptr;

    }

  }

  fclose(fp);



  set_brick_name(DN, "FORCE Digital Number brick");
  set_brick_open(DN, false);
  set_brick_format(DN, &pl2->gdalopt);
  set_brick_datatype(DN, _DT_USHORT_);
  set_brick_product(DN, "DN_");

  if ((fp_ = GDALOpen(meta->cal[0].fname, GA_ReadOnly)) == NULL){
    printf("unable to open for fetching projection from %s. ", meta->cal[0].fname); return FAILURE;}
  set_brick_proj(DN, (char*)GDALGetProjectionRef(fp_));
  GDALClose(fp_);

  for (b=0; b<nb; b++){
    set_brick_save(DN, b, true);
    set_brick_nodata(DN, b, 0);
    set_brick_date(DN, b, date);
    set_brick_scale(DN, b, 1);
    set_brick_unit(DN, b, "micrometers");
    get_brick_domain(DN, b, domain, NPOW_10);
    set_brick_bandname(DN, b, domain);
    if (b != b_temp){
      set_brick_wavelength(DN, b, wavelength(meta->cal[b].rsr_band));
    } else {
      set_brick_wavelength(DN, b, 11.0); // approximate wavelength for thermal
    }
  }

  set_brick_dirname(DN, pl2->d_temp);
  set_brick_provdir(DN, pl2->d_temp);
  #ifdef CMIX_FAS
  set_brick_dirname(DN, pl2->d_level1);
  #endif
  set_brick_filename(DN, "DIGITAL-NUMBERS");
  set_brick_par(DN, pl2->params->log);


  nchar = snprintf(meta->refsys, NPOW_04, "%03d%03d", path, row);
  if (nchar < 0 || nchar >= NPOW_04){
    printf("Buffer Overflow in assembling WRS-2\n"); return FAILURE;}


  #ifdef FORCE_DEBUG
  print_metadata(meta, nb);
  print_brick_info(DN);
  #endif


  if (check_metadata(pl2, meta, DN) != SUCCESS){
    printf("Unable to read parameter from MTL file!\n"); exit(1);}

  *dn = DN;
  return SUCCESS;
}


/** This function reads the Sentinel-2 metadata
--- pl2:    L2 parameters
--- meta:   metadata
--- dn:     Digital Number brick
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata_sentinel2(par_ll_t *pl2, meta_t *meta, brick_t **dn){
brick_t *DN = NULL;
FILE *fp = NULL;
char metaname[NPOW_10];
char  buffer[NPOW_13] = "\0";
char *tag = NULL;
char *tokenptr = NULL;
char *tokenptr2 = NULL;
char *tokenptr3 = NULL;
char *separator = " =\"<>";
char *separator2 = "-";
char *separator3 = "-T:.";
char d_top_1[NPOW_10];
char d_top_2[NPOW_10];
int b, nb;
int i = 0, j = 0, ii, jj;
int left, right;
int top, bottom;
int sv_nx = 0, sv_ny = 0;
int d = 0, nd;
float *s_sz = NULL, *s_sa = NULL;
float ***s_vz = NULL, ***s_va = NULL;
bool s = false, v = false, z = false, a = false;
char d_img[NPOW_10];
char id_img[NPOW_10];
char sensor[NPOW_04];
date_t date;
GDALDatasetH fp_;
int svgrid = 5000;
char domain[NPOW_10];
const char domains[13][NPOW_10] = {
  "ULTRABLUE", "BLUE", "GREEN", "RED", 
  "REDEDGE1", "REDEDGE2", "REDEDGE3", 
  "BROADNIR", "NIR", "WVP", "CIRRUS", 
  "SWIR1","SWIR2"
};
const char band_id[13][NPOW_03] = {
  "01", "02", "03", "04", "05", "06", 
  "07", "08", "8A", "09", "10", "11", "12", 
};


  /** initialize **/

  #ifdef FORCE_DEBUG
  printf("reading Sentinel-2 metadata\n");
  #endif

  meta->dtype = 16;
  meta->sat = 65535;


  nb = 13; // number of bands
  nd = 12; // number of detectors
  DN = allocate_brick(nb, 0, _DT_NONE_);
  set_brick_nprovenance(DN, 1);
  set_brick_provenance(DN, 0, pl2->d_level1);

  set_brick_res(DN, INT_MAX);
  set_brick_ncols(DN, 1);
  set_brick_nrows(DN, 1);

  meta->cal = allocate_calibration(nb);

  // re-init to 0 to be compatible with older baselines
  for (b=0; b<nb; b++) meta->cal[b].radd = 0;


  /** parse top-level xml **/
  directoryname(pl2->d_level1, d_top_1, NPOW_10);
  directoryname(d_top_1, d_top_2, NPOW_10);

  // scan directory for xml file
  if (findfile(d_top_2, "S2A", ".xml", metaname, NPOW_10) == FAILURE &&
      findfile(d_top_2, "MTD", ".xml", metaname, NPOW_10) == FAILURE){
    printf("Finding top-level S2 metadata file failed. ");
    return FAILURE;
  }

  #ifdef FORCE_DEBUG
  printf("top-level metadata: %s\n", metaname);
  #endif


  // open xml
  if ((fp = fopen(metaname, "r")) == NULL){
    printf("Unable to open S2 metadata file. "); return FAILURE;
  }

  // process line by line
  while (fgets(buffer, NPOW_13, fp) != NULL){

    // get tag
    tokenptr = strtok(buffer, separator);
    tag=tokenptr;

    // extract parameters by comparing tag
    while (tokenptr != NULL){

      tokenptr = strtok(NULL, separator);


      // Sentinel ID
      if (strcmp(tag, "SPACECRAFT_NAME") == 0){

        tokenptr2 = strtok(tokenptr, separator2);
        tokenptr2 = strtok(NULL, separator2);

        if ((atoi(tokenptr2+0)) != 2){
          printf("unknown/unsupported sensor ID! "); return FAILURE;
        }

        concat_string_2(sensor, NPOW_04, "SEN", tokenptr2, "");
        concat_string_2(id_img, NPOW_10, pl2->d_level1, "IMG_DATA", "/");

        // get filename
        for (b=0; b<nb; b++){

          set_brick_sensor(DN, b, sensor);

          copy_string(meta->cal[b].orig_band, NPOW_03, band_id[b]);
          set_brick_domain(DN, b, domains[b]);

          concat_string_3(id_img, NPOW_10, "_B", meta->cal[b].orig_band, ".jp2", "");

          if (findfile(d_img, id_img, NULL, meta->cal[b].fname, NPOW_10) == FAILURE){
            printf("Unable to find image %s. ", id_img); return FAILURE;}

        }


        if (strlen(pl2->b_level1) > 50){ // old, long naming convention
          strncpy(meta->refsys, pl2->b_level1+49, 6);
        } else { // new, compact naming convention
          strncpy(meta->refsys, pl2->b_level1+4, 6);
        }
        meta->refsys[6] = '\0';

      // product type
      } else if (strcmp(tag, "PROCESSING_LEVEL") == 0){
        if (strstr(tokenptr, "1C") == NULL){
          printf("Unknown processing level. "); return FAILURE;
        } else {
          meta->tier = 1;
        }

      // acquisition variables
      } else if (strcmp(tag, "PRODUCT_START_TIME") == 0){
        tokenptr3 = strtok(tokenptr, separator3);
        date.year = atoi(tokenptr3);
        tokenptr3 = strtok(NULL, separator3);
        date.month = atoi(tokenptr3);
        tokenptr3 = strtok(NULL, separator3);
        date.day = atoi(tokenptr3);
        date.doy = md2doy(date.month, date.day);
        tokenptr3 = strtok(NULL, separator3);
        date.hh = atoi(tokenptr3);
        tokenptr3 = strtok(NULL, separator3);
        date.mm = atoi(tokenptr3);
        tokenptr3 = strtok(NULL, separator3);
        date.ss = atoi(tokenptr3);
        date.tz = 0;
        date.week = doy2week(date.doy);
        date.ce  = doy2ce(date.doy, date.year);

      // scaling factor
      } else if (strcmp(tag, "QUANTIFICATION_VALUE") == 0){
        for (b=0; b<nb; b++) set_brick_scale(DN, b, atoi(tokenptr));
        while (strcmp(tokenptr, "/QUANTIFICATION_VALUE") != 0){
          if (atoi(tokenptr) != 0){
            for (b=0; b<nb; b++) set_brick_scale(DN, b, atoi(tokenptr));
          }
          tag = tokenptr;
          tokenptr = strtok(NULL, separator);
        }

      // additive scaling factor (since baseline 4.0)
      } else if (strcmp(tag, "RADIO_ADD_OFFSET") == 0){
        b = -1;
        while (strcmp(tokenptr, "/RADIO_ADD_OFFSET") != 0){
          if (b < 0){
            char_to_int(tokenptr, &b);
          } else if (b >= 0 || b < nb){
            char_to_float(tokenptr, &meta->cal[b].radd);
          }
          tag = tokenptr;
          tokenptr = strtok(NULL, separator);
        }
        #ifdef FORCE_DEBUG
        if (b >= 0) printf("additive scaling factor for band %d: %f\n", b, meta->cal[b].radd);
        #endif
      }

      // in case tag (key words) is not the first word in a line
      tag = tokenptr;

    }

  }

  fclose(fp);


  /** parse granule xml **/

  // scan directory for xml file
  if (findfile(pl2->d_level1, "S2A", ".xml", metaname, NPOW_10) == FAILURE &&
      findfile(pl2->d_level1, "MTD", ".xml", metaname, NPOW_10) == FAILURE){
    printf("Finding granule metadata file failed. "); return FAILURE;
  }

  #ifdef FORCE_DEBUG
  printf("granule-level metadata: %s\n", metaname);
  #endif

  // open xml
  if ((fp = fopen(metaname, "r")) == NULL){
    printf("Unable to open metadata file. "); return FAILURE;
  }

  // process line by line
  while (fgets(buffer, NPOW_13, fp) != NULL){

    // get tag
    tokenptr = strtok(buffer, separator);
    tag=tokenptr;

    // extract parameters by comparing tag
    while (tokenptr != NULL){

      tokenptr = strtok(NULL, separator);

      // nx/ny/res of highest resolution bands
      if (strcmp(tag, "resolution") == 0){
        if (atoi(tokenptr) < get_brick_res(DN)){
          set_brick_res(DN, atoi(tokenptr));
        }
      } else if (strcmp(tag, "NROWS") == 0){
        if (atoi(tokenptr) > get_brick_nrows(DN)) set_brick_nrows(DN, atoi(tokenptr));
      } else if (strcmp(tag, "NCOLS") == 0){
        if (atoi(tokenptr) > get_brick_ncols(DN)) set_brick_ncols(DN, atoi(tokenptr));
      } else if (strcmp(tag, "ULX") == 0){
        set_brick_ulx(DN, atof(tokenptr));
      } else if (strcmp(tag, "ULY") == 0){
        set_brick_uly(DN, atof(tokenptr));

      // sun/view grids
      } else if (strcmp(tag, "COL_STEP") == 0){
        while (atoi(tokenptr) == 0) tokenptr = strtok(NULL, separator);

        if (svgrid != atoi(tokenptr)){
          printf("SUN_VIEW_GRID is incompatible with Sentinel-2 metadata. "); return FAILURE;}

        sv_nx = ceil(get_brick_width(DN)/(float)svgrid) + 1;
        sv_ny = ceil(get_brick_height(DN)/(float)svgrid) + 1;
        if (s_sz == NULL) alloc((void**)&s_sz, sv_ny*sv_nx, sizeof(float));
        if (s_sa == NULL) alloc((void**)&s_sa, sv_ny*sv_nx, sizeof(float));
        if (s_vz == NULL) alloc_3D((void****)&s_vz, nb, nd, sv_ny*sv_nx, sizeof(float));
        if (s_va == NULL) alloc_3D((void****)&s_va, nb, nd, sv_ny*sv_nx, sizeof(float));
        //if (k_vz == NULL) alloc_2D((void***)&k_vz, sv_ny, sv_nx, sizeof(int));
        //if (k_va == NULL) alloc_2D((void***)&k_va, sv_ny, sv_nx, sizeof(int));
      } else if (strcmp(tag, "Sun_Angles_Grid") == 0){
        s = true;
      } else if (strcmp(tag, "/Sun_Angles_Grid") == 0){
        s = false;
      } else if (strcmp(tag, "Viewing_Incidence_Angles_Grids") == 0){
        v = true;
        b = 0;
        d = 0;
        while (tokenptr != NULL){
          if (strcmp(tokenptr, "bandId") == 0){
            tokenptr = strtok(NULL, separator);
            b = atoi(tokenptr);
          }
          if (strcmp(tokenptr, "detectorId") == 0){
            tokenptr = strtok(NULL, separator);
            d = atoi(tokenptr) - 1;
          }
          tokenptr = strtok(NULL, separator);
        }
      } else if (strcmp(tag, "/Viewing_Incidence_Angles_Grids") == 0){
        v = false;
      } else if (strcmp(tag, "Zenith") == 0){
        z = true; i = 0;
      } else if (strcmp(tag, "/Zenith") == 0){
        z = false;
      } else if (strcmp(tag, "Azimuth") == 0){
        a = true; i = 0;
      } else if (strcmp(tag, "/Azimuth") == 0){
        a = false;
      } else if (strcmp(tag, "VALUES") == 0 && i < sv_ny){

        j = 0;

        while (strcmp(tokenptr, "/VALUES") != 0 && j < sv_nx){

          if (s){
            if (z){
              if (strcmp(tokenptr, "NaN") !=0) s_sz[i*sv_ny+j] = atof(tokenptr);
            } else if (a){
              if (strcmp(tokenptr, "NaN") !=0) s_sa[i*sv_ny+j] = atof(tokenptr);
            }
          } else if (v){
            if (z){
              if (strcmp(tokenptr, "NaN") !=0){
                s_vz[b][d][i*sv_ny+j] += atof(tokenptr);
              }
            } else if (a){
              if (strcmp(tokenptr, "NaN") !=0){
                s_va[b][d][i*sv_ny+j] += atof(tokenptr);
              }
            }
          }

          tokenptr = strtok(NULL, separator);
          j++;

        }

        i++;

      }

      // in case tag (key words) is not the first word in a line
      tag = tokenptr;
    }

  }

  fclose(fp);


  // interpolate the grids
  // angles are given at grid intersections, we need one value per cell
  interpolate_sunview_grid(s_sz, sv_nx, sv_ny, meta->s2.nodata);
  interpolate_sunview_grid(s_sa, sv_nx, sv_ny, meta->s2.nodata);
  for (b=0; b<nb; b++){
    for (d=0; d<nd; d++){
      interpolate_sunview_grid(s_vz[b][d], sv_nx, sv_ny, meta->s2.nodata);
      interpolate_sunview_grid(s_va[b][d], sv_nx, sv_ny, meta->s2.nodata);
    }
  }
  sv_nx--;
  sv_ny--;

  // collapse view grids
  collapse_view_grid(s_vz, nb, nd, sv_nx, sv_ny, meta->s2.nodata);
  collapse_view_grid(s_va, nb, nd, sv_nx, sv_ny, meta->s2.nodata);


  // get image subset
  left = sv_nx-1; right  = 0;
  top  = sv_ny-1; bottom = 0;

  for (i=0; i<sv_ny; i++){
  for (j=0; j<sv_nx; j++){

    if (s_vz[0][0][i*sv_ny+j] != meta->s2.nodata && 
        s_va[0][0][i*sv_ny+j] != meta->s2.nodata){
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

  while (fmod(left*svgrid, 60) != 0 && left > 0) left--;
  while (fmod(top*svgrid,  60) != 0 && top  > 0) top--;
  //while (fmod(right*svgrid,  60) != 0 && right  < (sv_nx-2)) right++;
  //while (fmod(bottom*svgrid, 60) != 0 && bottom < (sv_nx-2)) bottom++;
  while (fmod(right*svgrid,  60) != 0 && right  < (sv_nx-1)) right++;
  while (fmod(bottom*svgrid, 60) != 0 && bottom < (sv_nx-1)) bottom++;

  if (!pl2->doreproj && !pl2->dotile){
    right  = sv_nx; left = 0;
    bottom = sv_ny; top  = 0;
  }

  //meta->s2.nx = right-left+1;
  //meta->s2.ny = bottom-top+1;
  meta->s2.nx = right-left;
  meta->s2.ny = bottom-top;

  if (meta->s2.nx <= 0 || meta->s2.ny <= 0){
    printf("no valid cell after subsetting. Abort.\n");
    free_brick(DN);
    exit(SUCCESS);
  }

  if (meta->s2.szen == NULL) alloc_2D((void***)&meta->s2.szen, meta->s2.ny, meta->s2.nx, sizeof(float));
  if (meta->s2.vzen == NULL) alloc_2D((void***)&meta->s2.vzen, meta->s2.ny, meta->s2.nx, sizeof(float));
  if (meta->s2.sazi == NULL) alloc_2D((void***)&meta->s2.sazi, meta->s2.ny, meta->s2.nx, sizeof(float));
  if (meta->s2.vazi == NULL) alloc_2D((void***)&meta->s2.vazi, meta->s2.ny, meta->s2.nx, sizeof(float));


  // copy values to final view sun grids
  for (i=0; i<meta->s2.ny; i++){
  for (j=0; j<meta->s2.nx; j++){

    ii = i+top;
    jj = j+left;

    meta->s2.szen[i][j] = s_sz[ii*sv_ny+jj];
    meta->s2.sazi[i][j] = s_sa[ii*sv_ny+jj];
    meta->s2.vzen[i][j] = s_vz[0][0][ii*sv_ny+jj];
    meta->s2.vazi[i][j] = s_va[0][0][ii*sv_ny+jj];


  }
  }


  meta->s2.left   = left   * svgrid/get_brick_res(DN);
  meta->s2.top    = top    * svgrid/get_brick_res(DN);
  meta->s2.right  = right  * svgrid/get_brick_res(DN);
  meta->s2.bottom = bottom * svgrid/get_brick_res(DN);

  if (meta->s2.right  > get_brick_ncols(DN)) meta->s2.right  = get_brick_ncols(DN);
  if (meta->s2.bottom > get_brick_nrows(DN)) meta->s2.bottom = get_brick_nrows(DN);

  free((void*)s_sz);
  free((void*)s_sa);
  free_3D((void***)s_vz, nb, nd);
  free_3D((void***)s_va, nb, nd);

  #ifdef FORCE_DEBUG
  printf("active image subset: UL %d/%d to LR %d/%d\n", meta->s2.left, meta->s2.top, meta->s2.right, meta->s2.bottom);
  printf("coarse cells: nx/ny %d/%d\n", meta->s2.nx, meta->s2.ny);
  #endif

  set_brick_ncols(DN, meta->s2.right-meta->s2.left);
  set_brick_nrows(DN, meta->s2.bottom-meta->s2.top);

  set_brick_ulx(DN, get_brick_x(DN, meta->s2.left));
  set_brick_uly(DN, get_brick_y(DN, meta->s2.top));

  set_brick_name(DN, "FORCE Digital Number brick");
  set_brick_open(DN, false);
  set_brick_format(DN, &pl2->gdalopt);
  set_brick_datatype(DN, _DT_USHORT_);
  set_brick_product(DN, "DN_");

  if ((fp_ = GDALOpen(meta->cal[0].fname, GA_ReadOnly)) == NULL){
    printf("unable to open %s. ", meta->cal[0].fname); return FAILURE;}
  set_brick_proj(DN, (char*)GDALGetProjectionRef(fp_));
  GDALClose(fp_);

  for (b=0; b<nb; b++){
    set_brick_save(DN, b, true);
    set_brick_nodata(DN, b, 0);
    set_brick_date(DN, b, date);
    set_brick_unit(DN, b, "micrometers");
    get_brick_domain(DN, b, domain, NPOW_10);
    set_brick_bandname(DN, b, domain);
    set_brick_wavelength(DN, b, wavelength(meta->cal[b].rsr_band));
  }

  set_brick_dirname(DN, pl2->d_temp);
  set_brick_provdir(DN, pl2->d_temp);
  #ifdef CMIX_FAS
  set_brick_dirname(DN, pl2->d_level1);
  #endif
  set_brick_filename(DN, "DIGITAL-NUMBERS");
  set_brick_par(DN, pl2->params->log);


  #ifdef FORCE_DEBUG
  print_metadata(meta, nb);
  print_brick_info(DN);
  #endif

  #ifdef FORCE_DEV
  printf("\nwarning: check_metadata disabled.\n");
  #endif
//  if (check_metadata(pl2, meta, DN) != SUCCESS){
//    printf("Unable to read parameter from MTL file!\n"); exit(1);}

  *dn = DN;
  return SUCCESS;
}


/** This function reads band-specific Landsat metadata like filenames and
+++ calibration parameters
--- d_level1: image directory
--- tag:      metadata tag
--- value:    metadata value
--- cal:      calibration coefficients
--- lid:      Landsat ID
--- type:     metadata type
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void parse_metadata_band(char *d_level1, char *tag, char *value, cal_t *cal, int lid, int type){
char string[NPOW_10], suffix[NPOW_10];


  if (lid == 7 && strcmp(cal->orig_band, "6") == 0){
    copy_string(suffix, NPOW_10, "_VCID_1");
  } else {
    suffix[0] = '\0';
  }

  if (type == 0){
    if (strcmp(cal->fname, "NULL") == 0){
      concat_string_3(string, NPOW_10, "FILE_NAME_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) concat_string_2(cal->fname, NPOW_10, d_level1, value, "/");
    }
  } else if (type == 1){
    if (cal->lmax ==  cal->fill){
      concat_string_3(string, NPOW_10, "RADIANCE_MAXIMUM_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->lmax = atof(value);
    }
  } else if (type == 2){
    if (cal->lmin == cal->fill){
      concat_string_3(string, NPOW_10, "RADIANCE_MINIMUM_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->lmin = atof(value);
    }
  } else if (type == 3){
    if (cal->qmax == cal->fill){
      concat_string_3(string, NPOW_10, "QUANTIZE_CAL_MAX_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->qmax = atof(value);
    }
  } else if (type == 4){
    if (cal->qmin == cal->fill){
      concat_string_3(string, NPOW_10, "QUANTIZE_CAL_MIN_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->qmin = atof(value);
    }
  } else if (type == 5){
    if (cal->rmul == cal->fill){
      concat_string_3(string, NPOW_10, "REFLECTANCE_MULT_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->rmul = atof(value);
    }
  } else if (type == 6){
    if (cal->radd == cal->fill){
      concat_string_3(string, NPOW_10, "REFLECTANCE_ADD_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->radd = atof(value);
    }
  } else if (type == 7){
    if (cal->k1 == cal->fill){
      concat_string_3(string, NPOW_10, "K1_CONSTANT_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->k1 = atof(value);
    }
  } else if (type == 8){
    if (cal->k2 == cal->fill){
      concat_string_3(string, NPOW_10, "K2_CONSTANT_BAND_", cal->orig_band, suffix, "");
      if (strcmp(tag, string) == 0) cal->k2 = atof(value);
    }
  }

  return;
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
void interpolate_sunview_grid(float *int_grid, int int_nx, int int_ny, float nodata){
int i, j, ii, jj;
int cell_nx, cell_ny;
int cell_p, int_p;
float *cell_grid = NULL;
float sum, num;


  cell_nx = int_nx - 1;
  cell_ny = int_ny - 1;

  alloc((void**)&cell_grid, cell_ny*cell_nx, sizeof(float));


  for (i=0, cell_p=0; i<cell_ny; i++){
  for (j=0; j<cell_nx; j++, cell_p++){

    sum = num = 0;

    for (ii=0; ii<=1; ii++){
    for (jj=0; jj<=1; jj++){

      int_p = (i+ii) * int_ny + (j+jj);

      if (int_grid[int_p] > 0){
        sum += int_grid[int_p];
        num++;
      }
    }
    }

    if (num > 0){
      cell_grid[cell_p] = sum/num;
    } else {
      cell_grid[cell_p] = nodata;
    }

  }
  }

  memmove(int_grid, cell_grid, cell_nx*cell_ny*sizeof(float));
  re_alloc((void**)&int_grid, int_nx*int_ny, cell_nx*cell_ny, sizeof(float));
  free((void*)cell_grid);

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
void collapse_view_grid(float ***grid, int nb, int nd, int nx, int ny, float nodata){
int b, d, i, j, p;
float sum, num;


  for (i=0, p=0; i<ny; i++){
  for (j=0; j<nx; j++, p++){

    sum = num = 0;

    for (b=0; b<nb; b++){
    for (d=0; d<nd; d++){

      if (grid[b][d][p] != nodata){
        sum += grid[b][d][p];
        num++;
      }

    }
    }

    if (num > 0){
      grid[0][0][p] = sum/num;
    } else {
      grid[0][0][p] = nodata;
    }

  }
  }
  

  return;
}



/** This function identifies the satellite mission, i.e. Landsat or Senti-
+++ nel-2
--- pl2:    L2 parameters
+++ Return: mission
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata_mission(par_ll_t *pl2){
int mission;
char metaname[NPOW_10];

  if (findfile(pl2->d_level1, "MTL", ".txt", metaname, NPOW_10) == SUCCESS){
    mission = LANDSAT;
    pl2->res = pl2->res_landsat;
  } else if (findfile(pl2->d_level1, "S2A", ".xml", metaname, NPOW_10) == SUCCESS ||
             findfile(pl2->d_level1, "MTD", ".xml", metaname, NPOW_10) == SUCCESS){
    mission = SENTINEL2;
    pl2->res = pl2->res_sentinel2;
  } else {
    printf("unknown Satellite Mission. "); return _UNKNOWN_;
  }

  #ifdef FORCE_DEBUG
  printf("\nMission: %d\n", mission);
  #endif

  return mission;
}


/** This function reads the metadata
--- pl2:      L2 parameters
--- metadata: metadata
--- dn:       Digital Number brick
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_metadata(par_ll_t *pl2, meta_t **metadata, brick_t **DN, int *mission){
int mission_id;
meta_t *meta = NULL;

  meta = allocate_metadata();
#ifdef FORCE_DEBUG
printf("there are still some things to do int meta. checking etc\n");
#endif
  if ((mission_id = parse_metadata_mission(pl2)) == _UNKNOWN_) return FAILURE;

  switch (mission_id){
    case LANDSAT:
      if (parse_metadata_landsat(pl2, meta, DN)  != SUCCESS) return FAILURE;
      break;
    case SENTINEL2:
      if (parse_metadata_sentinel2(pl2, meta, DN) != SUCCESS) return FAILURE;
      break;
    default:
      printf("unknown mission. ");
      return FAILURE;
  }

  *metadata = meta;
  *mission = mission_id;
  return SUCCESS;
}

