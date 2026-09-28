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
This file contains functions for reading Level 1 data
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-ll.h"



int init_level1(par_ll_t *pl2, rtd_t *rtd, meta_t *meta, brick_t **dn){

  if (pl2 == NULL || rtd == NULL || meta == NULL || dn == NULL || *dn != NULL){
    RETURN_ERROR("Invalid input.");
  }


  /** initialize Digital Number brick
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  brick_t *DN = NULL;

  DN = allocate_brick(meta->band_number, 0, _DT_NONE_);
  set_brick_dirname(DN, pl2->d_temp);
  set_brick_provdir(DN, pl2->d_temp);
  set_brick_filename(DN, "DIGITAL-NUMBERS");
  set_brick_name(DN, "FORCE Digital Number brick");
  set_brick_open(DN, false);
  set_brick_product(DN, "DN_");
  set_brick_par(DN, pl2->params->log);
  set_brick_format(DN, &pl2->gdalopt);
  set_brick_datatype(DN, _DT_USHORT_);

  set_brick_nprovenance(DN, 1);
  set_brick_provenance(DN, 0, pl2->d_level1);

  set_brick_res(DN, meta->res);
  set_brick_nrows(DN, meta->nrow);
  set_brick_ncols(DN, meta->ncol);
  set_brick_ulx(DN, meta->ulx);
  set_brick_uly(DN, meta->uly);

  char wkt[NPOW_10];
  epsg_to_wkt(meta->epsg, wkt);
  set_brick_proj(DN, wkt);

  for (int b=0; b<meta->band_number; b++){

    set_brick_sensor(DN, b, rtd->sensor_mapping.l2_sensor);

    set_brick_save(DN, b, true);
    set_brick_nodata(DN, b, meta->nodata);
    set_brick_date(DN, b, meta->date);
    set_brick_unit(DN, b, "micrometers");

    set_brick_domain(DN, b, rtd->band_mapping.domains[b]);

    char bandname[NPOW_10];
    concat_string_2(bandname, NPOW_10, 
      rtd->band_mapping.domains[b], rtd->band_mapping.l1_bands[b], " - B");
    set_brick_bandname(DN, b, bandname);

    set_brick_wavelength(DN, b, meta->wavelength[b] / 1000.0);

  }

  #ifdef CMIX_FAS
  set_brick_dirname(DN, pl2->d_level1);
  #endif

  #ifdef FORCE_DEBUG
  print_brick_info(DN);
  #endif

  *dn = DN;
  return SUCCESS;
}


/** This function reads all necessary or available Level 1 data
--- meta:    metadata
--- DN:      Digital Numbers
--- pl2:     L2 parameters
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_level1(meta_t *meta, brick_t *DN, par_ll_t *pl2){


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  int nb  = get_brick_nbands(DN);
  int nx  = get_brick_ncols(DN);
  int ny  = get_brick_nrows(DN);
  int nc  = get_brick_ncells(DN);
  double res = get_brick_res(DN);
  allocate_brick_bands(DN, nb, nc, _DT_USHORT_);

  ushort **dn_ = NULL;
  if ((dn_ = get_bands_ushort(DN)) == NULL) return FAILURE;

  CPLSetConfigOption("GDAL_PAM_ENABLED", "NO");
  CPLSetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS");
  //CPLPushErrorHandler(CPLQuietErrorHandler);

  int threads;

  if (pl2->ithread){
    if (pl2->nthread > nb){
      threads = nb;
    } else {
      threads = pl2->nthread;
    }
  } else {
    threads = 1;
  }

  int error = 0;

  #pragma omp parallel num_threads(threads) shared(dn_,nb,meta,nx,ny,res) reduction(+: error) default(none)
  {
 
    #pragma omp for
    for (int b=0; b<nb; b++){


      GDALDatasetH dataset;
      if ((dataset = GDALOpen(meta->image_path.string[b], GA_ReadOnly)) == NULL){
        printf("unable to open %s", meta->image_path.string[b]); 
        error++;
        continue;
      }// else {
        //CPLPopErrorHandler();
      //}
      
      #ifdef FORCE_DEBUG
      GDALDriverH driver = GDALGetDatasetDriver(dataset);
      printf("Driver: %s/%s\n", GDALGetDriverShortName(driver), GDALGetDriverLongName(driver));
      #endif

      // get image resolution
      double geotran[_GT_LEN_];
      GDALGetGeoTransform(dataset, geotran);
      double res_image = geotran[_GT_RES_];

      int xoff_disc_access = floor(meta->col_offset*res/res_image);
      int yoff_disc_access = floor(meta->row_offset*res/res_image);
      int nx_disc_access = floor(nx*res/res_image);
      int ny_disc_access = floor(ny*res/res_image);

      #ifdef FORCE_DEBUG
      printf("reading %d/%d pixels with offset %d/%d into buffer with %d/%d pixels\n", 
        nx_disc_access, ny_disc_access, xoff_disc_access, yoff_disc_access, nx, ny);
      #endif


      GDALRasterBandH band = GDALGetRasterBand(dataset, 1);
      if (GDALRasterIO(band, GF_Read, xoff_disc_access, yoff_disc_access, 
        nx_disc_access, ny_disc_access, dn_[b], 
        nx, ny, GDT_UInt16, 0, 0) == CE_Failure){
        printf("could not read %s. ", meta->image_path.string[b]); 
        error++;
        GDALClose(dataset);
        continue;
      }

      GDALClose(dataset);

    }

  }
  
  if (error > 0){
    printf("reading error. "); return FAILURE;}


  #ifdef FORCE_DEBUG
  print_brick_info(DN); set_brick_open(DN, OPEN_CREATE); write_brick(DN);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("read Level 1", TIME);
  #endif

  return SUCCESS;
}


/** This function detects extreme values and builds the nodata and satu-
+++ ration masks.
--- meta:   metadata
--- DN:     digital numbers
--- QAI:    Quality Assurance Information
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int bounds_level1(meta_t *meta, brick_t *DN, brick_t **QAI, par_ll_t *pl2){
int b, b_temp, b_cirrus, nb, p, nx, ny, nc;
brick_t *qai  = NULL;
ushort **dn_  = NULL;
small   *off_ = NULL;
int tvalid = 0;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  nb = get_brick_nbands(DN);
  nx = get_brick_ncols(DN);
  ny = get_brick_nrows(DN);
  nc = get_brick_ncells(DN);
  b_temp   = find_domain(DN,  "TEMP");
  b_cirrus = find_domain(DN,  "CIRRUS");

  // initialize a brick with general metadata
  qai = copy_brick(DN, 1, _DT_SHORT_);

  // set brick metadata
  set_brick_name(qai, "FORCE QAI brick");
  set_brick_product(qai, "QAI");
  set_brick_filename(qai, "QAI");
  set_brick_nodata(qai, 0, 1); 
  set_brick_wavelength(qai, 0, 1);
  set_brick_domain(qai, 0, "QAI");
  set_brick_bandname(qai, 0, "Quality assurance information");


  // get DN and OFF arrays for faster computation
  if ((dn_  = get_bands_ushort(DN))   == NULL) return FAILURE;

  alloc((void**)&off_, nc, sizeof(small));


  #pragma omp parallel private(b) shared(b_temp, b_cirrus, nb, nc, dn_, off_, qai, meta) reduction(+:tvalid) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=0; p<nc; p++){

      for (b=0; b<nb; b++){

        // if any layer void --> boundary
        if (b != b_cirrus && dn_[b][p] == 0){ off_[p] = true; break;}

        // if any (non-temp) layer saturated
        if (b != b_temp && dn_[b][p] >= meta->saturation){ set_saturation(qai, p, true); break;}

        // if temperature has any non-0 value
        if (b == b_temp) tvalid++;

      }
    }
  }


  // buffer one pixel (pixels are somehow contaminated. due to resampling?)
  if (pl2->bufnodata) buffer_(off_, nx, ny, 1);
  for (p=0; p<nc; p++) set_off(qai, p, off_[p]);
  free((void*)off_);

  if (b_temp >= 0 && tvalid == 0){
    printf("zero-filled temperature. "); return FAILURE;}

  if (impulse_noise_level1(meta, DN, qai, pl2) == FAILURE){
    printf("detecting impulse noise failed.\n"); return FAILURE;}


  #ifdef FORCE_DEBUG
  print_brick_info(qai); set_brick_open(qai, OPEN_CREATE); write_brick(qai);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("boundaries Level 1", TIME);
  #endif

  *QAI = qai;
  return SUCCESS;
}


/** This function attempts to detect and mask Impulse Noise, a phenomenon
+++ observed in 8bit Landsat data. The first three bands (RGB) are used
+++ to detect this. Note that IN is not confined to these bands and small
+++ IN won't be detected.. This function simply identifies the worst occu-
+++ rences.
--- meta:   metadata
--- DN:     digital numbers
--- QAI:    Quality Assurance Information
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int impulse_noise_level1(meta_t *meta, brick_t *DN, brick_t *QAI, par_ll_t *pl2){
int k, count = 0, b, bands[3], nb = 3;
int i, j, ii, jj, p, q, nx, ny;
double mx[3],varx[3], sd[3];
double  maxsd, max2sd;
ushort **dn_  = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  // impact noise was not observed yet in 16bit data
  char sensor[NPOW_10]; 
  get_brick_sensor(DN, 0, sensor, NPOW_10);
  if (!strings_equal(sensor, "LND04") ||
      !strings_equal(sensor, "LND05") ||
      !strings_equal(sensor, "LND07")) return SUCCESS;

  if (!pl2->impulse) return SUCCESS;

  nx = get_brick_ncols(DN);
  ny = get_brick_nrows(DN);

  if ((bands[0] = find_domain(DN,  "BLUE"))  < 0){
    printf("no BLUE band available.\n"); return FAILURE;}
  if ((bands[1] = find_domain(DN,  "GREEN")) < 0){
    printf("no GREEN band available.\n"); return FAILURE;}
  if ((bands[2] = find_domain(DN,  "RED"))   < 0){
    printf("no RED band available.\n"); return FAILURE;}


  if ((dn_ = get_bands_ushort(DN)) == NULL) return FAILURE;

  for (i=1; i<(ny-1); i++){
  for (j=1; j<(nx-1); j++){
    
    p = i*nx+j;

    if (get_off(QAI, p)) continue;

    k = 0;
    for (b=0; b<nb; b++) mx[b] = varx[b] = 0;

    for (ii=-1; ii<=1; ii++){
    for (jj=-1; jj<=1; jj++){

      q = nx*(i+ii)+j+jj;

      if (get_off(QAI, q)) continue;

      k++;

      if (k == 1){
        for (b=0; b<nb; b++) mx[b] = dn_[bands[b]][q];
      } else {
        for (b=0; b<nb; b++) var_recurrence(dn_[bands[b]][q], &mx[b], &varx[b], k);
      }

    }
    }


    if (k>0){

      for (b=0; b<nb; b++) sd[b] = standdev(varx[b], k);

      max2sd = maxsd = sd[0];
      for (b=1; b<nb; b++){
        if (sd[b] > maxsd){ max2sd = maxsd; maxsd = sd[b];}
      }
      if ((maxsd-max2sd) > 15){ set_off(QAI, p, true); count++;}

    }


  }
  }


  #ifdef FORCE_CLOCK
  proctime_print("Impulse Noise Level 1", TIME);
  #endif

  return SUCCESS;
}


/** This function converts all the DN bands to TOA reflectance or bright-
+++ ness Temperature. In case of Sentinel-2, TOA reflectance is first re-
+++ transformed to DNs before it is converted to TOA reflectance again.
+++ This is done to maintain a constant calibration between sensors and
+++ to retain the flexibility to e.g. use another E0 spectrum.
--- meta:    metadata
--- atc:     atmospheric correction factors
--- DN:      digital numbers
--- TOA:     Top of Atmosphere reflectance and temperature
--- QAI:     Quality Assurance Information
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_level1(meta_t *meta, atc_t *atc, brick_t *DN, brick_t **toa, brick_t *QAI){
brick_t  *TOA  = NULL;
ushort **dn_  = NULL;
short  **toa_ = NULL;
float    *sun_ = NULL;
int b, b_temp, b_cirrus, nb, nc, p, g;
short nodata;
float A, rad, dsun, pi_dsun2, tmp;
float toa_scale;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nb = get_brick_nbands(DN);
  nc = get_brick_ncells(DN);
  nodata = _FORCE_NO_DATA_;

  TOA = copy_brick(DN, nb, _DT_SHORT_);
  
  // temperature band?
  b_temp   = find_domain(TOA, "TEMP");
  b_cirrus = find_domain(DN,  "CIRRUS");

  // update metadata
  set_brick_name(TOA, "FORCE TOA brick");
  set_brick_product(TOA, "TOA");
  set_brick_filename(TOA, "TOA");

  for (b=0; b<nb; b++){

    set_brick_nodata(TOA, b, nodata);
    if (b != b_temp){
      set_brick_scale(TOA, b, 10000);
    } else {
      set_brick_scale(TOA, b, 100);
    }

  }


  // get brick arrays for faster computation
  if ((dn_  = get_bands_ushort(DN)) == NULL) return FAILURE;
  if ((toa_ = get_bands_short(TOA)) == NULL) return FAILURE;
  if ((sun_ = get_band_float(atc->xy_sun, cZEN)) == NULL) return FAILURE;


  /** TOA reflectance to TOA reflectance (Sentinel-2) 
  in early processing versions, scale factor was 1000, now 10000 **/
  if (meta->mission == SENTINEL2){

    for (b=0; b<nb; b++){

      toa_scale = get_brick_scale(TOA, b);
      
      #pragma omp parallel shared(b, nc, toa_scale, dn_, toa_, meta, QAI, nodata) default(none) 
      {

        #pragma omp for schedule(static)
        for (p=0; p<nc; p++){
          if (get_off(QAI, p)){ 
            toa_[b][p] = nodata; 
          } else {
            toa_[b][p] = (dn_[b][p] + meta->cal[b].reflectance.offset) / 
            meta->cal[b].reflectance.scale*toa_scale;
          }
        }
        
      }

    }

  /** digital numbers to TOA reflectance and brightness temperature (Landsat) **/
  } else {

    dsun = doy2dsun(get_brick_doy(TOA, 0));
    pi_dsun2  = M_PI*dsun*dsun;

    for (b=0; b<nb; b++){

      toa_scale = get_brick_scale(TOA, b);

      A = (meta->cal[b].radiance.lmax-meta->cal[b].radiance.lmin) / 
          (meta->cal[b].radiance.qmax-meta->cal[b].radiance.qmin);

      #pragma omp parallel private(rad, tmp, g) shared(b, b_temp, b_cirrus, nc, nodata, toa_scale, dn_, toa_, sun_, QAI, A, pi_dsun2, meta, atc) default(none) 
      {

        #pragma omp for schedule(guided)
        for (p=0; p<nc; p++){

          if (get_off(QAI, p)){ toa_[b][p] = nodata; continue;}

          // dn to radiance
          rad = A * (dn_[b][p]-meta->cal[b].radiance.qmin) + meta->cal[b].radiance.lmin;

          // radiance to brightness temperature in kelvin
          if (b == b_temp){

            tmp = meta->cal[b].temperature.k2/log((meta->cal[b].temperature.k1/rad)+1)*toa_scale;
            if (tmp < SHRT_MAX){
              toa_[b][p] = (short)tmp;
            } else {
              toa_[b][p] = SHRT_MAX;
            }

          // radiance to reflectance
          } else {

            g = convert_brick_p2p(QAI, atc->xy_sun, p);

            if (meta->cal[b].reflectance.scale <= 0){
              // old-style DN -> radiance -> reflectance conversion (should not happen anymore)
              tmp = rad*pi_dsun2 / (atc->E0[b]*sun_[g]);
            } else {
              // new DN -> reflectance conversion
              tmp = (meta->cal[b].reflectance.offset + meta->cal[b].reflectance.scale*dn_[b][p]) / sun_[g];
            }

            if (tmp < FLT_MIN){
              if (b == b_cirrus){
                toa_[b][p] = (short)0;
              } else {
                toa_[b][p] = (short)nodata;
                set_off(QAI, p, true);
              }
            } else {
              toa_[b][p] = (short)(tmp*toa_scale);
            }

          }

        }
        
      }

    }

  }


  #ifdef FORCE_DEBUG
  print_brick_info(TOA); set_brick_open(TOA, OPEN_CREATE); write_brick(TOA);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("Level 1 to TOA conversion", TIME);
  #endif

  *toa = TOA;
  return SUCCESS;
}

