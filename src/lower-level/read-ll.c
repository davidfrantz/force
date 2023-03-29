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


/** This function reads all necessary or available Level 1 data
--- meta:    metadata
--- mission: mission ID
--- DN:      Digital Numbers
--- pl2:     L2 parameters
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_level1(meta_t *meta, int mission, brick_t *DN, par_ll_t *pl2){
int b, nb, nx, ny, nc;
int nx_, ny_, xoff_ = 0, yoff_ = 0;
float res, res_;
double geotran[6];
ushort **dn_ = NULL;
GDALDatasetH dataset;
GDALRasterBandH band;
int error = 0;
int threads;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nb  = get_brick_nbands(DN);
  nx  = get_brick_ncols(DN);
  ny  = get_brick_nrows(DN);
  nc  = get_brick_ncells(DN);
  res = get_brick_res(DN);
  allocate_brick_bands(DN, nb, nc, _DT_USHORT_);
  if ((dn_ = get_bands_ushort(DN)) == NULL) return FAILURE;

  CPLSetConfigOption("GDAL_PAM_ENABLED", "NO");
  CPLSetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS");
  //CPLPushErrorHandler(CPLQuietErrorHandler);


  if (pl2->ithread){
    if (pl2->nthread > nb){
      threads = nb;
    } else {
      threads = pl2->nthread;
    }
  } else {
    threads = 1;
  }


  #pragma omp parallel  num_threads(threads) private(dataset,band,nx_,ny_,geotran,res_) firstprivate(xoff_,yoff_) shared(dn_,nb,meta,mission,nx,ny,res) reduction(+: error) default(none)
  {
 
    #pragma omp for
    for (b=0; b<nb; b++){

      if ((dataset = GDALOpen(meta->cal[b].fname, GA_ReadOnly)) == NULL){
        printf("unable to open %s. ", meta->cal[b].fname); error++;
      }// else {
        //CPLPopErrorHandler();
      //}
      
      #ifdef FORCE_DEBUG
      GDALDriverH driver = GDALGetDatasetDriver(dataset);
      printf("Driver: %s/%s\n", GDALGetDriverShortName(driver), GDALGetDriverLongName(driver));
      #endif

      // get number of pixels, GDAL handles conversion to nx, ny
      nx_ = GDALGetRasterXSize(dataset);
      ny_ = GDALGetRasterYSize(dataset);
      GDALGetGeoTransform(dataset, geotran);
      res_ = geotran[1];

      if (mission == SENTINEL2){
        xoff_ = floor(meta->s2.left*res/res_);
        yoff_ = floor(meta->s2.top*res/res_);
        nx_ = floor(nx*res/res_);
        ny_ = floor(ny*res/res_);
        #ifdef FORCE_DEBUG
        printf("reading %d/%d pixels with offset %d/%d into buffer with %d/%d pixels\n", 
          nx_, ny_, xoff_, yoff_, nx, ny);
        #endif
      }

      band = GDALGetRasterBand(dataset, 1);
      if (GDALRasterIO(band, GF_Read, xoff_, yoff_, nx_, ny_, dn_[b], 
        nx, ny, GDT_UInt16, 0, 0) == CE_Failure){
        printf("could not read %s. ", meta->cal[b].fname); error++;}

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
        if (b != b_temp && dn_[b][p] >= meta->sat){ set_saturation(qai, p, true); break;}

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
  if (meta->dtype != 8) return SUCCESS;
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
--- mission: mission ID
--- atc:     atmospheric correction factors
--- DN:      digital numbers
--- TOA:     Top of Atmosphere reflectance and temperature
--- QAI:     Quality Assurance Information
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_level1(meta_t *meta, int mission, atc_t *atc, brick_t *DN, brick_t **toa, brick_t *QAI){
brick_t  *TOA  = NULL;
ushort **dn_  = NULL;
short  **toa_ = NULL;
float    *sun_ = NULL;
int b, b_temp, b_cirrus, nb, nc, p, g;
short nodata;
float A, rad, tmp;
float dn_scale, toa_scale;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nb = get_brick_nbands(DN);
  nc = get_brick_ncells(DN);
  nodata = -9999;

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
  if (mission == SENTINEL2){

    for (b=0; b<nb; b++){
      
      dn_scale  = get_brick_scale(DN,  b);
      toa_scale = get_brick_scale(TOA, b);
      
      #pragma omp parallel shared(b, nc, dn_scale, toa_scale, dn_, toa_, meta) default(none) 
      {

        #pragma omp for schedule(static)
        for (p=0; p<nc; p++) toa_[b][p] = (dn_[b][p] + meta->cal[b].radd) / dn_scale*toa_scale;
        
      }

    }

  /** digital numbers to TOA reflectance and brightness temperature (Landsat) **/
  } else {

    for (b=0; b<nb; b++){

      toa_scale = get_brick_scale(TOA, b);

      A = (meta->cal[b].lmax-meta->cal[b].lmin) / 
          (meta->cal[b].qmax-meta->cal[b].qmin);

      #pragma omp parallel private(rad, tmp, g) shared(b, b_temp, b_cirrus, nc, nodata, toa_scale, dn_, toa_, sun_, QAI, A,  meta, atc) default(none) 
      {

        #pragma omp for schedule(guided)
        for (p=0; p<nc; p++){

          if (get_off(QAI, p)){ toa_[b][p] = nodata; continue;}


          // DN to radiance to brightness temperature in kelvin
          if (b == b_temp){

            rad = A * (dn_[b][p]-meta->cal[b].qmin) + meta->cal[b].lmin;
            tmp = meta->cal[b].k2/log((meta->cal[b].k1/rad)+1)*toa_scale;
            if (tmp < SHRT_MAX){
              toa_[b][p] = (short)tmp;
            } else {
              toa_[b][p] = SHRT_MAX;
            }

          // DN to reflectance
          } else {

            g = convert_brick_p2p(QAI, atc->xy_sun, p);
            tmp = (meta->cal[b].radd + meta->cal[b].rmul*dn_[b][p]) / sun_[g];

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

