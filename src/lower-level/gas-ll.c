/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2020 David Frantz

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
This file contains functions for handling Atmospheric Gas
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "gas-ll.h"

/** GNU Scientific Library (GSL) **/
#include <gsl/gsl_multimin.h>          // minimization functions 


/** This function computes a water vapor transmittace look-up-table, which
+++ is used for fast estimation of Sentinel-2 water vapor. The function 
+++ will exit successfully if atmospheric correction is disabled or if not
+++ Sentinel-2
--- meta:   metadata
--- atc:    atmospheric correction factors
+++ Return: SUCCESS / FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int wvp_transmitt_lut(meta_t *meta, atc_t *atc){
int b, nb, km, kw;
float w, m, m_min, m_max;
int km_min, km_max;
int nm = 101;
int nw = 701;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  _WVLUT_.nb = nb = get_brick_nbands(atc->xy_Tg);
  _WVLUT_.nw = nw;
  _WVLUT_.nm = nm;
  alloc_3D((void****)&_WVLUT_.val, _WVLUT_.nb, _WVLUT_.nw, _WVLUT_.nm, sizeof(float));


  if (atc->cosszen[0] < atc->cosvzen[0]){
    m_min = atc->cosszen[0]; 
  } else {
    m_min = atc->cosvzen[0];
  }
  km_min = floor(m_min*100.0);


  if (atc->cosszen[1] > atc->cosvzen[1]){
    m_max = atc->cosszen[1];
  } else {
    m_max = atc->cosvzen[1];
  }
  km_max = ceil(m_max*100.0);

  
  #ifdef FORCE_DEBUG
  printf("m min/max: %.2f %.2f\n", m_min, m_max);
  #endif

  #pragma omp parallel private(b, m, w) shared(nm, nw, nb, km_min, km_max, meta, _WVLUT_) default(none)
  {

    #pragma omp for collapse(2) schedule(guided)
    for (km=km_min; km<=km_max; km++){

      for (kw=0; kw<nw; kw++){

        m = km*0.01;
        w = kw*0.01;
    
        for (b=0; b<nb; b++){
          _WVLUT_.val[b][kw][km] = wvp_transmitt(w, m, meta->cal[b].rsr_band);
        }

      }
    }
    
  }


  #ifdef FORCE_CLOCK
  proctime_print("water vapor LUT", TIME);
  #endif

  return SUCCESS;
}


/** This minimizer function computes BOA reflectance for the reference and
+++ measurement channels and returns the absolute residual.
--- v:      current water vapor estimate
--- params: parameters used for minimization
+++ Return: Absolute residual of BOA reflectance (reference - measurement)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double fun_wvp(const gsl_vector *v, void *params){
float *p = (float *)params;
float w;
float nir, wvp;
int b_nir, b_wvp, b_nir_rsr, b_wvp_rsr;
float ms, mv, To_nir, To_wvp, rho_p_nir, rho_p_wvp;
float T_nir, T_wvp, s_nir, s_wvp;
float Tsw_nir, Tvw_nir, Tsw_wvp, Tvw_wvp, Tg_nir, Tg_wvp;
float tmp, sr_reference, sr_measure;
int kms, kmv, kw;


  w = gsl_vector_get(v, 0);

  nir       = p[0];
  wvp       = p[1];
  b_nir     = (int)p[2];
  b_wvp     = (int)p[3];
  b_nir_rsr = (int)p[4];
  b_wvp_rsr = (int)p[5];
  ms        = p[6];
  mv        = p[7];
  To_nir    = p[8];
  To_wvp    = p[9];
  rho_p_nir = p[10];
  rho_p_wvp = p[11];
  T_nir     = p[12];
  T_wvp     = p[13];
  s_nir     = p[14];
  s_wvp     = p[15];


  // negative water vapor values are not allowed
  if (w < 0.0) w = 0.0;

  // if water vapor > tabulated values, compute
  if (w > 7.0){

    Tsw_nir = wvp_transmitt(w, ms, b_nir_rsr);
    Tvw_nir = wvp_transmitt(w, mv, b_nir_rsr);
    Tsw_wvp = wvp_transmitt(w, ms, b_wvp_rsr);
    Tvw_wvp = wvp_transmitt(w, mv, b_wvp_rsr);

  // use tabulated values
  } else {

    kw =  (int)floor(w/0.01);
    kms = (int)floor(ms/0.01);
    kmv = (int)floor(mv/0.01);

    Tsw_nir = _WVLUT_.val[b_nir][kw][kms];
    Tvw_nir = _WVLUT_.val[b_nir][kw][kmv];
    Tsw_wvp = _WVLUT_.val[b_wvp][kw][kms];
    Tvw_wvp = _WVLUT_.val[b_wvp][kw][kmv];

  }


  Tg_nir  = Tsw_nir*Tvw_nir*To_nir;
  Tg_wvp  = Tsw_wvp*Tvw_wvp*To_wvp;

  tmp = (nir-rho_p_nir)/Tg_nir;
  sr_reference = tmp / (T_nir + s_nir*tmp);

  tmp = (wvp-rho_p_wvp)/Tg_wvp;
  sr_measure = tmp / (T_wvp + s_wvp*tmp);

  return fabs(sr_reference-sr_measure);
}


/** This function estimates water vapor in Sentinel-2 images. The function
+++ will only return SUCCESS for Landsat. Note that WVP and Tg won't be 
+++ allocated in this case. Water vapor is estimated for each 60m pixel
+++ using the complete radiative transfer assuming that BOA reflectance
+++ of the NIR reference channel @ 0.865µm and the NIR water vapor channel
+++ @ 0.945 should be equal. Nelder-Mead Simplex optimization is used.
+++ Water and shadow pixels will be set to the scene average, and a QAI
+++ flag is set in this case.
--- meta:   metadata
--- atc:    atmospheric correction factors
--- TOA:    TOA brick
--- QAI:    QAI brick
--- DEM:    DEM brick
+++ Return: Water vapor brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *water_vapor(meta_t *meta, atc_t *atc, brick_t *TOA, brick_t *QAI, brick_t *DEM){
int i, j, ii, jj, p, nx, ny, g, z, k;
int b_reference, b_measure;
float reference, measure, dem;
float w, w_avg;
double w_sum = 0, num = 0;
float param[16];
const gsl_multimin_fminimizer_type *T = NULL;
gsl_multimin_fminimizer *s = NULL;
gsl_vector *ss = NULL, *x = NULL;
gsl_multimin_function minex_func;
size_t iter;
int status;
double size;
brick_t *WVP = NULL;
short  *wvp_ = NULL;
small  *dem_ = NULL;
short **toa_ = NULL;
float *xy_ms = NULL;
float *xy_mv = NULL;
float *xy_Tvo_r = NULL;
float *xy_Tvo_m = NULL;
float *xy_Tso_r = NULL;
float *xy_Tso_m = NULL;
float **xyz_rho_p_r = NULL;
float **xyz_rho_p_m = NULL;
float **xyz_T_r = NULL;
float **xyz_T_m = NULL;
float **xyz_s_r = NULL;
float **xyz_s_m = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  if (wvp_transmitt_lut(meta, atc) != SUCCESS){
    printf("error in water vapor transmittance LUT. "); return NULL;}


  /** Water Vapor brick **/
  WVP = copy_brick(QAI, 1, _DT_SHORT_);
  set_brick_name(WVP, "FORCE Water Vapor");
  set_brick_product(WVP, "WVP");
  set_brick_filename(WVP, "WVP");
  set_brick_bandname(WVP, 0, "WVP");
  set_brick_nodata(WVP, 0, -9999);

  nx = get_brick_ncols(WVP);
  ny = get_brick_nrows(WVP);

  if ((wvp_ = get_band_short(WVP, 0))       == NULL) return NULL;
  if ((dem_ = get_band_small(DEM, 0))       == NULL) return NULL;
  if ((toa_ = get_bands_short(TOA))         == NULL) return NULL;
  if ((b_reference = find_domain(TOA, "NIR"))   < 0) return NULL;
  if ((b_measure   = find_domain(TOA, "VAPOR")) < 0) return NULL;
  
  if ((xy_ms       = get_band_float(atc->xy_sun,  cZEN)) == NULL) return NULL;
  if ((xy_mv       = get_band_float(atc->xy_view, cZEN)) == NULL) return NULL;
  if ((xy_Tvo_r    = get_band_float(atc->xy_Tvo, b_reference))   == NULL) return NULL;
  if ((xy_Tvo_m    = get_band_float(atc->xy_Tvo, b_measure))   == NULL) return NULL;
  if ((xy_Tso_r    = get_band_float(atc->xy_Tso, b_reference))   == NULL) return NULL;
  if ((xy_Tso_m    = get_band_float(atc->xy_Tso, b_measure))   == NULL) return NULL;
  if ((xyz_rho_p_r = atc_get_band_reshaped(atc->xyz_rho_p, b_reference))   == NULL) return NULL;
  if ((xyz_rho_p_m = atc_get_band_reshaped(atc->xyz_rho_p, b_measure))   == NULL) return NULL;
  if ((xyz_T_r     = atc_get_band_reshaped(atc->xyz_T, b_reference))   == NULL) return NULL;
  if ((xyz_T_m     = atc_get_band_reshaped(atc->xyz_T, b_measure))   == NULL) return NULL;
  if ((xyz_s_r     = atc_get_band_reshaped(atc->xyz_s, b_reference))   == NULL) return NULL;
  if ((xyz_s_m     = atc_get_band_reshaped(atc->xyz_s, b_measure))   == NULL) return NULL;


  #pragma omp parallel private(j, ii, jj, p, g, reference, measure, dem, z, k, w, x, T, ss, s, minex_func, param, iter, status, size) shared(nx, ny, b_reference, b_measure, toa_, wvp_, dem_, QAI, xy_ms, xy_mv, xy_Tvo_r, xy_Tvo_m, xy_Tso_r, xy_Tso_m, xyz_rho_p_r, xyz_rho_p_m, xyz_T_r, xyz_T_m, xyz_s_r, xyz_s_m, atc, meta, gsl_multimin_fminimizer_nmsimplex2) reduction(+: w_sum, num) default(none)
  {

  
    /** initialize optimizer
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    
    w = 3.0;

    // start value, is set to last value in loop
    x = gsl_vector_alloc(1);

    // initial step size of simplex
    ss = gsl_vector_alloc(1);
    gsl_vector_set(ss, 0, 0.25);

    // initialize method and iterate
    minex_func.n = 1;
    minex_func.f = fun_wvp;
    minex_func.params = param;
    T = gsl_multimin_fminimizer_nmsimplex2;
    s = gsl_multimin_fminimizer_alloc(T, 1);


    /**estimate water vapor for each 60m pixel
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i+=6){
    for (j=0; j<nx; j+=6){


      /** aggregate reference channel to 60m
      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

      reference = dem = k = 0;

      for (ii=0; ii<6; ii++){
      for (jj=0; jj<6; jj++){

        p = (i+ii)*nx+j+jj;

        if (get_off(QAI, p)   || get_shadow(QAI, p) || 
            get_water(QAI, p) || toa_[b_reference][p] < 1500) continue;

        reference += toa_[b_reference][p]/10000.0;
        dem       += dem_[p];
        k++;

      }
      }

      if (k == 0) continue;

      reference /= k;
      z = (int)roundf(dem/k);

      p = i*nx+j;

      measure = toa_[b_measure][p]/10000.0;

      g = convert_brick_ji2p(QAI, atc->xy_view, i, j);

      if (fequal(xy_mv[g], get_brick_nodata(atc->xy_view, cZEN))) continue;


      /** copy variables to param, and initialize minimizer
      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

      param[0]  = reference;
      param[1]  = measure;
      param[2]  = b_reference;
      param[3]  = b_measure;
      param[4]  = meta->cal[b_reference].rsr_band;
      param[5]  = meta->cal[b_measure].rsr_band;
      param[6]  = xy_ms[g];
      param[7]  = xy_mv[g];
      param[8]  = xy_Tso_r[g]*xy_Tvo_r[g];
      param[9]  = xy_Tso_m[g]*xy_Tvo_m[g];
      param[10] = xyz_rho_p_r[z][g];
      param[11] = xyz_rho_p_m[z][g];
      param[12] = xyz_T_r[z][g];
      param[13] = xyz_T_m[z][g];
      param[14] = xyz_s_r[z][g];
      param[15] = xyz_s_m[z][g];

      gsl_vector_set(x, 0, w);
      gsl_multimin_fminimizer_set(s, &minex_func, x, ss);


      /** optimize radiative transfer, estimate water vapor
      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

      iter = 0;

      do {
        iter++;
        status = gsl_multimin_fminimizer_iterate(s);

        if (status) break;

        size = gsl_multimin_fminimizer_size(s);
        status = gsl_multimin_test_size(size, 1e-2);

      } while (status == GSL_CONTINUE && iter < 100);

      if ((w = (float)gsl_vector_get(s->x, 0)) > 7 || w <= 0){
        w = 3; ;continue;}

      w_sum += w;
      num++;


      /** replicate values at original resolution
      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
      for (ii=0; ii<6; ii++){
      for (jj=0; jj<6; jj++){

        p = (i+ii)*nx+j+jj;

        if (get_off(QAI, p)) continue;

        wvp_[p] = (short)(w*1000);

      }
      }

    }
    }

    /** clean
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    gsl_vector_free(x); gsl_vector_free(ss); gsl_multimin_fminimizer_free(s);

  }

  free((void*)xyz_rho_p_r); free((void*)xyz_rho_p_m);
  free((void*)xyz_T_r);     free((void*)xyz_T_m);
  free((void*)xyz_s_r);     free((void*)xyz_s_m);

  /** fill water and shadow with mean water vapor, set QAI flag
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  if (num > 0){

    w_avg = w_sum/num;
    
    #ifdef FORCE_DEBUG
    printf("mean wvp: %f\n", w_avg);
    #endif

    #pragma omp parallel shared(nx, ny, w_avg, wvp_, QAI) default(none)
    {

      #pragma omp for
      for (p=0; p<nx*ny; p++){

        if (get_off(QAI, p) || wvp_[p] > 0) continue;

        set_vaporfill(QAI, p, true);
        wvp_[p] = (short)(w_avg*1000);

      }
      
    }

  }


  #ifdef FORCE_DEBUG
  print_brick_info(WVP); set_brick_open(WVP, OPEN_CREATE); write_brick(WVP);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("water vapor map", TIME);
  #endif


  return WVP;
}


/** Water vapor estimation
+++ This function computes the gaseous transmittance based on the esti-
+++ mated water vapor content. Gaseous Transmittance is scaled by 10000.
--- atc:    atmospheric correction factors
--- b:      band for which the transmittance is computed
--- WVP:    Water vapor brick
--- QAI:    QAI brick
+++ Return: Gaseous transmittance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *gas_transmittance(atc_t *atc, int b, brick_t *WVP, brick_t *QAI){
int i, j, ii, jj, p, nx, ny, g;
int kw, kms, kmv;
float w, ms, mv, Tsw, Tvw, Tso, Tvo, tg;
short *wvp_ = NULL;
short *Tg_  = NULL;
float *xy_ms = NULL;
float *xy_mv = NULL;
float *xy_Tso = NULL;
float *xy_Tvo = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  nx = get_brick_ncols(WVP);
  ny = get_brick_nrows(WVP);
  if ((wvp_ = get_band_short(WVP, 0))       == NULL) return NULL;
  
  if ((xy_ms       = get_band_float(atc->xy_sun, cZEN))   == NULL) return NULL;
  if ((xy_mv       = get_band_float(atc->xy_view, cZEN))   == NULL) return NULL;
  if ((xy_Tso       = get_band_float(atc->xy_Tso, b))   == NULL) return NULL;
  if ((xy_Tvo       = get_band_float(atc->xy_Tvo, b))   == NULL) return NULL;

  alloc((void**)&Tg_, nx*ny, sizeof(short));

  
  /**estimate water vapor for each 60m pixel
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  #pragma omp parallel private(j, p, ii, jj, g, w, kw, ms, mv, kms, kmv, Tsw, Tvw, Tso, Tvo, tg) shared(b, nx, ny, QAI, wvp_, WVP, _WVLUT_, Tg_, xy_ms, xy_mv, xy_Tso, xy_Tvo, atc) default(none) 
  {

    #pragma omp for schedule(guided)

    for (i=0; i<ny; i+=6){
    for (j=0; j<nx; j+=6){

      p = i*nx+j;

      g = convert_brick_ji2p(WVP, atc->xy_view, i, j);

      if (fequal(xy_mv[g], get_brick_nodata(atc->xy_view, cZEN))) continue;


      /** gaseous transmittance
      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

      w = wvp_[p]/1000.0;
      kw = (int)floor(w/0.01);

      ms = xy_ms[g];
      mv = xy_mv[g];
      kms = (int)floor(ms/0.01);
      kmv = (int)floor(mv/0.01);

      Tsw = _WVLUT_.val[b][kw][kms];
      Tvw = _WVLUT_.val[b][kw][kmv];
      Tso = xy_Tso[g];
      Tvo = xy_Tvo[g];
      tg  = gas_transmitt(Tsw, Tvw, Tso, Tvo);


      /** replicate values at original resolution
      +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
      for (ii=0; ii<6; ii++){
      for (jj=0; jj<6; jj++){

        p = (i+ii)*nx+j+jj;

        if (get_off(QAI, p)) continue;

        Tg_[p]  = (short)(tg*10000);

      }
      }

    }
    }

  }

  #ifdef FORCE_CLOCK
  proctime_print("gas transmittance map", TIME);
  #endif

  return Tg_;
}


/** Estimate ozone amount
+++ This function computes the atmospheric ozone amount as a function of 
+++ gaeographic position and DOY. Note that this is a very rough estimate.
--- lon:    Longitude
--- lat:    Latitude
--- doy:    Day-of-Year
+++ Return: ozone amount
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float ozone_amount(float lon, float lat, int doy){
float D, G, A, H, beta, C, F, I, DOY;
float o, omin;
D = 0.9856;
G = 20;
omin = 235;

  DOY = (float)doy;

  if (lat > 0) A = 150; else A = 100;
  if (lat > 0) H = 3; else H = 2;
  if (lat > 0) beta = 1.28; else beta = 1.5;
  if (lat > 0) C = 40; else C = 30;
  if (lat > 0) F = -30; else F = 152.625;

  if (lat > 0 && lon > 0) I = 20; else if (lat > 0 && lon < 0) I = 0; else I = -75;

  o = omin + (A+C*sin(D*(DOY+F)*_D2R_CONV_) + G*sin(H*(lon+I)*_D2R_CONV_)) * 
      sin(beta*lat*_D2R_CONV_)*sin(beta*lat*_D2R_CONV_);

  return(o/1000);
}


/** Water vapor from look up table
+++ This function reads water vapor estimates from a pre-compiled LUT or
+++ a constant value from the parameter file (see parameter file descrip-
+++ tion). This function is intended for sensors, which are not equipped
+++ with a water vapor sensitive band, e.g. Landsat. Two LUTs should be
+++ present. (1) daily LUTs (one per day), and (2) climatologies (one per
+++ month, i.e. 12 LUTs). Naming convention: (1) WVP_YYYY-MM-DD.txt, (2)
+++ WVP_0000-MM-00.txt. The files need to be plain text files, 4 columns 
+++ are separated by a blank space, no header. Columns are as follows: 
+++ Longitude, Latitude, Water vapor, Source/Number. Coordinates must be 
+++ given in geographic decimal degrees with negative values for South and
+++ West. Water vapor values are in cm, nodata value is 9999. The fourth
+++ column is either a three digit code indicating the source (e.g. MOD 
+++ or MYD for MODIS Terra or Aqua, respectively, TBD if nodata) for the
+++ daily LUTs (1) or the number of valid observations that were used to 
+++ compute the monthly climatology (2). The value of the line, which is
+++ closest to the image scene center is selected. Returned wvp may be -1 
+++ on failure
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
+++ Return: Water vapor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float water_vapor_from_lut(par_ll_t *pl2, atc_t *atc){
char fname[2][NPOW_10];
int nchar;
char buffer[NPOW_10] = "\0";
char *tokenptr = NULL;
const char *separator = " ";
char *source = NULL;
FILE *fp = NULL;
double center_x, center_y;
double site_x, site_y;
double diff_x, diff_y;
float wvp, avg, dist, min_dist = LONG_MAX;
int k, ne, nf;
int year, month, day;

  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  cite_me(_CITE_WVDB_);

  
  // scene center
  nf = get_brick_ncols(atc->xy_sun);
  ne = get_brick_nrows(atc->xy_sun);
  get_brick_geo(atc->xy_sun, nf/2, ne/2, &center_x, &center_y);
  year  = get_brick_year(atc->xy_sun, 0);
  month = get_brick_month(atc->xy_sun, 0);
  day   = get_brick_day(atc->xy_sun, 0);

  // initialize
  wvp = avg = -1;
  alloc((void**)&source, NPOW_02, sizeof(char));


  // use wvp from parameter file
  if (strcmp(pl2->d_wvp, "NULL") == 0){

    wvp = avg = pl2->wvp;
    copy_string(source, NPOW_02, "PRM");

    #ifdef FORCE_DEBUG
    printf("Use wvp from parameter file: %.2f\n", wvp);
    #endif

  // read wvp from LUT
  } else {

    // climatology LUT
    nchar = snprintf(fname[0], NPOW_10, "%s/WVP_%04d-%02d-%02d.txt", pl2->d_wvp,
      0, month, 0);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); return -1;}

      // daily LUT
    nchar = snprintf(fname[1], NPOW_10, "%s/WVP_%04d-%02d-%02d.txt", pl2->d_wvp,
      year, month, day);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); return -1;}

    for (k=0; k<2; k++){

      if (fileexist(fname[k])){
   
        if ((fp = fopen(fname[k], "r")) == NULL){
          printf("Unable to open LUT. "); return -1;}

        min_dist = LONG_MAX;

        // read line by line and find closest coordinate
        while (fgets(buffer, NPOW_10, fp) != NULL){

          buffer[strcspn(buffer, "\r\n#")] = 0;

          tokenptr = strtok(buffer, separator);

          site_x = atof(tokenptr); tokenptr = strtok(NULL, separator);
          site_y = atof(tokenptr); tokenptr = strtok(NULL, separator);
          diff_x = center_x-site_x; diff_y = center_y-site_y;

          dist = sqrt(diff_x*diff_x+diff_y*diff_y);

          // only use fairly near estimates (less than 1.5°)
          if (dist < min_dist && dist < 1.5){
            min_dist = dist;
            wvp = atof(tokenptr);
            if (k == 0){
              avg = wvp;
              copy_string(source, NPOW_02, "AVG");
            } else {
              tokenptr = strtok(NULL, separator);
              copy_string(source, NPOW_02, tokenptr);
            }
          }

        }

        fclose(fp);

      }

    }

    // if daily value has nodata, use average value
    if (wvp < 0 || wvp > 10) wvp = avg;

    #ifdef FORCE_DEBUG
    printf("Use wvp from LUT: %.2f, avg: %.2f, src: %s, diff to scene center: %.2f\n", 
      wvp, avg, source, min_dist);
    #endif

  }

  free((void*)source);

  #ifdef FORCE_CLOCK
  proctime_print("water vapor from LUT", TIME);
  #endif

  return wvp;
}

