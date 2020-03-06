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
This file contains functions for handling topographic effects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "topo-ll.h"


top_t *allocate_topography();
int init_topography(top_t *top);
int ocean_topography(stack_t *DEM);
int smooth_topography(stack_t *DEM);
int exposition_topography(stack_t *DEM, stack_t *EXP, stack_t *QAI);
int stats_topography(atc_t *atc, stack_t *DEM, stack_t *CDEM, stack_t *QAI);
int illumination_topography(atc_t *atc, stack_t *EXP, stack_t *ILL, stack_t *SKY, stack_t *QAI);


/** This function allocates the topographic variables
+++ Return: topographic variables (must be freed with free_topography)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
top_t *allocate_topography(){
top_t *top = NULL;


  alloc((void**)&top, 1, sizeof(top_t));
  init_topography(top);    

  return top;
}


/** This function initializes the topographic variables
--- top:    topographic variables
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int init_topography(top_t *top){

  top->dem = NULL;
  top->exp = NULL;
  top->ill = NULL;
  top->sky = NULL;
  top->c   = NULL;

  return SUCCESS;
}


/** This function detects oceans and sets these pixels to 0m a.s.l. as 
+++ ocean is often masked out in DEMs (but is valid data)
--- DEM:    Digital Elevation Model
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int ocean_topography(stack_t *DEM){
int i, j, ii, jj, p, np, nx, ny;
float *dem_ = NULL;
float nodata;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  nx = get_stack_ncols(DEM);
  ny = get_stack_nrows(DEM);
  nodata = get_stack_nodata(DEM, 0);
  if ((dem_ = get_band_float(DEM, 0)) == NULL) return FAILURE;

  #pragma omp parallel private(i, ii, p, np) shared(nx, ny, dem_, nodata) default(none) 
  {

    #pragma omp for schedule(static)
    for (j=0; j<nx; j++){
    for (i=0, ii=1; ii<ny; i++, ii++){
      p = i*nx+j; np = ii*nx+j;
      if (fequal(dem_[p], 0) && fequal(dem_[np], nodata)) dem_[np] = 0.0;
    }
    }

  }
  

  #pragma omp parallel private(i, ii, p, np) shared(nx, ny, dem_, nodata) default(none) 
  {

    #pragma omp for schedule(static)
    for (j=0; j<nx; j++){
    for (i=(ny-1), ii=(ny-2); ii>=0; i--, ii--){
      p = i*nx+j; np = ii*nx+j;
      if (fequal(dem_[p], 0) && fequal(dem_[np], nodata)) dem_[np] = 0.0;
    }
    }
  
  }

  
  #pragma omp parallel private(j, jj, p, np) shared(nx, ny, dem_, nodata) default(none) 
  {

    #pragma omp for schedule(static)
    for (i=0; i<ny; i++){
    for (j=0, jj=1; jj<nx; j++, jj++){
      p = i*nx+j; np = i*nx+jj;
      if (fequal(dem_[p], 0) && fequal(dem_[np], nodata)) dem_[np] = 0.0;
    }
    }
  
  }

  
  #pragma omp parallel private(j, jj, p, np) shared(nx, ny, dem_, nodata) default(none) 
  {

    #pragma omp for schedule(static)
    for (i=0; i<ny; i++){
    for (j=(nx-1), jj=(nx-2); jj>=0; j--, jj--){
      p = i*nx+j; np = i*nx+jj;
      if (fequal(dem_[p], 0) && fequal(dem_[np], nodata)) dem_[np] = 0.0;
    }
    }
    
  }


  #ifdef FORCE_CLOCK
  proctime_print("detect ocean from topography", TIME);
  #endif
  
  return SUCCESS;
}


/** This function smooths the DEM with a lowpass filter as there are often
+++ undesired effects in high resolution DEMs
--- DEM:    Digital Elevation Model
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int smooth_topography(stack_t *DEM){
int i, j, ii, jj, ni, nj, p, np, nx, ny, nc, k;
float *buf = NULL;
float sum, num;
float res;
float *dem_ = NULL;
float nodata;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  nx = get_stack_ncols(DEM);
  ny = get_stack_nrows(DEM);
  nc = get_stack_ncells(DEM);
  res = get_stack_res(DEM);
  nodata = get_stack_nodata(DEM, 0);
  if ((dem_ = get_band_float(DEM, 0)) == NULL) return FAILURE;


  if (res >= 30) return SUCCESS;
  
  #ifdef FORCE_DEV
  printf("smooth_topo should consider actual pixel size of DEM...\n");
  #endif

  k = 30/res;

  alloc((void**)&buf, nc, sizeof(float));

  #pragma omp parallel private(j, p, ii, jj, ni, nj, np, sum, num) shared(nx, ny, k, dem_, buf, nodata) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      p = i*nx+j;

      buf[p] = nodata;

      if (dem_[p] == nodata) continue;

      sum = num = 0;

      for (ii=-1*k; ii<=k; ii++){
      for (jj=-1*k; jj<=k; jj++){

        ni = i+ii; nj = j+jj;
        if (ni > ny-1 || ni < 0 || nj > nx-1 || nj < 0) continue;

        np = ni*nx+nj;
        if (fequal(dem_[np], nodata)) continue;

        sum += dem_[np];
        num++;

      }
      }

      if (num > 0) buf[p] = sum/num;

    }
    }

  }
  
  memmove(dem_, buf, nc*sizeof(float));
  free((void*)buf);
  
  
  #ifdef FORCE_CLOCK
  proctime_print("smooth topography", TIME);
  #endif

  return SUCCESS;
}


/** This function computes slope and aspect with the Horn (1981) method
--- DEM:    Digital Elevation Model
--- EXP:    Exposition
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int exposition_topography(stack_t *DEM, stack_t *EXP, stack_t *QAI){
int i, j, ii, jj, p, nx, ny, nc;
float devx, devy;
float tmp, dem[3][3];
ushort *slp_  = NULL;
ushort *asp_  = NULL;
float  *dem_  = NULL;
float nodata;
float slope, aspect;
float res;
bool valid;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  nx = get_stack_ncols(DEM);
  ny = get_stack_nrows(DEM);
  nc = get_stack_ncells(DEM);
  res = get_stack_res(DEM);
  nodata = (short)get_stack_nodata(DEM, 0);
  if ((dem_ = get_band_float(DEM, 0)) == NULL) return FAILURE;

  if ((slp_ = get_band_ushort(EXP, ZEN)) == NULL) return FAILURE;
  if ((asp_ = get_band_ushort(EXP, AZI)) == NULL) return FAILURE;


  #pragma omp parallel private(j, p, ii, jj, tmp, valid, devx, devy, slope, aspect, dem) shared(nx, ny, res, QAI, dem_, slp_, asp_, nodata) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=1; i<(ny-1); i++){
    for (j=1; j<(nx-1); j++){

      p = i*nx+j;

      if (get_off(QAI, p)) continue;

      valid = true;

      for (ii=-1; ii<=1; ii++){
      for (jj=-1; jj<=1; jj++){
        tmp = dem_[(i+ii)*nx+(j+jj)];
        if (fequal(tmp, nodata)){
          valid = false;
        } else {
          dem[ii+1][jj+1] = tmp;
        }
      }
      }


      if (valid){

        devx = ((dem[0][2]+2*dem[1][2]+dem[2][2]) - 
                (dem[0][0]+2*dem[1][0]+dem[2][0])) / (8*res);
        devy = ((dem[2][0]+2*dem[2][1]+dem[2][2]) - 
                (dem[0][0]+2*dem[0][1]+dem[0][2])) / (8*res);

        slope = atan(sqrt(devx*devx+devy*devy))*_R2D_CONV_;
        if (slope > 2) set_slope(QAI, p, true);

        if (slope > 0){
          aspect = atan2(devy, -1*devx)*_R2D_CONV_;
          if (aspect < 0){
            aspect = 90-aspect;
          } else if (aspect > 90){
            aspect = 360-aspect+90;
          } else {
            aspect = 90-aspect;
          }
        } else {
          slope = aspect = 0;
        }

        slp_[p] = (ushort)(slope*_D2R_CONV_*10000);
        asp_[p] = (ushort)(aspect*_D2R_CONV_*10000);

      } else {

        set_off(QAI, p, true);

      }

    }
    }
    
  }


  #pragma omp parallel shared(nx, nc, QAI, slp_, asp_) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=0; p<nc; p+=nx){

      if (get_off(QAI, p)) continue;
      
      if (get_off(QAI, p+1)){
        set_off(QAI, p, true);
      } else {
        slp_[p] = slp_[p+1];
        asp_[p] = asp_[p+1];
      }
    }
    
  }
  
  
  #pragma omp parallel shared(nx, nc, QAI, slp_, asp_) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=nx-1; p<nc; p+=nx){

      if (get_off(QAI, p)) continue;

      if (get_off(QAI, p-1)){
        set_off(QAI, p, true);
      } else {
        slp_[p] = slp_[p-1];
        asp_[p] = asp_[p-1];
      }
    }
    
  }
  
  
  #pragma omp parallel shared(nx, QAI, slp_, asp_) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=0; p<nx; p++){

      if (get_off(QAI, p)) continue;

      if (get_off(QAI, p+nx)){
        set_off(QAI, p, true);
      } else {
        slp_[p] = slp_[p+nx];
        asp_[p] = asp_[p+nx];
      }
    }
    
  }
  
  
  #pragma omp parallel shared(nx, ny, nc, QAI, slp_, asp_) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=(ny-1)*nx; p<nc; p++){

      if (get_off(QAI, p)) continue;

      if (get_off(QAI, p-nx)){
        set_off(QAI, p, true);
      } else {
        slp_[p] = slp_[p-nx];
        asp_[p] = asp_[p-nx];
      }
    }
    
  }

  
  #ifdef FORCE_CLOCK
  proctime_print("exposition from topography", TIME);
  #endif

  return SUCCESS;
}


/** This function computes elevation statistics (in km) and computes the
+++ binned DEM (100m classes)
--- atc:    atmospheric correction factors
--- DEM:    Digital Elevation Model
--- CDEM:   Binned Digital Elevation Model
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int stats_topography(atc_t *atc, stack_t *DEM, stack_t *CDEM, stack_t *QAI){
int p, nc;
float dem, mn = SHRT_MAX, mx = SHRT_MIN;
double sum = 0, num = 0;
float  *dem_  = NULL;
small  *cdem_ = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  

  nc = get_stack_ncells(DEM);
  if ((dem_ = get_band_float(DEM, 0)) == NULL) return FAILURE;
  if ((cdem_ = get_band_small(CDEM, 0)) == NULL) return FAILURE;


  #pragma omp parallel private(dem) shared(nc, QAI, dem_) reduction(+: sum, num) reduction(max: mx) reduction(min: mn) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)) continue;
      dem = dem_[p]/1000.0; // -> kilometer
      if (dem < mn) mn = dem;
      if (dem > mx) mx = dem;
      sum += dem; // -> kilometer
      num++;
    }
    
  }

  atc->dem.min = mn;
  atc->dem.max = mx;
  atc->dem.avg = (float)(sum/num);
  atc->dem.max += 0.001; // add 1m

  atc->dem.cnum = NPOW_08 - 1;
  atc->dem.step = (atc->dem.max-atc->dem.min)/atc->dem.cnum;
  
  
  #pragma omp parallel private(dem) shared(nc, QAI, dem_, cdem_, atc) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)) continue;
      dem = dem_[p]/1000.0; // -> kilometer
      cdem_[p] = (small)floor((dem-atc->dem.min)/atc->dem.step);
    }

  }


  if (atc->dem.min < -0.5 || atc->dem.max > 9){
    printf("DEM out of bounds: min %f max %f. ", 
    atc->dem.min, atc->dem.max); return FAILURE;
  }

 
  #ifdef FORCE_DEBUG
  printf("elevation stats:\n");
  printf(" avg: %+6.3f, min: %+6.3f, max: %+6.3f, step: %+6.3f, cnum: %d\n",
  atc->dem.avg, atc->dem.min, atc->dem.max, atc->dem.step, atc->dem.cnum);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("stats topography", TIME);
  #endif

  return SUCCESS;
}


/** This function computes the illumination angle and a simple sky view 
+++ factor
--- atc:    atmospheric correction factors
--- EXP:    Exposition
--- ILL:    Illumination angle
--- SKY:    Sky view factor
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int illumination_topography(atc_t *atc, stack_t *EXP, stack_t *ILL, stack_t *SKY, stack_t *QAI){
int p, g, nc;
float slp, asp;
ushort *slp_ = NULL;
ushort *asp_ = NULL;
short  *ill_ = NULL;
ushort *sky_ = NULL;
float **sun_ = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nc = get_stack_ncells(EXP);
  if ((slp_ = get_band_ushort(EXP, ZEN)) == NULL) return FAILURE;
  if ((asp_ = get_band_ushort(EXP, AZI)) == NULL) return FAILURE;
  if ((ill_ = get_band_short(ILL, 0)) == NULL) return FAILURE;
  if ((sky_ = get_band_ushort(SKY, 0)) == NULL) return FAILURE;
  if ((sun_ = get_bands_float(atc->xy_sun)) == NULL) return FAILURE;


  #pragma omp parallel private(g, slp, asp) shared(nc, QAI, slp_, asp_, ill_, sky_, sun_, atc) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){

      if (get_off(QAI, p)){ ill_[p] = -10000; continue;}

      g = convert_stack_p2p(QAI, atc->xy_sun, p);

      slp = slp_[p]/10000.0;
      asp = asp_[p]/10000.0;

      // illumination angle
      if (slp == 0){
        ill_[p] = (short)(sun_[cZEN][g]*10000);
      } else {
        ill_[p] = (short)(illumin(sun_[cZEN][g], sun_[sZEN][g], 
                              cos(slp), sin(slp), sun_[AZI][g], asp)*10000);
      }

      // set illumination QAI
      if (ill_[p] < 0){
        set_illumination(QAI, p, 3); // deep shadow
      } else if (ill_[p] < 1736.482){
        set_illumination(QAI, p, 2); // poor
      } else if (ill_[p] < 5735.764){
        set_illumination(QAI, p, 1); // moderate
      }

      // sky view factor (portion of the sky dome diffusing on to a tilted surface)
      sky_[p] = (ushort)(10000 - (slp/M_PI)*10000);

    }
    
  }


  #ifdef FORCE_CLOCK
  proctime_print("illumination condition", TIME);
  #endif

  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function frees the topographic variables
--- top:    topographic variables
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_topography(top_t *top){

  if (top == NULL) return;

  free_stack(top->dem);
  free_stack(top->exp);
  free_stack(top->ill);
  free_stack(top->sky);
  free_stack(top->c);

  free((void*)top); top = NULL;

  return;
}


/** This function compiles the basic set of topographic derivatives. This 
+++ includes reprojecting the DEM to the image extent/projection and com-
+++ puting slope and aspect
--- pl2:        L2 parameters
--- atc:        atmospheric correction factors
--- topography: topographic variables
--- QAI:        Quality Assurance Information
+++ Return:     SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int compile_topography(par_ll_t *pl2, atc_t *atc, top_t **topography, stack_t *QAI){
top_t *top = NULL;
stack_t *DEM = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  top = allocate_topography();


  /** Digital Elevation Model stack **/
  DEM = copy_stack(QAI, 1, _DT_FLOAT_);
  set_stack_name(DEM, "FORCE DEM stack");
  set_stack_product(DEM, "DEM");
  set_stack_filename(DEM, "DEM");
  set_stack_bandname(DEM, 0, "DEM");
  set_stack_nodata(DEM, 0, pl2->dem_nodata);


  // warp DEM to MEM or use flat DEM (z = 0m)
  if (strcmp(pl2->fdem, "NULL") != 0){
    if ((warp_from_disc_to_known_stack(1, pl2->nthread, pl2->fdem, DEM, 0, 0, pl2->dem_nodata)) != SUCCESS){
      printf("Reprojecting of DEM failed! "); return FAILURE;}
  }


  /** detect oceans and set to 0m a.s.l **/
  if ((ocean_topography(DEM)) != SUCCESS){
    printf("Compiling ocean DEM failed! "); return FAILURE;}


  /** smooth DEM **/
  if ((smooth_topography(DEM)) != SUCCESS){
    printf("Smoothing of DEM failed! "); return FAILURE;}


  #ifdef FORCE_DEBUG
  print_stack_info(DEM); set_stack_open(DEM, OPEN_CREATE); write_stack(DEM);
  #endif


  /** exposition stack **/
  top->exp = copy_stack(DEM, 2, _DT_USHORT_);
  set_stack_name(top->exp, "FORCE terrain exposition stack");
  set_stack_product(top->exp, "EXP");
  set_stack_filename(top->exp, "DEM-EXPOSITION");
  set_stack_bandname(top->exp, ZEN, "Slope");
  set_stack_bandname(top->exp, AZI, "Aspect");

  if ((exposition_topography(DEM, top->exp, QAI)) != SUCCESS){
    printf("Slope/aspect failed! "); return FAILURE;}
    
  #ifdef FORCE_DEBUG
  print_stack_info(top->exp); set_stack_open(top->exp, OPEN_CREATE); write_stack(top->exp);
  #endif

  
  /** calculate DEM stats + binned DEM **/
  top->dem = copy_stack(DEM, 1, _DT_SMALL_);
  set_stack_name(top->dem, "FORCE binned DEM stack");
  set_stack_product(top->dem, "BEM");
  set_stack_filename(top->dem, "DEM-BINNED");
  set_stack_bandname(top->dem, 0, "binned DEM");

  if ((stats_topography(atc, DEM, top->dem, QAI)) != SUCCESS){
    printf("Elevation statistics failed! "); return FAILURE;}

  #ifdef FORCE_DEBUG
  print_stack_info(top->dem); set_stack_open(top->dem, OPEN_CREATE); write_stack(top->dem);
  #endif


  /** illumination angle and sky view factor **/
  top->ill = copy_stack(DEM, 1, _DT_SHORT_);
  set_stack_name(top->ill, "FORCE Illumination angle stack");
  set_stack_product(top->ill, "ILL");
  set_stack_filename(top->ill, "DEM-ILLUMINATION");
  set_stack_bandname(top->ill, 0, "Illumination angle");

  top->sky = copy_stack(DEM, 1, _DT_USHORT_);
  set_stack_name(top->sky, "FORCE Sky View Factor stack");
  set_stack_product(top->sky, "SKY");
  set_stack_filename(top->sky, "DEM-SKY-VIEW");
  set_stack_bandname(top->sky, 0, "Sky View Factor");

  if (illumination_topography(atc, top->exp, top->ill, top->sky, QAI) != SUCCESS){
    printf("error in topographic correction. "); return FAILURE;}

  free_stack(DEM);


  #ifdef FORCE_DEBUG
  print_stack_info(top->ill); set_stack_open(top->ill, OPEN_CREATE); write_stack(top->ill);
  print_stack_info(top->sky); set_stack_open(top->sky, OPEN_CREATE); write_stack(top->sky);
  print_stack_info(QAI); set_stack_open(QAI, OPEN_CREATE); write_stack(QAI);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("compiled topography", TIME);
  #endif

  *topography = top;
  return SUCCESS;
}


/** This function computes the C-factor used for topographic correction. C
+++ is derived for the SWIR2 band and is then propagated through the spec-
+++ trum in the actual radiometric correction. C is derived for each pixel
+++ with sufficient illumination.
--- atc:    atmospheric correction factors
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information
--- DEM:    Digital Elevation model
--- EXP:    Exposition
--- ILL:    Illumination angle
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *cfactor_topography(atc_t *atc, stack_t *TOA, stack_t *QAI, stack_t *DEM, stack_t *EXP, stack_t *ILL){
int i, j, p, ii, jj, ip, jp, np, nx, ny, nc, g, z;
int b_sw2;
ushort s_min  = 350; // 2° slope
double mx, my;
double cov, varx, vary, num;
double offset, gain;
float c_, h0, f, cf;
float *cor_ = NULL;
float tmp, res;
float *swir_ = NULL;
float rho_p, tss, tsd, szen, ms;
int k, nk = 0, *K = NULL;
stack_t *CF = NULL;
ushort  *cf_ = NULL;
short *sw1_ = NULL;
short *sw2_ = NULL;
small *dem_ = NULL;
ushort *slp_ = NULL;
short *ill_ = NULL;
float *xy_szen = NULL;
float *xy_ms = NULL;
float **xyz_rho_p = NULL;
float **xyz_tss = NULL;
float **xyz_tsd = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  cite_me(_CITE_TOPCOR_);
  
  
  CF = copy_stack(QAI, 1, _DT_USHORT_);
  set_stack_name(CF, "FORCE C-factor stack");
  set_stack_product(CF, "CFC");
  set_stack_filename(CF, "DEM-C-FACTOR");
  set_stack_bandname(CF, 0, "C-Factor");
  set_stack_nodata(CF, 0, -9999);

  nx  = get_stack_ncols(QAI);
  ny  = get_stack_nrows(QAI);
  nc  = get_stack_ncells(QAI);
  res = get_stack_res(QAI);

  if ((cf_  = get_band_ushort(CF, 0))         == NULL) return NULL;
  if ((sw1_ = get_domain_short(TOA, "SWIR1")) == NULL) return NULL;
  if ((sw2_ = get_domain_short(TOA, "SWIR2")) == NULL) return NULL;
  if ((dem_ = get_band_small(DEM, 0))         == NULL) return NULL;
  if ((slp_ = get_band_ushort(EXP, ZEN))      == NULL) return NULL;
  if ((ill_ = get_band_short(ILL, 0))         == NULL) return NULL;
  if ((b_sw2 = find_domain(TOA, "SWIR2")) < 0) return NULL;
  
  if ((xy_szen     = get_band_float(atc->xy_sun,   ZEN)) == NULL) return NULL;
  if ((xy_ms       = get_band_float(atc->xy_sun,  cZEN)) == NULL) return NULL;
  if ((xyz_rho_p       = atc_get_band_reshaped(atc->xyz_rho_p, b_sw2))     == NULL) return NULL;
  if ((xyz_tss       = atc_get_band_reshaped(atc->xyz_tss, b_sw2))     == NULL) return NULL;
  if ((xyz_tsd       = atc_get_band_reshaped(atc->xyz_tsd, b_sw2))     == NULL) return NULL;

  
  /** kernel for sampling neighborhood **/
  alloc((void**)&K, 3000/res*2+1, sizeof(int));

  K[nk] = 3000/res; nk++;
  while (K[nk-1] > 100/res){
    K[nk] = K[nk-1]/sqrt(2);
    nk++;
  }

  if (K[nk-1] != 0) nk++;

  for (k=nk-2; k>=0; k--){
    K[nk] = -1*K[k];
    nk++;
  }

  #ifdef FORCE_DEBUG
  printf("sampling for estimating C:\n");
  for (k=0; k<nk; k++) printf("%d ", K[k]); printf("\n");
  #endif


  /** allocate memory **/
  alloc((void**)&cor_,  nc, sizeof(float));
  alloc((void**)&swir_, nc, sizeof(float));


  /** compute SWIR index **/
  
  #pragma omp parallel private(tmp) shared(nc, QAI, sw1_, sw2_, swir_) default(none) 
  {

    #pragma omp for schedule(guided)

    for (p=0; p<nc; p++){
      
      if (get_off(QAI, p)) continue;

      tmp = sw1_[p]/10000.0+sw2_[p]/10000.0;
      if (tmp == 0) swir_[p] = 0; else swir_[p] = (sw1_[p]/10000.0-sw2_[p]/10000.0)/tmp;

    }
    
  }


  /** estimate C for every pixel **/
  
  #pragma omp parallel private(j, p, ii, jj, ip, jp, np, g, z, rho_p, tss, tsd, szen, ms, f, h0, c_, num, mx, my, varx, vary, cov, gain, offset) shared(nx, ny, nk, K, b_sw2, QAI, sw2_, swir_, dem_, slp_, ill_, s_min, cor_, xyz_rho_p, xyz_tss, xyz_tsd, xy_szen, xy_ms, atc) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      p = i*nx+j;

      // only do for illuminated pixels
      if (get_off(QAI, p) || ill_[p] < 0) continue;

      g = convert_stack_ji2p(QAI, atc->xy_sun, i, j);
      z = dem_[p];

      // f-factor, h0-factor, theoretical C-factor
      rho_p = xyz_rho_p[z][g];
      tss   = xyz_tss[z][g];
      tsd   = xyz_tsd[z][g];
      szen  = xy_szen[g];
      ms    = xy_ms[g];

      f = f_factor(tss, tsd);
      h0 = (M_PI+2*szen)/(2.0*M_PI);
      c_ = c_factor_com(h0, f, ms);

      // only do for sloped pixels > 2°
      if (slp_[p] > s_min){

        num = mx = my = varx = vary = cov = 0.0;

        // sample neighborhood
        for (ii=0; ii<nk; ii++){
        for (jj=0; jj<nk; jj++){
          
          ip = i+K[ii]; jp = j+K[jj];
          if (ip < 0 || jp < 0 || ip > ny-1 || jp > nx-1) continue;
          np = ip*nx+jp;

          // only use illuminated and sloped pixels > 2°
          if (get_off(QAI, np) || ill_[np] < 0 || slp_[np] < s_min) continue;

          // only do for same land cover
          if (fabs(swir_[p]-swir_[np]) > 0.025) continue;

          num++;

          // linear regression
          if (fequal(num, 1)){
            mx = ill_[np]/10000.0; my = sw2_[np]/10000.0;
          } else {
            covar_recurrence(ill_[np]/10000.0, sw2_[np]/10000.0,
            &mx, &my, &varx, &vary, &cov, num);
          }

        }
        }


        if (num > 2){

          // regression parameters + c-factor
          linreg_coefs(mx, my, covariance(cov, num), 
            variance(varx, num), &gain, &offset);

          // if offset < path reflectance, 
          // use path reflectance as offset and recompute gain
          if (offset < rho_p){
            offset = rho_p;
            gain = (my-offset)/mx;
          }

          //if (gain < 0) continue;

          // shield against extreme C-values
          if (offset < 10*gain){
            cor_[p] = c_factor_emp(offset, gain);
          } else {
            cor_[p] = 10.0;
          }

        }

      }

      // use computed C if estimated C < 0
      if (cor_[p] < 0 || cor_[p] < c_) cor_[p] = c_;
      if (cor_[p] > USHRT_MAX/10000.0) cor_[p] = USHRT_MAX/10000.0;

    }
    }

  }

  free((void*)swir_);
  free((void*)xyz_rho_p); free((void*)xyz_tss); free((void*)xyz_tsd);
  

  /** smooth C-factor with lowpass **/

  #pragma omp parallel private(j, p, ip, jp, np, num, mx, cf) shared(nx, ny, QAI, ill_, cor_, cf_) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      
      p = i*nx+j;
      
      if (get_off(QAI, p) || ill_[p] < 0) continue;

        num = mx = 0.0;

        for (ip=(i-1); ip<=(i+1); ip++){
        for (jp=(j-1); jp<=(j+1); jp++){

          if (ip < 0 || jp < 0 || ip > ny-1 || jp > nx-1) continue;
          np = ip*nx+jp;

          if (get_off(QAI, np) || ill_[np] < 0) continue;

          mx += cor_[np];
          num++;

        }
        }

        if (num > 0) cf = mx/num*10000; else cf = cor_[p]*10000;
        if (cf > UINT_MAX) cf = UINT_MAX;
        if (num > 0) cf_[p] = (ushort)cf;

    }
    }
    
  }

  free((void*)cor_);
  free((void*)K);


  #ifdef FORCE_DEBUG
  print_stack_info(CF); set_stack_open(CF, OPEN_CREATE); write_stack(CF);
  #endif
  
  #ifdef FORCE_CLOCK
  proctime_print("topographic correction factors", TIME);
  #endif

  return CF;
}


/** Average elevation of coarse grid cell
+++ This function computes the average binned elevation in the given
+++ coarse grid cell
--- g:   cell of coarse grid
--- CDEM:   Binned Digital Elevation Model
--- FDEM:   Float  Digital Elevation Model
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int average_elevation_cell(int g, stack_t *CDEM, stack_t *FDEM, stack_t *QAI){
int i, j, ii, jj, p, cell_size, nx, ny;
double sum = 0, num = 0;
small *dem_ = NULL;


  // coarse cell to fine pos
  convert_stack_p2ji(CDEM, FDEM, g, &i, &j);

  // cellsize in fine pixels
  cell_size = floor(get_stack_res(CDEM)/get_stack_res(FDEM));
  
  nx = get_stack_ncols(FDEM);
  ny = get_stack_nrows(FDEM);

  if ((dem_  = get_band_small(FDEM, 0)) == NULL) return FAILURE;

  for (ii=i; ii<(i+cell_size); ii++){
  for (jj=j; jj<(j+cell_size); jj++){

    if (ii > ny-1 || jj > nx-1) continue;

    p = ii*nx+jj;
    if (get_off(QAI, p)) continue;

    sum += dem_[p];
    num++;

  }
  }

  if (num>0){
    set_stack(CDEM, 0, g, round(sum/num));
  } else {
    set_stack(CDEM, 0, g, 0);
  }
  
  return SUCCESS;
}

