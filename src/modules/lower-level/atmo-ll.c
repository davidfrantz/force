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
This file contains functions for atmospheric correction
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "atmo-ll.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdalgrid.h"       // GDAL gridder related entry points and defs


// interpolation weights
typedef struct {
int   gul, gur, gll, glr;
float wul, wur, wll, wlr;
float wtop, wdown;
} iweights_t;

iweights_t interpolation_weights(int j, int i, int nf, int ne, float res, float full_res, float *COARSE, float nodata);
float interpolate_coarse(iweights_t weight, float *COARSE);
int surface_reflectance(par_ll_t *pl2, atc_t *atc, int b, short *bck_, short *toa_, short *Tg_, short *boa_, small *dem_, short *ill_, ushort *sky_, ushort *cf_, brick_t *QAI);
short *background_reflectance(atc_t *atc, int b, short *toa_, short *Tg_, small *dem_, brick_t *QAI);
int atmo_angledep(par_ll_t *pl2, meta_t *meta, atc_t *atc, top_t *TOP, brick_t *QAI);
int atmo_elevdep(par_ll_t *pl2, atc_t *atc, brick_t *QAI, top_t *TOP);
int apply_aoi(brick_t *QAI, brick_t *AOI);
brick_t *compile_l2_qai(par_ll_t *pl2, cube_t *cube, brick_t *QAI);
brick_t *compile_l2_boa(par_ll_t *pl2, int mission, atc_t *atc, cube_t *cube, brick_t *TOA, brick_t *QAI, brick_t *WVP, top_t *TOP);
brick_t *compile_l2_dst(par_ll_t *pl2, cube_t *cube, brick_t *QAI);
brick_t *compile_l2_ovv(par_ll_t *pl2, brick_t *BOA, brick_t *QAI);
brick_t *compile_l2_vzn(par_ll_t *pl2, atc_t *atc, cube_t *cube, brick_t *QAI);
brick_t *compile_l2_hot(par_ll_t *pl2, cube_t *cube, brick_t *TOA, brick_t *QAI);
brick_t *compile_l2_aod(par_ll_t *pl2, atc_t *atc, cube_t *cube, brick_t *QAI, top_t *TOP);
brick_t *compile_l2_wvp(par_ll_t *pl2, atc_t *atc, cube_t *cube, brick_t *QAI, brick_t *WVP);
brick_t **compile_level2(par_ll_t *pl2, int mission, atc_t *atc, cube_t *cube, brick_t *TOA, brick_t *QAI, brick_t *WVP, top_t *TOP, int *nproduct);


/** This function computes the weights to interpolate the coarse atmos-
+++ pheric parameters
--- j:        column of full image
--- i:        row    of full image
--- nf:       number or columns in coarse grid
--- ne:       number or rows    in coarse grid
--- res:      resolution of coarse grid
--- full_res: resolution of full image
--- COARSE:   one coarse grid
--- nodata:   nodata of coarse grid
+++ Return:   weights
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
iweights_t interpolation_weights(int j, int i, int nf, int ne, float res, float full_res, float *COARSE, float nodata){
iweights_t weight;
float e_, f_;
int   e, f;
int   gleft, gright, gtop, gdown;
float cleft, cright, ctop, cdown;
bool vtop = true, vdown = true;
bool vul, vur, vll, vlr;


  e_ = i*full_res/res; e = floor(e_);
  f_ = j*full_res/res; f = floor(f_);

  if (f_-f < 0.5){ gleft = f-1; gright = f;} else { gleft = f; gright = f+1;}
  if (e_-e < 0.5){ gtop  = e-1; gdown  = e;} else { gtop  = e; gdown  = e+1;}

  if (gleft < 0) gleft = f;
  if (gtop < 0)  gtop  = e;
  if (gright >= nf) gright = f;
  if (gdown  >= ne) gdown  = e;


  cleft  = gleft  + 0.5;
  cright = gright + 0.5;
  ctop   = gtop   + 0.5;
  cdown  = gdown  + 0.5;

  weight.gul = gtop*nf  + gleft;
  weight.gur = gtop*nf  + gright;
  weight.gll = gdown*nf + gleft;
  weight.glr = gdown*nf + gright;

  vul = !fequal(COARSE[weight.gul], nodata);
  vur = !fequal(COARSE[weight.gur], nodata);
  vll = !fequal(COARSE[weight.gll], nodata);
  vlr = !fequal(COARSE[weight.glr], nodata);

  if (gleft == gright){
    weight.wul = 0.5; weight.wur = 0.5;
  } else if (vul && vur){
    weight.wul = (cright-f_)/(cright-cleft);
    weight.wur = (f_-cleft)/(cright-cleft);
  } else if (vul && !vur){
    weight.wul = 1; weight.wur = 0;
  } else if (!vul && vur){
    weight.wul = 0; weight.wur = 1;
  } else {
    weight.wul = 0; weight.wur = 0;
    vtop = false;
  }

  if (gleft == gright){
    weight.wll = 0.5; weight.wlr = 0.5;
  } else if (vll && vlr){
    weight.wll = (cright-f_)/(cright-cleft);
    weight.wlr = (f_-cleft)/(cright-cleft);
  } else if (vll && !vlr){
    weight.wll = 1; weight.wlr = 0;
  } else if (!vll && vlr){
    weight.wll = 0; weight.wlr = 1;
  } else {
    weight.wll = 0; weight.wlr = 0;
    vdown = false;
  }

  if (gtop == gdown){
    weight.wtop = 0.5; weight.wdown = 0.5;
  } else if (vtop && vdown){
    weight.wtop  = (cdown-e_)/(cdown-ctop);
    weight.wdown = (e_-ctop)/(cdown-ctop);
  } else if (vtop){
    weight.wtop = 1; weight.wdown = 0;
  } else if (vdown){
    weight.wtop = 0; weight.wdown = 1;
  } else {
    weight.wtop = 0; weight.wdown = 0;
  }

  //printf("g ul/ur/lr/ll: %d/%d/%d/%d\n", weight.gul, weight.gur, weight.glr, weight.gll);
  //printf("w ul/ur/lr/ll: %f/%f/%f/%f\n", weight.wul, weight.wur, weight.wlr, weight.wll);
  //printf("w top/down: %f/%f\n", weight.wtop, weight.wdown);

  return weight;  
}


/** This function interpolates the coarse atmospheric parameters
--- weight:   weight for the interpolation
--- COARSE:   one coarse grid
+++ Return:   interpolated value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float interpolate_coarse(iweights_t weight, float *COARSE){
float ul, ur, ll, lr;

  ul = COARSE[weight.gul];
  ur = COARSE[weight.gur];
  ll = COARSE[weight.gll];
  lr = COARSE[weight.glr];

  return weight.wtop  * (weight.wul*ul + weight.wur*ur) + 
         weight.wdown * (weight.wll*ll + weight.wlr*lr);
}


/** This function computes the surface reflectance
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
--- b:      band
--- bck_:   background reflectance
--- toa_:   TOA reflectance
--- Tg_:    gaseous transmittance
--- boa_:   BOA reflectance
--- dem_:   DEM
--- ill_:   illumination angle
--- sky_:   sky view factor
--- cf_:    topographic correction factor
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int surface_reflectance(par_ll_t *pl2, atc_t *atc, int b, short *bck_, short *toa_, short *Tg_, short *boa_, small *dem_, short *ill_, ushort *sky_, ushort *cf_, brick_t *QAI){
int i, j, p, nx, ny, ne, nf, z, b_sw2;
float fres, gres;
float A = 1.0;
float brdf = 1.0;
iweights_t weights;
float toa, bck, ref, tmp;
short nodata = -9999;
short vnodata;
float E0_, tss_sw2, tsd_sw2;
float sky, ill, cf, f, f0, h0; 
float T, Ts, tss, tsd, tvs, tvd;
float s, rho_p, szen, ms;
float tg, Tso, Tvo;
float  *xy_vz       = NULL;
float  *xy_sz       = NULL;
float  *xy_Tg       = NULL;
float  *xy_Tvo      = NULL;
float  *xy_Tso      = NULL;
float  *xy_brdf     = NULL;
float **xyz_T       = NULL;
float **xyz_Ts      = NULL;
float **xyz_tss     = NULL;
float **xyz_tsd     = NULL;
float **xyz_tvs     = NULL;
float **xyz_tvd     = NULL;
float **xyz_s       = NULL;
float **xyz_rho_p   = NULL;
float **xyz_tss_sw2 = NULL;
float **xyz_tsd_sw2 = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nx  = get_brick_ncols(QAI);
  ny  = get_brick_nrows(QAI);
  fres  = get_brick_res(QAI);
  nf  = get_brick_ncols(atc->xy_view);
  ne  = get_brick_nrows(atc->xy_view);
  gres  = get_brick_res(atc->xy_view);
  vnodata = get_brick_nodata(atc->xy_view, ZEN);
  if ((b_sw2 = find_domain(atc->xy_mod, "SWIR2")) < 0)   return FAILURE;

  if ((xyz_T       = atc_get_band_reshaped(atc->xyz_T,     b))     == NULL) return FAILURE;
  if ((xyz_Ts      = atc_get_band_reshaped(atc->xyz_Ts,    b))     == NULL) return FAILURE;
  if ((xyz_tss     = atc_get_band_reshaped(atc->xyz_tss,   b))     == NULL) return FAILURE;
  if ((xyz_tsd     = atc_get_band_reshaped(atc->xyz_tsd,   b))     == NULL) return FAILURE;
  if ((xyz_tvs     = atc_get_band_reshaped(atc->xyz_tvs,   b))     == NULL) return FAILURE;
  if ((xyz_tvd     = atc_get_band_reshaped(atc->xyz_tvd,   b))     == NULL) return FAILURE;
  if ((xyz_s       = atc_get_band_reshaped(atc->xyz_s,     b))     == NULL) return FAILURE;
  if ((xyz_rho_p   = atc_get_band_reshaped(atc->xyz_rho_p, b))     == NULL) return FAILURE;
  if ((xyz_tss_sw2 = atc_get_band_reshaped(atc->xyz_tss,   b_sw2)) == NULL) return FAILURE;
  if ((xyz_tsd_sw2 = atc_get_band_reshaped(atc->xyz_tsd,   b_sw2)) == NULL) return FAILURE;
  if ((xy_vz       = get_band_float(atc->xy_view, ZEN)) == NULL) return FAILURE;
  if ((xy_sz       = get_band_float(atc->xy_sun,  ZEN)) == NULL) return FAILURE;
  if ((xy_Tg       = get_band_float(atc->xy_Tg,   b))   == NULL) return FAILURE;
  if ((xy_Tvo      = get_band_float(atc->xy_Tvo,  b))   == NULL) return FAILURE;
  if ((xy_Tso      = get_band_float(atc->xy_Tso,  b))   == NULL) return FAILURE;
  if ((xy_brdf     = get_band_float(atc->xy_brdf, b))   == NULL) return FAILURE;
  
  
  if (pl2->dobrdf) cite_me(_CITE_BRDF_);


  #pragma omp parallel private(j, p, z, toa, weights, T, Ts, tss, tsd, tvs, tvd, s, rho_p, tss_sw2, tsd_sw2, tg, sky, ill, cf, szen, ms, Tso, Tvo, E0_, f, f0, h0, bck, tmp, ref) firstprivate(A, brdf) shared(b, b_sw2, nx, ny, nf, ne, gres, fres, vnodata, nodata, toa_, boa_, bck_, QAI, dem_, ill_, sky_, cf_, Tg_, atc, pl2, xyz_T, xyz_Ts, xyz_tss, xyz_tsd, xyz_tvs, xyz_tvd, xyz_s, xyz_rho_p, xyz_tss_sw2, xyz_tsd_sw2, xy_brdf, xy_vz, xy_sz, xy_Tg, xy_Tvo, xy_Tso) default(none) 
  {

    #pragma omp for schedule(guided)

    /** radiometric correction **/
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      p = i*nx+j;

      if (get_off(QAI, p)){ 
        boa_[p] = nodata; continue;}

      toa = toa_[p]/10000.0;


      if (pl2->doatmo){

        z = dem_[p];

        // smooth atc variables
        weights = interpolation_weights(j, i, nf, ne, gres, fres, xy_vz, vnodata);
        T       = interpolate_coarse(weights, xyz_T[z]);
        Ts      = interpolate_coarse(weights, xyz_Ts[z]);    
        tss     = interpolate_coarse(weights, xyz_tss[z]);
        tsd     = interpolate_coarse(weights, xyz_tsd[z]);
        tvs     = interpolate_coarse(weights, xyz_tvs[z]);
        tvd     = interpolate_coarse(weights, xyz_tvd[z]);
        s       = interpolate_coarse(weights, xyz_s[z]); 
        rho_p   = interpolate_coarse(weights, xyz_rho_p[z]); 
        tss_sw2 = interpolate_coarse(weights, xyz_tss[z]);
        tsd_sw2 = interpolate_coarse(weights, xyz_tsd[z]);
        if (pl2->dobrdf) brdf = interpolate_coarse(weights, xy_brdf); 

        if (Tg_ == NULL){
          tg = interpolate_coarse(weights, xy_Tg);
        } else {
          tg = Tg_[p]/10000.0;
        }

        // topographic correction factor
        if (pl2->dotopo){

          sky = sky_[p]/10000.0;
          ill = ill_[p]/10000.0;
          cf  = cf_[p]/10000.0;

          if (ill > 0){

            szen = interpolate_coarse(weights, xy_sz); 
            ms   = cos(szen);
            Tso  = interpolate_coarse(weights, xy_Tso); 
            Tvo  = interpolate_coarse(weights, xy_Tvo); 
            E0_ = atc->E0[b] * Tvo*Tso;
            f  = E0_*tss/(E0_*tsd);
            f0 = E0_*tss_sw2/(E0_*tsd_sw2);
            h0 = (M_PI+2*szen)/(2.0*M_PI);

            A = (ms+cf/f0*f/h0)/(ill+sky*cf/f0*f/h0);
            if (A < 0) A = -10000.0;

          } else A = 1.0;

        }

        if (pl2->doenv){

            // target reflectance
            bck = bck_[p]/10000.0;
            tmp = (1-bck*s);

            ref = A * brdf * 
                (toa/tg*tmp - rho_p*tmp - Ts*tvs*bck) / (Ts*tvd);

        } else {

          // homogeneous target reflectance
          tmp = (toa-rho_p)/tg;
          ref = A * brdf * tmp / (T + s*tmp);

          //printf("toa: %.2f, rho_p: %.4f, tg: %.4f, A: %.2f, brdf: %.2f, T: %.4f, s: %.4f\n", toa, rho_p, tg, A, brdf, T, s);

        }


      } else {

        // top-of-atmosphere
        ref = toa;

      }


      if (ref < 0.0) set_subzero(QAI,    p, true);
      if (ref > 1.0) set_saturation(QAI, p, true);

      if (pl2->erase_cloud && get_cloud(QAI, p) == 2){
        boa_[p] = nodata;
      } else if (ref < -1.0){
        boa_[p] = nodata;
        set_off(QAI, p, true);
      } else if (ref*10000.0 > SHRT_MAX){
        boa_[p] = (short)SHRT_MAX;
      } else {
        boa_[p] = (short)(ref*10000.0);
      }

    }
    }
    
  }

  free((void*)xyz_T);
  free((void*)xyz_Ts);
  free((void*)xyz_tss);
  free((void*)xyz_tsd);
  free((void*)xyz_tvs);
  free((void*)xyz_tvd);
  free((void*)xyz_s);
  free((void*)xyz_rho_p);
  free((void*)xyz_tss_sw2);
  free((void*)xyz_tsd_sw2);

  
  #ifdef FORCE_CLOCK
  proctime_print("compute surface reflectance", TIME);
  #endif

  return SUCCESS;
}


/** This function computes the background reflectance used for correcting
+++ adjacency effects
--- atc:    atmospheric correction factors
--- b:      band
--- toa_:    TOA reflectance
--- Tg_:    gaseous transmittance
--- dem_:   DEM
--- QAI:    Quality Assurance Information (modified)
+++ Return: Background reflectance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *background_reflectance(atc_t *atc, int b, short *toa_, short *Tg_, small *dem_, brick_t *QAI){
int i, j, p, nx, ny, nc, g, z, k;
float F[3], Fr, Fa, km[3], r[3], dF[3], sF, aod, mod;
float rho_p, T, s, F_, res;
float alpha[3], tmp, tg;
float old[3];
short  *ref = NULL;
float **avg = NULL;
short *bck = NULL;
float *imean = NULL;
float *jmean = NULL;
double sum, num;
float w_avg;
float *xy_Tg = NULL;
float **xyz_rho_p = NULL;
float **xyz_T = NULL;
float **xyz_s = NULL;
float **xyz_F = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  cite_me(_CITE_ADJACENCY_);

  
  nx  = get_brick_ncols(QAI);
  ny  = get_brick_nrows(QAI);
  nc  = get_brick_ncells(QAI);
  res = get_brick_res(QAI);

  if ((xyz_rho_p = atc_get_band_reshaped(atc->xyz_rho_p, b)) == NULL) return NULL;
  if ((xyz_T     = atc_get_band_reshaped(atc->xyz_T,     b)) == NULL) return NULL;
  if ((xyz_s     = atc_get_band_reshaped(atc->xyz_s,     b)) == NULL) return NULL;
  if ((xyz_F     = atc_get_band_reshaped(atc->xyz_F,     b)) == NULL) return NULL;
  if (Tg_ == NULL){
    if ((xy_Tg = get_band_float(atc->xy_Tg, b)) == NULL) return NULL;
  }
  
  /** allocate memory **/
  alloc((void**)&ref,   nc, sizeof(short));
  alloc((void**)&imean, ny, sizeof(float));
  alloc((void**)&jmean, nx, sizeof(float));
  alloc_2D((void***)&avg, 3, nc, sizeof(float));
  alloc((void**)&bck, nc, sizeof(short));


  /** radius of rings **/
  km[0] = 0.25;
  km[1] = 1.00;
  km[2] = 4.50;
  for (k=0; k<3; k++) r[k] = km[k]*1000/res;

  /** alpha for exponential moving average **/
  for (k=0; k<3; k++) alpha[k] = 2.0/(r[k]+1.0);

  #ifdef FORCE_DEBUG
  for (k=0; k<3; k++) printf("alpha for exponential moving average is %0.4f (ring %d)\n", alpha[k], k);
  #endif


  /** BOA reflectance without env. correction **/
  
  #pragma omp parallel private(g, z, tg, rho_p, T, s, tmp) shared(b, nc, ref, toa_, QAI, dem_, Tg_, xy_Tg, xyz_rho_p, xyz_T, xyz_s, atc, bck) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){

      if (get_off(QAI, p)) continue;

      g = convert_brick_p2p(QAI, atc->xy_sun, p);
      z = dem_[p];

      if (Tg_ == NULL){
        tg = xy_Tg[g];
      } else {
        tg = Tg_[p]/10000.0;
      }
      
      rho_p = xyz_rho_p[z][g];
      T     = xyz_T[z][g];
      s     = xyz_s[z][g];

      tmp = (toa_[p]/10000.0-rho_p)/tg;
      ref[p] = (short)(tmp / (T + s*tmp) * 10000);

    }

  }

  free((void*)xyz_rho_p); free((void*)xyz_T); free((void*)xyz_s);
  

  /** row means **/
  
  #pragma omp parallel private(j, p, sum, num) shared(nx, ny, imean, ref, QAI) default(none) 
  {

    #pragma omp for schedule(static)
    for (i=0; i<ny; i++){
      sum = num = 0;
      for (j=0, p=i*nx; j<nx; j++, p++){
        if (get_off(QAI, p) || get_cloud(QAI, p) > 0) continue;
        sum += ref[p]/10000.0;
        num++;
      }
      if (num > 0) imean[i] = (float)(sum/num);
    }

  }

  
  /** column means **/

  #pragma omp parallel private(i, p, sum, num) shared(nx, ny, jmean, ref, QAI) default(none) 
  {

    #pragma omp for schedule(static)
    for (j=0; j<nx; j++){
      sum = num = 0;
      for (i=0, p=j; i<ny; i++, p+=nx){
        if (get_off(QAI, p) || get_cloud(QAI, p) > 0) continue;
        sum += ref[p]/10000.0;
        num++;
      }
      if (num > 0) jmean[j] = (float)(sum/num);
    }
    
  }

  
  /** exponential moving average, left to right, then right to left **/
  
  #pragma omp parallel private(k, j, p, old) shared(nx, ny, imean, ref, QAI, avg, alpha) default(none) 
  {

    #pragma omp for schedule(static)
    for (i=0; i<ny; i++){

      for (k=0; k<3; k++) old[k] = imean[i];

      for (j=0, p=i*nx; j<nx; j++, p++){
        if (get_off(QAI, p)) continue;
        for (k=0; k<3; k++){
          if (get_cloud(QAI, p) > 0){
            avg[k][p] = old[k];
          } else {
            avg[k][p] = alpha[k]*ref[p]/10000.0 + (1.0-alpha[k])*old[k];
            old[k] = avg[k][p];
          }
        }
      }
      p--;

      for (j=nx-1; j>=0; j--, p--){
        if (get_off(QAI, p)) continue;
        for (k=0; k<3; k++){
          if (get_cloud(QAI, p) > 0){
            avg[k][p] = old[k];
          } else {
            avg[k][p] = alpha[k]*avg[k][p] + (1.0-alpha[k])*old[k];
            old[k] = avg[k][p];
          }
        }
      }

    }
    
  }
  

  /** exponential moving average, top to bottom, then bottom to top **/
  
  #pragma omp parallel private(k, i, p, old) shared(nx, ny, jmean, QAI, avg, alpha) default(none) 
  {

    #pragma omp for schedule(static)
    for (j=0; j<nx; j++){

      for (k=0; k<3; k++) old[k] = jmean[j];

      for (i=0, p=j; i<ny; i++, p+=nx){
        if (get_off(QAI, p)) continue;
        for (k=0; k<3; k++){
          if (get_cloud(QAI, p) > 0){
            avg[k][p] = old[k];
          } else {
            avg[k][p] = alpha[k]*avg[k][p] + (1.0-alpha[k])*old[k];
            old[k] = avg[k][p];
          }
        }
      }
      p-=nx;

      for (i=ny-1; i>=0; i--, p-=nx){
        if (get_off(QAI, p)) continue;
        for (k=0; k<3; k++){
          if (get_cloud(QAI, p) > 0){
            avg[k][p] = old[k];
          } else {
            avg[k][p] = alpha[k]*avg[k][p] + (1.0-alpha[k])*old[k];
            old[k] = avg[k][p];
          }
        }
      }

    }

  }

  
  /** relative influence of rings **/
  mod = mod_elev_scale(atc->mod[b], 1, atc->Hr);
  aod = atc->aod[b];

  for (k=0; k<3; k++){
    Fa = env_weight_aerosol(km[k]);
    Fr = env_weight_molecular(km[k]);
    F[k] = env_weight(aod, mod, Fa, Fr);
  }

  dF[0] = F[0]; dF[1] = F[1]-F[0]; dF[2] = F[2]-F[1];
  sF = dF[0]+dF[1]+dF[2];
  
  #ifdef FORCE_DEBUG
  for (k=0; k<3; k++) printf("relative influence of ring %d: %.2f\n", k, dF[k]/sF);
  #endif


  /** background reflectance **/
  
  #pragma omp parallel private(g, z, F_, w_avg) shared(b, nc, bck, ref, QAI, dem_, atc, dF, sF, xyz_F, avg) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){

      if (get_off(QAI, p)) continue;

      g = convert_brick_p2p(QAI, atc->xy_sun, p);
      z = dem_[p];

      F_ = xyz_F[z][g];

      w_avg = (dF[0]*avg[0][p] + dF[1]*avg[1][p] + dF[2]*avg[2][p])/sF;
      bck[p] = (short)((ref[p]/10000.0*F_ + (1-F_)*w_avg)*10000);

    }

  }
  
  free((void*)xyz_F);
  free((void*)ref);
  free((void*)imean);
  free((void*)jmean);
  free_2D((void**)avg, 3);


  #ifdef FORCE_CLOCK
  proctime_print("compute background reflectance", TIME);
  #endif
  
  return bck;
}


/** This function computes all atmospheric parameters that are a function 
+++ of view or sun angles.
--- pl2:    L2 parameters
--- meta:   metadata
--- atc:    atmospheric correction factors
--- TOP:    Topographic Derivatives
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int atmo_angledep(par_ll_t *pl2, meta_t *meta, atc_t *atc, top_t *TOP, brick_t *QAI){
int e, f, g, ne, nf, b, nb;
int dem, doy;
float ms, mv, psi, Hr;
float ozone;
double lon, lat;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  ne = get_brick_ncols(atc->xy_mod);
  nf = get_brick_nrows(atc->xy_mod);
  nb = get_brick_nbands(atc->xy_mod);

  #pragma omp parallel private(g, b, ms, mv, psi, Hr, doy, lon, lat, ozone, dem) shared(nb, ne, nf, meta, atc, pl2, QAI, TOP) default(none) 
  {

    #pragma omp for collapse(2) schedule(static)

    for (e=0; e<ne; e++){
    for (f=0; f<nf; f++){

      g = e*nf+f;

      if (is_brick_nodata(atc->xy_view, 0, g)) continue;

      // bi-directional correction parameter
      brdf_factor(atc->xy_sun, atc->xy_view, atc->xy_brdf, g);

      // relative air mass
      ms = get_brick(atc->xy_sun,  cZEN, g);
      mv = get_brick(atc->xy_view, cZEN, g);

      // cosine of backscattering angle
      psi = backscatter(ms, mv, 
        get_brick(atc->xy_sun, AZI, g), get_brick(atc->xy_view, AZI, g));
      set_brick(atc->xy_psi, 0, g, psi);

      // phase functions for molecular and aerosol scattering
      set_brick(atc->xy_Pr, 0, g, phase_molecular(psi));
      set_brick(atc->xy_Pa, 0, g, phase_aerosol(psi, atc->tthg.hg));

      // ozone amount (very rough approximation)
      get_brick_geo(atc->xy_sun, f, e, &lon, &lat);
      doy   = get_brick_doy(atc->xy_sun, 0);
      ozone = ozone_amount(lon, lat, doy);

      for (b=0; b<nb; b++){

        // gaseous transmittance
        set_brick(atc->xy_Tsw, b, g, wvp_transmitt(atc->wvp, ms, meta->cal[b].rsr_band));
        set_brick(atc->xy_Tvw, b, g, wvp_transmitt(atc->wvp, mv, meta->cal[b].rsr_band));
        set_brick(atc->xy_Tso, b, g, ozone_transmitt(ozone, ms, meta->cal[b].rsr_band));
        set_brick(atc->xy_Tvo, b, g, ozone_transmitt(ozone, mv, meta->cal[b].rsr_band));
        
        set_brick(atc->xy_Tg, b, g, gas_transmitt(
          get_brick(atc->xy_Tsw, b, g), get_brick(atc->xy_Tvw, b, g),
          get_brick(atc->xy_Tso, b, g), get_brick(atc->xy_Tvo, b, g)));

      }

      // average elevation
      average_elevation_cell(g, atc->xy_dem, TOP->dem, QAI);
      dem = get_brick(atc->xy_dem, 0, g);

      // correct rayleigh for elevation
      Hr = mod_elev_factor(dem*atc->dem.step+atc->dem.min+atc->dem.step/2);
      set_brick(atc->xy_Hr, 0, g, Hr);
      for (b=0; b<nb; b++) set_brick(atc->xy_mod, b, g, mod_elev_scale(atc->mod[b], 1, Hr));

      // Fresnel reflection for AOD from water retrieval 
      // --> assumed that the water body is flat -> incidence angle = szen
      set_brick(atc->xy_fresnel, 0, g, fresnel_reflection(get_brick(atc->xy_sun, ZEN, g)));

    }
    }

  }


  #ifdef FORCE_DEBUG
  print_brick_info(atc->xy_brdf);  set_brick_open(atc->xy_brdf,  OPEN_CREATE); write_brick(atc->xy_brdf);
  print_brick_info(atc->xy_psi); set_brick_open(atc->xy_psi, OPEN_CREATE); write_brick(atc->xy_psi);
  print_brick_info(atc->xy_Pr); set_brick_open(atc->xy_Pr, OPEN_CREATE); write_brick(atc->xy_Pr);
  print_brick_info(atc->xy_Pa); set_brick_open(atc->xy_Pa, OPEN_CREATE); write_brick(atc->xy_Pa);
  print_brick_info(atc->xy_Hr); set_brick_open(atc->xy_Hr, OPEN_CREATE); write_brick(atc->xy_Hr);
  print_brick_info(atc->xy_Ha); set_brick_open(atc->xy_Ha, OPEN_CREATE); write_brick(atc->xy_Ha);
  print_brick_info(atc->xy_Tsw); set_brick_open(atc->xy_Tsw, OPEN_CREATE); write_brick(atc->xy_Tsw);
  print_brick_info(atc->xy_Tvw); set_brick_open(atc->xy_Tvw, OPEN_CREATE); write_brick(atc->xy_Tvw);
  print_brick_info(atc->xy_Tso); set_brick_open(atc->xy_Tso, OPEN_CREATE); write_brick(atc->xy_Tso);
  print_brick_info(atc->xy_Tvo); set_brick_open(atc->xy_Tvo, OPEN_CREATE); write_brick(atc->xy_Tvo);
  print_brick_info(atc->xy_Tg); set_brick_open(atc->xy_Tg, OPEN_CREATE); write_brick(atc->xy_Tg);
  print_brick_info(atc->xy_mod); set_brick_open(atc->xy_mod, OPEN_CREATE); write_brick(atc->xy_mod);
  print_brick_info(atc->xy_fresnel); set_brick_open(atc->xy_fresnel, OPEN_CREATE); write_brick(atc->xy_fresnel);
  print_brick_info(atc->xy_dem); set_brick_open(atc->xy_dem, OPEN_CREATE); write_brick(atc->xy_dem);
  #endif


  #ifdef FORCE_CLOCK
  proctime_print("angle-dependent atm. modelling", TIME);
  #endif

  return SUCCESS;
}


/** This function computes all atmospheric parameters that are a function 
+++ of terrain elevation.
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
--- QAI:    Quality Assurance Information
--- TOP:    Topographic Derivatives
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int atmo_elevdep(par_ll_t *pl2, atc_t *atc, brick_t *QAI, top_t *TOP){
int e, f, g, ne, nf, b, nb, z, nz, p, nc, b_green;
float z0, zmed;
float ms, mv, Pr, Pa;
float od, mod, aod, Hr, Ha, rho_p;
float T, Ts, tsd, tss, Tv, tvd, tvs, F = 1.0;
small  *dem_ = NULL;
float **xyz_aod = NULL;
small *xy_interp = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  ne = get_brick_ncols(atc->xy_mod);
  nf = get_brick_nrows(atc->xy_mod);
  nb = get_brick_nbands(atc->xy_mod);
  nc = get_brick_ncells(QAI);
  if ((b_green = find_domain(atc->xy_mod, "GREEN")) < 0) return FAILURE;
  if ((dem_ =  get_band_small(TOP->dem, 0)) == NULL) return FAILURE;
  nz = NPOW_08;

  z0 = (atc->dem.min+atc->dem.step/2.0);

  #pragma omp parallel private(f, g, b, z, zmed, Pr, Pa, ms, mv, Ha, Hr, mod, aod, od, T, Ts, Tv, tsd, tss, tvd, tvs, rho_p) firstprivate(F) shared(nb, ne, nf, nz, z0, atc, pl2) default(none) 
  {

    //#pragma omp for collapse(2) schedule(static)
    #pragma omp for schedule(static)

    for (e=0; e<ne; e++){
    for (f=0; f<nf; f++){

      g = e*nf+f;
      
      if (is_brick_nodata(atc->xy_view, 0, g)) continue;

      // phase functions
      Pr = get_brick(atc->xy_Pr, 0, g);
      Pa = get_brick(atc->xy_Pa, 0, g);
      
      // relative air mass
      ms = get_brick(atc->xy_sun,  cZEN, g);
      mv = get_brick(atc->xy_view, cZEN, g);

      // for every possible elevation (100m steps)
      for (z=0, zmed=z0; z<nz; z++, zmed+=atc->dem.step){

        // correct mod for elevation
        Hr = mod_elev_factor(zmed);
        Ha = aod_elev_factor(zmed, atc->Hp);

        set_brick(atc->xyz_Hr[z], 0, g, Hr);
        set_brick(atc->xyz_Ha[z], 0, g, Ha);


        // down/up-welling scattering transmittances
        for (b=0; b<nb; b++){

          // correct mod and aod for elevation
          mod = mod_elev_scale(atc->mod[b], 1, Hr);
          set_brick(atc->xyz_mod[z], b, g, mod);

          if (atc->aodmap){
            aod = aod_elev_scale(get_brick(atc->xy_aod, b, g), atc->Ha, Ha);
          } else {
            aod = aod_elev_scale(atc->aod[b], atc->Ha, Ha);
          }
          set_brick(atc->xyz_aod[z], b, g, aod);

          // optical depth
          od = optical_depth(aod, mod);
          set_brick(atc->xyz_od[z], b, g, od);

          // scattering transmittance
          T = scatt_transmitt(aod, mod, od, ms, mv, 
                &Ts, &Tv, &tsd, &tss, &tvd, &tvs);
          set_brick(atc->xyz_T[z], b, g, T);
          set_brick(atc->xyz_Ts[z], b, g, Ts);
          set_brick(atc->xyz_Tv[z], b, g, Tv);
          set_brick(atc->xyz_tsd[z], b, g, tsd);
          set_brick(atc->xyz_tvd[z], b, g, tvd);
          set_brick(atc->xyz_tss[z], b, g, tss);
          set_brick(atc->xyz_tvs[z], b, g, tvs);

          // path reflectance
          rho_p = path_ref(pl2->domulti, atc->tthg.sob, aod, mod, od, Pa, Pr, tsd, tvd, ms, mv);
          set_brick(atc->xyz_rho_p[z], b, g, rho_p);

          // spherical albedo
          set_brick(atc->xyz_s[z], b, g, sphere_albedo(aod, mod, od));

          // environmental weighting function
          if (pl2->doenv) F = env_weight(aod, mod, atc->Fa, atc->Fr);
          set_brick(atc->xyz_F[z], b, g, F);

        }

      }

    }
    }
    
  }
  

  if ((xyz_aod   = atc_get_band_reshaped(atc->xyz_aod, b_green)) == NULL) return FAILURE;
  if ((xy_interp = get_band_small(atc->xy_interp, 0))            == NULL) return FAILURE;
  
  // high aerosol flag
  #pragma omp parallel private(g, z) shared(nc, QAI, dem_, xyz_aod, xy_interp, atc, pl2) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){

      if (get_off(QAI, p)) continue;
      g = convert_brick_p2p(QAI, atc->xy_sun, p);
      z = dem_[p];

      if (!atc->aodmap){
        set_aerosol(QAI, p, 3);
      } else if (xyz_aod[z][g] > 0.6){
        set_aerosol(QAI, p, 2);
      } else if (xy_interp[g]){
        set_aerosol(QAI, p, 1);
      }

    }

  }

  free((void*)xyz_aod);

  #ifdef FORCE_CLOCK
  proctime_print("elevation-dependent atm. modelling", TIME);
  #endif

  return SUCCESS;
}


/** This function applies the AOI and sets QAI to nodata
--- QAI:    Quality Assurance Information
--- AOI:    Area of Interest
+++ Return: SUCCESS/FAILURE/CANCEL
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int apply_aoi(brick_t *QAI, brick_t *AOI){
int nc, p;
small *aoi_ = NULL;


  if (AOI == NULL) return CANCEL;

  nc = get_brick_ncells(QAI);
  aoi_ = get_band_small(AOI, 0);

  #pragma omp parallel shared(QAI, aoi_, nc) default(none) 
  {
    
    #pragma omp for
    for (p=0; p<nc; p++){
      if (!aoi_[p]) set_off(QAI, p, true);
    }

  }

  free_brick(AOI);

  return SUCCESS;
}


/** This function compiles the BOA product ready to be output
--- pl2:    L2 parameters
--- mission: mission ID
--- atc:    atmospheric correction factors
--- cube:   data cube parameters
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information
--- WVP:    water vapor
--- TOP:    Topographic Derivatives
+++ Return: BOA brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_boa(par_ll_t *pl2, int mission, atc_t *atc, cube_t *cube, brick_t *TOA, brick_t *QAI, brick_t *WVP, top_t *TOP){
int p, nc;
int b, b_red, b_nir, b_sw1;
#ifndef ACIX
int b_, nb_, bands[10], ndomain = 10;
char domains[10][NPOW_10] = { "BLUE", "GREEN", "RED", 
                           "REDEDGE1", "REDEDGE2",
                           "REDEDGE3", "BROADNIR",
                           "NIR", "SWIR1", "SWIR2" };
#else
int b_, nb_, bands[12], ndomain = 12;
char domains[12][NPOW_10] = { "ULTRABLUE", "BLUE", "GREEN", "RED", 
                           "REDEDGE1", "REDEDGE2",
                           "REDEDGE3", "BROADNIR",
                           "NIR", "VAPOR", "SWIR1", "SWIR2" };
#endif
char fname[NPOW_10];
char product[NPOW_02];
char domain[NPOW_10];
char bandname[NPOW_10];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
short nodata = -9999;
small  *dem_     = NULL;
short  *ill_     = NULL;
ushort *sky_     = NULL;
ushort *cf_      = NULL;
short  *bck_     = NULL;
short  *Tg_      = NULL;
short  *toa_     = NULL;
short  *boa_     = NULL;
short  **boa__   = NULL;

brick_t *BOA = TOA;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  
  
  nc  = get_brick_ncells(BOA);
  if ((dem_ =  get_band_small(TOP->dem, 0))  == NULL) return NULL;

  if (pl2->dotopo){
    if ((ill_ =  get_band_short(TOP->ill,  0)) == NULL) return NULL;
    if ((sky_ =  get_band_ushort(TOP->sky, 0)) == NULL) return NULL;
    if ((cf_  =  get_band_ushort(TOP->c,   0)) == NULL) return NULL;
  }


  // number of BOA output bands
  for (b=0, nb_=0; b<ndomain; b++){
    if ((bands[nb_] = find_domain(BOA, domains[b])) >= 0) nb_++;
  }

  #ifdef FORCE_DEBUG
  printf("nb is %d, nb_ is %d\n", get_brick_nbands(BOA), nb_);
  #endif


  // final radiometric processing and band reordering
  for (b_=0; b_<nb_; b_++){

    if ((b = bands[b_]) < 0) continue;

    if ((toa_ = get_band_short(BOA, b))  == NULL) return NULL;
    if ((boa_ = get_band_short(BOA, b_)) == NULL) return NULL;

    if (pl2->doatmo && mission == SENTINEL2){
      if ((Tg_ = gas_transmittance(atc, b, WVP, QAI)) == NULL){
      printf("error in gas transmittance.\n"); return NULL;}
    } else Tg_ = NULL;
  
    if (pl2->doatmo && pl2->doenv){
      if ((bck_ = background_reflectance(atc, b, toa_, Tg_, dem_, QAI)) == NULL){
      printf("error in background reflectance.\n"); return NULL;}
    } else bck_ = NULL;
 
    if (surface_reflectance(pl2, atc, b, bck_, toa_, Tg_, boa_, dem_, ill_, sky_, cf_, QAI) == FAILURE){
    printf("error in surface reflectance.\n"); return NULL;}

    set_brick_wavelength(BOA, b_, get_brick_wavelength(BOA, b));
    get_brick_domain(BOA,   b, domain,   NPOW_10); set_brick_domain(BOA,   b_, domain);
    get_brick_bandname(BOA, b, bandname, NPOW_10); set_brick_bandname(BOA, b_, bandname);

    if (Tg_ != NULL) free((void*)Tg_);  
    Tg_ = NULL;
    if (bck_ != NULL) free((void*)bck_); 
    bck_ = NULL;

  }


  // force nodata in all bands
  if ((boa__ = get_bands_short(BOA)) == NULL) return NULL;
  
  #pragma omp parallel private(b_) shared(nc, nb_, QAI, boa__, nodata) default(none) 
  {
    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)){
        for (b_=0; b_<nb_; b_++) boa__[b_][p] = nodata;
      }
    }
  }


  // resize the brick
  if (reallocate_brick(BOA, nb_) == FAILURE){
    printf("error in reallocating brick.\n"); return NULL;}


  // reproj the data
  if (pl2->doreproj){
    if (warp_from_brick_to_unknown_brick(pl2->dotile, pl2->resample, pl2->nthread, BOA, cube) == FAILURE){
      printf("warping BOA failed.\n"); return NULL;}
  }


  // set metadata
  if (pl2->doatmo){
    copy_string(product, NPOW_02, "BOA");
  } else {
    copy_string(product, NPOW_02, "TOA");
  }
  set_brick_product(BOA, product);
  set_brick_name(BOA, "FORCE Level 2 Processing System");
  get_brick_compactdate(BOA, 0, date, NPOW_04);
  get_brick_sensor(BOA, 0, sensor, NPOW_04);
  
  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  
  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(BOA, fname);
  set_brick_open(BOA, OPEN_MERGE);
  set_brick_explode(BOA, false);
  set_brick_format(BOA, &pl2->gdalopt);

  if ((b_nir = find_domain(BOA, "NIR"))   < 0) return NULL;
  if ((b_sw1 = find_domain(BOA, "SWIR1")) < 0) return NULL;
  if ((b_red = find_domain(BOA, "RED"))   < 0) return NULL;

  
  #ifdef FORCE_CLOCK
  proctime_print("compile BOA", TIME);
  #endif

  return BOA;
}


/** This function compiles the QAI product ready to be output
--- pl2:    L2 parameters
--- cube:   data cube parameters
--- QAI:    Quality Assurance Information
+++ Return: QAI brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_qai(par_ll_t *pl2, cube_t *cube, brick_t *QAI){
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
brick_t *QA = QAI;
int p, nc;
short *qa_ = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  
  // reproj the data
  if (pl2->doreproj){
    if (warp_from_brick_to_unknown_brick(pl2->dotile, 0, pl2->nthread, QA, cube) == FAILURE){
      printf("warping QAI failed.\n"); return NULL;}
  }

  // make sure that OFF bit is set exclusively
  nc = get_brick_ncells(QA);
  qa_ = get_band_short(QA, 0);
  for (p=0; p<nc; p++){
    if (get_off(QA, p)) qa_[p] = 1;
  }
  

  // set metadata
  copy_string(product, NPOW_02, "QAI");
  set_brick_product(QA, product);
  set_brick_name(QA, "FORCE Level 2 Processing System");
  get_brick_sensor(QA, 0, sensor, NPOW_04);
  get_brick_compactdate(QA, 0, date, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  
  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(QA, fname);
  set_brick_open(QA, OPEN_UPDATE);
  set_brick_explode(QA, false);
  set_brick_format(QA, &pl2->gdalopt);
  set_brick_nodata(QA, 0, 1);
  set_brick_scale(QA, 0, 1);
  set_brick_wavelength(QA, 0, 1);
  set_brick_unit(QA, 0, "unknown");
  set_brick_domain(QA, 0, "QAI");
  set_brick_bandname(QA, 0, "Quality assurance information");

 
  #ifdef FORCE_CLOCK
  proctime_print("compile QAI", TIME);
  #endif

  return QA;
}


/** This function compiles the DST product ready to be output
--- pl2:    L2 parameters
--- cube:   data cube parameters
--- QAI:    Quality Assurance Information
+++ Return: DST brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_dst(par_ll_t *pl2, cube_t *cube, brick_t *QAI){
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
short nodata = -9999;
brick_t *DST = NULL;
short  *dst_ = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  DST = copy_brick(QAI, 1, _DT_SHORT_);
  set_brick_nodata(DST, 0, nodata);

  if ((dst_ = get_band_short(DST, 0)) == NULL) return NULL;

  cloud_distance(QAI, nodata, dst_);


  // set metadata
  copy_string(product, NPOW_02, "DST");
  set_brick_product(DST, product);
  set_brick_name(DST, "FORCE Level 2 Processing System");
  get_brick_compactdate(DST, 0, date, NPOW_04);
  get_brick_sensor(DST, 0, sensor, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  
  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(DST, fname);
  set_brick_open(DST, OPEN_MERGE);
  set_brick_explode(DST, false);
  set_brick_format(DST, &pl2->gdalopt);
  set_brick_scale(DST, 0, 1);
  set_brick_wavelength(DST, 0, 1);
  set_brick_unit(DST, 0, "unknown");
  set_brick_domain(DST, 0, product);
  set_brick_bandname(DST, 0, "Cloud/shadow/snow distance");


  #ifdef FORCE_CLOCK
  proctime_print("compile DST", TIME);
  #endif

  return DST;
}


/** This function compiles the OVV product ready to be output
--- pl2:    L2 parameters
--- cube:   BOA brick
--- QAI:    Quality Assurance Information
+++ Return: OVV brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_ovv(par_ll_t *pl2, brick_t *BOA, brick_t *QAI){
int i, j, p, i_, j_, p_, b;
int nx, ny, nx_, ny_, nc_;
double res, res_, step, scale;
gdalopt_t format;
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
brick_t *OVV = NULL;
short **ovv_ = NULL;
short  *red_ = NULL;
short  *green_ = NULL;
short  *blue_ = NULL;
enum { R, G, B };


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);
  res = get_brick_res(QAI);

  if (res < 1){
    res_ = 0.0015;
  } else {
    res_ = 150;
  }
  nx_ = nx*(res/res_);
  ny_ = ny*(res/res_);
  nc_ = nx_*ny_;

  step = res_/res;
  scale = 2500;

  OVV = copy_brick(QAI, 3, _DT_NONE_);
  default_gdaloptions(_FMT_JPEG_, &format);
  set_brick_format(OVV, &format);
  set_brick_res(OVV, res_);
  set_brick_ncols(OVV, nx_);
  set_brick_nrows(OVV, ny_);
  allocate_brick_bands(OVV, 3, nc_, _DT_SHORT_);
  for (b=0; b<3; b++) set_brick_nodata(OVV, b, 0);

  if ((ovv_   = get_bands_short(OVV)) == NULL) return NULL;
  if ((red_   = get_domain_short(BOA, "RED"))   == NULL) return NULL;
  if ((green_ = get_domain_short(BOA, "GREEN")) == NULL) return NULL;
  if ((blue_  = get_domain_short(BOA, "BLUE"))  == NULL) return NULL;


  #pragma omp parallel private(j_,p_,i,j,p) shared(nx,ny,nx_,ny_,step,scale,ovv_,red_,green_,blue_,QAI) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i_=0; i_<ny_; i_++){
    for (j_=0; j_<nx_; j_++){
      
      p_ = i_*nx_+j_;

      i = i_*step;
      j = j_*step;
      p = i*nx+j;

      if (get_off(QAI, p)){
        ovv_[R][p_] = 0;
        ovv_[G][p_] = 0;
        ovv_[B][p_] = 0;
      } else if (get_cloud(QAI, p) == 3){
        ovv_[R][p_] = 255;
        ovv_[G][p_] = 0;
        ovv_[B][p_] = 0;
      } else if (get_cloud(QAI, p) > 0){
        ovv_[R][p_] = 255;
        ovv_[G][p_] = 0;
        ovv_[B][p_] = 255;
      } else if (get_shadow(QAI, p)){
        ovv_[R][p_] = 0;
        ovv_[G][p_] = 255;
        ovv_[B][p_] = 255;
      } else if (get_snow(QAI, p)){
        ovv_[R][p_] = 255;
        ovv_[G][p_] = 255;
        ovv_[B][p_] = 0;
      } else if (get_saturation(QAI, p)){
        ovv_[R][p_] = 255;
        ovv_[G][p_] = 127;
        ovv_[B][p_] = 39;
      } else if (get_subzero(QAI, p)){
        ovv_[R][p_] = 34;
        ovv_[G][p_] = 177;
        ovv_[B][p_] = 76;
      } else {
        if (red_[p] < 0){
          ovv_[R][p_] = 0;
        } else if (red_[p] > scale){
          ovv_[R][p_] = 255;
        } else {
          ovv_[R][p_] = (small)(red_[p]/scale*255);
        }
        if (green_[p] < 0){
          ovv_[G][p_] = 0;
        } else if (green_[p] > scale){
          ovv_[G][p_] = 255;
        } else {
          ovv_[G][p_] = (small)(green_[p]/scale*255);
        }
        if (blue_[p] < 0){
          ovv_[B][p_] = 0;
        } else if (blue_[p] > scale){
          ovv_[B][p_] = 255;
        } else {
          ovv_[B][p_] = (small)(blue_[p]/scale*255);
        }
      }
    }
    }
  }


  // set metadata
  copy_string(product, NPOW_02, "OVV");
  set_brick_product(OVV, product);
  set_brick_name(OVV, "FORCE Level 2 Processing System");
  get_brick_compactdate(OVV, 0, date, NPOW_04);
  get_brick_sensor(OVV, 0, sensor, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(OVV, fname);
  set_brick_open(OVV, OPEN_UPDATE);
  set_brick_explode(OVV, false);
  
  for (b=0; b<3; b++){
    set_brick_scale(OVV, b, 1);
    set_brick_wavelength(OVV, b, 1);
    set_brick_unit(OVV, b, "unknown");
    set_brick_domain(OVV, b, product);
    set_brick_bandname(OVV, b, "Product overview");
  }


  #ifdef FORCE_CLOCK
  proctime_print("compile OVV", TIME);
  #endif

  return OVV;
}


/** This function compiles the VZN product ready to be output
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
--- cube:   data cube parameters
--- QAI:    Quality Assurance Information
+++ Return: VZN brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_vzn(par_ll_t *pl2, atc_t *atc, cube_t *cube, brick_t *QAI){
int e, f, g, i, j, p, ne, nf, ng, nx, ny, k = 0;
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
iweights_t weights;
short nodata = -9999;
float res, fres, gres;
brick_t *VZN      = NULL;
brick_t *COARSE   = NULL;
short   *vzn_     = NULL;
short   *coarse_  = NULL;
float   *fcoarse_ = NULL;
double  *grid_x   = NULL;
double  *grid_y   = NULL;
double  *grid_z   = NULL;
double  *grid_i   = NULL;
GDALGridAlgorithm eAlgorithm = GGA_InverseDistanceToAPower;
void *pOptions = NULL;
GDALDataType eOutputType = GDT_Float64;

  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif


  // copy coarse view zenith
  COARSE = copy_brick(atc->xy_view, 1, _DT_SHORT_);
  set_brick_nodata(COARSE, 0, nodata);
  ng = get_brick_ncells(COARSE);
  if ((coarse_ = get_band_short(COARSE, ZEN)) == NULL) return NULL;
  
  for (g=0; g<ng; g++){
    if (is_brick_nodata(atc->xy_view, ZEN, g)){
      coarse_[g] = nodata;
    } else {
      coarse_[g] = (short)(get_brick(atc->xy_view, ZEN, g)*_R2D_CONV_*100);
    }
  }
  
  
  res = cube->res;
  cube->res = get_brick_res(COARSE);

  // reproj the data
  if (pl2->doreproj){
    if (warp_from_brick_to_unknown_brick(pl2->dotile, pl2->resample, pl2->nthread, COARSE, cube) == FAILURE){
      printf("warping VZN failed.\n"); return NULL;}
  }

  cube->res = res;



  nf = get_brick_ncols(COARSE);
  ne = get_brick_nrows(COARSE);
  ng = get_brick_ncells(COARSE);
  gres  = get_brick_res(COARSE);
  if ((coarse_ = get_band_short(COARSE, ZEN)) == NULL) return NULL;

  ParseAlgorithmAndOptions(szAlgNameInvDist, &eAlgorithm, &pOptions);

  // allocate memory
  alloc((void**)&grid_x, ng, sizeof(double));
  alloc((void**)&grid_y, ng, sizeof(double));
  alloc((void**)&grid_z, ng, sizeof(double));
  alloc((void**)&grid_i, ng, sizeof(double));

  // copy to arrays
  for (e=0, g=0; e<ne; e++){
  for (f=0; f<nf; f++, g++){
    if (coarse_[g] == nodata) continue;
    grid_x[k] = f;
    grid_y[k] = e;
    grid_z[k] = coarse_[g];
    k++;
  }
  }

  // compute interpolation
  GDALGridCreate(eAlgorithm, pOptions, k, grid_x, grid_y, grid_z,
    0, nf-1, 0, ne-1, nf, ne, eOutputType, (void*)grid_i, NULL, NULL);

  // copy to float array
  alloc((void**)&fcoarse_, ng, sizeof(float));
  for (g=0; g<ng; g++){
    if (coarse_[g] == nodata){
      fcoarse_[g] = (float)grid_i[g];
    } else {
      fcoarse_[g] = (float)coarse_[g];
    }
  }

  // clean
  CPLFree(pOptions);
  free((void**)grid_x);
  free((void**)grid_y);
  free((void**)grid_z);
  free((void**)grid_i);
  free_brick(COARSE);
  

  // interpolate at full res
  VZN = copy_brick(QAI, 1, _DT_SHORT_);
  set_brick_nodata(VZN, 0, nodata);

  nx = get_brick_ncols(VZN);
  ny = get_brick_nrows(VZN);
  fres  = get_brick_res(VZN);
  if ((vzn_  = get_band_short(VZN, 0)) == NULL) return NULL;


  #pragma omp parallel private(j, p, weights) shared(nx, ny, nf, ne, fres, gres, QAI, vzn_, fcoarse_, nodata) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      p = i*nx+j;
      if (get_off(QAI, p)){
        vzn_[p] = nodata;
      } else {
        weights = interpolation_weights(j, i, nf, ne, gres, fres, fcoarse_, nodata);
        vzn_[p] = (short)(interpolate_coarse(weights, fcoarse_));
      }
    }
    }
  }
  
  free((void*)fcoarse_);

  // set metadata
  copy_string(product, NPOW_02, "VZN");
  set_brick_product(VZN, product);
  set_brick_name(VZN, "FORCE Level 2 Processing System");
  get_brick_compactdate(VZN, 0, date, NPOW_04);
  get_brick_sensor(VZN, 0, sensor, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(VZN, fname);
  set_brick_open(VZN, OPEN_MERGE);
  set_brick_explode(VZN, false);
  set_brick_format(VZN, &pl2->gdalopt);
  set_brick_scale(VZN, 0, 100);
  set_brick_wavelength(VZN, 0, 1);
  set_brick_unit(VZN, 0, "unknown");
  set_brick_domain(VZN, 0, product);
  set_brick_bandname(VZN, 0, "View zenith");


  #ifdef FORCE_CLOCK
  proctime_print("compile VZN", TIME);
  #endif

  return VZN;
}


/** This function compiles the HOT product ready to be output
--- pl2:    L2 parameters
--- cube:   data cube parameters
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information
+++ Return: HOT brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_hot(par_ll_t *pl2, cube_t *cube, brick_t *TOA, brick_t *QAI){
int p, nc;
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
short nodata = -9999;
short  *blue_    = NULL;
short  *red_     = NULL;
brick_t *HOT = NULL;
short *hot_ = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif


  HOT = copy_brick(QAI, 1, _DT_SHORT_);
  set_brick_nodata(HOT, 0, nodata);

  nc = get_brick_ncells(HOT);
  if ((hot_ = get_band_short(HOT, 0)) == NULL) return NULL;
  if ((blue_ = get_domain_short(TOA, "BLUE")) == NULL) return NULL;
  if ((red_  = get_domain_short(TOA, "RED"))  == NULL) return NULL;


  #pragma omp parallel shared(nc, QAI, hot_, blue_, red_, nodata) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)){
        hot_[p] = nodata;
      } else {
        hot_[p] = (short)(blue_[p] - 0.5*red_[p] - 800);
      }
    }
  }


  // reproj the data
  if (pl2->doreproj){
    if (warp_from_brick_to_unknown_brick(pl2->dotile, pl2->resample, pl2->nthread, HOT, cube) == FAILURE){
      printf("warping HOT failed.\n"); return NULL;}
  }


  // set metadata

  copy_string(product, NPOW_02, "HOT");
  set_brick_product(HOT, product);
  set_brick_name(HOT, "FORCE Level 2 Processing System");
  get_brick_compactdate(HOT, 0, date, NPOW_04);
  get_brick_sensor(HOT, 0, sensor, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(HOT, fname);
  set_brick_open(HOT, OPEN_MERGE);
  set_brick_explode(HOT, false);
  set_brick_format(HOT, &pl2->gdalopt);
  set_brick_wavelength(HOT, 0, 1);
  set_brick_unit(HOT, 0, "unknown");
  set_brick_domain(HOT, 0, product);
  set_brick_bandname(HOT, 0, "Haze optimized transform");

  
  #ifdef FORCE_CLOCK
  proctime_print("compile HOT", TIME);
  #endif

  return HOT;
}


/** This function compiles the AOD product ready to be output
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
--- cube:   data cube parameters
--- QAI:    Quality Assurance Information
--- TOP:    Topographic Derivatives
+++ Return: AOD brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_aod(par_ll_t *pl2, atc_t *atc, cube_t *cube, brick_t *QAI, top_t *TOP){
int i, j, p, nx, ny, ne, nf, z;
int b_green;
float fres, gres;
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
float wvl;
iweights_t weights;
short nodata = -9999;
short vnodata;
small  *dem_     = NULL;
brick_t *AOD = NULL;
short  *aod_ = NULL;
float **xy_aod_ = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif



  AOD = copy_brick(QAI, 1, _DT_SHORT_);
  set_brick_nodata(AOD, 0, nodata);

  nx = get_brick_ncols(AOD);
  ny = get_brick_nrows(AOD);
  fres  = get_brick_res(QAI);
  if ((aod_ = get_band_short(AOD, 0)) == NULL) return NULL;
  if ((dem_ =  get_band_small(TOP->dem, 0)) == NULL) return NULL;

  if ((b_green = find_domain(atc->xy_aod, "GREEN"))   < 0) return NULL;
  wvl = get_brick_wavelength(atc->xy_aod, b_green);

  nf  = get_brick_ncols(atc->xy_aod);
  ne  = get_brick_nrows(atc->xy_aod);
  gres  = get_brick_res(atc->xy_aod);
  vnodata = get_brick_nodata(atc->xy_aod, b_green);
  if ((xy_aod_ = atc_get_band_reshaped(atc->xyz_aod, b_green)) == NULL) return NULL;


  #pragma omp parallel private(j, p, z, weights) shared(nx, ny, nf, ne, gres, fres, b_green, QAI, dem_, aod_, xy_aod_, nodata, vnodata, pl2) default(none) 
  {

    #pragma omp for schedule(guided)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){
      p = i*nx+j;
      z = dem_[p];
      if (get_off(QAI, p) || !pl2->doatmo){
        aod_[p] = nodata;
      } else {
        weights = interpolation_weights(j, i, nf, ne, gres, fres, xy_aod_[z], vnodata);
        aod_[p] = (short)(interpolate_coarse(weights, xy_aod_[z])*1000);
      }
    }
    }
  }
  free((void*)xy_aod_);


  // reproj the data
  if (pl2->doreproj){
    if (warp_from_brick_to_unknown_brick(pl2->dotile, pl2->resample, pl2->nthread, AOD, cube) == FAILURE){
      printf("warping AOD failed.\n"); return NULL;}
  }


  // set metadata
  copy_string(product, NPOW_02, "AOD");
  set_brick_product(AOD, product);
  set_brick_name(AOD, "FORCE Level 2 Processing System");
  get_brick_compactdate(AOD, 0, date, NPOW_04);
  get_brick_sensor(AOD, 0, sensor, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(AOD, fname);
  set_brick_open(AOD, OPEN_MERGE);
  set_brick_explode(AOD, false);
  set_brick_format(AOD, &pl2->gdalopt);
  set_brick_scale(AOD, 0, 1000);
  set_brick_wavelength(AOD, 0, wvl);
  set_brick_unit(AOD, 0, "micrometers");
  set_brick_domain(AOD, 0, product);
  set_brick_bandname(AOD, 0, "Aerosol optical depth");


  #ifdef FORCE_CLOCK
  proctime_print("compile AOD", TIME);
  #endif

  return AOD;
}


/** This function compiles the WVP product ready to be output
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
--- cube:   data cube parameters
--- QAI:    Quality Assurance Information
--- WVP:    Water vapor
+++ Return: WVP brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_l2_wvp(par_ll_t *pl2, atc_t *atc, cube_t *cube, brick_t *QAI, brick_t *WVP){
int p, nc;
char fname[NPOW_10];
char product[NPOW_02];
char sensor[NPOW_04];
char date[NPOW_04];
int nchar;
short nodata = -9999;
brick_t *WV = WVP;
short *wvp_ = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif
  
  
  if (WV == NULL){
    
    WV = copy_brick(QAI, 1, _DT_SHORT_);
    set_brick_nodata(WV, 0, nodata);

    nc = get_brick_ncells(WV);
    if ((wvp_ = get_band_short(WV, 0)) == NULL) return NULL;

    #pragma omp parallel shared(nc, QAI, wvp_, nodata, atc, pl2) default(none) 
    {

      #pragma omp for schedule(static)
      for (p=0; p<nc; p++){
        if (get_off(QAI, p) || !pl2->doatmo){
          wvp_[p] = nodata;
        } else {
          wvp_[p] = (short)(atc->wvp*1000);
        }
      }
    }

  }

  // reproj the data
  if (pl2->doreproj){
    if (warp_from_brick_to_unknown_brick(pl2->dotile, pl2->resample, pl2->nthread, WV, cube) == FAILURE){
      printf("warping WVP failed.\n"); return NULL;}
  }
  

  // set metadata
  copy_string(product, NPOW_02, "WVP");
  set_brick_product(WV, product);
  set_brick_name(WV, "FORCE Level 2 Processing System");
  get_brick_compactdate(WV, 0, date, NPOW_04);
  get_brick_sensor(WV, 0, sensor, NPOW_04);

  nchar = snprintf(fname, NPOW_10, "%s_LEVEL2_%s_%s", date, sensor, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  #ifdef ACIX
  nchar = snprintf(fname, NPOW_10, "%s_%s", pl2->b_level1, product);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  #endif
  set_brick_filename(WV, fname);
  set_brick_open(WV, OPEN_MERGE);
  set_brick_explode(WV, false);
  set_brick_format(WV, &pl2->gdalopt);
  set_brick_scale(WV, 0, 1000);
  set_brick_wavelength(WV, 0, 1);
  set_brick_unit(WV, 0, "unknown");
  set_brick_domain(WV, 0, product);
  set_brick_bandname(WV, 0, "Water vapor");

  #ifdef FORCE_CLOCK
  proctime_print("compile WVP", TIME);
  #endif
  
  return WV;
}

    
/** This function compiles the Level 2 products ready to be output
--- pl2:      L2 parameters
--- mission:  mission ID
--- atc:      atmospheric correction factors
--- cube:     data cube parameters
--- TOA:      TOA reflectance
--- QAI:      Quality Assurance Information
--- WVP:      water vapor
--- TOP:      Topographic Derivatives
--- nproduct: number of products
+++ Return: array of product bricks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_level2(par_ll_t *pl2, int mission, atc_t *atc, cube_t *cube, brick_t *TOA, brick_t *QAI, brick_t *WVP, top_t *TOP, int *nproduct){
int nprod, p_boa, p_qai, p_dst, p_vzn, p_hot, p_aod, p_wvp, p_ovv;
brick_t **LEVEL2 = NULL;


  #ifdef FORCE_CLOCK 
  time_t TIME; time(&TIME);
  #endif


  
  nprod = 0;
  p_qai = nprod++;
  p_boa = nprod++;
  if (pl2->odst){ p_dst = nprod++;} else { p_dst = -1;}
  if (pl2->ovzn){ p_vzn = nprod++;} else { p_vzn = -1;}
  if (pl2->ohot){ p_hot = nprod++;} else { p_hot = -1;}
  if (pl2->oaod){ p_aod = nprod++;} else { p_aod = -1;}
  if (pl2->owvp){ p_wvp = nprod++;} else { p_wvp = -1;}
  if (pl2->oovv){ p_ovv = nprod++;} else { p_ovv = -1;}
  
  #ifdef FORCE_DEBUG
  printf("compiling %d products\n", nprod);
  #endif


  alloc((void**)&LEVEL2, nprod, sizeof(brick_t*));





  if (p_hot >= 0){
    if ((LEVEL2[p_hot] = compile_l2_hot(pl2, cube, TOA, QAI)) == NULL){
      printf("error in compiling L2 HOT. "); return NULL;}}
      
  if (p_aod >= 0){
    if ((LEVEL2[p_aod] = compile_l2_aod(pl2, atc, cube, QAI, TOP)) == NULL){
      printf("error in compiling L2 AOD. "); return NULL;}}

  // do BOA near the end (TOA is altered within)
  if ((LEVEL2[p_boa] = compile_l2_boa(pl2, mission, atc, cube, TOA, QAI, WVP, TOP)) == NULL){
    printf("error in compiling L2 BOA. "); return NULL;}

  // do WVP after BOA (WVP is altered within)
  if (p_wvp >= 0){
    if ((LEVEL2[p_wvp] = compile_l2_wvp(pl2, atc, cube, QAI, WVP)) == NULL){
      printf("error in compiling L2 WVP. "); return NULL;}} else free_brick(WVP);

  // do QAI at the very end (QAI is altered within)
  if ((LEVEL2[p_qai] = compile_l2_qai(pl2, cube, QAI)) == NULL){
    printf("error in compiling L2 QAI. "); return NULL;}

  if (p_dst >= 0){
    if ((LEVEL2[p_dst] = compile_l2_dst(pl2, cube, LEVEL2[p_qai])) == NULL){
      printf("error in compiling L2 DST. "); return NULL;}}

  if (p_vzn >= 0){
    if ((LEVEL2[p_vzn] = compile_l2_vzn(pl2, atc, cube, LEVEL2[p_qai])) == NULL){
    printf("error in compiling L2 VZN. "); return NULL;}}

  if (p_ovv >= 0){
    if ((LEVEL2[p_ovv] = compile_l2_ovv(pl2, LEVEL2[p_boa], LEVEL2[p_qai])) == NULL){
    printf("error in compiling L2 OVV. "); return NULL;}}

    // free some memory
  free_topography(TOP);


  #ifdef FORCE_CLOCK
  proctime_print("compile Level 2 products", TIME);
  #endif

  *nproduct = nprod;
  return LEVEL2;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** Radiometric module, main function
+++ This function is the radiometric core module of FORCE L2PS, and does 
+++ radiometric correction. This includes estimating AOD, water vapor, 
+++ environment effect, topographic correction and atmospheric correction.
--- pl2:    L2 parameters
--- meta:    metadata
--- mission: mission ID
--- atc:     atmospheric correction factors
--- cube:    data cube parameters
--- TOA:     TOA reflectance
--- QAI:     Quality Assurance Information
--- TOP:     Topographic Derivatives
--- nprod:   number of products
+++ Return:  array of product bricks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **radiometric_correction(par_ll_t *pl2, meta_t *meta, int mission, atc_t *atc, cube_t *cube, brick_t *TOA, brick_t *QAI, brick_t *AOI, top_t *TOP, int *nprod){
int b, nb;
brick_t  *WVP    = NULL;
brick_t **L2 = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nb = get_brick_nbands(TOA);


  if (pl2->doatmo){
    
    cite_me(_CITE_RADCOR_);
    cite_me(_CITE_RADTRAN_);
    cite_me(_CITE_ATMVAL_);


    /** elevation stats (mean, max/min, # of 100m classes)
    +++ rayleigh scattering @ sea level
    +++ elevation correction factors for optical depths
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    for (b=0; b<nb; b++) atc->mod[b] = molecular_optical_depth(atc->wvl[b]);
    atc->Hr = mod_elev_factor(atc->dem.avg);

    #ifdef FORCE_DEBUG
    print_fvector(atc->mod, "rayleigh @ sea level", nb, 1, 4);
    printf("Hr: %.4f\n", atc->Hr);
    #endif

    /** precipitable water vapor
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (mission == LANDSAT){
      if ((atc->wvp = water_vapor_from_lut(pl2, atc)) < 0){
        printf("Cannot read wvp from LUT. "); return NULL;}
    } else atc->wvp = 0.0;

    /** angle-dependent coarse-grid atmospheric modelling
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    atmo_angledep(pl2, meta, atc, TOP, QAI);

    /** compile AOD, use image-based water/shadow targets, refine by DODB, 
    +++ use external values (one or several options are possible)
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (compile_aod(pl2, meta, atc, TOA, QAI, TOP) == FAILURE){
      printf("error in AOD module.\n"); return NULL;}

    /** elevation-dependent coarse-grid atmospheric modelling
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    atmo_elevdep(pl2, atc, QAI, TOP);
    
    /** water vapor and gaseous transmittance estimation
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (mission == SENTINEL2){
      if ((WVP = water_vapor(meta, atc, TOA, QAI, TOP->dem)) == NULL){
        printf("error in water vapor estimation. "); return NULL;}
    } else WVP = NULL;

    /** estimate topographic correction factor
    +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
    if (pl2->dotopo){
      if ((TOP->c = cfactor_topography(atc, TOA, QAI, 
                      TOP->dem, TOP->exp, TOP->ill)) == NULL){
          printf("error in topographic correction. "); return NULL;}
    }

  }

  /** Apply AOI mask
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  if (apply_aoi(QAI, AOI) == FAILURE){
    fprintf(stderr, "Failed to apply AOI mask\n");
    return NULL;
  }


  /** Level 2 datasets
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  if ((L2 = compile_level2(pl2, mission, atc, cube, TOA, QAI, WVP, TOP, nprod)) == NULL){
    printf("error in compiling Level 2 products. "); return NULL;}


  /** clean
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  free_wvlut();


  #ifdef FORCE_DEBUG
  int prod;
  for (prod=0; prod<(*nprod); prod++){ print_brick_info(L2[prod]); write_brick(L2[prod]);}
  #endif


  #ifdef FORCE_CLOCK
  proctime_print("radiometric module", TIME);
  #endif

  return L2;
}

