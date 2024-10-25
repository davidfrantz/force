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
This file contains functions for handling Aerosol Optical Depth
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "aod-ll.h"

/** GNU Scientific Library (GSL) **/
#include <gsl/gsl_multimin.h>          // minimization functions 

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdalgrid.h"       // GDAL gridder related entry points and defs


/** This function reads externally provided AOD values, e.g. to use an AOD
+++ climatology. The function also sets the global AOD fallback, which is 
+++ used as external value, if no AOD LUTs are provided.
--- pl2:    L2 parameters
--- atc:    atmospheric correction factors
+++ Return: aod spectrum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *aodfileread(par_ll_t *pl2, atc_t *atc){
FILE *fp = NULL;
char  buffer[NPOW_10] = "\0";
char *tokenptr = NULL;
const char *separator = " ";
char fname[NPOW_10];
int nchar;
int b, nb; 
float co[3] = { -2.8665, -1.4787, -0.0486 };
double center_x, center_y;
double site_x, site_y;
double diff_x, diff_y;
float dist, min_dist = LONG_MAX;
int ne, nf;
int doy;
float *aod_lut = NULL;



  nb = get_brick_nbands(atc->xy_aod);
  alloc((void**)&aod_lut, nb, sizeof(float));


  /** global fallback
  +** *******************************************************************/

  // AOD derived from global AERONET data (median AOD)
  for (b=0; b<nb; b++) aod_lut[b] = exp(co[0]) * 
                          pow(atc->wvl[b], co[1] + co[2]*atc->lwvl[b]);

  #ifdef FORCE_DEBUG
  print_fvector(aod_lut,   "\nglobal aod fallback", nb, 1, 3);
  #endif

  // if AOD directory not given, continue with global fallback
  if ((strcmp(pl2->d_aod, "NULL") == 0)) return aod_lut;

  cite_me(_CITE_AODFALLBACK_);


  /** spatio-temporal fallback
  +** *******************************************************************/
  
  // scene center
  nf = get_brick_ncols(atc->xy_aod);
  ne = get_brick_nrows(atc->xy_aod);
  get_brick_geo(atc->xy_aod, nf/2, ne/2, &center_x, &center_y);
  doy   = get_brick_doy(atc->xy_aod, 0);


  // daily LUT
  nchar = snprintf(fname, NPOW_10, "%s/AOD_%03d.txt", pl2->d_aod, doy);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}

  // open aod file
  if ((fp = fopen(fname, "r")) == NULL){
      printf("Unable to open aod file!\n"); return NULL;}

  // process line by line
  while (fgets(buffer, NPOW_10, fp) != NULL){

    tokenptr = strtok(buffer, separator);

    site_x = atof(tokenptr); tokenptr = strtok(NULL, separator);
    site_y = atof(tokenptr); tokenptr = strtok(NULL, separator);
    diff_x = center_x-site_x; diff_y = center_y-site_y;

    dist = sqrt(diff_x*diff_x+diff_y*diff_y);
    if (dist < min_dist){
      min_dist = dist;
      co[0] = atof(tokenptr); tokenptr = strtok(NULL, separator);
      co[1] = atof(tokenptr); tokenptr = strtok(NULL, separator);
      co[2] = atof(tokenptr);
    }

  }

  fclose(fp);

  // AOD derived from LUT
  for (b=0; b<nb; b++) aod_lut[b] = exp(co[0]) * 
                        pow(atc->wvl[b], co[1] + co[2]*atc->lwvl[b]);

  #ifdef FORCE_DEBUG
  print_fvector(aod_lut,   "\naod from file", nb, 1, 3);
  #endif

  return aod_lut;
}


/** This function identifies candidate dark targets and extracts object-
+++ related information. For each object, the target location, radius, 
+++ size, reflectance and environment reflectance is tabulated. There are
+++ different methods for identifying water/shadow/vegetation targets. The
+++ function returns the dark target information and number of targets.
--- atc:    atmospheric correction factors
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information
--- TOP:    Topographic Derivatives
--- type:   Target type (water/shadow/vegetation)
--- DOBJ:   Extracted targets (returned)
+++ Return: Number of candidate targets
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int extract_dark_target(atc_t *atc, brick_t *TOA, brick_t *QAI, top_t *TOP, int type, darkobj_t **DOBJ){
int b, bb, nb;
int blue, green, red, nir, sw2;
int i, j, p, nx, ny, nc, o, cell_size;
float z;
bool   valid;
small  *LAPSE    = NULL; // lapsing spectra
small *TARGET   = NULL; // candidate targets
small *INVERT   = NULL; // inverse of candidate targets
ushort *DISTANCE = NULL; // distance to "shore" of target
int   *SEGMENT  = NULL; // target segmentation
int num;                // number of candidate targets
double *esum = NULL;    // sum for env. reflectance mean
double *esum_priv = NULL;    // sum for env. reflectance mean
double *tsum = NULL;    // sum for target reflectance mean
double *csum = NULL;    // sum for illumin. angle mean
double *ssum = NULL;    // sum for sky view mean
double *zsum = NULL;    // sum for elevation mean
double tn = 0, en = 0;  // ctr for env. / target reflectance mean
float *etoa = NULL;     // global env. reflectance
darkobj_t *dobj = NULL; // dark targets

small  *dem_   = NULL;
ushort *slp_   = NULL;
short  *ill_   = NULL;
ushort *sky_   = NULL;
short **toa_   = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  // shadow type is disabled until further notice
  if (type == _AOD_SHD_){
    *DOBJ = dobj;
    return 0;
  }


  nb = get_brick_nbands(TOA);
  nx = get_brick_ncols(TOA);
  ny = get_brick_nrows(TOA);
  nc = get_brick_ncells(TOA);

  cell_size = floor(get_brick_res(atc->xy_aod)/get_brick_res(TOA));

  if ((dem_   = get_band_small(TOP->dem,  0))   == NULL) return FAILURE;
  if ((slp_   = get_band_ushort(TOP->exp, ZEN)) == NULL) return FAILURE;
  if ((ill_   = get_band_short(TOP->ill,  0))   == NULL) return FAILURE;
  if ((sky_   = get_band_ushort(TOP->sky, 0))   == NULL) return FAILURE;
  if ((toa_   = get_bands_short(TOA))           == NULL) return FAILURE;
  
  if ((blue  = find_domain(TOA, "BLUE"))  < 0) return FAILURE; 
  if ((green = find_domain(TOA, "GREEN")) < 0) return FAILURE; 
  if ((red   = find_domain(TOA, "RED"))   < 0) return FAILURE; 
  if ((nir   = find_domain(TOA, "NIR"))   < 0) return FAILURE; 
  if ((sw2   = find_domain(TOA, "SWIR2")) < 0) return FAILURE; 


  /** allocate memory
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  
  alloc((void**)&TARGET,  nc, sizeof(small));
  alloc((void**)&INVERT,  nc, sizeof(small));
  alloc((void**)&LAPSE,   nc, sizeof(small));  

  alloc((void**)&SEGMENT, nc, sizeof(int));
  
  alloc((void**)&esum, nb, sizeof(double));
  alloc((void**)&etoa, nb, sizeof(float));


  /** identify candidate targets
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  #pragma omp parallel private(z) shared(nc, type, blue, red, nir, sw2, INVERT, TARGET, QAI, toa_, slp_, ill_, dem_, atc) default(none) 
  {

    #pragma omp for schedule(guided)

    for (p=0; p<nc; p++){
      
      if (get_off(QAI, p)){ 
        INVERT[p] = true; continue;}
        
      z = dem_[p]*atc->dem.step+atc->dem.min+atc->dem.step/2;

      switch (type){
        case _AOD_WAT_:                         // water target:
          if (toa_[sw2][p] < 250 &&           // swir2 < 2.5%
              toa_[red][p] < 2000 &&          // red < 20%
              toa_[blue][p] > toa_[nir][p] && // blue > nir
              ((z < 1 && slp_[p] < 870) ||    // slope < 1° if z > 1km
               (z > 1 && slp_[p] < 175))){    // slope < 5° if z < 1km
            TARGET[p] = true;
          } else {
            INVERT[p] = true;
          }
          break;
        case _AOD_SHD_:                         // shadow target:
          if (toa_[sw2][p] < 500 &&           // swir2 < 5%
              toa_[red][p] < 2000 &&          // red < 20%
              toa_[blue][p] > toa_[nir][p] && // blue > nir
              ill_[p] < 5000 &&               // illumination angle > 60°
              slp_[p] > 870){                 // slope > 5°
            TARGET[p] = true;
          } else {
            INVERT[p] = true;
          }
          break;
        case _AOD_VEG_:                         // vegetation target:
          if (toa_[sw2][p] < 500 &&           // swir2 < 5%
              toa_[red][p] < 2000 &&          // red < 20%
              toa_[blue][p] < toa_[nir][p] && // blue < nir
              (toa_[nir][p]-toa_[red][p])/    // NDVI > 0.4
              (float)(toa_[nir][p]+toa_[red][p]) > 0.4 && 
              slp_[p] < 2620){                // slope < 15°
            TARGET[p] = true;
          } else {
            INVERT[p] = true;
          }
          break;
        default:
          printf("unknown dark target type.\n"); exit(1);
      }

    }

  }


  /** target segmentation
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  // sub-divide the targets. This ensures that a target cannot run through
  // the complete image, e.g. a river
  
  #pragma omp parallel private(j, p) shared(nx, ny, cell_size, TARGET) default(none)
  {

    #pragma omp for schedule(static)
  
    for (i=0; i<ny; i+=cell_size){
    for (j=0; j<nx; j++){
      p = i*nx+j;
      TARGET[p] = false;
    }
    }

  }

  #pragma omp parallel private(i, p) shared(nx, ny, cell_size, TARGET) default(none)
  {

    #pragma omp for schedule(static)
  
    for (j=0; j<nx; j+=cell_size){
    for (i=0; i<ny; i++){
      p = i*nx+j;
      TARGET[p] = false;
    }
    }

  }

  // segmentate targets
  num = connectedcomponents_(TARGET, SEGMENT, nx, ny);
  free((void*)TARGET);


  // distance to "shore"
  DISTANCE = dist_transform_(INVERT, nx, ny);
  free((void*)INVERT);


  /** find lapsing spectra
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  #pragma omp parallel private(b, bb, valid) shared(nb, nc, sw2, type, QAI, SEGMENT, DISTANCE, toa_, LAPSE, atc) default(none)
  {

    #pragma omp for schedule(guided)

    for (p=0; p<nc; p++){

      if (get_off(QAI, p)) continue;
      if (SEGMENT[p] < 1) continue;
      if ((get_shadow(QAI, p) || get_cloud(QAI, p) > 0) && type != _AOD_SHD_) continue;
      if (type == _AOD_VEG_ && DISTANCE[p] < 2) continue;
      
      for (b=1, valid=true; b<nb; b++){

        if (!atc->aod_bands[b]) continue;
        //if (atc->wvl[b] >= 0.7 && b != sw2) continue;
        if (type == _AOD_VEG_ && b == sw2) continue;

        for (bb=b-1; bb>=0; bb--){
          if (!atc->aod_bands[bb]) continue;
          //if (atc->wvl[bb] >= 0.7) continue;
          if (toa_[b][p] > toa_[bb][p]){ valid = false; break;}
        }
        
        if (!valid) break;
      }

      LAPSE[p] = valid;

    }
    
  }


  /** scene average: global environment
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  #pragma omp parallel private(b, esum_priv) shared(nb, nc, QAI, toa_, atc, esum) reduction(+: en) default(none)
  {
    
    alloc((void**)&esum_priv, nb, sizeof(double));

    #pragma omp for schedule(guided)

    for (p=0; p<nc; p++){

      if (get_off(QAI, p) || get_shadow(QAI, p) || get_cloud(QAI, p) > 0) continue;

      for (b=0; b<nb; b++){
        if (!atc->aod_bands[b]) continue;
        //if (atc->wvl[bb] >= 0.7) continue;
        esum_priv[b] += toa_[b][p]/10000.0; // rescale to prevent overflow
      }

      en++;

    }
    
    #pragma omp critical
    {
      for (b=0; b<nb; b++) esum[b] +=  esum_priv[b];
    }
    
    free((void*)esum_priv);
    
  }

  if (en > 0){
    for (b=0; b<nb; b++) etoa[b] = (float)(esum[b]/en);
  } else {
    for (b=0; b<nb; b++) etoa[b] = 0.0;
  }
  free((void*)esum);

  #ifdef FORCE_DEBUG
  print_fvector(etoa, "scene environment", nb, 1, 4);
  #endif


  /** allocate and initialize darkobject struct
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  alloc((void**)&dobj, num, sizeof(darkobj_t));
  for (o=0; o<num; o++){
    dobj[o].imin = ny; dobj[o].jmin = nx;
    dobj[o].k = 0; dobj[o].r = 0;
    dobj[o].ms = dobj[o].mv = dobj[o].razi = 0;
    alloc((void**)&dobj[o].ttoa, nb, sizeof(float));
    alloc((void**)&dobj[o].etoa, nb, sizeof(float));
    alloc((void**)&dobj[o].aod,  nb, sizeof(float));
    alloc((void**)&dobj[o].est,  nb, sizeof(float));
  }

  alloc((void**)&zsum, num, sizeof(double));
  alloc((void**)&csum, num, sizeof(double));
  alloc((void**)&ssum, num, sizeof(double));


  /** extract information, 1st pass, loop over image
  +++ determine bounding box, object size, mean illumin. angle, elevation
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  for (i=0; i<ny; i++){
  for (j=0; j<nx; j++){

    p = i*nx+j;

    if (get_off(QAI, p)) continue;
    if ((o = SEGMENT[p]-1) < 0) continue;
    if ((get_shadow(QAI, p) || get_cloud(QAI, p) > 0) && type != _AOD_SHD_) continue;
    if (!LAPSE[p]) continue;

    csum[o] += ill_[p]/10000.0; // rescale to prevent overflow
    ssum[o] += sky_[p]/10000.0; // rescale to prevent overflow
    zsum[o] += (dem_[p]*atc->dem.step+atc->dem.min+atc->dem.step/2);

    // object radius
    if (DISTANCE[p] > dobj[o].r) dobj[o].r = DISTANCE[p];

    // object location
    if (i > dobj[o].imax) dobj[o].imax = i;
    if (j > dobj[o].jmax) dobj[o].jmax = j;
    if (i < dobj[o].imin) dobj[o].imin = i;
    if (j < dobj[o].jmin) dobj[o].jmin = j;

    // pixel number
    dobj[o].k++;

  }
  }


  /** extract information, 2nd pass, loop over objects
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  #pragma omp parallel private(i, j, p, b, en, tn, tsum, esum) shared(num, nb, nx, ny, dobj, type, etoa, toa_, QAI, SEGMENT, LAPSE, atc, csum, zsum, ssum) default(none)
  {
    
    alloc((void**)&tsum, nb, sizeof(double));
    alloc((void**)&esum, nb, sizeof(double));

    #pragma omp for schedule(guided)

    for (o=0; o<num; o++){

      // ignore small and narrow objects
      if (dobj[o].k < 10){
        dobj[o].valid = false;
        continue;
      } else if (dobj[o].r < 3){
        dobj[o].valid = false;
        continue;
      } else {
        dobj[o].valid = true;
      }

      en = tn = 0.0;
      for (b=0; b<nb; b++) tsum[b] = 0.0;
      for (b=0; b<nb; b++) esum[b] = 0.0;


      // use 3x object radius for environment reflectance
      for (i=dobj[o].imin-3*dobj[o].r; i<=dobj[o].imax+3*dobj[o].r; i++){
      for (j=dobj[o].jmin-3*dobj[o].r; j<=dobj[o].jmax+3*dobj[o].r; j++){
        
        if (i < 0 || j < 0 || i >= ny || j >= nx) continue;
        p = i*nx+j;

        if (get_off(QAI, p)) continue;
        if ((get_shadow(QAI, p) || get_cloud(QAI, p) > 0) && type != _AOD_SHD_) continue;

        // no target --> environment reflectance
        if (SEGMENT[p] == 0){

          for (b=0; b<nb; b++){
            if (!atc->aod_bands[b]) continue;
            //if (atc->wvl[bb] >= 0.7) continue;
            esum[b] += toa_[b][p]/10000.0; // rescale to prevent overflow
          }
          en++;

        // if target and pixel is lapsing --> target reflectance
        } else if (SEGMENT[p]-1 == o && LAPSE[p]){

          for (b=0; b<nb; b++){
            if (!atc->aod_bands[b]) continue;
            //if (atc->wvl[bb] >= 0.7) continue;
            tsum[b] += toa_[b][p]/10000.0; // rescale to prevent overflow
          }
          tn++;

        }

      }
      }

      // if less than 10 pixels were valid targets, skip
      if (tn < 10){
        dobj[o].valid = false;
        continue;
      }

      // if less than 10 pixels were valid environment, use global one
      if (en < 10){
        for (b=0; b<nb; b++) esum[b] = etoa[b];
        en = 1;
      }

      // mean illumination angle and elevation
      dobj[o].cosi = (float)(csum[o]/dobj[o].k);
      dobj[o].sky  = (float)(ssum[o]/dobj[o].k);
      dobj[o].z    = (float)(zsum[o]/dobj[o].k);

      // MOD elevation correction factors
      dobj[o].Hr = mod_elev_factor(dobj[o].z);

      // target centroid
      dobj[o].i = (int)((dobj[o].imax+dobj[o].imin)/2);
      dobj[o].j = (int)((dobj[o].jmax+dobj[o].jmin)/2);

      // target centroid in coarse grid
      convert_brick_ji2jip(QAI, atc->xy_aod, dobj[o].i, dobj[o].j, &dobj[o].e, &dobj[o].f, &dobj[o].g);
      
      // sun angles
      dobj[o].ms   = get_brick(atc->xy_sun,  cZEN, dobj[o].g);
      dobj[o].mv   = get_brick(atc->xy_view, cZEN, dobj[o].g);
      dobj[o].razi = get_brick(atc->xy_view,  AZI, dobj[o].g) - 
                     get_brick(atc->xy_sun,   AZI, dobj[o].g);


      // target      reflectance
      // environment reflectance
      for (b=0; b<nb; b++){
        if (!atc->aod_bands[b]) continue;
        //if (atc->wvl[bb] >= 0.7) continue;
        dobj[o].etoa[b] = (float)(esum[b]/en);
        dobj[o].ttoa[b] = (float)(tsum[b]/tn);
      }

    }
    
    free((void*)tsum);
    free((void*)esum); 

  }
  

  // clean
  free((void*)LAPSE); 
  free((void*)SEGMENT); 
  free((void*)DISTANCE);
  free((void*)etoa);
  free((void*)csum); free((void*)zsum); free((void*)ssum);

  #ifdef FORCE_DEBUG
  printf("%d potential type-%d targets\n", num, type);
  #endif
  
  #ifdef FORCE_CLOCK
  proctime_print("dark target extraction", TIME);
  #endif

  *DOBJ = dobj;
  return num;
}


#ifdef FORCE_DEBUG

#include "cpl_multiproc.h"  // CPL Multi-Threading

/** This function prints the dark objects
--- dobj:   Extracted targets (returned)
--- o:      Object ID
--- nb:     number of spectral bands
--- type:   Target type (water/shadow/vegetation)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_dark_object(darkobj_t *dobj, int o, int nb, int type, char *dlog){
int b;
char fname[NPOW_10];
int nchar;
char *lock = NULL;
FILE *fp = NULL;


  nchar = snprintf(fname, NPOW_10, "%s/dark-targets-type%d.txt", dlog, type);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); exit(1);}

  if ((lock = (char*)CPLLockFile(fname, 60)) == NULL){
    printf("Unable to lock %s (timeout: %ds).\n", fname, 60);
    return;}

  fp = fopen(fname, "a");

  fprintf(fp, "obj %04d, n %06d, r %02d, CG-X/Y/G %04d/%04d/%04d, COSI %.2f, Z %01.3f, Z-correction: %.3f\n", 
    o+1, dobj[o].k, dobj[o].r, dobj[o].e, dobj[o].f, dobj[o].g, 
    dobj[o].cosi, dobj[o].z, dobj[o].Ha);
  fprintf(fp, " mv/ms %.3f/%.3f, rel-azi %.3f\n", dobj[o].mv, dobj[o].ms, dobj[o].razi);
  fprintf(fp, "   target: "); for (b=0; b<nb; b++) fprintf(fp, " %.3f", dobj[o].ttoa[b]); fprintf(fp, "\n");
  fprintf(fp, "   environ:"); for (b=0; b<nb; b++) fprintf(fp, " %.3f", dobj[o].etoa[b]); fprintf(fp, "\n");
  fprintf(fp, "   aod img:"); for (b=0; b<nb; b++) fprintf(fp, " %.3f", dobj[o].est[b]);  fprintf(fp, "\n");
  fprintf(fp, "   aod fit:"); for (b=0; b<nb; b++) fprintf(fp, " %.3f", dobj[o].aod[b]);  fprintf(fp, "\n");
  fprintf(fp, "   fitcoef:"); for (b=0; b<3;  b++) fprintf(fp, " %.3f", dobj[o].coef[b]); fprintf(fp, "\n");
  fprintf(fp, "   Rsq %01.2f, spec # %d, fit # %d\n\n", dobj[o].rsq, dobj[o].lib_id, dobj[o].ang_fit);
  
  fclose(fp);
  CPLUnlockFile(lock);

  return;
}

#endif


/** This function attempts to estimates AOD over each valid dark target.
+++ This is done on the object-level. A reference library based method is
+++ used, where AOD is found by comparing the target TOA reflectance with
+++ the atmospherically contaminated reference spectra. The contamination
+++ is increased iteratively. The AOD is propagated through the spectrum
+++ by either fitting a 2nd order polynomial or a linear function to the
+++ logarithm of AOD and wavelengths. The fit with highest R² wins. For 
+++ each coarse grid cell, all successful AOD estimates are averaged using
+++ the square of R² as weight.
--- pl2:    L2 parameters
--- meta:   metadata
--- atc:    atmospheric correction factors
--- res:    resolution
--- dobj:   Dark targets (deallocated within)
--- num:    Number of candidate targets
--- type:   Target type (water/shadow/vegetation)
+++ Return: Number of successful targets
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int aod_from_target(par_ll_t *pl2, meta_t *meta, atc_t *atc, float res, darkobj_t *dobj, int num, int type){
speclib_t *lib = NULL;
int b, bb, nb, naod, c, o, w, wbest, k = 0;
int ultrablue, blue, green, red, sw2;
int err = 0;
float **aodest = NULL;
float **aodlog = NULL;
float **aodfit = NULL;
float rsq, rsqbest, mean, vres, vaod;
bool valid;
int fit, fitbest;
float coef[3], coefbest[3];


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nb = get_brick_nbands(atc->xy_aod);

  ultrablue  = find_domain(atc->xy_aod, "ULTRABLUE");
  if ((blue  = find_domain(atc->xy_aod, "BLUE"))  < 0) return FAILURE; 
  if ((green = find_domain(atc->xy_aod, "GREEN")) < 0) return FAILURE; 
  if ((red   = find_domain(atc->xy_aod, "RED"))   < 0) return FAILURE; 
  if ((sw2   = find_domain(atc->xy_aod, "SWIR2")) < 0) return FAILURE; 


  // shadow type is disabled until further notice
  if (type == _AOD_SHD_){
    //*map_aod = map_avg;
    return 0;}


  /** initialize spectral library **/
  switch (type){
    case _AOD_WAT_:
      lib = water_lib(nb, meta);
      break;
    case _AOD_SHD_:
      lib = land_lib(nb, meta);
      break;
    case _AOD_VEG_:
      lib = veg_lib(nb, blue, green, red);
      for (b=0; b<nb; b++) atc->aod_bands[b] = false;
      atc->aod_bands[blue]  = true;
      atc->aod_bands[green] = true;
      atc->aod_bands[red]   = true;
      atc->aod_bands[sw2]   = true;
      break;
  }

  for (b=0, naod=0; b<nb; b++){
    if (atc->aod_bands[b]) naod++;
    //if (atc->wvl[bb] < 0.7) naod++;
  }
  //naod++; // include SWIR2



  #ifdef FORCE_DEBUG
  printf("estimate aod from type-%d targets\n", type);
  #endif


  /** estimate AOD for every target
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  #pragma omp parallel private(b, bb, c, w, wbest, rsqbest, vres, vaod, rsq, aodest, aodlog, aodfit, valid, mean, fit, fitbest, coef, coefbest) shared(nb, res, ultrablue, blue, red, sw2, num, naod, lib, dobj, type, atc, pl2) reduction(+: k, err) default(none) 
  {

    /** allocate memory **/
    alloc_2D((void***)&aodest, lib->n, nb, sizeof(float)); // estimated AOD, band-wise
    alloc_2D((void***)&aodlog, lib->n, nb, sizeof(float)); // log of estimated AOD
    alloc_2D((void***)&aodfit, lib->n, nb, sizeof(float)); // fitted AOD

    #pragma omp for schedule(guided)
    for (o=0; o<num; o++){

      if (!dobj[o].valid) continue;

      memset(coefbest, 0, 3*sizeof(float));

      /** estimate band-wise aod that matches best with each pair of target 
      +++ and library reflectance **/
      if (aod_lib_to_target(atc, res, pl2->domulti, dobj[o], lib, type, aodest) == FAILURE){
        printf("error in library/target AOD optimization.\n");
        err++; continue;}
      //for (w=0; w<lib->n; w++){aodest[w][0] = 0.2; for (b=1; b<nb; b++) aodest[w][b] = aodest[w][b]*0.8;}

      /** fit aerosol model to each pair of target and library spectrum **/
      for (w=0, wbest=-1, rsqbest=0; w<lib->n; w++){

        // skip spectrum if any estimated AOD < 0
        for (b=0, valid=true; b<nb; b++){
          if (!atc->aod_bands[b]) continue;
          if (aodest[w][b] < 0){ valid = false; break;}
        }
        if (!valid) continue;

        // if vegetation target and red AOD > blue AOD, skip
        if (type == _AOD_VEG_ && aodest[w][red] > aodest[w][blue]) continue;

        // fix SWIR2 to low value (derived from AERONET analysis)
        aodest[w][sw2] = 0.0172;

        // if blue AOD < assumed SWIR2 AOD, skip
        if (aodest[w][blue] <= aodest[w][sw2]) continue;

        // if ultrablue AOD < blue AOD, skip
        if (ultrablue >= 0 && atc->aod_bands[ultrablue] &&
            aodest[w][ultrablue] < aodest[w][blue]) continue;

        // compute logarithm of aod + mean of AOD over all bands
        for (b=0, mean=0; b<nb; b++){

          if (!atc->aod_bands[b]) continue;

          // if any AOD < assumed SWIR2 AOD, force all following bands to this value
          if (aodest[w][b] <= aodest[w][sw2]){
            for (bb=b; bb<nb; bb++){
              if (!atc->aod_bands[bb]) continue;
              aodest[w][bb] = aodest[w][sw2];
            }
          }

          aodlog[w][b] = log(aodest[w][b]);
          mean += aodest[w][b];
        }
        mean /= (float)naod;


        /** fit polynomial AOD model (King and Byrne) **/

        if (type != _AOD_VEG_){

          fit = 0;
          aod_polynomial_fit(atc, naod, aodlog[w], &coef[0], &coef[1], &coef[2]);
          for (b=0; b<nb; b++) aodfit[w][b] = coef[0] * pow(atc->wvl[b], coef[1]+coef[2]*atc->lwvl[b]);

          /** some reasonability checks **/
          valid = true;
          for (b=1, valid = true; b<nb; b++){
            if (aodfit[w][b] > aodfit[w][b-1] && 
                aodfit[w][b] > 0.01 &&
                aodfit[w][b-1] > 0.01) valid = false;
            if (aodfit[w][b] < -0.01 &&
                aodfit[w][b-1] < -0.01) valid = false;
          }

          if (coef[0] < 0 || coef[0] > 0.3) valid = false;
          if (coef[1] > 0 || coef[1] < -3)  valid = false;
          if (coef[2] > 0) valid = false;
        
        } else valid = false;


        /** if polynomial fit was not successful or if we have a vegetation 
        +++ target, try to fit linear AOD model (Angstrom) **/

        if (!valid){

          fit = 1; coef[2] = 0;
          aod_linear_fit(atc, aodlog[w], &coef[0], &coef[1]);
          for (b=0; b<nb; b++) aodfit[w][b] = coef[0] * pow(atc->wvl[b], coef[1]);

          /** some reasonability checks **/
          valid = true;
          for (b=1, valid = true; b<nb; b++){
            if (aodfit[w][b] > aodfit[w][b-1] && 
              aodfit[w][b] > 0.01 &&
              aodfit[w][b-1] > 0.01) valid = false;
            if (aodfit[w][b] < -0.01 &&
              aodfit[w][b-1] < -0.01) valid = false;
          }

          if (coef[0] < 0 || coef[0] > 0.3) valid = false;
          if (coef[1] > 0 || coef[1] < -3)  valid = false;

        }

        /** skip if neither fit was successful **/
        if (!valid) continue;

        /** compute R² of fit, and find best spectrum **/
        for (b=0, vres=0, vaod=0; b<nb; b++){
          if (!atc->aod_bands[b]) continue;
          vres += (aodest[w][b]-aodfit[w][b])*(aodest[w][b]-aodfit[w][b]);
          vaod += (aodest[w][b]-mean)*(aodest[w][b]-mean);}
        if ((rsq = 1.0-vres/vaod) > rsqbest){
          rsqbest = rsq;
          wbest = w;
          fitbest = fit;
          for (c=0; c<3; c++) coefbest[c] = coef[c];
        }

      }

      /** if no valid fit, skip **/
      if (wbest < 0 || rsqbest < 0.1){
        dobj[o].valid = false;
        continue;
      }

      /** copy best aod **/
      for (b=0; b<nb; b++){
        dobj[o].aod[b] = aodfit[wbest][b];
        dobj[o].est[b] = aodest[wbest][b];
      }

      /** copy coefs, rsq, spectrum ID and type of fit **/
      for (c=0; c<3; c++) dobj[o].coef[c] = coefbest[c];
      dobj[o].rsq  = rsqbest;
      dobj[o].lib_id = wbest;
      dobj[o].ang_fit = fitbest;

      /** number of successful targets **/
      k++;

      #ifdef FORCE_DEBUG
      print_dark_object(dobj, o, nb, type, pl2->d_log);
      #endif

    }

    free_2D((void**)aodest, lib->n);
    free_2D((void**)aodlog, lib->n);
    free_2D((void**)aodfit, lib->n);

  }

  if (err > 0){ printf("error in estimating AOD.\n"); return -1;}

  free_2D((void**)lib->s, lib->n);
  free((void*)lib);

  
  #ifdef FORCE_CLOCK
  proctime_print("object-based AOD estimation", TIME);
  #endif

  return k;
}


struct elev_minimizer_params { int n; double *z; double *aod; };


/** Minimizer for estimating dependency of AOD on elevation
+++ This function minimizes slope
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double aod_elevation_minimizer(double x, void *p){
struct elev_minimizer_params *params = (struct elev_minimizer_params*)p;
int k;
float xsum = 0, ysum = 0, xm, ym;
float csum = 0, vsum = 0, cv, vr;
float slope;


  // mean
  for (k=0; k<params->n; k++){
    xsum += params->z[k];
    ysum += params->aod[k]/exp(-params->z[k]/x);
  }
  xm = xsum/params->n;
  ym = ysum/params->n;

  //covariance and variance
  for (k=0; k<params->n; k++){
    csum += (params->z[k]-xm)*(params->aod[k]/exp(-params->z[k]/x)-ym);
    vsum += (params->z[k]-xm)*(params->z[k]-xm);
  }
  cv = csum/params->n;
  vr = vsum/params->n;

  slope = cv/vr;
  
  #ifdef FORCE_DEBUG
  printf("elev minimizer: mean z %f, mean aod %f, cov %f, var %f, slope %f, x %f\n",
    xm, ym, cv, vr, slope, x);
  #endif

  return fabs(slope);
}


/** This function estimates the elevation dependency of AOD, and computes
+++ elevation correction factors
--- atc:    atmospheric correction factors
--- dark:   Dark targets
--- green:  green band ID
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int aod_elev_dependency(atc_t *atc, dark_t *dark, int green){
const gsl_min_fminimizer_type *T = gsl_min_fminimizer_brent;
gsl_min_fminimizer *s = NULL;
gsl_function minex_func;

struct elev_minimizer_params param;

size_t iter = 0;
int status;
int o, k = 0;
int g, ng;
float dem, Ha;
float zmin = INT_MAX;
float zmax = INT_MIN;

double minim = 50;
double lower = 0.1;
double upper = 1000;


  param.n = dark->nwat+dark->nshd+dark->nveg;
  
  // almost flat AOD
  atc->Hp = 1000;

  if (param.n > 3 && atc->dem.max-atc->dem.min > 0.1){
    
    alloc((void**)&param.z,   param.n, sizeof(double));
    alloc((void**)&param.aod, param.n, sizeof(double));
    
    for (o=0; o<dark->kwat; o++){
      if (!dark->wat[o].valid) continue;
      param.z[k]   = dark->wat[o].z;
      if (param.z[k] < zmin) zmin = param.z[k];
      if (param.z[k] > zmax) zmax = param.z[k];
      param.aod[k] = dark->wat[o].aod[green];
      k++;
    }

    for (o=0; o<dark->kshd; o++){
      if (!dark->shd[o].valid) continue;
      param.z[k]   = dark->shd[o].z;
      if (param.z[k] < zmin) zmin = param.z[k];
      if (param.z[k] > zmax) zmax = param.z[k];
      param.aod[k] = dark->shd[o].aod[green];
      k++;
    }

    for (o=0; o<dark->kveg; o++){
      if (!dark->veg[o].valid) continue;
      param.z[k]   = dark->veg[o].z;
      if (param.z[k] < zmin) zmin = param.z[k];
      if (param.z[k] > zmax) zmax = param.z[k];
      param.aod[k] = dark->veg[o].aod[green];
      k++;
    }

    if (zmax-zmin > 0.1){

      /* Initialize method and iterate */
      minex_func.function = &aod_elevation_minimizer;
      minex_func.params   = &param;

      s = gsl_min_fminimizer_alloc(T);

      gsl_set_error_handler_off();
      status = gsl_min_fminimizer_set(s, &minex_func, minim, lower, upper);
      gsl_set_error_handler(NULL);
   
   
      if (status != GSL_EINVAL){
   
        #ifdef FORCE_DEBUG
        printf("%5s [%9s, %9s] %9s\n", 
                "iter", "lower", "upper", "min");
        printf ("%lu [%.3f, %.3f] %.3f \n",
                iter, lower, upper, minim);
        #endif
                
        do {
          iter++;
          status = gsl_min_fminimizer_iterate(s);
          
          minim = gsl_min_fminimizer_x_minimum(s);
          lower = gsl_min_fminimizer_x_lower(s);
          upper = gsl_min_fminimizer_x_upper(s);

          status = gsl_min_test_interval(lower, upper, 0.001, 0.0);

          #ifdef FORCE_DEBUG        
          if (status == GSL_SUCCESS) printf ("Converged:\n");
          printf ("%lu [%.3f, %.3f] %.3f \n", iter, lower, upper, minim);
          #endif

        } while (status == GSL_CONTINUE && iter < 100);

        #ifdef FORCE_DEBUG
        printf("converged after %lu iterations. Factor is %f\n", iter, minim);
        #endif
        
        if (minim < 1.2){
          #ifdef FORCE_DEBUG
          printf("unreasonably strong elev dependency.. reset to 1.2\n");
          #endif
          minim = 1.2;
        }
        atc->Hp = minim;
      
      } else {
        #ifdef FORCE_DEBUG
        printf("no min found. use ~flat AOD\n");
        #endif
      }

      gsl_min_fminimizer_free(s);

    }

    free((void*)param.z);
    free((void*)param.aod);

    // AOD scaling height
  
  }

  // AOD scaling factor for scene average
  atc->Ha = aod_elev_factor(atc->dem.avg, atc->Hp);

  // AOD scaling factor for grid cells
  ng = get_brick_ncells(atc->xy_aod);
  for (g=0; g<ng; g++){
    if (is_brick_nodata(atc->xy_view, 0, g)) continue;
    dem = get_brick(atc->xy_dem, 0, g);
    Ha = aod_elev_factor(dem*atc->dem.step+atc->dem.min+atc->dem.step/2, atc->Hp);
    set_brick(atc->xy_Ha, 0, g, Ha);
  }

  // AOD scaling factor for dark targets
  for (o=0; o<dark->kwat; o++){
    if (!dark->wat[o].valid) continue;
    dark->wat[o].Ha = aod_elev_factor(dark->wat[o].z, atc->Hp);
  }

  for (o=0; o<dark->kshd; o++){
    if (!dark->shd[o].valid) continue;
    dark->shd[o].Ha = aod_elev_factor(dark->shd[o].z, atc->Hp);
  }

  for (o=0; o<dark->kveg; o++){
    if (!dark->veg[o].valid) continue;
    dark->veg[o].Ha = aod_elev_factor(dark->veg[o].z, atc->Hp);
  }


  return SUCCESS;
}


/** This function compiles the water library used for estimating AOD
--- nb:     number of bands
--- meta:   metadata
+++ Return: Spectral library
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
speclib_t *water_lib(int nb, meta_t *meta){
speclib_t *lib;
int i, b, b_rsr, w;
double v, s;

  alloc((void**)&lib, 1, sizeof(speclib_t));

  lib->n = _AERO_WATERLIB_DIM_[0];
  alloc_2D((void***)&lib->s, lib->n, nb, sizeof(float));

  for (i=0; i<lib->n; i++){
    for (b=0; b<nb; b++){
      b_rsr = meta->cal[b].rsr_band;
      for (w=0, v=0, s=0; w<_AERO_WATERLIB_DIM_[1]; w++){
        v += _RSR_[b_rsr][w]*_AERO_WATERLIB_[i][w];
        s += _RSR_[b_rsr][w];
      }
      if (s > 0) lib->s[i][b] = v/s;
    }
  }

  return lib;
}


/** This function compiles the land library used for estimating AOD
+++ Note that this functionality is currently unavailable in this version
+++ of FORCE. To be updated.
--- nb:     number of bands
--- meta:   metadata
+++ Return: Spectral library
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
speclib_t *land_lib(int nb, meta_t *meta){
speclib_t *lib;
int k = 1;

  alloc((void**)&lib, 1, sizeof(speclib_t));

  lib->n = k;
  alloc_2D((void***)&lib->s, lib->n, nb, sizeof(float));

  return lib;
}


/** This function compiles the vegetation library used for estimating AOD
--- nb:     number of bands
--- blue:   blue band ID
--- green:  green band ID
--- red:    red band ID
+++ Return: Spectral library
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
speclib_t *veg_lib(int nb, int blue, int green, int red){
speclib_t *lib;
int k;
float rb, g;


  for (rb=0.01, k=0; rb<=0.05; rb+=0.005){
  for (g=0.01; g<=0.05; g+=0.005){
      k++; 
  }
  }
  
  alloc((void**)&lib, 1, sizeof(speclib_t));

  lib->n = k;
  alloc_2D((void***)&lib->s, lib->n, nb, sizeof(float));

  for (rb=0.01, k=0; rb<=0.05; rb+=0.005){
  for (g=0.01; g<=0.05; g+=0.005){
      lib->s[k][blue]  = rb;
      lib->s[k][green] = rb+g;
      lib->s[k][red]   = rb;
      k++;
  }
  }

  return lib;
}


/** This function returns the AOD, at which the target TOA reflectance and
+++ the atmospherically contaminated reference spectra match best. AOD is
+++ estimated by increasing the contamination iteratively. The function 
+++ processes one target.
--- atc:    atmospheric correction factors
--- res:    resolution
--- multi:  multi or single scattering?
--- dobj:   Dark target
--- lib:    Spectral library
--- type:   Target type (water/shadow/vegetation)
--- aodest: Estimated AOD
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int aod_lib_to_target(atc_t *atc, float res, bool multi, darkobj_t dobj, speclib_t *lib, int type, float **aodest){
int b, nb, g, w;
int blue, green, red, sw2;
float aod, mod, od, Hr, Pr, Pa, tmp;
float ms, mv;
float Tso, Tvo, Tg;
float T, Ts, Tv, tvd, tsd, tvs, tss, s;
float rhop, rhoe, rhow, rhoc, rhob, fresnel;
float Fr, Fa, F, km;
float match, *bestmatch = NULL;
float brdf;
float E0_, Eg, Egc, k;


  g = dobj.g;
  
  nb = get_brick_nbands(atc->xy_aod);
  ms = get_brick(atc->xy_sun,  cZEN, g);
  mv = get_brick(atc->xy_view, cZEN, g);
  Pr = get_brick(atc->xy_Pr, 0, g);
  Pa = get_brick(atc->xy_Pa, 0, g);
  fresnel = get_brick(atc->xy_fresnel, 0, g);
  if ((blue  = find_domain(atc->xy_aod, "BLUE"))  < 0) return FAILURE; 
  if ((green = find_domain(atc->xy_aod, "GREEN")) < 0) return FAILURE; 
  if ((red   = find_domain(atc->xy_aod, "RED"))   < 0) return FAILURE; 
  if ((sw2   = find_domain(atc->xy_aod, "SWIR2")) < 0) return FAILURE; 


  alloc((void**)&bestmatch, lib->n, sizeof(float));

  /** do separately for each band **/
  for (b=0; b<nb; b++){

    if (!atc->aod_bands[b] || b == sw2) continue;

    // BRDF adjustment for vegetation targets
    brdf = get_brick(atc->xy_brdf, b, g);

    // initilaize best match
    for (w=0; w<lib->n; w++) bestmatch[w] = INT_MAX;

    // exoatmospheric solar irradiance, ozone corrected
    Tso = get_brick(atc->xy_Tso, b, g);
    Tvo = get_brick(atc->xy_Tvo, b, g);
    E0_ = atc->E0[b] * Tso*Tvo;

    // total  gaseous transmittance
    Tg = get_brick(atc->xy_Tg, b, g);

    // elevation scale factor for MOD
    mod = get_brick(atc->xy_mod, b, g);
    Hr = get_brick(atc->xy_Hr, 0, g);
    mod = mod_elev_scale(mod, Hr, dobj.Hr);


    /** iteratively increase AOD **/
    for (aod=-0.2; aod<3; aod+=0.001){

      // total optical depth
      od = optical_depth(aod, mod);

      // scattering transmittances
      T = scatt_transmitt(aod, mod, od, ms, mv, 
        &Ts, &Tv, &tsd, &tss, &tvd, &tvs);

      // global irradiance at ground
      Eg = E0_*tsd + E0_*tss;

      // anisotropy index
      k  = (E0_*tsd)/atc->E0[b];

      // topographic correction factor
      if (dobj.cosi > 0){
        Egc = E0_*tsd*dobj.cosi/ms +
              E0_*tss*k*dobj.cosi/ms +
              E0_*tss*(1-k)*dobj.sky;
      } else {
        Egc = E0_*tss*(1-k)*dobj.sky;
      }

      // path reflectance
      rhop = path_ref(multi, atc->tthg.sob, aod, mod, od, Pa, Pr, 
        tsd, tvd, ms, mv);
  
      // spherical albedo
      s = sphere_albedo(aod, mod, od);

      // reasonability check
      if (dobj.etoa[b]-rhop < -0.05){
        break;
      }

      // simplified reflectance environment
      tmp = (dobj.etoa[b]-rhop)/Tg;
      rhoe = tmp / (T + s*tmp);

      // environmental weighting function, F(r)
      km = dobj.r*res/1000.0+res/1000.0/2;
      Fr = env_weight_molecular(km);
      Fa = env_weight_aerosol(km);
      F = env_weight(aod, mod, Fa, Fr);


      /** compute TOA reference reflectance and compare to 
      +++ measured TOA reflectance **/

      for (w=0; w<lib->n; w++){

        // reference reflectance including Fresnel
        if (type == _AOD_WAT_){
          rhow = (lib->s[w][b] + fresnel*tss)/Ts;
        } else {
          rhow =  lib->s[w][b];
        }

        // reference reflectance including topography and BRDF
        if (type == _AOD_SHD_) rhow = rhow*Egc/Eg;
        if (type == _AOD_VEG_) rhow = rhow*Egc/(brdf*Eg);

        // background reflectance
        rhob = rhow*F+(1-F)*rhoe;

        // computed TOA reflectance
        rhoc = Tg*(rhop+(Ts*(tvd*rhow+tvs*rhob))/(1-rhob*s));

        // match similarity
        match = fabs(rhoc-dobj.ttoa[b]);
        if (match < bestmatch[w]){
          bestmatch[w] = match;
          aodest[w][b] = aod;
        }

      }


    }

  }

  free((void*)bestmatch);

  return SUCCESS;
}


/** This function fits the Angstrom expression to the AOD spectrum and es-
+++ timates the turbidity coefficient (a0) and Angstrom exponent (a1).
--- atc:    atmospheric correction factors
--- logaod: logarithm of estimated AOD
--- a0:     turbidity coefficient (returned)
--- a1:     Angstrom exponent     (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int aod_linear_fit(atc_t *atc, float *logaod, float *a0, float *a1){
int b, nb, k;
float xsum = 0, ysum = 0, xm, ym;
float csum = 0, vsum = 0, cv, vr;
float slope, inter;


  nb = get_brick_nbands(atc->xy_aod);

  // mean
  for (b=0, k=0; b<nb; b++){
    if (!atc->aod_bands[b]) continue;
    xsum += atc->lwvl[b];
    ysum += logaod[b];
    k++;
  }
  xm = xsum/k;
  ym = ysum/k;

  //covariance and variance
  for (b=0; b<nb; b++){
    if (!atc->aod_bands[b]) continue;
    csum += (atc->lwvl[b]-xm)*(logaod[b]-ym);
    vsum += (atc->lwvl[b]-xm)*(atc->lwvl[b]-xm);
  }
  cv = csum/k;
  vr = vsum/k;

  slope = cv/vr;
  inter = ym - slope*xm;

  *a0 = exp(inter);
  *a1 = slope;

  return SUCCESS;
}


/** Minimizer for 2nd order polynomial fit between ln(aod) and ln(wvl)
+++ This function minimizes the RMSE of the polynomial fit.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double aod_polynomial_minimizer(const gsl_vector *v, void *params){
float a0, a1, a2;
int b, bb = 0, nb;
float *p = (float *)params;
double y, lwvl, laod, tmp = 0;

  a0 = gsl_vector_get(v, 0);
  a1 = gsl_vector_get(v, 1);
  a2 = gsl_vector_get(v, 2);
  nb = p[bb++];

  for (b=0; b<nb; b++){
    lwvl = p[bb++];
    laod = p[bb++];
    y = a0 + a1*lwvl + a2*lwvl*lwvl;
    tmp += (y-laod)*(y-laod);
  }

  return sqrt(tmp/(float)nb);
}


/** 2nd order polynomial fit between ln(aod) and ln(wvl)
+++ This function fits a modified Angstrom expression to the AOD spectrum,
+++ which accounts for curvature in the AOD spectrum. The function estima-
+++ tes the turbidity coefficient (a0), Angstrom exponent (a1) and a term
+++ for describing the curvature (a2).
--- atc:    atmospheric correction factors
--- naod:   number of bands for AOD estimation
--- logaod: logarithm of estimated AOD
--- a0:     turbidity coefficient (returned)
--- a1:     Angstrom exponent     (returned)
--- a2:     curvature coefficient (returned)
+++ Return: GSL fit status
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int aod_polynomial_fit(atc_t *atc, int naod, float *logaod, float *a0, float *a1, float *a2){
const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
gsl_multimin_fminimizer *s = NULL;
gsl_vector *ss, *x;
gsl_multimin_function minex_func;
int b, bb, nb;
float *param;
size_t iter = 0;
int status;
double size;


  nb = get_brick_nbands(atc->xy_aod);

  alloc((void**)&param, naod*2+1, sizeof(float));

  param[0] = naod;
  for (b=0, bb=1; b<nb; b++){
    if (!atc->aod_bands[b]) continue;
    param[bb++] = atc->lwvl[b];
    param[bb++] = logaod[b];
  }


  /* Starting point */
  x = gsl_vector_alloc (3);
  gsl_vector_set (x, 0, 0);
  gsl_vector_set (x, 1, 0);
  gsl_vector_set (x, 2, 0);

  /* Set initial step sizes */
  ss = gsl_vector_alloc (3);
  gsl_vector_set (ss, 0, 0.01);
  gsl_vector_set (ss, 1, 0.01);
  gsl_vector_set (ss, 2, 0.02);

  /* Initialize method and iterate */
  minex_func.n = 3;
  minex_func.f = aod_polynomial_minimizer;
  minex_func.params = param;

  s = gsl_multimin_fminimizer_alloc(T, 3);
  gsl_multimin_fminimizer_set(s, &minex_func, x, ss);

  do {
    iter++;
    status = gsl_multimin_fminimizer_iterate(s);

    if (status) break;

    size = gsl_multimin_fminimizer_size(s);
    status = gsl_multimin_test_size(size, 1e-2);

  } while (status == GSL_CONTINUE && iter < 100);

  *a0 = exp(gsl_vector_get(s->x, 0));
  *a1 = gsl_vector_get(s->x, 1);
  *a2 = gsl_vector_get(s->x, 2);

  free((void*)param);
  gsl_vector_free(x);
  gsl_vector_free(ss);
  gsl_multimin_fminimizer_free(s);

  return status;
}


/** This function builds the continuous AOD field at low resolution. The 
+++ sparse AOD maps of the individual estimators are averaged per grid 
+++ cell. Interpolation and lowpass filtering is applied to provide a con-
+++ tinuos field.
--- atc:    atmospheric correction factors
--- dark:   dark target container
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int aod_map(atc_t *atc, dark_t *dark){
int b, nb, g, ng, o;
float w, aod;
float **map = NULL;
float *weight = NULL;

  nb = get_brick_nbands(atc->xy_aod);
  ng = get_brick_ncells(atc->xy_aod);

  // alocate memory
  alloc_2D((void***)&map, nb, ng, sizeof(float));
  alloc((void**)&weight, ng, sizeof(float));


  // add water measurements to map
  for (o=0; o<dark->kwat; o++){
    if (!dark->wat[o].valid) continue;
    g = dark->wat[o].g;
    w = dark->wat[o].rsq*dark->wat[o].rsq;
    for (b=0; b<nb; b++){
      aod = aod_elev_scale(dark->wat[o].aod[b], dark->wat[o].Ha, atc->Ha);
      map[b][g] += aod*w;
    }
    weight[g] += w;
  }
  
  // add shade measurements to map
  for (o=0; o<dark->kshd; o++){
    if (!dark->shd[o].valid) continue;
    g = dark->shd[o].g;
    w = dark->shd[o].rsq*dark->shd[o].rsq;
    for (b=0; b<nb; b++){
      aod = aod_elev_scale(dark->shd[o].aod[b], dark->shd[o].Ha, atc->Ha);
      map[b][g] += aod*w;
    }
    weight[g] += w;
  }
  
  // add veg measurements to map
  for (o=0; o<dark->kveg; o++){
    if (!dark->veg[o].valid) continue;
    g = dark->veg[o].g;
    w = dark->veg[o].rsq*dark->veg[o].rsq;
    for (b=0; b<nb; b++){
      aod = aod_elev_scale(dark->veg[o].aod[b], dark->veg[o].Ha, atc->Ha);
      map[b][g] += aod*w;
    }
    weight[g] += w;
  }


  /** compute AOD average for coarse grid cells **/
  for (g=0; g<ng; g++){
    for (b=0; b<nb; b++){
      if (weight[g] > 0){ 
        map[b][g] /= weight[g];
      } else {
        set_brick(atc->xy_interp, 0, g, true);
      }
    }
  }


  /** clean **/

  free((void*)weight);
  
 
  // interpolate AOD map
  if (interpolate_aod_map(atc, map) != SUCCESS){
    printf("interpolating between AOD failed.\n"); return FAILURE;}

  free_2D((void**)map, nb);

  return SUCCESS;
}


/** This function interpolates the average AOD maps that may contain no-
+++ data values. Inverse Distance Weighting is used, this reduces extra-
+++ polation artifacts. Lowpass filtering is applied to the interpolation
+++ to further smooth the map.
--- atc:     atmospheric correction factors
--- map_aod: averaged AOD map
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int interpolate_aod_map(atc_t *atc, float **map_aod){
int b, nb, green;
int e, f, g, ee, ff, ne, nf, ng;
int k = 0;
GDALGridAlgorithm eAlgorithm = GGA_InverseDistanceToAPower;
void *pOptions = NULL;
GDALDataType eOutputType = GDT_Float64;
double  sum,  num;
double gsum, gnum;
float aod;
double  *aod_x   = NULL;
double  *aod_y   = NULL;
double **aod_z   = NULL;
double *interpol = NULL;


  nb = get_brick_nbands(atc->xy_aod);
  nf = get_brick_ncols(atc->xy_aod);
  ne = get_brick_nrows(atc->xy_aod);
  ng = get_brick_ncells(atc->xy_aod);
  if ((green = find_domain(atc->xy_aod, "GREEN")) < 0) return FAILURE; 

  ParseAlgorithmAndOptions(szAlgNameInvDist, &eAlgorithm, &pOptions);

  // number of cells with AOD estimate
  for (g=0, k=0; g<ng; g++){
    if (map_aod[green][g] > 0) k++;
  }

  // allocate memory
  alloc((void**)&aod_x, k, sizeof(double));
  alloc((void**)&aod_y, k, sizeof(double));
  alloc_2D((void***)&aod_z, nb, k, sizeof(double));
  alloc((void**)&interpol, ng, sizeof(double));


  // copy AOD estimates to array
  for (e=0, k=0, g=0; e<ne; e++){
  for (f=0; f<nf; f++, g++){
    if (map_aod[green][g] > 0){
      for (b=0; b<nb; b++){
        aod_z[b][k] = map_aod[b][g];
      }
      aod_x[k] = f;
      aod_y[k] = e;
      k++;
    }
  }
  }


  // interpolate and smooth AOD
  for (b=0; b<nb; b++){

    // compute interpolation
    GDALGridCreate(eAlgorithm, pOptions, k, aod_x, aod_y, aod_z[b],
      0, nf-1, 0, ne-1, nf, ne, eOutputType, (void*)interpol, NULL, NULL);

    sum = num = 0;

    // lowpass filter
    for (e=0, g=0; e<ne; e++){
    for (f=0; f<nf; f++, g++){

      if (get_brick(atc->xy_view, ZEN, g) < 0) continue;

      gsum = gnum = 0;
      for (ee=-1; ee<=1; ee++){
      for (ff=-1; ff<=1; ff++){
        if (e+ee < 0 || f+ff < 0 || e+ee >= ne || f+ff >= nf) continue;
        gsum += interpol[(e+ee)*nf + (f+ff)];
        gnum++;
      }
      }
      
      if (gnum > 0) aod = (float)(gsum/gnum); else aod = 0.0;
      set_brick(atc->xy_aod, b, g, aod); 

      if (aod > 0){
        sum += aod;
        num++;
      }

    }
    }

    if (num > 0) atc->aod[b] = (float)(sum/num); else atc->aod[b] = 0.0;

  }

  // clean
  CPLFree(pOptions);
  free((void*)aod_x);
  free((void*)aod_y);
  free_2D((void**)aod_z, nb);
  free((void*)interpol);

  return SUCCESS;
}


/** This function is the entry point for AOD estimation. The fallback is
+++ either set to a global constant or to an externally provided spectrum.
+++ The Dark Object Persistance is compiled if demanded; if not, the comp-
+++ lete image is declared persitent. Dark targets are extracted from the
+++ image, and AOD is inferred on a per-object basis. The objects are av-
+++ eraged, and interpolated to produce an AOD map. Finally AOD is logged.
--- pl2:    L2 parameters
--- meta:   metadata
--- atc:    atmospheric correction factors
--- TOA:    TOA reflectance
--- QAI:    Quality Assurance Information
--- TOP:    Topographic Derivatives
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int compile_aod(par_ll_t *pl2, meta_t *meta, atc_t *atc, brick_t *TOA, brick_t *QAI, top_t *TOP){
int b, nb, o;
int green, sw2;
float res;
float *aod_lut = NULL;
//int *aod_bands = NULL;
dark_t dark;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  

  nb = get_brick_nbands(TOA);
  res = get_brick_res(TOA);
  if ((green = find_domain(TOA, "GREEN")) < 0) return FAILURE; 
  if ((sw2   = find_domain(TOA, "SWIR2")) < 0) return FAILURE; 


  /** read external AOD file
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  if ((aod_lut = aodfileread(pl2, atc)) == NULL) return FAILURE;

  if (!pl2->doaod){
    for (b=0; b<nb; b++) atc->aod[b] = aod_lut[b];
    free((void*)aod_lut);
    return SUCCESS;
  }

  cite_me(_CITE_AODEST_);


  /** estimate AOD from image
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  // use all visible bands + swir2 for AOD estimation
  for (b=0; b<nb; b++){
    if (atc->wvl[b] < 0.7 || b == sw2) atc->aod_bands[b] = true;
  }

#ifdef FORCE_DEBUG
printf("aod_bands as local variable?\n");
#endif
  // extract dark targets from image and tabulate necessary information
  dark.kwat = extract_dark_target(atc, TOA, QAI, TOP, _AOD_WAT_, &dark.wat);
  dark.kshd = extract_dark_target(atc, TOA, QAI, TOP, _AOD_SHD_, &dark.shd);
  dark.kveg = extract_dark_target(atc, TOA, QAI, TOP, _AOD_VEG_, &dark.veg);

  #ifdef FORCE_CLOCK
  proctime_print("DT extracted", TIME);
  #endif
  // object-based estimation of AOD
  if ((dark.nwat = aod_from_target(pl2, meta, atc, res, dark.wat, dark.kwat, _AOD_WAT_)) < 0) return FAILURE;
  if ((dark.nshd = aod_from_target(pl2, meta, atc, res, dark.shd, dark.kshd, _AOD_SHD_)) < 0) return FAILURE;
  if ((dark.nveg = aod_from_target(pl2, meta, atc, res, dark.veg, dark.kveg, _AOD_VEG_)) < 0) return FAILURE;
  
  // estimate elevation-dependency
  aod_elev_dependency(atc, &dark, green);
  

  #ifdef FORCE_CLOCK
  proctime_print("AOD estimated", TIME);
  #endif
  // compile AOD map or use fallback
  if ((dark.nwat+dark.nshd+dark.nveg) > 0){
    if (aod_map(atc, &dark) == FAILURE){
      printf("error in computing AOD map\n"); return FAILURE;}
    atc->aodmap = true;
  } else {
    for (b=0; b<nb; b++) atc->aod[b] = aod_lut[b];
    atc->aodmap = false;
  }
  #ifdef FORCE_CLOCK
  proctime_print("AOD map", TIME);
  #endif

  free((void*)aod_lut);
  
  for (o=0; o<dark.kwat; o++){
    free((void*)dark.wat[o].ttoa); free((void*)dark.wat[o].etoa);
    free((void*)dark.wat[o].aod);    
    free((void*)dark.wat[o].est);    
  }
  free((void*)dark.wat);
  
  for (o=0; o<dark.kshd; o++){
    free((void*)dark.shd[o].ttoa); free((void*)dark.shd[o].etoa);
    free((void*)dark.shd[o].aod);    
    free((void*)dark.shd[o].est);    
  }
  free((void*)dark.shd);
  
  for (o=0; o<dark.kveg; o++){
    free((void*)dark.veg[o].ttoa); free((void*)dark.veg[o].etoa);
    free((void*)dark.veg[o].aod);    
    free((void*)dark.veg[o].est);    
  }
  free((void*)dark.veg);


  /** print for logfile **/
  printf("AOD: %06.4f. # of targets: %d/%d. ", 
    atc->aod[green], dark.nwat, dark.nveg);
    
  #ifdef FORCE_DEBUG
  print_brick_info(atc->xy_aod);    set_brick_open(atc->xy_aod,    OPEN_CREATE); write_brick(atc->xy_aod);
  print_brick_info(atc->xy_interp); set_brick_open(atc->xy_interp, OPEN_CREATE); write_brick(atc->xy_interp);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("AOD estimation", TIME);
  #endif

  return SUCCESS;
}

