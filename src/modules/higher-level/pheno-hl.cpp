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
This file contains functions for phenometrics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "pheno-hl.h"


#ifdef SPLITS

#include <splits.h>

using namespace splits;

int pheno(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int y_index, int year_min, par_lsp_t *lsp);
int pheno_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_lsp_t *lsp);

/** This function derives phenometrics from an interpolated time series
+++ for one target year.
--- ts:       pointer to instantly useable TSA image arrays
--- mask_:    mask image
--- nc:       number of cells
--- ni:       number of interpolation steps
--- nodata:   nodata value
--- y_index:  index of target year
--- year_min: first year in the complete time series
--- lsp:      pheno parameters
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int pheno(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int y_index, int year_min, par_lsp_t *lsp){
int l, nlsp = 26;
int year;
int p;
int i, ii, i_, i0, i1, ni_;
char cdat0[NPOW_10];
char cdat1[NPOW_10];
int nchar, error = 0;
float *y    = NULL;
float *yhat = NULL;
float *w    = NULL;
float *doy  = NULL;
float dce, ce0;
float ymax;
int doymax, yoff;
bool southern = false;
bool valid;
float dseg;
int nseg;
float x_left, x_right, x;
float y_left, y_right;
Spline *spl;


  valid = false;
  for (l=0; l<nlsp; l++){
    if (ts->lsp_[l] != NULL) valid = true;
  }
  if (ts->spl_ != NULL) valid = true;
  
  if (!valid) return CANCEL;


  // which hemisphere?
  switch (lsp->hemi){
    case _HEMI_NORTH_:
      southern = false; break; // N
    case _HEMI_SOUTH_:
      southern = true;  break; // S
    case _HEMI_MIXED_:
      southern = false; break; // decide later
  }
  
  // number of segments
  dseg = lsp->nseg/365.0;
  nseg = (int)round((365-lsp->dprev)*dseg + lsp->nseg + lsp->dnext*dseg);
  
  year = year_min+y_index+1;
  

  #pragma omp parallel private(l,i,ii,i0,i1,ni_,x_left,x_right,y_left,y_right,valid,doymax,yoff,x,i_,ymax,y,yhat,w,doy,dce,ce0,cdat0,cdat1,spl,nchar) firstprivate(southern) shared(mask_,ts,nc,ni,y_index,year_min,nodata,lsp,nseg,year,nlsp) reduction(+: error) default(none)
  {

    // allocate
    alloc((void**)&y,    ni,  sizeof(float));
    alloc((void**)&yhat, ni,  sizeof(float));
    alloc((void**)&w,    ni,  sizeof(float));
    alloc((void**)&doy,  ni,  sizeof(float));
    
   
    #pragma omp for
    for (p=0; p<nc; p++){
      
      // init with nodata
      for (l=0; l<nlsp; l++){
        if (ts->lsp_[l] != NULL) ts->lsp_[l][y_index][p] = nodata;
      }

      if (mask_ != NULL && !mask_[p]) continue;


      
      i0 = -1; i1 = -1;
      ymax = INT_MIN;
      valid = false;
      doymax = 182;
      yoff = 0;

      /** copy x/y/w to working variables **/
      for (i=0, ii=0; i<ni; i++){

        // subset by time
        if (ts->d_tsi[i].year == year ||
           (ts->d_tsi[i].year == year-1 && ts->d_tsi[i].doy >= lsp->dprev) || 
           (ts->d_tsi[i].year == year+1 && ts->d_tsi[i].doy <= lsp->dnext)){

          if (i0 < 0) i0 = i;
          i1 = i;

          // copy DOY
          doy[ii] = ts->d_tsi[i].doy;

          // linearly interpolate y-value, give half weight
          // if nothing to interpolate, assign nodata, and give 0 weight
          if (ts->tsi_[i][p] == nodata){
            
            x_left = x_right = INT_MIN;
            y_left = y_right = nodata;
            x = ts->d_tsi[i].ce;
            
            for (i_=i-1; i_>=0; i_--){
              if (ts->d_tsi[i_].year < year-1 ||
                 (ts->d_tsi[i_].year == year-1 && ts->d_tsi[i_].doy < lsp->dprev)){
                break;
              }
              if (ts->tsi_[i_][p] != nodata){
                x_left = ts->d_tsi[i_].ce;
                y_left = ts->tsi_[i_][p];
                break;
              }
            }
            for (i_=i+1; i_<ni; i_++){
              if (ts->d_tsi[i_].year > year+1 ||
                 (ts->d_tsi[i_].year == year+1 && ts->d_tsi[i_].doy > lsp->dnext)){
                break;
              }
              if (ts->tsi_[i_][p] != nodata){
                x_right = ts->d_tsi[i_].ce;
                y_right = ts->tsi_[i_][p];
                break;
              }
            }
            
            if (x_left > 0 && x_right > 0){
              y[ii] = (y_left*(x_right-x) + y_right*(x-x_left))/(x_right-x_left);
              w[ii] = 0.5;
            } else {
              y[ii] = nodata;
              w[ii] = 0.0;
            }

          // copy y-value, determine max y, give max weight
          } else {

            y[ii] = ts->tsi_[i][p];
            w[ii] = 1.0;

            if (y[ii] > ymax){
              ymax = y[ii];
              doymax = ts->d_tsi[i].doy;
            }

          }

          ii++;

        }

      }

      // length of compiled time series
      ni_ = ii;



      if (ymax > lsp->minval){

        /** time step in days, and first day **/
        dce   = ts->d_tsi[i1].ce - ts->d_tsi[i0].ce;
        ce0   = ts->d_tsi[i0].ce - year_min*365-365;
        //printf("ce0, dce: %f, %f\n", ce0, dce);

        /** allow switching hemispheres **/
        if (lsp->hemi == _HEMI_MIXED_ && (doymax < 90 || doymax > 275)){
          southern = true;}
        if (southern) yoff = -1;
        //printf("doymax: %d\n", doymax);

        /** compile temporal domaion **/
        nchar = snprintf(cdat0, NPOW_10, "%d/%d", ts->d_tsi[i0].doy, ts->d_tsi[i0].year);
        if (nchar < 0 || nchar >= NPOW_10){ 
          printf("Buffer Overflow in assembling date\n"); error++; continue;}
        nchar = snprintf(cdat1, NPOW_10, "%d/%d", ts->d_tsi[i1].doy, ts->d_tsi[i1].year);
        if (nchar < 0 || nchar >= NPOW_10){ 
          printf("Buffer Overflow in assembling date\n"); error++; continue;}

        Date dat0(cdat0); 
        Date dat1(cdat1); 
        Domain domain(dat0, dat1, doy, ni_);
        //printf("Domain: %s - %s\n", cdat0, cdat1);
      
        /** fit spline **/
        if ((spl = create_spline(domain, y, w, UNIFORM_BSPLINE, nseg, 3)) == NULL){
          printf("NULL spline returned\n"); continue;} // segfault if using quadratic spline.. bug, I guess..

        /** evaluate spline **/
        evaluate(spl, domain, yhat); // pheno values look odd if spline is not evaluated.. idk, bug?
        if (ts->spl_ != NULL){
          for (i=0; i<ni_; i++) ts->spl_[i+i0][p] = (short)yhat[i];
        }
        

        /** derive the LSP **/
        std::map<int, Pheno_set> season;
        season = phenology(spl, domain, southern, lsp->start, PHENOLOGY_RAW);
        Pheno_set ph = season[year+yoff];

        /** sanity check **/
        if (ph.peak_val         > lsp->minval &&
            ph.amplitude        > lsp->minamp && 
            ph.doy_early_min    < ph.doy_start_green && 
            ph.doy_start_green  < ph.doy_peak &&
            ph.doy_peak         < ph.doy_end_green && 
            ph.doy_end_green    < ph.doy_late_min) valid = true;


        /** copy LSP if all OK **/
        if (valid){
          if (lsp->use[_LSP_DEM_]) ts->lsp_[_LSP_DEM_][y_index][p] = (short)(ph.doy_early_min*dce+ce0);   // days since 1st LSP year
          if (lsp->use[_LSP_DSS_]) ts->lsp_[_LSP_DSS_][y_index][p] = (short)(ph.doy_start_green*dce+ce0); // days since 1st LSP year
          if (lsp->use[_LSP_DRI_]) ts->lsp_[_LSP_DRI_][y_index][p] = (short)(ph.doy_early_flex*dce+ce0);  // days since 1st LSP year
          if (lsp->use[_LSP_DPS_]) ts->lsp_[_LSP_DPS_][y_index][p] = (short)(ph.doy_peak*dce+ce0);        // days since 1st LSP year
          if (lsp->use[_LSP_DFI_]) ts->lsp_[_LSP_DFI_][y_index][p] = (short)(ph.doy_late_flex*dce+ce0);   // days since 1st LSP year
          if (lsp->use[_LSP_DES_]) ts->lsp_[_LSP_DES_][y_index][p] = (short)(ph.doy_end_green*dce+ce0);   // days since 1st LSP year
          if (lsp->use[_LSP_DLM_]) ts->lsp_[_LSP_DLM_][y_index][p] = (short)(ph.doy_late_min*dce+ce0);    // days since 1st LSP year
          if (lsp->use[_LSP_LTS_]) ts->lsp_[_LSP_LTS_][y_index][p] = (short)(ph.min_min_duration*dce);    // days
          if (lsp->use[_LSP_LGS_]) ts->lsp_[_LSP_LGS_][y_index][p] = (short)(ph.green_duration*dce);      // days
          if (lsp->use[_LSP_VEM_]) ts->lsp_[_LSP_VEM_][y_index][p] = (short)(ph.early_min_val);           // index value
          if (lsp->use[_LSP_VSS_]) ts->lsp_[_LSP_VSS_][y_index][p] = (short)(ph.start_green_val);         // index value
          if (lsp->use[_LSP_VRI_]) ts->lsp_[_LSP_VRI_][y_index][p] = (short)(ph.early_flex_val);          // index value
          if (lsp->use[_LSP_VPS_]) ts->lsp_[_LSP_VPS_][y_index][p] = (short)(ph.peak_val);                // index value
          if (lsp->use[_LSP_VFI_]) ts->lsp_[_LSP_VFI_][y_index][p] = (short)(ph.late_flex_val);           // index value
          if (lsp->use[_LSP_VES_]) ts->lsp_[_LSP_VES_][y_index][p] = (short)(ph.end_green_val);           // index value
          if (lsp->use[_LSP_VLM_]) ts->lsp_[_LSP_VLM_][y_index][p] = (short)(ph.late_min_val);            // index value
          if (lsp->use[_LSP_VBL_]) ts->lsp_[_LSP_VBL_][y_index][p] = (short)(ph.latent_val);              // index value
          if (lsp->use[_LSP_VSA_]) ts->lsp_[_LSP_VSA_][y_index][p] = (short)(ph.amplitude);               // index value
          if (lsp->use[_LSP_IST_]) ts->lsp_[_LSP_IST_][y_index][p] = (short)(ph.min_min_integral*dce*0.001); // days * index value * 10
          if (lsp->use[_LSP_IBL_]) ts->lsp_[_LSP_IBL_][y_index][p] = (short)(ph.latent_integral*dce*0.001);  // days * index value * 10
          if (lsp->use[_LSP_IBT_]) ts->lsp_[_LSP_IBT_][y_index][p] = (short)(ph.total_integral*dce*0.001);   // days * index value * 10
          if (lsp->use[_LSP_IGS_]) ts->lsp_[_LSP_IGS_][y_index][p] = (short)(ph.green_integral*dce*0.001);   // days * index value * 10
          if (lsp->use[_LSP_RAR_]) ts->lsp_[_LSP_RAR_][y_index][p] = (short)(ph.greenup_rate/dce);        // index value / days
          if (lsp->use[_LSP_RAF_]) ts->lsp_[_LSP_RAF_][y_index][p] = (short)(ph.senescence_rate/dce);     // index value / days
          if (lsp->use[_LSP_RMR_]) ts->lsp_[_LSP_RMR_][y_index][p] = (short)(ph.early_flex_rate/dce);     // index value / days
          if (lsp->use[_LSP_RMF_]) ts->lsp_[_LSP_RMF_][y_index][p] = (short)(ph.late_flex_rate/dce);      // index value / days
        }

        destroy_spline(spl);

      }

    }

    /** clean **/
    free((void*)y); free((void*)yhat);
    free((void*)w); free((void*)doy);

  }

  if (error > 0) return FAILURE;
  
  return SUCCESS;
}


/** This function derives phenometrics from an interpolated time series
+++ for each year.
--- ts:       pointer to instantly useable TSA image arrays
--- mask_:    mask image
--- nc:       number of cells
--- ni:       number of interpolation steps
--- nodata:   nodata value
--- year_min: first year in the complete time series
--- year_max: last  year in the complete time series
--- lsp:      pheno parameters
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int pheno_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_lsp_t *lsp){
int l, nlsp = 26;
int year;
int p;
int i, ii, i_, i0, i1, ni_;
char cdat0[NPOW_10];
char cdat1[NPOW_10];
int nchar, error = 0;
float *y    = NULL;
float *yhat = NULL;
float *w    = NULL;
float *doy  = NULL;
float dce, ce0;
float ymax;
int doymax, yoff;
bool southern = false;
bool valid;
float dseg;
int nseg;
float x_left, x_right, x;
float y_left, y_right;
Spline *spl;


  valid = false;
  for (l=0; l<nlsp; l++){
    if (ts->lsp_[l] != NULL) valid = true;
  }
  if (ts->spl_ != NULL) valid = true;
  
  if (!valid) return CANCEL;


  // which hemisphere?
  switch (lsp->hemi){
    case _HEMI_NORTH_:
      southern = false; break; // N
    case _HEMI_SOUTH_:
      southern = true;  break; // S
    case _HEMI_MIXED_:
      southern = false; break; // decide later
  }
  
  // number of segments
  dseg = lsp->nseg/365.0;
  nseg = (int)round((365-lsp->dprev)*dseg + lsp->nseg*lsp->ny + lsp->dnext*dseg);


  #pragma omp parallel private(l,i,ii,i0,i1,ni_,x_left,x_right,y_left,y_right,year,valid,doymax,yoff,x,i_,ymax,y,yhat,w,doy,dce,ce0,cdat0,cdat1,spl,nchar) firstprivate(southern) shared(mask_,ts,nc,ni,year_min,year_max,nodata,lsp,nseg,nlsp) reduction(+: error) default(none)
  {

    // allocate
    alloc((void**)&y,    ni,  sizeof(float));
    alloc((void**)&yhat, ni,  sizeof(float));
    alloc((void**)&w,    ni,  sizeof(float));
    alloc((void**)&doy,  ni,  sizeof(float));
    
   
    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (l=0; l<nlsp; l++){
          if (ts->lsp_[l] != NULL){
            for (year=0; year<lsp->ny; year++) ts->lsp_[l][year][p] = nodata;
          }
        }
        continue;
      }

      
      i0 = -1; i1 = -1;
      ymax = INT_MIN;
      doymax = 182;
      yoff = 0;

      /** copy x/y/w to working variables **/
      for (i=0, ii=0; i<ni; i++){

        // subset by time
        if ((ts->d_tsi[i].year > year_min && ts->d_tsi[i].year < year_max) ||
           (ts->d_tsi[i].year == year_min && ts->d_tsi[i].doy >= lsp->dprev) || 
           (ts->d_tsi[i].year == year_max && ts->d_tsi[i].doy <= lsp->dnext)){

          if (i0 < 0) i0 = i;
          i1 = i;

          // copy DOY
          doy[ii] = ts->d_tsi[i].doy;

          // linearly interpolate y-value, give half weight
          // if nothing to interpolate, assign nodata, and give 0 weight
          if (ts->tsi_[i][p] == nodata){
            
            x_left = x_right = INT_MIN;
            y_left = y_right = nodata;
            x = ts->d_tsi[i].ce;
            
            for (i_=i-1; i_>=0; i_--){
              if (ts->d_tsi[i_].year < year_min ||
                 (ts->d_tsi[i_].year == year_min && ts->d_tsi[i_].doy < lsp->dprev)){
                break;
              }
              if (ts->tsi_[i_][p] != nodata){
                x_left = ts->d_tsi[i_].ce;
                y_left = ts->tsi_[i_][p];
                break;
              }
            }
            for (i_=i+1; i_<ni; i_++){
              if (ts->d_tsi[i_].year > year_max ||
                 (ts->d_tsi[i_].year == year_max && ts->d_tsi[i_].doy > lsp->dnext)){
                break;
              }
              if (ts->tsi_[i_][p] != nodata){
                x_right = ts->d_tsi[i_].ce;
                y_right = ts->tsi_[i_][p];
                break;
              }
            }
            
            if (x_left > 0 && x_right > 0){
              y[ii] = (y_left*(x_right-x) + y_right*(x-x_left))/(x_right-x_left);
              w[ii] = 0.5;
            } else {
              y[ii] = nodata;
              w[ii] = 0.0;
            }

          // copy y-value, determine max y, give max weight
          } else {

            y[ii] = ts->tsi_[i][p];
            w[ii] = 1.0;

            if (y[ii] > ymax){
              ymax = y[ii];
              doymax = ts->d_tsi[i].doy;
            }

          }

          ii++;

        }

      }

      // length of compiled time series
      ni_ = ii;



      /** nodata if deriving LSP failed **/
      for (l=0; l<nlsp; l++){
        if (ts->lsp_[l] != NULL){
          for (year=0; year<lsp->ny; year++) ts->lsp_[l][year][p] = nodata;
        }
      }


      /** derive LSP **/
      if (ymax > lsp->minval){

        /** time step in days, and first day **/
        dce   = ts->d_tsi[i1].ce - ts->d_tsi[i0].ce;
        ce0   = ts->d_tsi[i0].ce - year_min*365-365;
        //printf("ce0, dce: %f, %f\n", ce0, dce);

        /** allow switching hemispheres **/
        if (lsp->hemi == _HEMI_MIXED_ && (doymax < 90 || doymax > 275)){
          southern = true;}
        if (southern) yoff = -1;
        //printf("doymax: %d\n", doymax);

        /** compile temporal domaion **/
        nchar = snprintf(cdat0, NPOW_10, "%d/%d", ts->d_tsi[i0].doy, ts->d_tsi[i0].year);
        if (nchar < 0 || nchar >= NPOW_10){ 
          printf("Buffer Overflow in assembling date\n"); error++; continue;}
        nchar = snprintf(cdat1, NPOW_10, "%d/%d", ts->d_tsi[i1].doy, ts->d_tsi[i1].year);
        if (nchar < 0 || nchar >= NPOW_10){ 
          printf("Buffer Overflow in assembling date\n"); error++; continue;}
        
        Date dat0(cdat0); 
        Date dat1(cdat1); 
        Domain domain(dat0, dat1, doy, ni_);
        //printf("Domain: %s - %s\n", cdat0, cdat1);
      
        /** fit spline **/
        if ((spl = create_spline(domain, y, w, UNIFORM_BSPLINE, nseg, 3)) == NULL){
          printf("NULL spline returned\n");} // segfault if using quadratic spline.. bug, I guess..

        /** evaluate spline **/
        evaluate(spl, domain, yhat); // pheno values look odd if spline is not evaluated.. idk, bug?
        for (i=0; i<ni_; i++) ts->spl_[i+i0][p] = (short)yhat[i];

        /** derive the LSP **/
        std::map<int, Pheno_set> season;
        season = phenology(spl, domain, southern, lsp->start, PHENOLOGY_RAW);
        //printf("Number of seasons: %d\n", (int)season.size());

        for (year=0; year<lsp->ny; year++){

          Pheno_set ph = season[year_min+year+1+yoff];
          //printf("%d %f %f %f\n", year_min+year+1+yoff, ph.doy_early_min, ph.doy_peak, ph.doy_late_min);
          //printf("%d %f %f %f\n", year_min+year+1+yoff, season[year_min+year+1+yoff].doy_early_min, season[year_min+year+1+yoff].doy_peak, season[year_min+year+1+yoff].doy_late_min);
          valid = false;

          /** sanity check **/
          if (ph.peak_val > lsp->minval &&
              ph.amplitude > lsp->minamp && 
              ph.doy_early_min < ph.doy_start_green && 
              ph.doy_start_green < ph.doy_peak &&
              ph.doy_peak < ph.doy_end_green && 
              ph.doy_end_green < ph.doy_late_min) valid = true;


          /** copy LSP if all OK **/
          if (valid){
            if (lsp->use[_LSP_DEM_]) ts->lsp_[_LSP_DEM_][year][p] = (short)(ph.doy_early_min*dce+ce0);   // days since 1st LSP year
            if (lsp->use[_LSP_DSS_]) ts->lsp_[_LSP_DSS_][year][p] = (short)(ph.doy_start_green*dce+ce0); // days since 1st LSP year
            if (lsp->use[_LSP_DRI_]) ts->lsp_[_LSP_DRI_][year][p] = (short)(ph.doy_early_flex*dce+ce0);  // days since 1st LSP year
            if (lsp->use[_LSP_DPS_]) ts->lsp_[_LSP_DPS_][year][p] = (short)(ph.doy_peak*dce+ce0);        // days since 1st LSP year
            if (lsp->use[_LSP_DFI_]) ts->lsp_[_LSP_DFI_][year][p] = (short)(ph.doy_late_flex*dce+ce0);   // days since 1st LSP year
            if (lsp->use[_LSP_DES_]) ts->lsp_[_LSP_DES_][year][p] = (short)(ph.doy_end_green*dce+ce0);   // days since 1st LSP year
            if (lsp->use[_LSP_DLM_]) ts->lsp_[_LSP_DLM_][year][p] = (short)(ph.doy_late_min*dce+ce0);    // days since 1st LSP year
            if (lsp->use[_LSP_LTS_]) ts->lsp_[_LSP_LTS_][year][p] = (short)(ph.min_min_duration*dce);    // days
            if (lsp->use[_LSP_LGS_]) ts->lsp_[_LSP_LGS_][year][p] = (short)(ph.green_duration*dce);      // days
            if (lsp->use[_LSP_VEM_]) ts->lsp_[_LSP_VEM_][year][p] = (short)(ph.early_min_val);           // index value
            if (lsp->use[_LSP_VSS_]) ts->lsp_[_LSP_VSS_][year][p] = (short)(ph.start_green_val);         // index value
            if (lsp->use[_LSP_VRI_]) ts->lsp_[_LSP_VRI_][year][p] = (short)(ph.early_flex_val);          // index value
            if (lsp->use[_LSP_VPS_]) ts->lsp_[_LSP_VPS_][year][p] = (short)(ph.peak_val);                // index value
            if (lsp->use[_LSP_VFI_]) ts->lsp_[_LSP_VFI_][year][p] = (short)(ph.late_flex_val);           // index value
            if (lsp->use[_LSP_VES_]) ts->lsp_[_LSP_VES_][year][p] = (short)(ph.end_green_val);           // index value
            if (lsp->use[_LSP_VLM_]) ts->lsp_[_LSP_VLM_][year][p] = (short)(ph.late_min_val);            // index value
            if (lsp->use[_LSP_VBL_]) ts->lsp_[_LSP_VBL_][year][p] = (short)(ph.latent_val);              // index value
            if (lsp->use[_LSP_VSA_]) ts->lsp_[_LSP_VSA_][year][p] = (short)(ph.amplitude);               // index value
            if (lsp->use[_LSP_IST_]) ts->lsp_[_LSP_IST_][year][p] = (short)(ph.min_min_integral*dce*0.001); // days * index value * 10
            if (lsp->use[_LSP_IBL_]) ts->lsp_[_LSP_IBL_][year][p] = (short)(ph.latent_integral*dce*0.001);  // days * index value * 10
            if (lsp->use[_LSP_IBT_]) ts->lsp_[_LSP_IBT_][year][p] = (short)(ph.total_integral*dce*0.001);   // days * index value * 10
            if (lsp->use[_LSP_IGS_]) ts->lsp_[_LSP_IGS_][year][p] = (short)(ph.green_integral*dce*0.001);   // days * index value * 10
            if (lsp->use[_LSP_RAR_]) ts->lsp_[_LSP_RAR_][year][p] = (short)(ph.greenup_rate/dce);        // index value / days
            if (lsp->use[_LSP_RAF_]) ts->lsp_[_LSP_RAF_][year][p] = (short)(ph.senescence_rate/dce);     // index value / days
            if (lsp->use[_LSP_RMR_]) ts->lsp_[_LSP_RMR_][year][p] = (short)(ph.early_flex_rate/dce);     // index value / days
            if (lsp->use[_LSP_RMF_]) ts->lsp_[_LSP_RMF_][year][p] = (short)(ph.late_flex_rate/dce);      // index value / days
          }
          
        }

        destroy_spline(spl);

      }
     
    }

    /** clean **/
    free((void*)y); free((void*)yhat);
    free((void*)w); free((void*)doy);

  }

  if (error > 0) return FAILURE;

  return SUCCESS;
}


#endif


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function derives phenometrics from an interpolated time series.
+++ The function will fail if FORCE was not compiled with SPLITS (see in-
+++ stallation instructions). Do not expect this function to work if you
+++ have fairly sparse time series. In addition, the time series needs to
+++ be long enough (at least extend into the previous and next year). Phe-
+++ nometrics are derived for each given year.
--- ts:     pointer to instantly useable TSA image arrays
--- mask_:  mask image
--- nc:     number of cells
--- ni:     number of interpolation steps
--- nodata: nodata value
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tsa_pheno(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_hl_t *phl){


  if (phl->tsa.lsp.ospl +
      phl->tsa.lsp.olsp +
      phl->tsa.lsp.otrd +
      phl->tsa.lsp.ocat == 0) return SUCCESS;
  
  #ifndef SPLITS
  
  printf("Cannot compute phenometrics.\n");
  printf("SPLITS is not available.\n");
  printf("Install SPLITS and re-compile (see user guide)\n");
  return FAILURE;
  
  #else

  int y;

  cite_me(_CITE_SPLITS_);

  //if (0 > 1){
  //  if (pheno_ts(ts, mask_, nc, ni, nodata, 
  //    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, &phl->tsa.lsp) == FAILURE) return FAILURE;
  //} else {
    for (y=0; y<phl->tsa.lsp.ny; y++){
      if (pheno(ts, mask_, nc, ni, nodata, 
        y, phl->date_range[_MIN_].year, &phl->tsa.lsp) == FAILURE) return FAILURE;
    }
  //}
  
  #endif


  return SUCCESS;
}

