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
This file contains functions for polarmetrics
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "polar-hl.h"


int polar_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_pol_t *pol);


/** This function derives phenometrics from an interpolated time series
                          +++ for each year.
                          --- ts:       pointer to instantly useable TSA image arrays
                          --- mask_:    mask image
                          --- nc:       number of cells
                          --- ni:       number of interpolation steps
                          --- nodata:   nodata value
                          --- year_min: first year in the complete time series
                          --- year_max: last  year in the complete time series
                          --- pol:      pheno parameters
                          +++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int polar_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_pol_t *pol){
int l, npol = 15;
int year;
int p;
int i, ii, i_, i0, i1, ni_;
char cdat0[NPOW_10];
char cdat1[NPOW_10];
int nchar, error = 0;
float *v    = NULL;
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
float ce_left, ce_right, ce;
float v_left, v_right;
Spline *spl;


  valid = false;
  for (l=0; l<npol; l++){
    if (ts->pol_[l] != NULL) valid = true;
  }

  if (!valid) return CANCEL;




  #pragma omp parallel private(l,i,ii,i0,i1,ni_,ce_left,ce_right,v_left,v_right,year,valid,doymax,yoff,ce,i_,ymax,v,w,doy,dce,ce0,cdat0,cdat1,spl,nchar) firstprivate(southern) shared(mask_,ts,nc,ni,year_min,year_max,nodata,pol,nseg,npol) reduction(+: error) default(none)
  {

    // allocate
    alloc((void**)&v,     ni,  sizeof(float));
    alloc((void**)&doy,   ni,  sizeof(float));
    alloc((void**)&rad,   ni,  sizeof(float));
    alloc((void**)&pol_x, ni,  sizeof(float));
    alloc((void**)&pol_y, ni,  sizeof(float));


    #pragma omp for
    for (p=0; p<nc; p++){

      /** nodata if deriving POL failed **/
      for (l=0; l<npol; l++){
        if (ts->pol_[l] != NULL){
          for (year=0; year<pol->ny; year++) ts->pol_[l][year][p] = nodata;
        }
      }
      
      if (mask_ != NULL && !mask_[p]) continue;



      valid = true;
      mean_pol_x = 0;
      mean_pol_y = 0;


      /** copy ce/v to working variables 
      +++ and interpolate linearly to make sure **/
      for (i=0; i<ni; i++){

        // copy DOY
        doy[i] = ts->d_tsi[i].doy;
        rad[i] = ts->d_tsi[i].doy/365*2*M_PI;

        // linearly interpolate v-value
        if (ts->tsi_[i][p] == nodata){
          
          ce_left = ce_right = INT_MIN;
          v_left = v_right = nodata;
          ce = ts->d_tsi[i].ce;
          
          for (i_=i-1; i_>=0; i_--){
            if (ts->tsi_[i_][p] != nodata){
              ce_left = ts->d_tsi[i_].ce;
              v_left = ts->tsi_[i_][p];
              break;
            }
          }
          for (i_=i+1; i_<ni; i_++){
            if (ts->tsi_[i_][p] != nodata){
              ce_right = ts->d_tsi[i_].ce;
              v_right = ts->tsi_[i_][p];
              break;
            }
          }
          
          if (ce_left > 0 && ce_right > 0){
            v[i] = (v_left*(ce_right-ce) + v_right*(ce-ce_left))/(ce_right-ce_left);
          } else if (ce_left > 0){
            v[i] = v_left;
          } else if (ce_right > 0){
            v[i] = v_right;
          } else {
            valid = false;
          }

        // copy v-value
        } else {

          v[i] = ts->tsi_[i][p];

        }

        pol_x[i] = v[i]*cos(rad[i]);
        pol_y[i] = v[i]*sin(rad[i]);

        mean_pol_x += pol_x[i];
        mean_pol_y += pol_y[i];

      }

      if (!valid) continue;

      // mean of polar coordinates
      mean_pol_x /= ni;
      mean_pol_y /= ni;

      // average vector
      mean_rad = atan2(mean_pol_y, mean_pol_x);
      if (mean_rad <= 0) mean_rad += 2*M_PI;
      mean_v = sqrt(mean_pol_x*mean_pol_x + mean_pol_y*mean_pol_y);

      // diametric opposite of average vector = start of phenological year
      if (mean_rad < M_PI){
        theta = mean_rad + M_PI;
      } else {
        theta = mean_rad - M_PI;
      }

      // yoff = 0; // year offset, probably use?


      for (year=0; year<pol->ny; year++){

        sum_ann = 0;
        n       = 0;

        // extract annual values
        // cumulative values
        for (i=i1; i<ni; i++){

          if (ts->d_tsi[i].year <  year_min+year) continue;
          if (ts->d_tsi[i].year == year_min+year && rad[i] < theta) continue;
          if (ts->d_tsi[i].year >  year_min+year && rad[i] => theta) break;


          rad_ann[n] = rad[i];
          v_ann[n]   = v[i];

          sum_ann += v[i];
          cum_v_ann[n++] = sum_ann;

        }


        rad_start_grow = rad_early_grow = rad_mid_grow = rad_end_grow = rad_late_grow = rad_len_grow = -1;
        v_start_grow   = v_early_grow   = v_mid_grow   = v_end_grow   = v_late_grow                  = -1;
        mean_v_grow = var_v_grow = sd_v_grow = n_grow = 0;
        mean_pol_x_grow   = mean_pol_y_grow   = 0;
        mean_pol_x_spring = mean_pol_y_spring = 0;
        mean_pol_x_fall   = mean_pol_y_fall   = 0;

        for (i=0; i<n; i++){
          cum_v[i] /= sum;
          if (cum_v[i] >= 0.150 && rad_start_grow < 0){ rad_start_grow = rad_ann[i]; v_start_grow = v_ann[i];}
          if (cum_v[i] >= 0.500 && rad_mid_grow   < 0){ rad_mid_grow   = rad_ann[i]; v_mid_grow   = v_ann[i];}
          if (cum_v[i] >= 0.800 && rad_end_grow   < 0){ rad_end_grow   = rad_ann[i]; v_end_grow   = v_ann[i];}
          if (cum_v[i] >= 0.150 && cum_v[i] < 0.800){
            var_recurrence(v_ann[i], &mean_v_grow, &var_v_grow, ++n_grow);
            mean_pol_x_grow   += v_ann[i]*cos(rad_ann[i]);
            mean_pol_y_grow   += v_ann[i]*sin(rad_ann[i]);
            n_grow
          }
          if (cum_v[i] >= 0.150 && cum_v[i] < 0.500){
            mean_pol_x_spring += v_ann[i]*cos(rad_ann[i]);
            mean_pol_y_spring += v_ann[i]*sin(rad_ann[i]);
            n_spring++;
          }
          if (cum_v[i] >= 0.500 && cum_v[i] < 0.800){
            mean_pol_x_fall   += v_ann[i]*cos(rad_ann[i]);
            mean_pol_y_fall   += v_ann[i]*sin(rad_ann[i]);
            n_fall++;
          }
        }

        rad_len_grow = rad_end - rad_start;
        sd_v_grow    = standdev(var_v_grow, n);

        mean_pol_x_grow   /= n_grow;
        mean_pol_y_grow   /= n_grow;
        mean_pol_x_spring /= n_spring;
        mean_pol_y_spring /= n_spring;
        mean_pol_x_fall   /= n_fall;
        mean_pol_y_fall   /= n_fall;

        mean_rad_grow = atan2(mean_pol_y_grow, mean_pol_x_grow);
        if (mean_rad_grow <= 0) mean_rad_grow += 2*M_PI;
        mean_v_grow = sqrt(mean_pol_x_grow*mean_pol_x_grow + mean_pol_y_grow*mean_pol_y_grow);

        mean_rad_spring = atan2(mean_pol_y_spring, mean_pol_x_spring);
        if (mean_rad_spring <= 0) mean_rad_spring += 2*M_PI;
        mean_v_spring = sqrt(mean_pol_x_spring*mean_pol_x_spring + mean_pol_y_spring*mean_pol_y_spring);

        mean_rad_fall = atan2(mean_pol_y_fall, mean_pol_x_fall);
        if (mean_rad_fall <= 0) mean_rad_fall += 2*M_PI;
        mean_v_fall = sqrt(mean_pol_x_fall*mean_pol_x_fall + mean_pol_y_fall*mean_pol_y_fall);



        valid = false;

        // sanity check?
        // if () valid = true;

        valid = true;

        /** copy POL if all OK **/
        if (valid){
          if (pol->odem) ts->pol_[_POL_DEM_][year][p] = (short)(ph.doy_early_min*dce+ce0);   // days since 1st POL year
          if (pol->odss) ts->pol_[_POL_DSS_][year][p] = (short)(ph.doy_start_green*dce+ce0); // days since 1st POL year
          if (pol->odri) ts->pol_[_POL_DRI_][year][p] = (short)(ph.doy_early_flex*dce+ce0);  // days since 1st POL year
          if (pol->odps) ts->pol_[_POL_DPS_][year][p] = (short)(ph.doy_peak*dce+ce0);        // days since 1st POL year
          if (pol->odfi) ts->pol_[_POL_DFI_][year][p] = (short)(ph.doy_late_flex*dce+ce0);   // days since 1st POL year
          if (pol->odes) ts->pol_[_POL_DES_][year][p] = (short)(ph.doy_end_green*dce+ce0);   // days since 1st POL year
          if (pol->odlm) ts->pol_[_POL_DLM_][year][p] = (short)(ph.doy_late_min*dce+ce0);    // days since 1st POL year
          if (pol->olts) ts->pol_[_POL_LTS_][year][p] = (short)(ph.min_min_duration*dce);    // days
          if (pol->olgs) ts->pol_[_POL_LGS_][year][p] = (short)(ph.green_duration*dce);      // days
          if (pol->ovem) ts->pol_[_POL_VEM_][year][p] = (short)(ph.early_min_val);           // index value
          if (pol->ovss) ts->pol_[_POL_VSS_][year][p] = (short)(ph.start_green_val);         // index value
          if (pol->ovri) ts->pol_[_POL_VRI_][year][p] = (short)(ph.early_flex_val);          // index value
          if (pol->ovps) ts->pol_[_POL_VPS_][year][p] = (short)(ph.peak_val);                // index value
          if (pol->ovfi) ts->pol_[_POL_VFI_][year][p] = (short)(ph.late_flex_val);           // index value
          if (pol->oves) ts->pol_[_POL_VES_][year][p] = (short)(ph.end_green_val);           // index value
          if (pol->ovlm) ts->pol_[_POL_VLM_][year][p] = (short)(ph.late_min_val);            // index value
          if (pol->ovbl) ts->pol_[_POL_VBL_][year][p] = (short)(ph.latent_val);              // index value
          if (pol->ovsa) ts->pol_[_POL_VSA_][year][p] = (short)(ph.amplitude);               // index value
          if (pol->oist) ts->pol_[_POL_IST_][year][p] = (short)(ph.min_min_integral*dce*0.001); // days * index value * 10
          if (pol->oibl) ts->pol_[_POL_IBL_][year][p] = (short)(ph.latent_integral*dce*0.001);  // days * index value * 10
          if (pol->oibt) ts->pol_[_POL_IBT_][year][p] = (short)(ph.total_integral*dce*0.001);   // days * index value * 10
          if (pol->oigs) ts->pol_[_POL_IGS_][year][p] = (short)(ph.green_integral*dce*0.001);   // days * index value * 10
          if (pol->orar) ts->pol_[_POL_RAR_][year][p] = (short)(ph.greenup_rate/dce);        // index value / days
          if (pol->oraf) ts->pol_[_POL_RAF_][year][p] = (short)(ph.senescence_rate/dce);     // index value / days
          if (pol->ormr) ts->pol_[_POL_RMR_][year][p] = (short)(ph.early_flex_rate/dce);     // index value / days
          if (pol->ormf) ts->pol_[_POL_RMF_][year][p] = (short)(ph.late_flex_rate/dce);      // index value / days
        }
        
      }


    }

    /** clean **/
    free((void*)v);  free((void*)doy);
    free((void*)pol_x); free((void*)pol_y);

  }

  if (error > 0) return FAILURE;

  return SUCCESS;
}


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
int tsa_polar(tsa_t *ts, small *mask_, int nc, int ni, short nodata, par_hl_t *phl){


  //if (phl->tsa.pol.ospl +
  //    phl->tsa.pol.opol +
  //    phl->tsa.pol.otrd +
  //    phl->tsa.pol.ocat == 0) return SUCCESS;

  //cite_me(_CITE_POLAR_);


  if (polar_ts(ts, mask_, nc, ni, nodata, 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, &phl->tsa.pol) == FAILURE) return FAILURE;



  return SUCCESS;
}

