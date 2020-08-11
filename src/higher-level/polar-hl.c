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


enum { _RAD_, _VAL_, _CUM_, _YEAR_, _DOY_, _CE_, _SEASON_, _PCX_, _PCY_, _COORD_LEN_ };

void polar_coords(float r, float v, float yr, float polar_array[_COORD_LEN_]);
void polar_vector(float x, float y, float yr, float doy_theta, float polar_array[_COORD_LEN_]);
void identify_seasons(float **polar, int ni, int istep, float doy_theta);
void accumulate_seasons(float **polar, int ni);

int polar_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_tsi_t *tsi, par_pol_t *pol);




void polar_coords(float r, float v, float yr, float polar_array[_COORD_LEN_]){
float doy;


  doy = r*365.0/(2.0*M_PI);

  polar_array[_RAD_]  = r;
  polar_array[_VAL_]  = v;
  polar_array[_YEAR_] = yr;
  polar_array[_DOY_]  = doy;
  polar_array[_CE_]   = doy2ce(doy, yr);
  polar_array[_PCX_]  = v*cos(r);
  polar_array[_PCY_]  = v*sin(r);

  return;
}


void polar_vector(float x, float y, float yr, float doy_theta, float polar_array[_COORD_LEN_]){
float r, v, doy;


  r = atan2(y, x);
  if (r <= 0) r += 2*M_PI;
  v = sqrt(x*x + y*y);
  
  doy = r*365.0/(2.0*M_PI);

  polar_array[_RAD_]  = r;
  polar_array[_VAL_]  = v;
  polar_array[_YEAR_] = yr;
  polar_array[_DOY_]  = doy;

  if (doy > doy_theta){
    polar_array[_CE_] = doy2ce(doy, yr);
  } else {
    polar_array[_CE_] = doy2ce(doy, yr+1);
  }

  polar_array[_PCX_]  = x;
  polar_array[_PCY_]  = y;

  return;
}


void identify_seasons(float **polar, int ni, int istep, float doy_theta){
int i, s = -1, y = 0;
float rstep = istep/365.0*2.0*M_PI;
float ce_theta;


  for (i=0; i<ni; i++){

    ce_theta = doy2ce(doy_theta, y);

    if (polar[i][_CE_] >= ce_theta && 
        polar[i][_CE_]-ce_theta <= istep){ s++; y++;}

    polar[i][_SEASON_] = s;

  }

  return;
}

void accumulate_seasons(float **polar, int ni){
int i, s;
float sum;


  polar[0][_CUM_] = polar[0][_VAL_];
  s = polar[0][_SEASON_];

  for (i=1; i<ni; i++){

    if (polar[i][_SEASON_] != s){
      polar[i][_CUM_] = polar[i][_VAL_];
      s = polar[i][_SEASON_];
    } else {
      polar[i][_CUM_] = polar[i-1][_CUM_] + polar[i][_VAL_];
    }

  }

  sum = polar[ni-1][_CUM_];
  s   = polar[ni-1][_SEASON_];

  // absolute sum to percent
  for (i=(ni-1); i>=0; i--){

    if (polar[i][_SEASON_] != s){
      sum = polar[i][_CUM_];
      s   = polar[i][_SEASON_];
    }

    polar[i][_CUM_] /= sum;

  }

  return;
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
                          --- pol:      pheno parameters
                          +++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int polar_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_tsi_t *tsi, par_pol_t *pol){
int l;
int year;
int p;
int i, i_, i0;
int s, y;
float r, v;
bool valid;
float ce_left, ce_right, ce;
float v_left, v_right;

enum { _START_, _MID_, _END_, _EVENT_LEN_ };
enum { _LONGTERM_, _THISYEAR_, _EARLY_, _GROW_, _LATE_, _WINDOW_LEN_};

float theta, doy_theta, ce_theta;
float theta_now, doy_theta_now, ce_theta_now;

float timing[_EVENT_LEN_][_COORD_LEN_];
float vector[_WINDOW_LEN_][_COORD_LEN_];
float mean_window[_WINDOW_LEN_][2];
int   n_window[_WINDOW_LEN_];
double recurrence[2];

float **polar = NULL;



  valid = false;
  for (l=0; l<_POL_LENGTH_; l++){
    if (ts->pol_[l] != NULL) valid = true;
  }

  if (!valid) return CANCEL;




  //#pragma omp parallel private(l,i,i0,ni_,ce_left,ce_right,v_left,v_right,year,valid,ce,v,doy) firstprivate(southern) shared(mask_,ts,nc,ni,year_min,year_max,nodata,pol,nseg) default(none)
  {

    // allocate
    alloc_2D((void***)&polar, ni, _COORD_LEN_, sizeof(float));


    //#pragma omp for
    for (p=0; p<nc; p++){

      /** nodata if deriving POL failed **/
      for (l=0; l<_POL_LENGTH_; l++){
        if (ts->pol_[l] != NULL){
          for (year=0; year<pol->ny; year++) ts->pol_[l][year][p] = nodata;
        }
      }
      
      if (mask_ != NULL && !mask_[p]) continue;



      valid = true;
      memset(mean_window[_LONGTERM_], 0, 2*sizeof(float));


      /** copy doy/v to working variables 
      +++ and interpolate linearly to make sure **/
      for (i=0; i<ni; i++){

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
            v = (v_left*(ce_right-ce) + v_right*(ce-ce_left))/(ce_right-ce_left);
          } else if (ce_left > 0){
            v = v_left;
          } else if (ce_right > 0){
            v = v_right;
          } else {
            v = nodata;
            valid = false;
          }

        // copy v-value
        } else {

          v = ts->tsi_[i][p];

        }

        r = ts->d_tsi[i].doy/365.0*2.0*M_PI;

if (p == 375639) printf("doy: %d\n", ts->d_tsi[i].doy);
if (p == 375639) printf("r:   %f\n", r);
if (p == 375639) printf("v:   %f\n", v);
if (v < 0) v = 0;
        polar_coords(r, v, ts->d_tsi[i].year-year_min, polar[i]);
if (p == 375639) printf("x:   %f\n", polar[i][_PCX_]);
if (p == 375639) printf("y:   %f\n", polar[i][_PCY_]);

        mean_window[_LONGTERM_][_X_] += polar[i][_PCX_];
        mean_window[_LONGTERM_][_Y_] += polar[i][_PCY_];

      }

      if (!valid) continue;

if (p == 375639) printf("valid pixel.\n");

      // mean of polar coordinates
      mean_window[_LONGTERM_][_X_] /= ni;
      mean_window[_LONGTERM_][_Y_] /= ni;
if (p == 375639) printf("mean pol x/y: %f %f\n", mean_window[_LONGTERM_][_X_], mean_window[_LONGTERM_][_Y_]);

      // multi-annual average vector
      polar_vector(mean_window[_LONGTERM_][_X_], mean_window[_LONGTERM_][_Y_], 0, 0, vector[_LONGTERM_]);

      // diametric opposite of average vector = start of phenological year
      if (vector[_LONGTERM_][_RAD_] < M_PI){
        theta = vector[_LONGTERM_][_RAD_] + M_PI;
      } else {
        theta = vector[_LONGTERM_][_RAD_] - M_PI;
      }
      doy_theta = (theta*365.0/(2.0*M_PI));
      
if (p == 375639) printf("avg:   %f %f %f\n", vector[_LONGTERM_][_RAD_], vector[_LONGTERM_][_DOY_], vector[_LONGTERM_][_VAL_]);
if (p == 375639) printf("theta: %f %f\n", theta, doy_theta);


      identify_seasons(polar, ni, tsi->step, doy_theta);

      accumulate_seasons(polar, ni);


      for (s=0, i0=0; s<pol->ns; s++){

        memset(timing,      0, sizeof(float)*_EVENT_LEN_*_COORD_LEN_);
        memset(mean_window, 0, sizeof(float)*_WINDOW_LEN_*2);
        memset(n_window,    0, sizeof(float)*_WINDOW_LEN_);
        memset(recurrence,  0, sizeof(double)*2);

        if (doy_theta < 182) y = s; else y = s+1;
        ce_theta = doy2ce(doy_theta, s);


        for (i=i0; i<ni; i++){
if (p == 375639) printf("season: %2.0f, rad: %.2f, val: %7.2f, year: %2.0f, ce: %6.0f, doy: %3.0f, x: %7.2f, y: %7.2f, cum: %7.2f\n", 
polar[i][_SEASON_], polar[i][_RAD_], polar[i][_VAL_], polar[i][_YEAR_], polar[i][_CE_], polar[i][_DOY_], polar[i][_PCX_], polar[i][_PCY_], polar[i][_CUM_]);
          
          if (polar[i][_SEASON_] < s) continue;
          if (polar[i][_SEASON_] > s){ i0 = i; break; }

          // start of growing season
          if (polar[i][_CUM_] >= pol->start && timing[_START_][_CUM_] == 0){
            memcpy(timing[_START_], polar[i], sizeof(float)*_COORD_LEN_);}

          // mid of growing season
          if (polar[i][_CUM_] >= pol->mid   && timing[_MID_][_CUM_]   == 0){
            memcpy(timing[_MID_],   polar[i], sizeof(float)*_COORD_LEN_);}

          // end of growing season
          if (polar[i][_CUM_] >= pol->end   && timing[_END_][_CUM_]   == 0){
            memcpy(timing[_END_],   polar[i], sizeof(float)*_COORD_LEN_);}

          // mean, sd of val + average vector of growing season
          if (polar[i][_CUM_] >= pol->start && 
              polar[i][_CUM_] <  pol->end){
            var_recurrence(polar[i][_VAL_], &recurrence[0], &recurrence[1], ++n_window[_GROW_]);
            mean_window[_GROW_][_X_] += polar[i][_PCX_];
            mean_window[_GROW_][_Y_] += polar[i][_PCY_];
          }

          // average vector of early growing season part
          if (polar[i][_CUM_] >= pol->start && 
              polar[i][_CUM_] <  pol->mid){
            mean_window[_EARLY_][_X_] += polar[i][_PCX_];
            mean_window[_EARLY_][_Y_] += polar[i][_PCY_];
            n_window[_EARLY_]++;
          }
          
          // average vector of late growing season part
          if (polar[i][_CUM_] >= pol->mid && 
              polar[i][_CUM_] <  pol->end){
            mean_window[_LATE_][_X_] += polar[i][_PCX_];
            mean_window[_LATE_][_Y_] += polar[i][_PCY_];
            n_window[_LATE_]++;
          }

          mean_window[_THISYEAR_][_X_] += polar[i][_PCX_];
          mean_window[_THISYEAR_][_Y_] += polar[i][_PCY_];
          n_window[_THISYEAR_]++;

        }
        
        

        mean_window[_THISYEAR_][_X_]  /= n_window[_THISYEAR_];
        mean_window[_THISYEAR_][_Y_]  /= n_window[_THISYEAR_];
        polar_vector(mean_window[_THISYEAR_][_X_],  mean_window[_THISYEAR_][_Y_], s, doy_theta, vector[_THISYEAR_]);

        if (vector[_THISYEAR_][_RAD_] < M_PI){
          theta_now = vector[_THISYEAR_][_RAD_] + M_PI;
        } else {
          theta_now = vector[_THISYEAR_][_RAD_] - M_PI;
        }
        doy_theta_now = (theta_now*365.0/(2.0*M_PI));
        
        if (doy_theta_now > doy_theta){
          ce_theta_now = doy2ce(doy_theta_now, s);
        } else {
          ce_theta_now = doy2ce(doy_theta_now, s+1);
        }



        mean_window[_GROW_][_X_]  /= n_window[_GROW_];
        mean_window[_GROW_][_Y_]  /= n_window[_GROW_];
        mean_window[_EARLY_][_X_] /= n_window[_EARLY_];
        mean_window[_EARLY_][_Y_] /= n_window[_EARLY_];
        mean_window[_LATE_][_X_]  /= n_window[_LATE_];
        mean_window[_LATE_][_Y_]  /= n_window[_LATE_];

        polar_vector(mean_window[_GROW_][_X_],  mean_window[_GROW_][_Y_],  s, doy_theta, vector[_GROW_]);
        polar_vector(mean_window[_EARLY_][_X_], mean_window[_EARLY_][_Y_], s, doy_theta, vector[_EARLY_]);
        polar_vector(mean_window[_LATE_][_X_],  mean_window[_LATE_][_Y_],  s, doy_theta, vector[_LATE_]);



        // sanity check?
        //valid = false;
        // if () valid = true;
        //valid = true;



if (p == 375639) printf("season: %d, year %d\n", s, y);
if (p == 375639) printf("mean, sd, and n: %f, %f, %d\n", recurrence[0], standdev(recurrence[1], n_window[_GROW_]), n_window[_GROW_]);



        /** copy POL if all OK **/
        //if (valid){
          //if (pol->odem) ts->pol_[_POL_DEM_][y][p] = (short)0;
          if (pol->odss) ts->pol_[_POL_DSS_][y][p] = (short)timing[_START_][_CE_];
          if (pol->odms) ts->pol_[_POL_DMS_][y][p] = (short)timing[_MID_][_CE_];
          if (pol->odes) ts->pol_[_POL_DES_][y][p] = (short)timing[_END_][_CE_];
          if (pol->odev) ts->pol_[_POL_DEV_][y][p] = (short)vector[_EARLY_][_CE_];
          if (pol->odav) ts->pol_[_POL_DAV_][y][p] = (short)vector[_GROW_][_CE_];
          if (pol->odlv) ts->pol_[_POL_DLV_][y][p] = (short)vector[_LATE_][_CE_];
          //if (pol->odlm) ts->pol_[_POL_DLM_][y][p] = (short)0;
          if (pol->olgs) ts->pol_[_POL_LGS_][y][p] = (short)(timing[_END_][_CE_] - timing[_START_][_CE_]);
          if (pol->olbv) ts->pol_[_POL_LBV_][y][p] = (short)(vector[_LATE_][_CE_] - vector[_EARLY_][_CE_]);
          //if (pol->ovem) ts->pol_[_POL_VEM_][y][p] = (short)0;
          if (pol->ovss) ts->pol_[_POL_VSS_][y][p] = (short)timing[_START_][_VAL_];
          if (pol->ovms) ts->pol_[_POL_VMS_][y][p] = (short)timing[_MID_][_VAL_];
          if (pol->oves) ts->pol_[_POL_VES_][y][p] = (short)timing[_END_][_VAL_];
          if (pol->ovev) ts->pol_[_POL_VEV_][y][p] = (short)vector[_EARLY_][_VAL_];
          if (pol->ovav) ts->pol_[_POL_VAV_][y][p] = (short)vector[_GROW_][_VAL_];
          if (pol->ovlv) ts->pol_[_POL_VLV_][y][p] = (short)vector[_LATE_][_VAL_];
          //if (pol->ovlm) ts->pol_[_POL_VLM_][y][p] = (short)0;
          //if (pol->ovbl) ts->pol_[_POL_VBL_][y][p] = (short)0;
          if (pol->ovga) ts->pol_[_POL_VGA_][y][p] = (short)recurrence[0];
          if (pol->ovgv) ts->pol_[_POL_VGV_][y][p] = (short)standdev(recurrence[1], n_window[_GROW_]);
          //if (pol->odpy) ts->pol_[_POL_DPY_][y][p] = (short)(ce_theta_now - ce_theta);
          if (pol->odpy) ts->pol_[_POL_DPY_][y][p] = (short)(ce_theta);
          //if (pol->oist) ts->pol_[_POL_IST_][y][p] = (short)0;
          //if (pol->oibl) ts->pol_[_POL_IBL_][y][p] = (short)0;
          //if (pol->oibt) ts->pol_[_POL_IBT_][y][p] = (short)0;
          //if (pol->oigs) ts->pol_[_POL_IGS_][y][p] = (short)0;
          //if (pol->orar) ts->pol_[_POL_RAR_][y][p] = (short)0;
          //if (pol->oraf) ts->pol_[_POL_RAF_][y][p] = (short)0;
          //if (pol->ormr) ts->pol_[_POL_RMR_][y][p] = (short)0;
          //if (pol->ormf) ts->pol_[_POL_RMF_][y][p] = (short)0;
        //}
        
      }


    }

    free_2D((void**)polar, ni);

  }


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


  if (phl->tsa.pol.opct +
      phl->tsa.pol.opol +
      phl->tsa.pol.otrd +
      phl->tsa.pol.ocat == 0) return SUCCESS;

  cite_me(_CITE_POL_);


  if (polar_ts(ts, mask_, nc, ni, nodata, 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, &phl->tsa.tsi, &phl->tsa.pol) == FAILURE) return FAILURE;



  return SUCCESS;
}

