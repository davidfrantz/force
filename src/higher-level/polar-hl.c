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

// should rather be a struct..
//enum { _RAD_, _VAL_, _CUM_, _YEAR_, _DOY_, _CE_, _SEASON_, _PCX_, _PCY_, _COORD_LEN_ };
typedef struct {
  float rad;
  float val;
  float cum;
  float pcx;
  float pcy;
  int   doy;
  int   year;
  int   season;
  int   ce;
} polar_t;

void print_polar(polar_t *polar);
void polar_coords(float r, float v, float yr, polar_t *polar);
void polar_vector(float x, float y, polar_t *polar);
void ce_from_polar_vector(float yr, polar_t *theta, polar_t *polar);
void identify_regular_seasons(polar_t *polar, int ni, int istep, polar_t *theta);
polar_t *identify_variable_seasons(polar_t *polar, int ni, int istep, par_pol_t *pol, polar_t *theta, bool print);
void accumulate_seasons(polar_t *polar, int ni);

int polar_ts(tsa_t *ts, small *mask_, int nc, int ni, short nodata, int year_min, int year_max, par_tsi_t *tsi, par_pol_t *pol);


void print_polar(polar_t *polar){
  
  
  printf("polar coordinate:\n");
  printf("  season: %d, year: %d, ce: %d, doy: %d\n",
    polar->season, polar->year, polar->ce, polar->doy);
  printf("  rad: %.2f, val: %7.2f, , x: %7.2f, y: %7.2f, cum: %7.2f\n", 
    polar->rad, polar->val, polar->pcx, polar->pcy, polar->cum);

  return;
}

void polar_coords(float r, float v, float yr, polar_t *polar){
float doy;


  doy = r*365.0/(2.0*M_PI);

  polar->rad   = r;
  polar->val   = v;
  polar->year = yr;
  polar->doy  = doy;
  polar->ce    = doy2ce(doy, yr);
  polar->pcx   = v*cos(r);
  polar->pcy   = v*sin(r);

  return;
}


void polar_vector(float x, float y, polar_t *polar){
float r, v, doy;


  r = atan2(y, x);
  if (r <= 0) r += 2*M_PI;
  v = sqrt(x*x + y*y);
  
  doy = r*365.0/(2.0*M_PI);

  polar->rad  = r;
  polar->val  = v;
  polar->doy  = doy;


  polar->pcx  = x;
  polar->pcy  = y;

  return;
}


void ce_from_polar_vector(float yr, polar_t *theta, polar_t *polar){


  polar->year = yr;

  if (polar->doy > theta->doy){
    polar->ce = doy2ce(polar->doy, yr);
  } else {
    polar->ce = doy2ce(polar->doy, yr+1);
  }

  return;
}


void identify_regular_seasons(polar_t *polar, int ni, int istep, polar_t *theta){
int i, s = -1, y = 0;
//float rstep = istep/365.0*2.0*M_PI;


  for (i=0; i<ni; i++){

    theta->ce = doy2ce(theta->doy, y);

    if (polar[i].ce >= theta->ce && 
        polar[i].ce-theta->ce <= istep){ s++; y++;}

    polar[i].season = s;

  }

  return;
}


polar_t *identify_variable_seasons(polar_t *polar, int ni, int istep, par_pol_t *pol, polar_t *theta, bool print){
int s, i, i0, ii, i1;
float mean_pct[2], n_pct;
polar_t *alpha0 = NULL; // mean vector in pre-structured phenological year
polar_t *theta0 = NULL; // diametric opposite of alpha0 = fine-tuned start of phenological year
float opposite;
int *diff_season = NULL;
int ce_shift, i_shift, d_shift, d_seas;


  alloc((void**)&theta0, pol->ns, sizeof(polar_t));
  if (!pol->adaptive) return theta0;

  alloc((void**)&alpha0, pol->ns, sizeof(polar_t));


  // fine-tune the start of phenological year per season
  for (s=0, i0=0; s<pol->ns; s++){

    memset(mean_pct, 0, sizeof(float)*2);
    n_pct = 0;

    for (i=i0; i<ni; i++){

      if (polar[i].season < s) continue;
      if (polar[i].season > s){ i0 = i; break; }

      mean_pct[_X_] += polar[i].pcx;
      mean_pct[_Y_] += polar[i].pcy;
      n_pct++;

    }

    mean_pct[_X_]  /= n_pct;
    mean_pct[_Y_]  /= n_pct;
    polar_vector(mean_pct[_X_],  mean_pct[_Y_], &alpha0[s]);

    if (alpha0[s].rad < M_PI){
      theta0[s].rad = alpha0[s].rad + M_PI;
    } else {
      theta0[s].rad = alpha0[s].rad - M_PI;
    }
    opposite = alpha0[s].rad - M_PI; // opposite with sign

    theta0[s].doy = (theta0[s].rad*365.0/(2.0*M_PI));

    if (opposite < 0 && alpha0[s].rad - theta->rad >= 0){
      theta0[s].year = s-1;
      theta0[s].ce   = doy2ce(theta0[s].doy, s-1);
    } else if (opposite > 0 && alpha0[s].rad - theta->rad < 0){
      theta0[s].year = s+1;
      theta0[s].ce   = doy2ce(theta0[s].doy, s+1);
    } else {
      theta0[s].year = s;
      theta0[s].ce   = doy2ce(theta0[s].doy, s);
    }
if (print) printf("season %d. alpha: %f %d. updated theta: %f %d %d\n", s, alpha0[s].rad, alpha0[s].doy, theta0[s].rad, theta0[s].doy, theta0[s].ce);
  }


  alloc((void**)&diff_season, ni, sizeof(int));

  s = polar[0].season;

  for (i=1; i<ni; i++){

    if (polar[i].season != s){

      s = polar[i].season;

      if (s >= pol->ns) break;

      theta->ce = doy2ce(theta->doy, s);

      ce_shift  = theta0[s].ce - theta->ce;
      i_shift = floor(abs(ce_shift)/(float)istep);
      d_shift = (ce_shift > 0) ?  1 : -1;
      d_seas  = (ce_shift > 0) ? -1 :  1;
      
      if (print) printf("season %d: shift is %d days, %d positions in %d direction. Adding %d to season\n",
        s, ce_shift, i_shift, d_shift, d_seas);

      if (ce_shift > 0){

        for (ii=0; ii<i_shift; ii++){
          i1 = i+ii;
          if (i1 >= ni) break;
          diff_season[i1] = -1;
        }

      } else {

        for (ii=0; ii<i_shift; ii++){
          i1 = i-ii-1;
          if (i1 < 0) break;
          diff_season[i1] = 1;
        }

      }
      
      
      //while (i_shift > 0){
      //  ii = i+i_shift*d_shift;
      //  if (ii < 0 || ii >= ni) break;
      //  diff_season[ii] = d_seas;
      //  i_shift--;
      //}

    }

  }


  // eventually update season
  for (i=0; i<ni; i++) polar[i].season += diff_season[i];
  
  free((void*)diff_season);
  free((void*)alpha0);


  return theta0;
}


void accumulate_seasons(polar_t *polar, int ni){
int i, s;
float sum;


  polar[0].cum = polar[0].val;
  s = polar[0].season;

  for (i=1; i<ni; i++){

    if (polar[i].season != s){
      polar[i].cum = polar[i].val;
      s = polar[i].season;
    } else {
      polar[i].cum = polar[i-1].cum + polar[i].val;
    }

  }

  sum = polar[ni-1].cum;
  s   = polar[ni-1].season;

  // absolute sum to percent
  for (i=(ni-1); i>=0; i--){

    if (polar[i].season != s){
      sum = polar[i].cum;
      s   = polar[i].season;
    }

    polar[i].cum /= sum;

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

enum { _LEFT_, _START_, _MID_, _SIJSIJNSI_, _END_, _RIGHT_, _EVENT_LEN_ };
enum { _ALPHA_, _THETA_, _EARLY_, _GROW_, _LATE_, _WINDOW_LEN_ };

polar_t timing[_EVENT_LEN_];
polar_t vector[_WINDOW_LEN_];
float mean_window[_WINDOW_LEN_][2];
int   n_window[_WINDOW_LEN_];
double recurrence[2];

polar_t *polar = NULL;
polar_t *theta0 = NULL;


  valid = false;
  for (l=0; l<_POL_LENGTH_; l++){
    if (ts->pol_[l] != NULL) valid = true;
  }

  if (!valid) return CANCEL;




  //#pragma omp parallel private(l,i,i0,ni_,ce_left,ce_right,v_left,v_right,year,valid,ce,v,doy) firstprivate(southern) shared(mask_,ts,nc,ni,year_min,year_max,nodata,pol,nseg) default(none)
  {

    // allocate
    alloc((void**)&polar, ni, sizeof(polar_t));


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
      memset(mean_window[_ALPHA_], 0, 2*sizeof(float));


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

        if (p == 404173) printf("doy: %d\n", ts->d_tsi[i].doy);
        if (p == 404173) printf("r:   %f\n", r);
        if (p == 404173) printf("v:   %f\n", v);
        if (v < 0) v = 0;
        polar_coords(r, v, ts->d_tsi[i].year-year_min, &polar[i]);
        if (p == 404173) printf("x:   %f\n", polar[i].pcx);
        if (p == 404173) printf("y:   %f\n", polar[i].pcy);
        
        if (pol->opct) ts->pcx_[i][p] = (short)polar[i].pcx;
        if (pol->opct) ts->pcy_[i][p] = (short)polar[i].pcy;

        mean_window[_ALPHA_][_X_] += polar[i].pcx;
        mean_window[_ALPHA_][_Y_] += polar[i].pcy;

      }

      if (!valid) continue;

      if (p == 404173) printf("valid pixel.\n");

      // mean of polar coordinates
      mean_window[_ALPHA_][_X_] /= ni;
      mean_window[_ALPHA_][_Y_] /= ni;
      if (p == 404173) printf("mean pol x/y: %f %f\n", mean_window[_ALPHA_][_X_], mean_window[_ALPHA_][_Y_]);

      // multi-annual average vector
      polar_vector(mean_window[_ALPHA_][_X_], mean_window[_ALPHA_][_Y_], &vector[_ALPHA_]);

      // diametric opposite of average vector = start of phenological year
      if (vector[_ALPHA_].rad < M_PI){
        vector[_THETA_].rad = vector[_ALPHA_].rad + M_PI;
      } else {
        vector[_THETA_].rad = vector[_ALPHA_].rad - M_PI;
      }
      vector[_THETA_].doy = (vector[_THETA_].rad*365.0/(2.0*M_PI));

      if (p == 404173) printf("avg:   %f %d %f\n", vector[_ALPHA_].rad, vector[_ALPHA_].doy, vector[_ALPHA_].val);
      if (p == 404173) printf("theta: %f %d %d\n", vector[_THETA_].rad, vector[_THETA_].doy, vector[_THETA_].ce);


      identify_regular_seasons(polar, ni, tsi->step, &vector[_THETA_]);

      if (p == 404173){
        theta0 = identify_variable_seasons(polar, ni, tsi->step, pol, &vector[_THETA_], true);
      } else {
        theta0 = identify_variable_seasons(polar, ni, tsi->step, pol, &vector[_THETA_], false);
      }

      accumulate_seasons(polar, ni);

        
      if (p == 404173) for (i=0; i<ni; i++) print_polar(&polar[i]);

      for (s=0, i0=0; s<pol->ns; s++){

        memset(&timing,     0, sizeof(polar_t)*_EVENT_LEN_);
        memset(mean_window, 0, sizeof(float)*_WINDOW_LEN_*2);
        memset(n_window,    0, sizeof(float)*_WINDOW_LEN_);
        memset(recurrence,  0, sizeof(double)*2);

        if (vector[_THETA_].doy < 182) y = s; else y = s+1;
        vector[_THETA_].ce = doy2ce(vector[_THETA_].doy, s);


        for (i=i0; i<ni; i++){
          
          if (polar[i].season < s) continue;
          if (polar[i].season > s){ i0 = i; break; }

          // start of phenological year
          if (polar[i].cum > 0 && timing[_LEFT_].cum == 0){
            memcpy(&timing[_LEFT_], &polar[i], sizeof(polar_t));}

          // end of phenological year
          if (polar[i].cum == 1){
            if (i+1 < ni){
              memcpy(&timing[_RIGHT_], &polar[i+1], sizeof(polar_t));
            } else {
              memcpy(&timing[_RIGHT_], &polar[i],   sizeof(polar_t));
            }
          }

          // start of growing season
          if (polar[i].cum >= pol->start && timing[_START_].cum == 0){
            memcpy(&timing[_START_], &polar[i], sizeof(polar_t));}

          // mid of growing season
          if (polar[i].cum >= pol->mid   && timing[_MID_].cum   == 0){
            memcpy(&timing[_MID_],   &polar[i], sizeof(polar_t));}

          // end of growing season
          if (polar[i].cum >= pol->end   && timing[_END_].cum   == 0){
            memcpy(&timing[_END_],   &polar[i], sizeof(polar_t));}

          // mean, sd of val + average vector of growing season
          if (polar[i].cum >= pol->start && 
              polar[i].cum <  pol->end){
            var_recurrence(polar[i].val, &recurrence[0], &recurrence[1], ++n_window[_GROW_]);
            mean_window[_GROW_][_X_] += polar[i].pcx;
            mean_window[_GROW_][_Y_] += polar[i].pcy;
          }

          // max of season
          if (polar[i].val > timing[_SIJSIJNSI_].val){
            memcpy(&timing[_SIJSIJNSI_],   &polar[i], sizeof(polar_t));}

          // average vector of early growing season part
          if (polar[i].cum >= pol->start && 
              polar[i].cum <  pol->mid){
            mean_window[_EARLY_][_X_] += polar[i].pcx;
            mean_window[_EARLY_][_Y_] += polar[i].pcy;
            n_window[_EARLY_]++;
          }
          
          // average vector of late growing season part
          if (polar[i].cum >= pol->mid && 
              polar[i].cum <  pol->end){
            mean_window[_LATE_][_X_] += polar[i].pcx;
            mean_window[_LATE_][_Y_] += polar[i].pcy;
            n_window[_LATE_]++;
          }

        }


        mean_window[_GROW_][_X_]  /= n_window[_GROW_];
        mean_window[_GROW_][_Y_]  /= n_window[_GROW_];
        mean_window[_EARLY_][_X_] /= n_window[_EARLY_];
        mean_window[_EARLY_][_Y_] /= n_window[_EARLY_];
        mean_window[_LATE_][_X_]  /= n_window[_LATE_];
        mean_window[_LATE_][_Y_]  /= n_window[_LATE_];

        polar_vector(mean_window[_GROW_][_X_],  mean_window[_GROW_][_Y_], &vector[_GROW_]);
        polar_vector(mean_window[_EARLY_][_X_], mean_window[_EARLY_][_Y_],&vector[_EARLY_]);
        polar_vector(mean_window[_LATE_][_X_],  mean_window[_LATE_][_Y_], &vector[_LATE_]);

        ce_from_polar_vector(s, &vector[_THETA_], &vector[_GROW_]);
        ce_from_polar_vector(s, &vector[_THETA_], &vector[_EARLY_]);
        ce_from_polar_vector(s, &vector[_THETA_], &vector[_LATE_]);



        // sanity check?
        //valid = false;
        // if () valid = true;
        //valid = true;



        if (p == 404173) printf("season: %d, year %d\n", s, y);
        if (p == 404173) printf("mean, sd, and n: %f, %f, %d\n", recurrence[0], standdev(recurrence[1], n_window[_GROW_]), n_window[_GROW_]);

        // date parameters
        if (pol->use[_POL_DEM_]) ts->pol_[_POL_DEM_][y][p] = (short)timing[_LEFT_].ce;
        if (pol->use[_POL_DSS_]) ts->pol_[_POL_DSS_][y][p] = (short)timing[_START_].ce;
        if (pol->use[_POL_DMS_]) ts->pol_[_POL_DMS_][y][p] = (short)timing[_MID_].ce;
        if (pol->use[_POL_DPS_]) ts->pol_[_POL_DPS_][y][p] = (short)timing[_SIJSIJNSI_].ce;
        if (pol->use[_POL_DES_]) ts->pol_[_POL_DES_][y][p] = (short)timing[_END_].ce;
        if (pol->use[_POL_DLM_]) ts->pol_[_POL_DLM_][y][p] = (short)timing[_RIGHT_].ce;
        if (pol->use[_POL_DEV_]) ts->pol_[_POL_DEV_][y][p] = (short)vector[_EARLY_].ce;
        if (pol->use[_POL_DAV_]) ts->pol_[_POL_DAV_][y][p] = (short)vector[_GROW_].ce;
        if (pol->use[_POL_DLV_]) ts->pol_[_POL_DLV_][y][p] = (short)vector[_LATE_].ce;
        if (pol->use[_POL_DPY_]) ts->pol_[_POL_DPY_][y][p] = (short)(vector[_THETA_].ce);
        if (pol->use[_POL_DPV_]) ts->pol_[_POL_DPV_][y][p] = (short)(theta0[s].ce - vector[_THETA_].ce);

        // length paramaters
        if (pol->use[_POL_LGS_]) ts->pol_[_POL_LGS_][y][p] = (short)(timing[_END_].ce   - timing[_START_].ce);
        if (pol->use[_POL_LGV_]) ts->pol_[_POL_LGV_][y][p] = (short)(vector[_LATE_].ce  - vector[_EARLY_].ce);
        if (pol->use[_POL_LTS_]) ts->pol_[_POL_LTS_][y][p] = (short)(timing[_RIGHT_].ce - timing[_LEFT_].ce);

        // value parameters
        if (pol->use[_POL_VEM_]) ts->pol_[_POL_VEM_][y][p] = (short)timing[_LEFT_].val;
        if (pol->use[_POL_VSS_]) ts->pol_[_POL_VSS_][y][p] = (short)timing[_START_].val;
        if (pol->use[_POL_VMS_]) ts->pol_[_POL_VMS_][y][p] = (short)timing[_MID_].val;
        if (pol->use[_POL_VPS_]) ts->pol_[_POL_VPS_][y][p] = (short)timing[_SIJSIJNSI_].val;
        if (pol->use[_POL_VLM_]) ts->pol_[_POL_VLM_][y][p] = (short)timing[_RIGHT_].val;
        if (pol->use[_POL_VES_]) ts->pol_[_POL_VES_][y][p] = (short)timing[_END_].val;
        if (pol->use[_POL_VEV_]) ts->pol_[_POL_VEV_][y][p] = (short)vector[_EARLY_].val;
        if (pol->use[_POL_VAV_]) ts->pol_[_POL_VAV_][y][p] = (short)vector[_GROW_].val;
        if (pol->use[_POL_VLV_]) ts->pol_[_POL_VLV_][y][p] = (short)vector[_LATE_].val;
        if (pol->use[_POL_VSA_]) ts->pol_[_POL_VSA_][y][p] = (short)(timing[_SIJSIJNSI_].val - 
          (timing[_START_].val+timing[_END_].val)/2.0);
        if (pol->use[_POL_VPA_]) ts->pol_[_POL_VPA_][y][p] = (short)(timing[_SIJSIJNSI_].val - timing[_MID_].val);
        if (pol->use[_POL_VBL_]) ts->pol_[_POL_VBL_][y][p] = (short)((timing[_LEFT_].val+timing[_RIGHT_].val)/2.0);
        if (pol->use[_POL_VGA_]) ts->pol_[_POL_VGA_][y][p] = (short)recurrence[0];
        if (pol->use[_POL_VGV_]) ts->pol_[_POL_VGV_][y][p] = (short)standdev(recurrence[1], n_window[_GROW_]);


      }

      if (theta0 != NULL) free((void*)theta0); theta0 = NULL;

    }

    free((void*)polar);

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

