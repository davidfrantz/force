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
This file contains functions for CSO procesing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "cso-hl.h"


brick_t **compile_cso(ard_t *ard, cso_t *cs, par_hl_t *phl, cube_t *cube, int nt, int nw, int *nproduct);
brick_t *compile_cso_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the bricks, in which CSO results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- cs:       pointer to instantly useable CSO image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nt:       number of ARD products over time
--- nw:       number of windows for CSO
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for CSO results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_cso(ard_t *ard, cso_t *cs, par_hl_t *phl, cube_t *cube, int nt, int nw, int *nproduct){
brick_t **CSO = NULL;
short nodata = SHRT_MIN;
int w, q;
int year, month;
date_t date;
char fdate[NPOW_10];
int o, nprod = phl->cso.sta.nmetrics;
int error = 0;

char prodname[NPOW_08][NPOW_03];
int nchar;
short ***ptr[NPOW_08];


  if (phl->cso.sta.num > -1) copy_string(prodname[phl->cso.sta.num], NPOW_03, "NUM");
  if (phl->cso.sta.min > -1) copy_string(prodname[phl->cso.sta.min], NPOW_03, "MIN");
  if (phl->cso.sta.max > -1) copy_string(prodname[phl->cso.sta.max], NPOW_03, "MAX");
  if (phl->cso.sta.rng > -1) copy_string(prodname[phl->cso.sta.rng], NPOW_03, "RNG");
  if (phl->cso.sta.iqr > -1) copy_string(prodname[phl->cso.sta.iqr], NPOW_03, "IQR");
  if (phl->cso.sta.avg > -1) copy_string(prodname[phl->cso.sta.avg], NPOW_03, "AVG");
  if (phl->cso.sta.std > -1) copy_string(prodname[phl->cso.sta.std], NPOW_03, "STD");
  if (phl->cso.sta.skw > -1) copy_string(prodname[phl->cso.sta.skw], NPOW_03, "SKW");
  if (phl->cso.sta.krt > -1) copy_string(prodname[phl->cso.sta.krt], NPOW_03, "KRT");
  for (q=0; q<phl->cso.sta.nquantiles; q++){
    nchar = snprintf(prodname[phl->cso.sta.qxx[q]], NPOW_03, "Q%02.0f", phl->cso.sta.q[q]*100);
    if (nchar < 0 || nchar >= NPOW_03){
      printf("Buffer Overflow in assembling prodname\n"); return NULL;}
  }


  for (o=0; o<nprod; o++) ptr[o] = &cs->cso_[o];


  alloc((void**)&CSO, nprod, sizeof(brick_t*));

  // alloc dates, use one additional date to have right boundary
  if (nw > 0) alloc((void**)&cs->d_cso, nw+1, sizeof(date_t)); else cs->d_cso = NULL;



  for (o=0; o<nprod; o++){

    if ((CSO[o] = compile_cso_brick(ard[0].QAI, nw, true, prodname[o], phl)) == NULL || (  *ptr[o] = get_bands_short(CSO[o])) == NULL){
      printf("Error compiling %s product. ", prodname[o]); error++;
    } else {
      init_date(&date);
      year = phl->date_range[_MIN_].year;
      month = phl->date_range[_MIN_].month;

      for (w=0; w<=nw; w++){
        if (month > 12){ year++; month -= 12;}
        set_date(&date, year, month, 1);
        copy_date(&date, &cs->d_cso[w]);
        compact_date(date.year, date.month, date.day, fdate, NPOW_10);
//printf("W: "); print_date(&date);
        if (w < nw){
          set_brick_nodata(CSO[o], w, nodata);
          set_brick_wavelength(CSO[o], w, w+1);
          set_brick_date(CSO[o], w, date);
          set_brick_domain(CSO[o], w, fdate);
          set_brick_bandname(CSO[o], w, fdate);
        }
        month += phl->cso.step;
      }

    }
  }


  if (error > 0){
    printf("%d compiling CSO product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(CSO[o]);
    free((void*)CSO);
    if (cs->d_cso != NULL){ free((void*)cs->d_cso); cs->d_cso = NULL;}
    return NULL;
  }

  *nproduct = nprod;
  return CSO;
}


/** This function compiles a CSO brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for CSO result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_cso_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;


  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Clear-Sky Observations");
  set_brick_product(brick, prodname);
  
  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);

  nchar = snprintf(fname, NPOW_10, "%04d-%04d_%03d-%03d-%02d_HL_CSO_%s_%s", 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, 
    phl->doy_range[_MIN_], phl->doy_range[_MAX_], 
    phl->cso.step, phl->sen.target, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);
  

  if (write){
    set_brick_open(brick, OPEN_BLOCK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, phl->format);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
    set_brick_date(brick, b, date);
  }

  return brick;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the clear-sky observations module
--- ard:       ARD
--- mask:      mask image
--- nt:        number of ARD products over time
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with CSO results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **clear_sky_observations(ard_t *ard, brick_t *mask, int nt, par_hl_t *phl, cube_t *cube, int *nproduct){
cso_t cs;
brick_t **CSO;
small *mask_ = NULL;
int o, w, k, n, q, p, nprod = 0;
int t, t_left;
int d_ce, ce, ce_left;
int *t0 = NULL, *t1 = NULL;
int month, year;
int nc;
int nw;
short nodata = SHRT_MIN;
short minimum, maximum;
short q25_, q75_;
double mean, var;
double skew, kurt;
double skewscaled, kurtscaled;
float *q_array = NULL;
bool alloc_q_array = false;


  cite_me(_CITE_CSO_);


  // import bricks
  nc = get_brick_chunkncells(ard[0].QAI);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return NULL;}
  }


  // number of steps
  nw = 0;
  year  = phl->date_range[_MIN_].year;
  month = phl->date_range[_MIN_].month;
  
  while (year < phl->date_range[_MAX_].year ||
        (year <= phl->date_range[_MAX_].year && month < phl->date_range[_MAX_].month)){
    nw++;
    if ((month += phl->cso.step) > 12){ year++; month -= 12;}
  }
 
  //printf("nw: %d\n", nw);


  // compile products + bricks
  if ((CSO = compile_cso(ard, &cs, phl, cube, nt, nw, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile CSO products!\n"); 
    free((void*)CSO);
    *nproduct = 0;
    return NULL;
  }

  
  // first and last t per window

  alloc((void**)&t0, nw, sizeof(int));
  alloc((void**)&t1, nw, sizeof(int));

  for (w=0, t_left=0; w<nw; w++){

    t0[w] = t1[w] = -1;

    for (t=t_left; t<nt; t++){

      ce = get_brick_ce(ard[t].QAI, 0);

      if (ce >= cs.d_cso[w].ce && ce < cs.d_cso[w+1].ce){
        if (t0[w] < 0) t0[w] = t;
        t1[w] = t;
      } else if (ce >= cs.d_cso[w+1].ce){
        break;
      }

    }

    if (t1[w] >= 0) t_left = t1[w];
    
    //printf("t0: %d, t1: %d\n", t0[w], t1[w]);
    
  }




  
  if (phl->cso.sta.quantiles || phl->cso.sta.iqr > -1) alloc_q_array = true;
  

  #pragma omp parallel private(o,t,w,minimum,maximum,q,q_array,mean,var,skew,kurt,n,k,skewscaled,kurtscaled,q25_,q75_,d_ce,ce,ce_left) shared(mask_,cs,nc,nw,nt,nodata,alloc_q_array,phl,t0,t1,nprod,ard) default(none)
  {

    if (alloc_q_array) alloc((void**)&q_array, nt+1, sizeof(float));

    #pragma omp for
    for (p=0; p<nc; p++){

      if (mask_ != NULL && !mask_[p]){
        for (o=0; o<nprod; o++){
          for (w=0; w<nw; w++) cs.cso_[o][w][p] = nodata;
        }
        continue;
      }


      for (w=0; w<nw; w++){

        mean = var = skew = kurt = n = k = 0;
        skewscaled = kurtscaled = 0;
        minimum = SHRT_MAX; maximum = SHRT_MIN;
        q25_ = q75_ = SHRT_MIN;

        if (t0[w] > -1){

          ce_left = cs.d_cso[w].ce;

          for (t=t0[w]; t<=t1[w]; t++){

            // check nodata
            if (!ard[t].msk[p]) continue;
            
            ce = get_brick_ce(ard[t].QAI, 0);

            // if current date is larger than previous date (incl. left window boundary),
            // include dt in stats
            if (ce > ce_left){
              d_ce = ce - ce_left;
              if (alloc_q_array) q_array[k] = (float)d_ce;
              kurt_recurrence(d_ce, &mean, &var, &skew, &kurt, ++k);
              if (d_ce < minimum) minimum = d_ce;
              if (d_ce > maximum) maximum = d_ce;
              ce_left = ce;
            }

            n++;

          }

          // if current date is smaller than right window boundary,
          // include dt of right boundary in stats
          if ((d_ce = cs.d_cso[w+1].ce - ce_left) > 0){
            if (alloc_q_array) q_array[k] = (float)d_ce;
            kurt_recurrence(d_ce, &mean, &var, &skew, &kurt, ++k);
            if (d_ce < minimum) minimum = d_ce;
            if (d_ce > maximum) maximum = d_ce;
          }

        } else {

          minimum = maximum = mean = cs.d_cso[w+1].ce-cs.d_cso[w].ce;

        }


        if (n > 0){
          skewscaled = skewness(var, skew, n)*1000;
          kurtscaled = (kurtosis(var, kurt, n)-3)*1000;
          if (skewscaled < -30000) skewscaled = -30000;
          if (skewscaled >  30000) skewscaled =  30000;
          if (kurtscaled < -30000) kurtscaled = -30000;
          if (kurtscaled >  30000) kurtscaled =  30000;
        }
        if (phl->cso.sta.num > -1) cs.cso_[phl->cso.sta.num][w][p] = n;
        if (phl->cso.sta.min > -1) cs.cso_[phl->cso.sta.min][w][p] = minimum;
        if (phl->cso.sta.max > -1) cs.cso_[phl->cso.sta.max][w][p] = maximum;
        if (phl->cso.sta.rng > -1) cs.cso_[phl->cso.sta.rng][w][p] = maximum-minimum;
        if (phl->cso.sta.avg > -1) cs.cso_[phl->cso.sta.avg][w][p] = (short)mean;
        if (phl->cso.sta.std > -1) cs.cso_[phl->cso.sta.std][w][p] = (short)standdev(var, n);
        if (phl->cso.sta.skw > -1) cs.cso_[phl->cso.sta.skw][w][p] = (short)skewscaled;
        if (phl->cso.sta.krt > -1) cs.cso_[phl->cso.sta.krt][w][p] = (short)kurtscaled;

        if (phl->cso.sta.quantiles){
          for (q=0; q<phl->cso.sta.nquantiles; q++){
            cs.cso_[phl->cso.sta.qxx[q]][w][p] = (short)quantile(q_array, n, phl->cso.sta.q[q]);
            if (phl->cso.sta.q[q] == 0.25) q25_ = cs.cso_[phl->cso.sta.qxx[q]][w][p];
            if (phl->cso.sta.q[q] == 0.75) q75_ = cs.cso_[phl->cso.sta.qxx[q]][w][p];
          }
        }

        if (phl->cso.sta.iqr > -1){
          if (q25_ == SHRT_MIN) q25_ = (short)quantile(q_array, n, 0.25);
          if (q75_ == SHRT_MIN) q75_ = (short)quantile(q_array, n, 0.75);
          cs.cso_[phl->cso.sta.iqr][w][p] = q75_-q25_;
        }

      
      }

    }
    
    if (alloc_q_array) free((void*)q_array);

  }




  // clean temporal information
  if (cs.d_cso != NULL){ free((void*)cs.d_cso); cs.d_cso = NULL;}
  free((void*)t0);
  free((void*)t1);


  *nproduct = nprod;
  return CSO;
}

