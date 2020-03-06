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
This file contains functions for ImproPhing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "improphe-hl.h"


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** Predict medium resolution data (ImproPhe method)
+++ This function is the core method of ImproPhe and predicts the HR data
+++ on the basis of HR reflectance, HR texture, MR texture and the MR da-
+++ ta. Neighboring pixels of the same class are used for the prediction.
--- hr_:       hr data
--- hr_text_:  hr texture
--- mr_:       mr data
--- mr_tex_:   mr texture
--- pred_:     prediction
--- KDIST:     kernel distance
--- nodata_hr: nodata of hr_
--- nodata_mr: nodata of mr_
--- i:         row
--- j:         column
--- p:         pixel
--- nx:        number of columns
--- ny:        number of rows
--- h:         prediction radius
--- nb_hr:     number of bands in hr_
--- nb_mr:     number of bands in mr_
--- nk:        number of kernel pixels
--- mink:      minimum number of pixels for good prediction
+++ Return:    SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int improphe(float **hr_, float *hr_tex_, float **mr_, float **mr_tex_, short **pred_, float **KDIST, float nodata_hr, short nodata_mr, int i, int j, int p, int nx, int ny, int h, int nb_hr, int nb_mr, int nk, int mink){
int ki, kj; // iterator for KDIST
int ii, jj; // iterator for kernel relative to center
int ni, nj, np; // pixel positions
int b_hr, b_mr, s; // iterator for bands, MR parameters, cutoff-threshold
int wn = 0, k; // iterator for neighbours

double D;     // pixel distance
double S, SS; // spectral similarity + rescaled S
double T, TT; // subpixel heterogeneity + rescaled T
double U, UU; // MR heterogeneity + rescaled U
double W;     // pixel weight

// S, T, U, C of all valid neighbours
double *Srecord = NULL, *Trecord = NULL, **Urecord = NULL, **Mrecord = NULL;
// S, T, U range of all valid neighbours
double Srange[2][5], Trange[2][5], ***Urange = NULL;//[2][MAX_STR_LEN][5];

double *weightxdata = NULL, *weight = NULL;

double sum;          // temporary variables to calculate S
float Sthr[5] = { 0.025, 0.05, 0.1, 0.2, 0.4 }; // cutoff-threshold for S
int Sn[5] = { 0, 0, 0, 0, 0 };  // number of neighbours for each cutoff-threshold
int ns = 5, Sc, *Sclass;             // index for cutoff-threshold

bool skip;



  // if any MR parameter == nodata, skip
  for (b_mr=0, skip=false; b_mr<nb_mr; b_mr++){
    if (mr_[b_mr][p] == nodata_mr){
      skip = true; break;}
  }
  if (skip) return SUCCESS;


  // allocate
  alloc((void**)&Sclass, nk, sizeof(int));
  alloc((void**)&Srecord, nk, sizeof(double));
  alloc((void**)&Trecord, nk, sizeof(double));
  alloc_2D((void***)&Mrecord, nb_mr, nk, sizeof(double));
  alloc_2D((void***)&Urecord, nb_mr, nk, sizeof(double));
  alloc_3D((void****)&Urange, 2, nb_mr, ns, sizeof(double));
  alloc((void**)&weightxdata, nb_mr, sizeof(double));
  alloc((void**)&weight, nb_mr, sizeof(double));


  // initialize
  for (s=0; s<ns; s++){
    Srange[0][s] = Trange[0][s] = INT_MAX; 
    Srange[1][s] = Trange[1][s] = INT_MIN;
    for (b_mr=0; b_mr<nb_mr; b_mr++){ 
      Urange[0][b_mr][s] = INT_MAX; 
      Urange[1][b_mr][s] = INT_MIN;
    }
  }
  for (b_mr=0; b_mr<nb_mr; b_mr++) weightxdata[b_mr] = weight[b_mr] = 0.0;


  /** for each neighbour **/
  for (ii=-h, ki=0; ii<=h; ii++, ki++){
  for (jj=-h, kj=0; jj<=h; jj++, kj++){

    ni  = i+ii; nj = j+jj;
    if (ni < 0 || nj < 0 || ni >= ny || nj >= nx){
      continue;}
    np = nx*ni+nj;

    // if not in circular kernel, skip
    if ((D = KDIST[ki][kj]) > h) continue;

    // if HR == nodata, skip
    if (hr_[0][np] == nodata_hr) continue;

    // if any MR parameter == nodata, skip
    for (b_mr=0, skip=false; b_mr<nb_mr; b_mr++){
      if (mr_[b_mr][np] == nodata_mr){
        skip = true; break;}
    }
    if (skip) continue;

    // specral distance (MAE)
    for (b_hr=0, sum=0; b_hr<nb_hr; b_hr++) sum += fabs(hr_[b_hr][p]-hr_[b_hr][np]);
    S = sum/(float)nb_hr;

    // get cutoff-threshold index for S
    for (s=0, Sc=-1; s<ns; s++){
      if (S < Sthr[s]){ Sc = s; break;}
    }
    if (Sc < 0) continue;
    Sclass[wn] = Sc;

    // keep S + determine range
    Srecord[wn] = S;
    if (S > 0){
      for (s=Sc; s<ns; s++){
        if (S > Srange[1][s]) Srange[1][s] = S;
        if (S < Srange[0][s]) Srange[0][s] = S;
      }
    }

    // keep MR data
    for (b_mr=0; b_mr<nb_mr; b_mr++) Mrecord[b_mr][wn] = mr_[b_mr][np];

    // keep HR texture + determine range
    T = hr_tex_[np];
    Trecord[wn] = T;
    if (T > 0){
      for (s=Sc; s<ns; s++){
        if (T > Trange[1][s]) Trange[1][s] = T;
        if (T < Trange[0][s]) Trange[0][s] = T;
      }
    }

    // keep MR texture + determine range
    for (b_mr=0; b_mr<nb_mr; b_mr++){
      U = mr_tex_[b_mr][np];
      Urecord[b_mr][wn] = U;
      if (U > 0){
        for (s=Sc; s<ns; s++){
          if (U > Urange[1][b_mr][s]) Urange[1][b_mr][s] = U;
          if (U < Urange[0][b_mr][s]) Urange[0][b_mr][s] = U;
        }
      }
    }

    wn++; // number of valid neighbours
    for (s=Sc; s<ns; s++) Sn[s]++; // number of valid neighbours in S-class

  }
  }

  // if no valid neighbour... damn.. use MR
  if (wn == 0){
    for (b_mr=0; b_mr<nb_mr; b_mr++) pred_[b_mr][p] = mr_[b_mr][p];
    free((void*)Sclass);
    free((void*)Srecord);
    free((void*)Trecord);
    free_2D((void**)Urecord, nb_mr);
    free_2D((void**)Mrecord, nb_mr);
    free((void*)weightxdata);
    free((void*)weight);
    free_3D((void***)Urange, 2, nb_mr);
    return SUCCESS;
  }

  // determine the spectral similarity cutoff threshold
  for (s=0, Sc=-1; s<ns; s++){
    if (Sn[s] >= mink){ Sc = s; break;}
  }
  if (Sc < 0) Sc = ns-1;


  // compute pixel weight
  for (k=0; k<wn; k++){

    if (Sclass[k] > Sc) continue;

    SS = rescale_weight(Srecord[k], Srange[0][Sc], Srange[1][Sc]);
    TT = rescale_weight(Trecord[k], Trange[0][Sc], Trange[1][Sc]);

    for (b_mr=0; b_mr<nb_mr; b_mr++){

      UU = rescale_weight(
        Urecord[b_mr][k], Urange[0][b_mr][Sc], Urange[1][b_mr][Sc]);

      W = 1/(SS*TT*UU); 

      weightxdata[b_mr] += W*Mrecord[b_mr][k];
      weight[b_mr] += W;

    }
  }

  // prediction -> weighted mean
  for (b_mr=0; b_mr<nb_mr; b_mr++) pred_[b_mr][p] = (float)(weightxdata[b_mr]/weight[b_mr]);


  // tidy up things
  free((void*)Sclass);
  free((void*)Srecord);
  free((void*)Trecord);
  free_2D((void**)Urecord, nb_mr);
  free_2D((void**)Mrecord, nb_mr);
  free((void*)weightxdata);
  free((void*)weight);
  free_3D((void***)Urange, 2, nb_mr);

  return SUCCESS;
}


/** This function rescales the neighbor pixel weights (R --> R').
--- weight:    weight R
--- minweight: minimum weight within kernel
--- maxweight: maximum weight within kernel
+++ Return:    rescaled weight R'
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double rescale_weight(double weight, double minweight, double maxweight){
double normweight;

  if (weight == 0){
    normweight = 1;
  } else if (minweight == maxweight){
    normweight = 1;
  } else {
    normweight = 1+exp(25 * 
      ((weight-minweight)/(maxweight-minweight))-7.5);
  }

  return normweight;
}


/** This function computes seasonal averages of Level 2 data for the Im-
+++ proPhe code. If one seasonal window has no valid data, the overall 
+++ mean is used instead. The averages are copied into the 3x3 mosaic ima-
+++ ges used in ImproPhe.
--- ard:    ARD
--- mask_:  mask image
--- nb:     number of bands
--- nc:     number of pixels
--- nt:     number of ARD products over time
--- nodata: nodata value
--- nwin:   number of seasonal windows
--- dwin:   doy boundaries of seasonal windows
--- ywin:   year that seasonal window refers to (-1 for multi-annual)
+++ Return: seasonal average
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short **average_season(ard_t *ard, small *mask_, int nb, int nc, int nt, short nodata, int nwin, int *dwin, int ywin){
double **avg = NULL;
int *k = NULL;
int t, p, b, w, wb, doy, year;
bool ok;
short **seasonal_avg_ = NULL;


  alloc_2D((void***)&seasonal_avg_, nwin*nb, nc, sizeof(short));


  #pragma omp parallel private(avg,k,t,p,w,b,wb,ok,doy,year) shared(seasonal_avg_,ard,mask_,nodata,nb,nc,nt,nwin,dwin,ywin) default(none)
  {

    // allocate memory
    alloc((void**)&k,       nwin+1, sizeof(int));
    alloc_2D((void***)&avg, nwin+1, nb, sizeof(double));


    #pragma omp for
    for (p=0; p<nc; p++){
      
      if (mask_ != NULL && !mask_[p]){
        for (wb=0; wb<nwin*nb; wb++) seasonal_avg_[wb][p] = nodata;
        continue;
      }


      // initialize
      for (w=0; w<=nwin; w++){
        for (b=0; b<nb; b++) avg[w][b] = 0.0;
        k[w] = 0.0;
      }

      // compute stats
      for (t=0; t<nt; t++){

        if (!ard[t].msk[p]) continue;

        year = get_stack_year(ard[t].DAT, 0);
        doy  = get_stack_doy(ard[t].DAT,  0);

        // get seasonal window
        for (w=0, ok=false; w<nwin; w++){
          
          // within year window
          if (dwin[w] < dwin[w+1] &&
              doy >= dwin[w] && doy < dwin[w+1] && (year == ywin || ywin < 0)){
            ok = true; break;}

          // window extends to previous year
          if (dwin[w] > dwin[w+1] &&
              ((doy >= dwin[w] && doy <= 365 && (year == ywin-1 || ywin < 0)) ||
               (doy < dwin[w+1] && (year == ywin || ywin < 0)))){
            ok = true; break;}
              
          // window extends to next year
          if (dwin[w] > dwin[w+1] &&
              ((doy >= dwin[w] && doy <= 365 && (year == ywin || ywin < 0)) ||
               (doy < dwin[w+1] && (year == ywin+1 || ywin < 0)))){
            ok = true; break;}

        }

        // not in any window, skip
        if (!ok) continue;

        // sum up for mean
        for (b=0; b<nb; b++){

          avg[w][b]             += ard[t].dat[b][p];
          avg[nwin][b] += ard[t].dat[b][p];

        }

        k[w]++; k[nwin]++;

      }
      
      if (k[nwin] == 0){
        
        for (wb=0; wb<nwin*nb; wb++) seasonal_avg_[wb][p] = nodata;

      } else {

        for (w=0, wb=0; w<nwin; w++){

          if (k[w] > 0){

            // seasonal mean
            for (b=0; b<nb; b++, wb++) seasonal_avg_[wb][p] = (short)(avg[w][b]/k[w]);

          } else {

            // overall mean
            for (b=0; b<nb; b++, wb++) seasonal_avg_[wb][p] = (short)(avg[nwin][b]/k[nwin]);

          }

        }

      }

    }


    // clean
    free_2D((void**)avg, nwin+1);
    free((void*)k);

  }


  return seasonal_avg_;
}


/** This function computes the spatial mean and standard deviation and 
+++ standardizes the input image.
--- data:   image (modified within)
--- nodata: nodata value
--- nc:     number of pixels
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int standardize_float(float *data, float nodata, int nc){
int p;
double avg = 0, var = 0, k = 0;

  // calculate stats
  for (p=0; p<nc; p++){

    if (data[p] == nodata) continue;

    if (++k == 1){
      avg = data[p];
    } else {
      var_recurrence(data[p], &avg, &var, k);
    }

  }

  // standardize
  for (p=0; p<nc; p++){
    if (data[p] == nodata) continue;
    data[p] = (data[p]-(float)avg)/(float)standdev(var, k);
  }

  return SUCCESS;
}


/** Focal standard deviation filter for multi-/single-band images
+++ This function computes the spatial standard deviation as a measure for
+++ texture. If a multiband image is given, the pixel-based standard devi-
+++ ation of the band with highest standard deviation is returned.
--- DAT:    single-/multi-band image
--- nodata: nodata value of image
--- h:      radius of kernel
--- nx:     number of columns
--- ny:     number of rows
--- nb:     number of bands to be used
--- bstart: first band to be used
+++ Return: standard deviation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *focal_sd(float **DAT, float nodata, int h, int nx, int ny, int nb, int bstart){
int i, j, ii, jj, ni, nj, b, nc = nx*ny;
double sd_0, sd_max;
double avg, var, k, dat;
float *sd = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  alloc((void**)&sd, nc, sizeof(float));

  #pragma omp parallel private(b, sd_0, sd_max, avg, var, k, dat, ii, jj, ni, nj) shared(bstart,h,nb,nx,ny,sd,DAT,nodata) default(none)
  {

    #pragma omp for collapse(2) schedule(dynamic,1)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      for (b=bstart, sd_max=0; b<(bstart+nb); b++){

        avg = var = k = 0.0;

        for (ii=-h; ii<=h; ii++){
        for (jj=-h; jj<=h; jj++){

          ni = i+ii; nj = j+jj;

          if (ni < 0 || nj < 0 || ni >= ny || nj >= nx){
            continue;}

          if ((dat = (double)DAT[b][ni*nx+nj]) == nodata){
            continue;}

          if (++k == 1){
            avg = dat;
          } else {
            var_recurrence(dat, &avg, &var, k);
          }
        }
        }

        if ((sd_0 = standdev(var, k))  > sd_max) sd_max = sd_0;

      }

      sd[nx*i+j] = sd_max;

    }
    }

  }


  #ifdef FORCE_CLOCK
  proctime_print("computing focal sd", TIME);
  #endif

  return sd;
}

