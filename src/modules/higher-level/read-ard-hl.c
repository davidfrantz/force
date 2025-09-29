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
This file contains functions for reading ARD
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-ard-hl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdal.h"           // public (C callable) GDAL entry points


int reduce_psf(short *hr, int nx, int ny, int nc, short *lr, int NX, int NY, int NC, short nodata);
int date_ard(date_t *date, char *bname);
int product_ard(char product[], int size, char *bname);
int sensor_ard(int *sid, sen_t *sen, char *bname);
int list_mask(int tx, int ty, par_hl_t *phl, dir_t *dir);
int list_ard(int tx, int ty, sen_t *sen, par_hl_t *phl, dir_t *dir);
int list_ard_filter_ce(int cemin, int cemax, dir_t dir);


/** Reduce spatial resolution using an approximate Point Spread Function
+++ This function will convolve the full-res image with a Gaussian Lowpass
+++ with filter size based upon the two resolutions. Afterward, the image
+++ is reduced using boxcar averaging.
--- hr:     full-res image
--- nx:     number of x-pixels (full-res)
--- ny:     number of y-pixels (full-res)
--- nc:     number of cells (full-res)
--- lr:     low-res image (modified within)
--- NX:     number of x-pixels (low-res)
--- NY:     number of y-pixels (low-res)
--- NC:     number of cells (low-res)
--- nodata: nodata value
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int reduce_psf(short *hr, int nx, int ny, int nc, short *lr, int NX, int NY, int NC, short nodata){
int nk;
int i, j, ii, jj, ni, nj, p, np;
double sum, num, scale = 10000.0;
float **kernel = NULL;
float *GAUSS= NULL;
float sigma, r;
float i0, i1, j0, j1, iw, jw, w;

//printf("reduce_psf: no need to convolve the complete image\n");

  /** Convolute band with gaussian kernel **/

  r = nx/(float)NX;

  if ((nk = r*2) % 2 == 0) nk++;

  sigma = find_sigma(r);

  if (gauss_kernel(nk, sigma, &kernel) != SUCCESS){
    printf("Could not generate kernel. "); return FAILURE;}

  alloc((void**)&GAUSS, nc, sizeof(float));

  #pragma omp parallel private(p,np,ii,jj,ni,nj,sum,num) shared(nx,ny,GAUSS,hr,kernel,nk,nodata,scale) default(none)
  {

    #pragma omp for collapse(2) schedule(dynamic,1)
    for (i=0; i<ny; i++){
    for (j=0; j<nx; j++){

      p = i*nx+j;
      
      GAUSS[p] = nodata;

      if (hr[p] == nodata) continue;

      sum = num = 0;

      for (ii=0; ii<nk; ii++){
      for (jj=0; jj<nk; jj++){

        ni = -(nk-1)/2 + ii + i; nj = -(nk-1)/2 + jj + j;
        if (ni < 0 || ni >= ny || nj < 0 || nj >= nx) continue;
        np = ni*nx+nj;
        
        if (hr[np] == nodata) continue;

        sum += (hr[np]/scale)*kernel[ii][jj];
        num += kernel[ii][jj];

      }
      }

      if (num > 0) GAUSS[p] = sum/num;

    }
    }
    
  }

  free_2D((void**)kernel, nk);


  /** Reduce spatial resolution **/

  #pragma omp parallel private(p,np,ii,jj,ni,nj,i0,i1,j0,j1,iw,jw,w,sum,num) shared(nx,ny,NX,NY,GAUSS,lr,r,nodata,scale) default(none)
  {

    #pragma omp for collapse(2) schedule(dynamic,1)
    for (i=0; i<NY; i++){
    for (j=0; j<NX; j++){
      
      p = i*NX+j;

      sum = num = 0;

      for (ii=0; ii<r; ii++){
      for (jj=0; jj<r; jj++){

        ni = floor(i*r)+ii;
        nj = floor(j*r)+jj;
        
        if (ni >= ny || nj >= nx) continue;
        np = ni*nx+nj;

        if (GAUSS[np] == nodata) continue;

        // compute contribution of hr pixel to lr pixel
        i1 = (ni+1)/r-i; if (i1 > 1.0) i1 = 1.0;
        j1 = (nj+1)/r-j; if (j1 > 1.0) j1 = 1.0;
        i0 = ni/r-i; if (i0 < 0.0) i0 = 0.0;
        j0 = nj/r-j; if (j0 < 0.0) j0 = 0.0;
        iw = i1-i0;
        jw = j1-j0;
        w = iw*jw;

        // weighted mean
        sum += GAUSS[np]*w;
        num += w;

      }
      }

      if (num > 0) lr[p] = (short)(sum/num*scale); else lr[p] = nodata;

    }
    }
    
  }

  free((void*)GAUSS);

  return SUCCESS;
}


/** This function extracts the ARD date from the file's basename, i.e.
+++ acquisition year, month, day, day of year, week and days since CE.
--- date:   date struct (returned)
--- bname:  basename of ARD image
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int date_ard(date_t *date, char *bname){
char cy[5], cm[3], cd[3];
date_t d;


  strncpy(cy, bname,   4); cy[4] = '\0';
  strncpy(cm, bname+4, 2); cm[2] = '\0';
  strncpy(cd, bname+6, 2); cd[2] = '\0';

  init_date(&d);
  set_date(&d, atoi(cy), atoi(cm), atoi(cd));
  
  
  #ifdef FORCE_DEBUG
  printf("date is: %04d (Y), %02d (M), %02d (D), %03d (DOY), %02d (W), %d (CE)\n",
    d.year, d.month, d.day, d.doy, d.week, d.ce);
  #endif

  if (d.year < 1900   || d.year > 2100)   return FAILURE;
  if (d.month < 1     || d.month > 12)    return FAILURE;
  if (d.day < 1       || d.day > 31)      return FAILURE;
  if (d.doy < 1       || d.doy > 365)     return FAILURE;
  if (d.week < 1      || d.week > 52)     return FAILURE;
  if (d.ce < 1900*365 || d.ce > 2100*365) return FAILURE;

  *date = d;
  return SUCCESS;
}


/** This function extracts the ARD product from the file's basename.
--- product: buffer for the product
--- size:    length of the buffer
--- bname:   basename of ARD image
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int product_ard(char product[], int size, char *bname){


  if (size < 4){
    printf("array is too short for getting product.\n");
    product[0] = '\0';
    return FAILURE;
  }

  strncpy(product, bname+22, 3); product[3] = '\0';

  return SUCCESS;
}


/** This function extracts the ARD sensor from the file's basename,
+++ e.g. LND08 and relates it to the ARD sensor dictionary.
--- sid:    sensor ID (returned)
--- sen:    sensor parameters
--- bname:  basename of ARD image
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int sensor_ard(int *sid, sen_t *sen, char *bname){
int s;
char cs[6];


  strncpy(cs, bname+16, 5); cs[5] = '\0';

  for (s=0; s<sen->n; s++){
    if (strstr(cs, sen->sensor[s]) != NULL){
      *sid = s;
      #ifdef FORCE_DEBUG
      printf("sensor is: %s, ID is %d\n", cs, s);
      #endif
      return SUCCESS;
    }
  }

  return FAILURE;
}


/** This function lists all files in the mask directory, and matches the
+++ entries against the requested mask basename. The dir_t struct must be
+++ freed on success.
--- tx:     tile X-ID
--- ty:     tile Y-ID
--- phl:    HL parameters
--- dir:    directory listing (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int list_mask(int tx, int ty, par_hl_t *phl, dir_t *dir){
int m;
int nchar;
dir_t d;


  // current directory
  nchar = snprintf(d.name, NPOW_10, "%s/X%04d_Y%04d", phl->d_mask, tx, ty);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return FAILURE;}

  // directory listing
  if ((d.N = scandir(d.name, &d.LIST, 0, alphasort)) < 0){
    return FAILURE;}

  // masks
  alloc_2D((void***)&d.list, d.N, NPOW_10, sizeof(char));

  for (m=0, d.n=0; m<d.N; m++){

    if (strcmp(d.LIST[m]->d_name, phl->b_mask) == 0){
      copy_string(d.list[d.n++], NPOW_10, d.LIST[m]->d_name);
      break;
    }

  }

  if (d.n<1){
    free_2D((void**)d.list, d.N);
    free_2D((void**)d.LIST, d.N);
    d.list = NULL;
    d.LIST = NULL;
    return FAILURE;
  }

  *dir = d;
  return SUCCESS;
}


/** This function lists all ARD main products (BAP, BOA, TOA, SIG) files 
+++ in the lower level directory. Only requested sensors are listed. Only 
+++ the requested time frame is listed. The dir_t struct must be freed on 
+++ success.
--- tx:     tile X-ID
--- ty:     tile Y-ID
--- sen:    sensor parameters
--- phl:    HL parameters
--- dir:    directory listing (returned)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int list_ard(int tx, int ty, sen_t *sen, par_hl_t *phl, dir_t *dir){
int i, s;
bool vs;
date_t date;
dir_t d;
char ext[NPOW_10];
int nchar;


  // current directory
  nchar = snprintf(d.name, NPOW_10, "%s/X%04d_Y%04d", phl->d_lower, tx, ty);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return FAILURE;}

  #ifdef FORCE_DEBUG
  printf("scanning %s for files\n", d.name);
  #endif

  // directory listing
  if ((d.N = scandir(d.name, &d.LIST, 0, alphasort)) < 0){
    return FAILURE;}

  #ifdef FORCE_DEBUG
  printf("found %d files, filtering now\n", d.N);
  #endif

  // reflectance products
  alloc_2D((void***)&d.list, d.N, NPOW_10, sizeof(char));

  for (i=0, d.n=0; i<d.N; i++){

    // filter expected extensions    
    extension(d.LIST[i]->d_name, ext, NPOW_10);
    if (strcmp(ext, ".dat") != 0 &&
        strcmp(ext, ".bsq") != 0 &&
        strcmp(ext, ".bil") != 0 &&
        strcmp(ext, ".tif") != 0 &&
        strcmp(ext, ".vrt") != 0) continue;

    // filter product type
    if (strstr(d.LIST[i]->d_name, sen->main_product) == NULL) continue;

    // filter sensor list
    for (s=0, vs=false; s<sen->n; s++){
      if (strstr(d.LIST[i]->d_name, sen->sensor[s]) != NULL){
        #ifdef FORCE_DEBUG
        printf("sensor is: %s\n", sen->sensor[s]);
        #endif
        vs = true; 
        break;
      }
    }
    if (!vs) continue;


    // filter dates
    date_ard(&date, d.LIST[i]->d_name);
    if (date.ce < phl->date_range[_MIN_].ce) continue;
    if (date.ce > phl->date_range[_MAX_].ce) continue;
    if (!phl->date_doys[date.doy]) continue;

    // special case: use de-orbiting Landsat 7?
    if (strstr(d.LIST[i]->d_name, "LND07") != NULL &&
        date.ce > phl->date_ignore_lnd07.ce) continue;

    // if we are still here, copy
    copy_string(d.list[d.n++], NPOW_10, d.LIST[i]->d_name);

  }

  if (d.n<1){
    free_2D((void**)d.list, d.N);
    free_2D((void**)d.LIST, d.N);
    d.list = NULL;
    d.LIST = NULL;
    return FAILURE;
  }

  #ifdef FORCE_DEBUG
  printf("%d datasets in here. Proceed.\n", d.n);
  #endif

  *dir = d;
  return SUCCESS;
}


/** This function filters the ARD list. All datasets that are within
+++ the requested temporal range are retained. The range must be given in
+++ days since CE (no-leap-year approximation). The ARD list is modi-
+++ fied (re-ordered and shortened). If there is no dataset left, FAILURE
+++ is returned. The dir_t struct must be freed on success.
--- cemin:  start of range
--- cemax:  end of range
--- dir:    directory listing
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int list_ard_filter_ce(int cemin, int cemax, dir_t dir){
int t;
date_t date;
char **list;
int n;


  alloc_2D((void***)&list, dir.n, NPOW_10, sizeof(char));

  for (t=0, n=0; t<dir.n; t++){
    date_ard(&date, dir.list[t]);
    if (date.ce < cemin || date.ce > cemax) continue;
    copy_string(list[n++], NPOW_10, dir.list[t]);
  }

  for (t=0; t<dir.n; t++){
    if (t >= n){
      copy_string(dir.list[t], NPOW_10, "NULL");
    } else {
      copy_string(dir.list[t], NPOW_10, list[t]);
    }
  }
  free_2D((void**)list, dir.n);

  if ((dir.n = n)<1){
    free_2D((void**)dir.list, dir.N);
    free_2D((void**)dir.LIST, dir.N);
    return FAILURE;
  }

  #ifdef FORCE_DEBUG
  printf("\n%d datasets within ce limits\n", dir.n);
  #endif

  return SUCCESS;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function reads the processing mask (if given).
--- success: success of the mask retrieval
--- ibytes:  bytes read
--- tx:      tile X-ID
--- ty:      tile Y-ID
--- chunk:   chunk number
--- cube:    datacube parameters, e.g. resolution
--- phl:     HL parameters
+++ Return: image brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *read_mask(int *success, off_t *ibytes, int tile[], int chunk[], cube_t *cube, par_hl_t *phl){
brick_t *MASK = NULL;
small *mask_ = NULL;
int nc, p;
char fname[NPOW_10];
dir_t dir;
int n = 0;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  #ifdef FORCE_DEBUG
  printf("Reading mask X%04d_Y%04d\n", tile[_X_], tile[_Y_]);
  #endif

  if (strcmp(phl->d_mask, "NULL") == 0){
    *success = SUCCESS;
    return NULL;
  }

  // get mask file listing, skip if tile is empty
  if (list_mask(tile[_X_], tile[_Y_], phl, &dir) == FAILURE){
    #ifdef FORCE_DEBUG
    printf("No data in here (no mask). Skip.\n");
    #endif
    *success = CANCEL;
    return NULL;
  }


  // read mask
  concat_string_2(fname, NPOW_10, dir.name, dir.list[0], "/");
  if ((MASK = read_chunk(fname, _ARD_MSK_, NULL, 1, 1, 255, _DT_SMALL_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, false, 0, 0)) == NULL){
      printf("Error reading mask %s. ", fname); *success = FAILURE; return NULL;}
  if (phl->radius > 0){
    if ((MASK = add_chunks(fname, _ARD_MSK_, NULL, 1, 1, 255, _DT_SMALL_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, false, phl->radius, MASK)) == NULL){
      printf("Error adding masks %s. ", fname); *success = FAILURE; return NULL;}
  }
  //if ((MASK = read_mask_chunk(fname, 255, chunk, tx, ty, cube)) == NULL){
  //    printf("Error reading mask %s. ", fname); *success = FAILURE; return NULL;}

  free_2D((void**)dir.list, dir.N); dir.list = NULL;
  free_2D((void**)dir.LIST, dir.N); dir.LIST = NULL;

  nc = get_brick_chunkncells(MASK);
  if ((mask_ = get_band_small(MASK, 0)) == NULL){
    printf("Error getting processing mask."); 
    free_brick(MASK);
    *success = FAILURE; return NULL;}

  (*ibytes) += get_brick_size(MASK);

  // count and make sure that mask_ is binary
  for (p=0; p<nc; p++){
    if (mask_[p] == 1){
      n++;
    } else {
      mask_[p] = 0;
    }
  }

  if (n == 0){
    //printf("no valid pixel in mask in this chunk. skip this chunk!\n");
    free_brick(MASK);
    *success = CANCEL;
    return NULL;
  }

  #ifdef FORCE_CLOCK
  proctime_print("read mask", TIME);
  #endif

  *success = SUCCESS;
  return MASK;
}


/** This function reads the features.
--- ibytes: bytes read
--- nt:     number of features read (returned)
--- tx:     tile X-ID
--- ty:     tile Y-ID
--- chunk:  chunk number
--- cube:   datacube parameters, e.g. resolution
--- phl:    HL parameters
+++ Return: ARD
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ard_t *read_features(off_t *ibytes, int *nt, int tile[], int chunk[], cube_t *cube, par_hl_t *phl){
int f, p;
char fname[NPOW_10];
int nchar;
ard_t *features = NULL;
int error = 0;
off_t bytes = 0;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  #ifdef FORCE_DEBUG
  printf("Reading X%04d_Y%04d\n", tile[_X_], tile[_Y_]);
  #endif


  // check if all features do exist
  for (f=0; f<phl->ftr.nfeature; f++){
    nchar = snprintf(fname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_lower, tile[_X_], tile[_Y_], phl->ftr.bname[f]);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); 
      *nt = 0; 
      return NULL;
    }
    if (!fileexist(fname)){
      *nt = 0;
      return NULL;
    }
  }


  alloc((void**)&features, phl->ftr.nfeature, sizeof(ard_t));


  #pragma omp parallel private(fname,p,nchar) shared(features,phl,cube,chunk,tile) reduction(+: error, bytes) default(none)
  {

    #pragma omp for
    for (f=0; f<phl->ftr.nfeature; f++){

      nchar = snprintf(fname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_lower, tile[_X_], tile[_Y_], phl->ftr.bname[f]);
      if (nchar < 0 || nchar >= NPOW_10){
        printf("Buffer Overflow in assembling filename\n"); error++; continue;}
  

      // read feature
      if ((features[f].DAT = read_chunk(fname, _ARD_FTR_, NULL, phl->ftr.band[f], 1, phl->ftr.nodata, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, phl->psf, 0, 0)) == NULL ||
          (features[f].dat = get_bands_short(features[f].DAT)) == NULL){
        printf("Error reading feature %s. ", fname); error++; continue;}
      if (phl->radius > 0){
        if ((features[f].DAT = add_chunks(fname, _ARD_FTR_, NULL, phl->ftr.band[f], 1, phl->ftr.nodata, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, phl->psf, phl->radius, features[f].DAT)) == NULL ||
            (features[f].dat = get_bands_short(features[f].DAT)) == NULL){
          printf("Error adding feature %s. ", fname); error++; continue;}
      }
      bytes += get_brick_size(features[f].DAT);
      
      // compile a 0-filled QAI brick, processing must continue..
      if ((features[f].QAI = copy_brick(features[f].DAT, 1, _DT_SHORT_)) == NULL || 
          (features[f].qai = get_band_short(features[f].QAI, 0)) == NULL){
        printf("Error compiling feature %s.", fname); error++; continue;}
        
      int nc = get_brick_chunkncells(features[f].DAT);
      
      for (p=0; p<nc; p++){
        if (features[f].dat[0][p] == phl->ftr.nodata) set_off(features[f].QAI, p, true);
      }

      features[f].AUX = NULL; features[f].aux = NULL;

    }

  }
  
  
  if (error > 0){
    printf("%d reading errors. ", error); 
    free_ard(features, phl->ftr.nfeature);
    *nt = -1;
    return NULL;
  }

  #ifdef FORCE_CLOCK
  proctime_print("read features", TIME);
  #endif

  (*ibytes) += bytes;
  *nt = phl->ftr.nfeature;
  return features;
}


/** This function reads continuous field images, e.g. LSP.
--- ibytes: bytes read
--- nt:     number of images read (returned)
--- tx:     tile X-ID
--- ty:     tile Y-ID
--- chunk:  chunk number
--- cube:   datacube parameters, e.g. resolution
--- phl:    HL parameters
+++ Return: ARD
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ard_t *read_confield(off_t *ibytes, int *nt, int tile[], int chunk[], cube_t *cube, par_hl_t *phl){
int f;
char fname[NPOW_10];
int nchar;
ard_t *con = NULL;
int error = 0;
off_t bytes = 0;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  #ifdef FORCE_DEBUG
  printf("Reading X%04d_Y%04d\n", tile[_X_], tile[_Y_]);
  #endif


  // check if all features do exist
  for (f=0; f<phl->con.n; f++){
    nchar = snprintf(fname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->con.dname, tile[_X_], tile[_Y_], phl->con.fname[f]);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); 
      *nt = 0; 
      return NULL;
    }
    if (!fileexist(fname)){
      *nt = 0;
      return NULL;
    }
  }


  alloc((void**)&con, phl->con.n, sizeof(ard_t));


  #pragma omp parallel private(fname,nchar) shared(con,phl,cube,chunk,tile) reduction(+: error, bytes) default(none)
  {

    #pragma omp for
    for (f=0; f<phl->con.n; f++){

      nchar = snprintf(fname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->con.dname, tile[_X_], tile[_Y_], phl->con.fname[f]);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling filename\n"); error++; continue;}

      // read continuous field
      if ((con[f].DAT = read_chunk(fname, _ARD_FTR_, NULL, 1, -1, phl->con.nodata, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, false, 0, 0)) == NULL ||
          (con[f].dat = get_bands_short(con[f].DAT)) == NULL){
        printf("Error reading continuous field %s. ", fname); error++; continue;}
      if (phl->radius > 0){
        if ((con[f].DAT = add_chunks(fname, _ARD_FTR_, NULL, 1, -1, phl->con.nodata, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, false, phl->radius, con[f].DAT)) == NULL ||
            (con[f].dat = get_bands_short(con[f].DAT)) == NULL){
          printf("Error adding continuous field products %s. ", fname); error++; continue;}
      }
      bytes += get_brick_size(con[f].DAT);

      // compile a 0-filled QAI brick, processing must continue..
      if ((con[f].QAI = copy_brick(con[f].DAT, 1, _DT_SHORT_)) == NULL || 
          (con[f].qai = get_band_short(con[f].QAI, 0)) == NULL){
        printf("Error compiling continuous field %s.", fname); error++; continue;}


      con[f].AUX = NULL; con[f].aux = NULL;

    }

  }
  
  
  if (error > 0){
    printf("%d reading errors. ", error); 
    free_ard(con, phl->con.n);
    *nt = -1;
    return NULL;
  }

  #ifdef FORCE_CLOCK
  proctime_print("read con", TIME);
  #endif

  (*ibytes) += bytes;
  *nt = phl->con.n;
  return con;
}


/** This function reads all L2 ARD, which are needed to do the required
+++ processing.
--- ibytes: bytes read
--- nt:     number of datasets read (returned)
--- tx:     tile X-ID
--- ty:     tile Y-ID
--- chunk:  chunk number
--- cube:   datacube parameters, e.g. resolution
--- sen:    sensor parameters
--- phl:    HL parameters
+++ Return: ARD
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ard_t *read_ard(off_t *ibytes, int *nt, int tile[], int chunk[], cube_t *cube, sen_t *sen, par_hl_t *phl){
int t, b, p, nb, nc;
char fname[NPOW_10];
char bname[NPOW_10];
char temp[NPOW_10];
dir_t dir;
ard_t *ard = NULL;
int error = 0;
off_t bytes = 0;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  #ifdef FORCE_DEBUG
  printf("Reading X%04d_Y%04d\n", tile[_X_], tile[_Y_]);
  #endif


  // get ARD file listing, skip if tile is empty
  if (list_ard(tile[_X_], tile[_Y_], sen, phl, &dir) == FAILURE){
    #ifdef FORCE_DEBUG
    printf("No data in here (datasets). Skip.\n");
    #endif
    *nt = 0;
    return NULL;
  }


  alloc((void**)&ard, dir.n, sizeof(ard_t));
  

  #pragma omp parallel private(bname,fname,temp,p,b,nc,nb) shared(ard,dir,phl,sen,cube,chunk,tile) reduction(+: error, bytes) default(none)
  {

    #pragma omp for
    for (t=0; t<dir.n; t++){

      // read main product
      if (phl->prd.ref && error == 0){
        copy_string(bname, 1024, dir.list[t]);
        concat_string_2(fname, NPOW_10, dir.name, bname, "/");
        if ((ard[t].DAT = read_chunk(fname, _ARD_REF_, sen, 0, 0, -9999, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, phl->psf, 0, 0)) == NULL ||
            (ard[t].dat = get_bands_short(ard[t].DAT)) == NULL){
          printf("Error reading main product %s. ", fname); error++; continue;}
        if (phl->radius > 0){
          if ((ard[t].DAT = add_chunks(fname, _ARD_REF_, sen, 0, 0, -9999, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, phl->psf, phl->radius, ard[t].DAT)) == NULL ||
              (ard[t].dat = get_bands_short(ard[t].DAT)) == NULL){
            printf("Error adding main products %s. ", fname); error++; continue;}
        }
        bytes += get_brick_size(ard[t].DAT);
      } else {
        ard[t].DAT = NULL;
        ard[t].dat = NULL;
      }


      // read quality product
      if (phl->prd.qai && error == 0){
        copy_string(bname, 1024, dir.list[t]); // clean copy
        replace_string(bname, sen->main_product, sen->quality_product, NPOW_10);
        concat_string_2(fname, NPOW_10, dir.name, bname, "/");
        if (strcmp(sen->quality_product, "NULL") != 0){
          if ((ard[t].QAI = read_chunk(fname, _ARD_AUX_, sen, 1, 1, 1, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, false, 0, 0)) == NULL ||
              (ard[t].qai = get_band_short(ard[t].QAI, 0)) == NULL){
            printf("Error reading QAI product %s. ", fname); error++; continue;}
          if (phl->radius > 0){
            if ((ard[t].QAI = add_chunks(fname, _ARD_AUX_, sen, 1, 1, 1, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, false, phl->radius, ard[t].QAI)) == NULL ||
                (ard[t].qai = get_band_short(ard[t].QAI, 0)) == NULL){
              printf("Error adding QAI products %s. ", fname); error++; continue;}
          }
          bytes += get_brick_size(ard[t].QAI);
        } else {
          if ((ard[t].QAI = copy_brick(ard[t].DAT, 1, _DT_SHORT_)) == NULL || 
              (ard[t].qai = get_band_short(ard[t].QAI, 0)) == NULL){
            printf("Error compiling feature %s.", fname); error++; continue;}
            
          nc = get_brick_chunkncells(ard[t].DAT);
          nb = get_brick_nbands(ard[t].DAT);
          
          for (p=0; p<nc; p++){
          for (b=0; b<nb; b++){
            if (ard[t].dat[b][p] == -9999) set_off(ard[t].QAI, p, true);
          }
          }

        }
      } else {
        ard[t].QAI = NULL;
        ard[t].qai = NULL;
      }

      if (phl->prd.aux && error == 0){

        if (ard[t].DAT == NULL){
          printf("Error reading AUX products. Main product not available. "); 
          error++; 
          continue;
        }

        ard[t].AUX = copy_brick(ard[t].DAT, phl->sen.n_aux_products, _DT_SHORT_);
        ard[t].aux = get_bands_short(ard[t].AUX);

        for (int prd=0; prd<phl->sen.n_aux_products; prd++){

          brick_t *aux_brick = NULL;

          copy_string(bname, 1024, dir.list[t]); // clean copy
          replace_string(bname, sen->main_product, phl->sen.aux_products[prd], NPOW_10);
          concat_string_2(fname, NPOW_10, dir.name, bname, "/");
          if ((aux_brick = read_chunk(fname, _ARD_AUX_, sen, 1, 1, -9999, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, phl->psf, 0, 0)) == NULL){
            printf("Error reading %s product %s. ", phl->sen.aux_products[prd], fname); error++; continue;}
          if (phl->radius > 0){
            if ((aux_brick = add_chunks(fname, _ARD_AUX_, sen, 1, 1, -9999, _DT_SHORT_, phl->chunk_size, chunk, cube->tile_size, tile, cube->resolution, phl->psf, phl->radius, aux_brick)) == NULL){
              printf("Error adding %s products %s. ", phl->sen.aux_products[prd], fname); error++; continue;}
          }
          
          // copy to multiband aux brick in ARD
          short *aux_band = get_band_short(aux_brick, 0);
          memcpy(ard[t].aux[prd], aux_band, get_brick_chunkncells(aux_brick) * get_brick_byte(aux_brick));
          set_brick_bandname(ard[t].AUX, prd, phl->sen.aux_products[prd]);
          set_brick_domain(ard[t].AUX, prd, phl->sen.aux_products[prd]);

          bytes += get_brick_size(aux_brick);
          free_brick(aux_brick);

        }

      }

    }

  }
  
  if (error > 0){
    printf("%d reading errors. ", error); 
    free_ard(ard, dir.n);
    *nt = -1;
    return NULL;
  }

  // copy all provenance to 1st date
  if (phl->prd.ref){
    get_brick_provenance(ard[0].DAT, 0, temp, NPOW_10);
    set_brick_nprovenance(ard[0].DAT, dir.n);
    set_brick_provenance(ard[0].DAT, 0, temp);
    for (t=1; t<dir.n; t++){
      get_brick_provenance(ard[t].DAT, 0, temp, NPOW_10);
      set_brick_provenance(ard[0].DAT, t, temp);
    }
  }

  free_2D((void**)dir.list, dir.N); dir.list = NULL;
  free_2D((void**)dir.LIST, dir.N); dir.LIST = NULL;
  

  #ifdef FORCE_CLOCK
  proctime_print("read ARD", TIME);
  #endif

  (*ibytes) += bytes;
  *nt = dir.n;
  return ard;
}


/** This function reads a chunk of ARD-styled data, and returns a brick. 
+++ GDAL takes care of image decimation / replication if the image has a 
+++ different spatial resolution than expected. A PSF aggregation can also
+++ be used.
--- file:      filename
--- ard_type:  type of ARD, e.g. L2 reflectance or feature
--- sen:       sensor parameters
--- read_b:    if not ARD reflectance, what is the first band to read?
--- read_nb:   if not ARD reflectance, how many bands to read?
--- nodata:    nodata value
--- datatype:  datatype for brick
--- chunk:     chunk number
--- tx:        tile X-ID
--- ty:        tile Y-ID
--- cube:      datacube parameters, e.g. resolution
--- psf:       use PSF?
--- partial_x: only read part of the chunk (width)
--- partial_y: only read part of the chunk (height)
+++ Return:    image brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *read_chunk(char *file, int ard_type, sen_t *sen, int read_b, int read_nb, short nodata, int datatype, double chunk_size[], int chunk[], double tile_size[], int tile[], double resolution, bool psf, double partial_x, double partial_y){
brick_t *brick  = NULL;
short   *brick_short_ = NULL;
small   *brick_small_ = NULL;
GDALDatasetH dataset;
GDALRasterBandH band;
gdalopt_t format;

short *read_buf  = NULL;
short *psf_buf   = NULL;

int sid = 0;

int b, nbands, nb = 0, offb = read_b, p;
int nx, ny, nc;
int xoff_disc, yoff_disc;
int nx_read, ny_read, nc_read;
int b_brick;
int b_disc;
int nx_disc, ny_disc, nc_disc;
double res_disc;
double geotran_disc[_GT_LEN_];
const char *projection_disc = NULL;

char bname[NPOW_10];
char prd[NPOW_02] = "TBD";
date_t date;

double width, height, x_offset, y_offset;
double tol = 5e-3;


  //CPLSetConfigOption("GDAL_PAM_ENABLED", "NO");
  //CPLPushErrorHandler(CPLQuietErrorHandler);

  width  = chunk_size[_X_];
  height = chunk_size[_Y_];
  x_offset = chunk[_X_]*chunk_size[_X_];
  y_offset = chunk[_Y_]*chunk_size[_Y_];

  if (fabs(partial_x) > tol){
    width = fabs(partial_x);
    if (partial_x < 0) x_offset = chunk_size[_X_]+partial_x;
  }
  
  if (fabs(partial_y) > tol){
    height = fabs(partial_y);
    if (partial_y < 0) y_offset = y_offset+chunk_size[_Y_]+partial_y;
  }



  basename_with_ext(file, bname, NPOW_10);
  
  if (ard_type == _ARD_REF_ || ard_type == _ARD_AUX_){
    if (date_ard(&date, bname) != SUCCESS){
      printf("getting date of ARD failed (%s)\n", bname); 
      exit(FAILURE);
    }
    if (product_ard(prd, NPOW_02, bname) != SUCCESS){
      printf("getting product of ARD failed (%s)\n", bname); 
      exit(FAILURE);
    }
    if (sen != NULL){
      if (sensor_ard(&sid, sen, bname) != SUCCESS){
        printf("getting sensor of ARD failed (%s)\n", bname); 
        exit(FAILURE);
      }
    }
  } else init_date(&date);
    

  if (fmod(width, resolution) > tol){
    printf("requested width %f must be a multiple of RESOLUTION %f (%f > %f). ", width, resolution, fmod(width, resolution), tol);
    return NULL;
  }
  if (fmod(height, resolution) > tol){
    printf("requested height %f must be a multiple of RESOLUTION %f (%f > %f). ", height, resolution, fmod(height, resolution), tol);
    return NULL;
  }
  
  nx = (int)(width/resolution);
  ny = (int)(height/resolution);
  nc = nx*ny;

  

  if (ard_type == _ARD_REF_){
    for (b=0; b<sen->n_bands; b++){
      if (sen->band_number[sid][b] >= 0) nb++;
    }
    nbands = sen->n_bands;
  } else {
    nb     = read_nb;
    nbands = read_nb;
  }
  #ifdef FORCE_DEBUG
  printf("reading %d bands.\n", nb);
  #endif



  /** read data
  +++ case 1: target res = image res:
  +++         read @ target res using NN
  +++ case 2: target res < image res && psf = false:
  +++         read @ target res using NN
  +++ case 3: target res < image res && psf = true:
  +++         read @ image res using NN,
  +++         reduce to target res using PSF
  +++ case 4: target res > image res:
  +++         read @ target res using NN
  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  
  dataset = GDALOpenEx(file, GDAL_OF_READONLY, NULL, NULL, NULL);
  //CPLPopErrorHandler();
  
  if (dataset == NULL){
    printf("unable to open %s. ", file); return NULL;}

  if (nb < 0 || nbands < 0){
    nb = nbands = GDALGetRasterCount(dataset);
    offb = 1;
  }

  // number of pixels, resolution and projection in image on disc

  GDALGetGeoTransform(dataset, geotran_disc); 
  res_disc = geotran_disc[_GT_RES_];
  projection_disc = GDALGetProjectionRef(dataset);

  #ifdef FORCE_DEBUG
  print_dvector(geotran_disc, "Geotransformation", _GT_LEN_, 10, 2);
  #endif
  
  if (fmod(width, res_disc) > tol){
    printf("requested image width %f must be a multiple of image resolution %f (%f > %f). ", width, res_disc, fmod(width, res_disc), tol);
    return NULL;
  }
  if (fmod(height, res_disc) > tol){
    printf("requested image height %f must be a multiple of image resolution %f (%f > %f). ", height, res_disc, fmod(height, res_disc), tol);
    return NULL;
  }
  if (fmod(y_offset, res_disc) > tol){
    printf("requested image offset %f must be a multiple of image resolution %f (%f > %f). ", y_offset, res_disc, fmod(y_offset, res_disc), tol);
    return NULL;
  }
  if (fmod(x_offset, res_disc) > tol){
    printf("requested image offset %f must be a multiple of image resolution %f (%f > %f). ", x_offset, res_disc, fmod(x_offset, res_disc), tol);
    return NULL;
  }

  nx_disc = (int)(width/res_disc);
  ny_disc = (int)(height/res_disc);
  nc_disc = nx_disc*ny_disc;
  xoff_disc = (int)(x_offset/res_disc);
  yoff_disc = (int)(y_offset/res_disc);
  

  #ifdef FORCE_DEBUG
  printf("read %d x %d pixels with offset %d / %d at resolution %f\n", nx_disc, ny_disc, xoff_disc, yoff_disc, res_disc);
  printf("copy to %d x %d pixels with offset %d / %d at resolution %f\n", nx, ny, 0, 0, resolution);
  #endif


  // depending on PSF flag, read image part at full or reduced resolution
  if (psf && nc_disc > nc){
    nx_read = nx_disc; ny_read = ny_disc; nc_read = nc_disc;
    alloc((void**)&psf_buf, nc, sizeof(short));
    #ifdef FORCE_DEBUG
    printf("reduce with psf: %s\n", bname);
    #endif
  } else {
    nx_read = nx; ny_read = ny; nc_read = nc;
    #ifdef FORCE_DEBUG
    printf("input using NN:  %s\n", bname);
    #endif
  }

  #ifdef FORCE_DEBUG
  printf("reading %d pixels, converting them into %d RAM pixels\n", nc_read, nc);
  #endif

  alloc((void**)&read_buf, nc_read, sizeof(short));
  brick = allocate_brick(nb, nc, datatype);

  for (b=0, b_brick=0; b<nbands; b++){

    if (ard_type == _ARD_REF_){
      if ((b_disc = sen->band_number[sid][b])  < 0) continue;
      if ((b_disc = sen->band_number[sid][b]) == 0){
        set_brick_domain(brick, b_brick, sen->band_names[b]);
        b_brick++;
        continue;
      }
    } else {
      b_disc = offb+b;
    }

    #ifdef FORCE_DEBUG
    printf("read band %d to %d, %d bands in total\n", b_disc, b_brick, nb);
    #endif

    if (datatype == _DT_SMALL_){
      if ((brick_small_ = get_band_small(brick, b_brick)) == NULL) return NULL;
    } else if (datatype == _DT_SHORT_){
      if ((brick_short_ = get_band_short(brick, b_brick)) == NULL) return NULL;
    } else {
      printf("unsupported datatype. "); return NULL;
    }

    for (p=0; p<nc_read; p++) read_buf[p] = nodata;

    band = GDALGetRasterBand(dataset, b_disc);
    if (GDALRasterIO(band, GF_Read, 
      xoff_disc, yoff_disc, nx_disc, ny_disc, 
      read_buf, nx_read, ny_read, GDT_Int16, 0, 0) == CE_Failure){
      printf("could not read image.\n"); return NULL;}

    if (psf && nc_disc > nc){
      for (p=0; p<nc; p++) psf_buf[p] = nodata;
      reduce_psf(read_buf, nx_disc, ny_disc, nc_disc, psf_buf, nx, ny, nc, nodata);
      if (datatype == _DT_SMALL_){
        for (p=0; p<nc; p++) brick_small_[p] = psf_buf[p];
      } else if (datatype == _DT_SHORT_){
        for (p=0; p<nc; p++) brick_short_[p] = psf_buf[p];
      } else {
        printf("unsupported datatype. "); return NULL;
      }
    } else {
      if (datatype == _DT_SMALL_){
        for (p=0; p<nc; p++) brick_small_[p] = read_buf[p];
      } else if (datatype == _DT_SHORT_){
        for (p=0; p<nc; p++) brick_short_[p] = read_buf[p];
      } else {
        printf("unsupported datatype. "); return NULL;
      }
    }
    
    if (ard_type == _ARD_REF_){
      set_brick_domain(brick, b_brick, sen->band_names[b]);
      set_brick_bandname(brick, b_brick, sen->band_names[b]);
    }

    b_brick++;

  }


// compile brick correctly
  set_brick_geotran(brick,    geotran_disc);
  set_brick_res(brick,        resolution);
  set_brick_proj(brick,       projection_disc);
  set_brick_ncols(brick,      (int)(tile_size[_X_]/resolution));
  set_brick_nrows(brick,      (int)(tile_size[_Y_]/resolution));
  set_brick_chunkncols(brick, (int)(chunk_size[_X_]/resolution));
  set_brick_chunknrows(brick, (int)(chunk_size[_Y_]/resolution));
  set_brick_chunk_dim_x(brick,  (int)(tile_size[_X_]/chunk_size[_X_]));
  set_brick_chunk_dim_y(brick,  (int)(tile_size[_Y_]/chunk_size[_Y_]));
  set_brick_chunkx(brick,     chunk[_X_]);
  set_brick_chunky(brick,     chunk[_Y_]);
  set_brick_tilex(brick,      tile[_X_]);
  set_brick_tiley(brick,      tile[_Y_]);

  set_brick_filename(brick, "DONOTOUTPUT");
  set_brick_dirname(brick, "DONOTOUTPUT");
  set_brick_provdir(brick, "DONOTOUTPUT");
  set_brick_product(brick, prd);
  set_brick_sensorid(brick, sid);
  set_brick_name(brick, "FORCE Level 2 ARD");

  set_brick_nprovenance(brick, 1);
  set_brick_provenance(brick, 0, file);

  default_gdaloptions(_FMT_GTIFF_, &format);

  set_brick_open(brick,   OPEN_FALSE);
  set_brick_format(brick, &format);
  
  //printf("some of the ARD metadata should be read from disc. TBI\n");
  for (b=0; b<nb; b++) set_brick_nodata(brick, b, nodata);
  for (b=0; b<nb; b++) set_brick_scale(brick, b, 10000);
  for (b=0; b<nb; b++) set_brick_date(brick, b, date);
  if(sen != NULL){
    for (b=0; b<nb; b++) set_brick_sensor(brick, b, sen->sensor[sid]);
  }

  GDALClose(dataset);
  if (read_buf != NULL){ free((void*)read_buf); read_buf = NULL;}
  if (psf_buf  != NULL){ free((void*)psf_buf);  psf_buf  = NULL;}

  //CSLDestroy(open_options);

  return brick;
}


/** This function adds a partial chunk of ARD-styled data to the brick we
+++ already have. This is needed for functions that use kernel based pro-
+++ cessing and need to incorporate data from neighoring chunks and/or
+++ tiles to generate seamless output.
--- file:     filename
--- ard_type: type of ARD, e.g. L2 reflectance or feature
--- sen:      sensor parameters
--- read_b:   if not ARD reflectance, what is the first band to read?
--- read_nb:  if not ARD reflectance, how many bands to read?
--- nodata:   nodata value
--- datatype: datatype for brick
--- chunk:    chunk number
--- tx:       tile X-ID
--- ty:       tile Y-ID
--- cube:     datacube parameters, e.g. resolution
--- psf:      use PSF?
--- radius:   distance (in projection units) that need to be added
--- ARD:      image brick we already have in memory (freed within)
+++ Return:   extended image brick
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *add_chunks(char *file, int ard_type, sen_t *sen, int read_b, int read_nb, short nodata, int datatype, double chunk_size[], int chunk_central[], double tile_size[], int tile_central[], double resolution, bool psf, double distance_to_add, brick_t *ARD){
brick_t *add  = NULL;
small  **add_small_ = NULL;
short  **add_short_ = NULL;
brick_t *brick  = NULL;
small  **brick_small_ = NULL;
short  **brick_short_ = NULL;
int pixels_to_add;
int nbands;
int mosaic_pixels[2], chunk_pixels[2];
int chunk_to_add_relative[2], chunk_to_add[2], chunk_layout[2];
int tile_to_add[2];
char fname[NPOW_10], *pch = NULL;;
char c_tc[NPOW_04];
char c_tn[NPOW_04];
int nchar;


  nbands = get_brick_nbands(ARD);

  pixels_to_add = (int)(distance_to_add/resolution);
  mosaic_pixels[_X_] = (int)(chunk_size[_X_]/resolution) + pixels_to_add*2;
  mosaic_pixels[_Y_] = (int)(chunk_size[_Y_]/resolution) + pixels_to_add*2;

  chunk_pixels[_X_] = (int)(chunk_size[_X_]/resolution);
  chunk_pixels[_Y_] = (int)(chunk_size[_Y_]/resolution);

  chunk_layout[_X_] = (int)(tile_size[_X_]/chunk_size[_X_]);
  chunk_layout[_Y_] = (int)(tile_size[_Y_]/chunk_size[_Y_]);


  #ifdef FORCE_DEBUG
  printf("brick with added edges has %d x %d pixels = %d cells\n",
    mosaic_pixels[_X_], mosaic_pixels[_Y_], mosaic_pixels[_X_]*mosaic_pixels[_Y_]);
  #endif
  

  brick = copy_brick(ARD, nbands, _DT_NONE_);
  set_brick_chunkncols(brick, mosaic_pixels[_X_]);
  set_brick_chunknrows(brick, mosaic_pixels[_Y_]);
  allocate_brick_bands(brick, nbands, mosaic_pixels[_X_] * mosaic_pixels[_Y_], datatype);
  if (datatype == _DT_SMALL_){
    if ((brick_small_ = get_bands_small(brick)) == NULL) return NULL;
  } else if (datatype == _DT_SHORT_){
    if ((brick_short_ = get_bands_short(brick)) == NULL) return NULL;
  } else {
    printf("unsupported datatype. "); return NULL;
  }
  
  // copy file name
  copy_string(fname, NPOW_10, file);

  nchar = snprintf(c_tc, NPOW_04, "X%04d_Y%04d", tile_central[_X_], tile_central[_Y_]);
  if (nchar < 0 || nchar >= NPOW_04){ 
    printf("Buffer Overflow in assembling tile\n"); return NULL;}


  for (chunk_to_add_relative[_Y_]=-1; chunk_to_add_relative[_Y_]<=1; chunk_to_add_relative[_Y_]++){
  for (chunk_to_add_relative[_X_]=-1; chunk_to_add_relative[_X_]<=1; chunk_to_add_relative[_X_]++){
    

    chunk_to_add[_Y_] = chunk_central[_Y_] + chunk_to_add_relative[_Y_];
    chunk_to_add[_X_] = chunk_central[_X_] + chunk_to_add_relative[_X_];

    if (chunk_to_add[_Y_] < 0){
      chunk_to_add[_Y_] = chunk_layout[_Y_] - 1;
      tile_to_add[_Y_] = tile_central[_Y_] - 1;
    } else if (chunk_to_add[_Y_] >= chunk_layout[_Y_]){
      chunk_to_add[_Y_] = 0;
      tile_to_add[_Y_] = tile_central[_Y_] + 1;
    } else {
      tile_to_add[_Y_] = tile_central[_Y_];
    }

     if (chunk_to_add[_X_] < 0){
      chunk_to_add[_X_] = chunk_layout[_X_] - 1;
      tile_to_add[_X_] = tile_central[_X_] - 1;
    } else if (chunk_to_add[_X_] >= chunk_layout[_X_]){
      chunk_to_add[_X_] = 0;
      tile_to_add[_X_] = tile_central[_X_] + 1;
    } else {
      tile_to_add[_X_] = tile_central[_X_];
    }


    if (chunk_to_add_relative[_Y_] == 0 && chunk_to_add_relative[_X_] == 0){

      add = ARD;
      if (datatype == _DT_SMALL_){
        if ((add_small_ = get_bands_small(ARD)) == NULL) return NULL;
      } else if (datatype == _DT_SHORT_){
        if ((add_short_ = get_bands_short(ARD)) == NULL) return NULL;
      } else {
        printf("unsupported datatype. "); return NULL;
      }

    } else {

      // new filename
      nchar = snprintf(c_tn, NPOW_04, "X%04d_Y%04d", tile_to_add[_X_], tile_to_add[_Y_]);
      if (nchar < 0 || nchar >= NPOW_04){ 
        printf("Buffer Overflow in assembling tile\n"); return NULL;}
      
      if ((pch = strstr(fname, c_tc)) == NULL){
        printf("error in assembling filename for neighboring chunk.\n"); return NULL;
      } else strncpy(pch, c_tn, 11);

      copy_string(c_tc, NPOW_04, c_tn);
          
      
      #ifdef FORCE_DEBUG
      printf("\nneighboring chunk %d %d:\n", chunk_to_add_relative[_X_], chunk_to_add_relative[_Y_]);
      printf("X%04d_Y%04d -> X%04d_Y%04d\n", tile_central[_X_], tile_central[_Y_], tile_to_add[_X_], tile_to_add[_Y_]);
      printf("Chunk X:%d Y:%d -> Chunk X:%d Y:%d\n", chunk_central[_X_], chunk_central[_Y_], chunk_to_add[_X_], chunk_to_add[_Y_]);
      printf("Radius %.0f %.0f\n", chunk_to_add_relative[_X_]*distance_to_add, chunk_to_add_relative[_Y_]*distance_to_add);
      printf("%s\n", fname);
      #endif

      
      // read the partial chunk
      if (fileexist(fname)){
        #ifdef FORCE_DEBUG
        printf("file exists. read chunk.\n");
        #endif
        if ((add  = read_chunk(fname, ard_type, sen, read_b, read_nb, nodata, datatype, chunk_size, chunk_to_add, tile_size, tile_to_add, resolution, psf, chunk_to_add_relative[_X_]*distance_to_add, chunk_to_add_relative[_Y_]*distance_to_add)) == NULL){
          printf("Error reading neighoring product %s. ", fname); return NULL;}
        if (datatype == _DT_SMALL_){
          if ((add_small_ = get_bands_small(add)) == NULL) return NULL;
        } else if (datatype == _DT_SHORT_){
          if ((add_short_ = get_bands_short(add)) == NULL) return NULL;
        } else {
          printf("unsupported datatype. "); return NULL;
        }
      } else {
        #ifdef FORCE_DEBUG
        printf("file doesn't exists. NULL.\n");
        #endif
        add = NULL;
        add_small_ = NULL;
        add_short_ = NULL;
      }
      
    }

    int mosaic_insert[2];
    int mosaic_offset[2];

    if (chunk_to_add_relative[_X_] <  0){ 
      mosaic_insert[_X_] = pixels_to_add; 
      mosaic_offset[_X_] = 0; 
    } else if (chunk_to_add_relative[_X_] == 0){ 
      mosaic_insert[_X_] = chunk_pixels[_X_]; 
      mosaic_offset[_X_] = pixels_to_add; 
    } else if (chunk_to_add_relative[_X_] >  0){
      mosaic_insert[_X_] = pixels_to_add;
      mosaic_offset[_X_] = mosaic_pixels[_X_] - pixels_to_add; 
    }

    if (chunk_to_add_relative[_Y_] <  0){ 
      mosaic_insert[_Y_] = pixels_to_add;
      mosaic_offset[_Y_] = 0; 
    } else if (chunk_to_add_relative[_Y_] == 0){ 
      mosaic_insert[_Y_] = chunk_pixels[_Y_];
      mosaic_offset[_Y_] = pixels_to_add; 
    } else if (chunk_to_add_relative[_Y_] >  0){ 
      mosaic_insert[_Y_] = pixels_to_add;
      mosaic_offset[_Y_] = mosaic_pixels[_Y_] - pixels_to_add; 
    }


    #ifdef FORCE_DEBUG
    printf("adding %d x %d pixels with %d / %d offset\n", 
      mosaic_insert[_X_], mosaic_insert[_Y_], 
      mosaic_offset[_X_], mosaic_offset[_Y_]); 
    #endif


    // copy the values
    for (int row=0; row<mosaic_insert[_Y_]; row++){
    for (int col=0; col<mosaic_insert[_X_]; col++){

      int neighbor_cell = row*mosaic_insert[_X_] + col;
      int mosaic_cell = (row+mosaic_offset[_Y_])*mosaic_pixels[_X_] + col+mosaic_offset[_X_];

      if (datatype == _DT_SMALL_){
        if (add_small_ == NULL){
          for (int b=0; b<nbands; b++) brick_small_[b][mosaic_cell] = nodata;
        } else {
          for (int b=0; b<nbands; b++) brick_small_[b][mosaic_cell] = add_small_[b][neighbor_cell];
        }
      } else if (datatype == _DT_SHORT_){
        if (add_short_ == NULL){
          for (int b=0; b<nbands; b++) brick_short_[b][mosaic_cell] = nodata;
        } else {
          for (int b=0; b<nbands; b++) brick_short_[b][mosaic_cell] = add_short_[b][neighbor_cell];
        }
      } else {
        printf("unsupported datatype. "); return NULL;
      }

    }
    }

    free_brick(add);
    add = NULL;
    add_small_ = NULL;
    add_short_ = NULL;

  }
  }


  #ifdef FORCE_DEBUG
  printf("\ndone adding this dataset\n\n");
  #endif

  return brick;
}


/** This function frees the ARD
--- ard:    ARD
--- nt:     number of datasets
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int free_ard(ard_t *ard, int nt){
int t;

  for (t=0; t<nt; t++){
    if (ard[t].DAT != NULL){ free_brick(ard[t].DAT); ard[t].DAT = NULL;}
    if (ard[t].QAI != NULL){ free_brick(ard[t].QAI); ard[t].QAI = NULL;}
    if (ard[t].MSK != NULL){ free_brick(ard[t].MSK); ard[t].MSK = NULL;}
    if (ard[t].AUX != NULL){ free_brick(ard[t].AUX); ard[t].AUX = NULL;}
  }
  free((void*)ard);
  ard = NULL;

  return SUCCESS;
}

