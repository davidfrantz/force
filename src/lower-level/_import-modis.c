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
This program imports MODIS 09 GA products to FORCE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/stack-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/quality-cl.h"
//#include "../lower-level/cube-ll.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points

/** OpenMP **/
//#include <omp.h> // multi-platform shared memory multiprocessing




////-------------------------------------------------------------------
//// convert integer to bit array
//void int2bit(int x, int *bin, int size){
//int quotient = x, i;
//
//  for (i=0; i<size; i++){
//    bin[i]= quotient % 2;
//    quotient = quotient / 2;
//  }
//}
//
////-------------------------------------------------------------------
//// get integer value of a bit word
//int bit2int(int *bin, int from, int len){
//int x, k, i;
//
//  for (i=0, k=0, x=0; i<len; i++, k++){
//    x += bin[from+i]*pow(2,k);
//  }
//  return x;
//}

int get_modqa(ushort *modqa_, int index, int p, int bitfields){
int i;
short val = 0;


  for (i=0; i<bitfields; i++) val |= (short)(1 << i);

  return (short)(modqa_[p] >> index) & val;
}


void set_meta(stack_t *STACK, date_t *date, double geotran[6], int nx, int ny, const char *proj, int tx, int ty, int sid, const char *sensor, const char *prd, const char *dout){
char fname[NPOW_10];
int nchar;


  nchar = snprintf(fname, NPOW_10, "%04d%02d%02d_LEVEL2_%s_%s", date->year, date->month, date->day, sensor, prd);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling output name name\n"); exit(FAILURE);}

  set_stack_open(STACK, 1);
  set_stack_format(STACK, _FMT_GTIFF_);
  set_stack_explode(STACK, 0);

  set_stack_res(STACK, geotran[1]);
  set_stack_ulx(STACK, geotran[0]);
  set_stack_uly(STACK, geotran[3]);
  set_stack_ncols(STACK, nx);
  set_stack_nrows(STACK, ny);
  set_stack_chunkncols(STACK, nx);
  set_stack_chunknrows(STACK, ny/10);
  set_stack_nchunks(STACK, 10);
  set_stack_proj(STACK, proj);
  set_stack_tilex(STACK, tx);
  set_stack_tiley(STACK, ty);

  set_stack_product(STACK, prd);
  set_stack_sensorid(STACK, sid);

  set_stack_name(STACK, "FORCE Level 2 MODIS Import");
  set_stack_dirname(STACK, dout);
  set_stack_filename(STACK, fname);

  return;
}


void set_meta_band(stack_t *STACK, int b, int scale, short nodata, const char *sensor, const char *domain, float wvl, const char *unit, date_t *date){


  set_stack_scale(STACK, b, scale);
  set_stack_nodata(STACK, b, nodata);
  set_stack_sensor(STACK, b, sensor);
  set_stack_domain(STACK, b, domain);
  set_stack_wavelength(STACK, b, wvl);
  set_stack_unit(STACK, b, unit);
  set_stack_date(STACK, b, *date);
  set_stack_save(STACK, b, 1);
  set_stack_bandname(STACK, b, domain);

  return;
}


void write_modcube(char *dout, char *proj){
cube_t cube;


  copy_string(cube.dname, NPOW_10, dout);
  copy_string(cube.proj,  NPOW_10, proj);

  cube.origin_geo.x = -124.291447;
  cube.origin_geo.y = 90.000000;
  cube.origin_map.x = -20015109.354000;
  cube.origin_map.y = 10007554.677000;
  cube.tilesize     = 1111950.521000;
  cube.chunksize    = 111195.052100;

  if (write_datacube_def(&cube) == FAILURE){
    printf("Writing datacube definition failed\n\n"); exit(FAILURE);}

  return;
}


void compile_qai(stack_t *QAI, ushort *modqa_, short **boa_, int nc, int nb, int sid, short nodata){
int p, b;


  for (p=0; p<nc; p++){

    if (boa_[0][p] < -100){
      for (b=0; b<nb; b++) boa_[b][p] = nodata;
      set_off(QAI, p, 1);
    }

    if (get_modqa(modqa_,  0, p, 2) >  0) set_cloud(QAI, p, 2);   // cloudy, mixed or not set
    if (get_modqa(modqa_,  2, p, 1) >  0) set_shadow(QAI, p, 1);  // cloud shadow
    if (get_modqa(modqa_,  3, p, 3) != 1) set_water(QAI, p, 1);   // any water type
    if (get_modqa(modqa_,  6, p, 2) == 0) set_aerosol(QAI, p, 3); // aerosol climatology
    if (get_modqa(modqa_,  6, p, 2) >  1) set_aerosol(QAI, p, 2); // high aerosol
    if (get_modqa(modqa_,  8, p, 2) >  1) set_cloud(QAI, p, 3);   // cirrus
    if (get_modqa(modqa_, 10, p, 1) >  1) set_cloud(QAI, p, 2);   // internal cloud algo
    if (get_modqa(modqa_, 12, p, 1) >  1) set_snow(QAI, p, 1);    // snow/ice
    if (get_modqa(modqa_, 13, p, 1) >  1) set_cloud(QAI, p, 1);   // adjacent to cloud
    if (get_modqa(modqa_, 15, p, 1) >  1) set_snow(QAI, p, 1);    // internal snow algo

    for (b=0; b<nb; b++){
      if (b == 5 && sid == _SEN_MOD02_) continue;
      if (boa_[b][p] < 0){     set_subzero(QAI, p, 1);    break;}
      if (boa_[b][p] > 10000){ set_saturation(QAI, p, 1); break;}
    }

  }

  return;
}


int main ( int argc, char *argv[] ){


char *finp = NULL;
char *dout = NULL;
char binp[NPOW_10];
char ext[NPOW_10];
char dtile[NPOW_10];

char sen_[4], sensor[6];
char year_[5], doy_[4];
char tx_[3], ty_[3];
int year, doy, month, day;
int tx, ty;
int sid;


date_t date;

GDALDatasetH fp, fs;
GDALRasterBandH band;
char **sds = NULL;
char *sdsname = NULL;
char KeyName[NPOW_10];

int b, nb = 7, b_ref[7] = { 14, 15, 12, 13, 16, 17, 18 }, b_qai = 2;
int nchar;

int nx, ny, nc, nx_, ny_;
stack_t *BOA = NULL;
short **boa_ = NULL;
stack_t *QAI = NULL;
ushort *modqa_ = NULL;

char domain[7][NPOW_10] = { "BLUE", "GREEN", "RED", "NIR", "SWIR0", "SWIR1", "SWIR2" };
float wvl[7] = { 0.469, 0.555, 0.645, 0.858, 1.240, 1.640, 2.130 };
double geotran[6];
const char *proj_;
char proj[NPOW_10];
short nodata = -9999;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 3){
    printf("usage: %s hdf-image out-dir\n\n", argv[0]);
    return FAILURE;
  }


  // input args
  finp = argv[1];
  dout = argv[2];

  // basename and extension
  basename_with_ext(finp, binp, NPOW_10);
  extension2(finp, ext, NPOW_10);

  // some tests on input parameters
  if (!fileexist(finp)){
    printf("input file %s does not exist.\n\n", finp); return FAILURE;}

  if (!fileexist(dout)){
    printf("output directory %s does not exist.\n\n", dout); return FAILURE;}

  if (strstr(binp, "09GA") == NULL){
    printf("input file %s is not a MODIS 09GA product.\n\n", binp); return FAILURE;}

  if (strcmp(ext, ".hdf") != 0){
    printf("input file %s is not a MODIS hdf file.\n\n", ext); return FAILURE;}

  // register GDAL drivers  
  GDALAllRegister();

  // info from filename
  strncpy(sen_,  binp,    3); sen_[3]  = '\0';
  strncpy(year_, binp+9,  4); year_[4] = '\0'; year = atoi(year_);
  strncpy(doy_,  binp+13, 3); doy_[3]  = '\0'; doy  = atoi(doy_);
  strncpy(tx_,   binp+18, 2); tx_[2]   = '\0'; tx   = atoi(tx_);
  strncpy(ty_,   binp+21, 2); ty_[2]   = '\0'; ty   = atoi(ty_);
  
  if (strcmp(sen_, "MOD") == 0){
    copy_string(sensor, 6, "MOD01");
    sid = _SEN_MOD01_;
  } else if (strcmp(sen_, "MYD") == 0){
    copy_string(sensor, 6, "MOD02");
    sid = _SEN_MOD02_;
  } else {
    printf("no MODIS sensor detected: %s.\n\n", sen_); return FAILURE;
  }


  doy2md(doy, &month, &day); // not accurate in leap years
  set_date(&date, year, month, day);
  
  nchar = snprintf(dtile, NPOW_10, "%s/X%04d_Y%04d", dout, tx, ty);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling output directory\n");return FAILURE;}

  if (!fileexist(dtile)) createdir(dtile);

  // open input dataset
  if ((fp = GDALOpen(finp, GA_ReadOnly)) == NULL){
    printf("unable to open %s\n\n", finp); return FAILURE;}

  // get SDS listing
  sds = GDALGetMetadata(fp, "SUBDATASETS");
  if (CSLCount(sds) == 0){
    printf("unable to retrieve SDS list from %s\n\n", finp);return FAILURE;}
  //CSLPrint(sds, NULL);
  


  for (b=0; b<nb; b++){

    nchar = snprintf(KeyName, NPOW_10, "SUBDATASET_%d_NAME", b_ref[b]);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling hdf sds name\n");return FAILURE;}

    sdsname = CPLStrdup(CSLFetchNameValue(sds, KeyName));
    if ((fs = GDALOpen(sdsname, GA_ReadOnly)) == NULL){
      printf("unable to open SDS %s\n\n", sdsname); return FAILURE;}

    nx = GDALGetRasterXSize(fs);
    ny = GDALGetRasterYSize(fs);
    nc = nx*ny;

    proj_ = GDALGetProjectionRef(fs);
    copy_string(proj, NPOW_10, proj_);
    GDALGetGeoTransform(fs, geotran);

    if (BOA == NULL){
      BOA = allocate_stack(nb, nc, _DT_SHORT_);
      if ((boa_ = get_bands_short(BOA)) == NULL) return FAILURE;
    }

    band = GDALGetRasterBand(fs, 1);
    if (GDALRasterIO(band, GF_Read,  0, 0, nx, ny, 
          boa_[b], nx, ny, GDT_Int16, 0, 0) == CE_Failure){
      printf("could not read image %s. ", KeyName); return FAILURE;}
    CPLFree(sdsname);
    GDALClose(fs);
    
    set_meta_band(BOA, b, 10000, nodata, sensor, domain[b], wvl[b], "micrometers", &date);

  }


  nchar = snprintf(KeyName, NPOW_10, "SUBDATASET_%d_NAME", b_qai);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling hdf sds name\n");return FAILURE;}

  sdsname = CPLStrdup(CSLFetchNameValue(sds, KeyName));
  if ((fs = GDALOpen(sdsname, GA_ReadOnly)) == NULL){
    printf("unable to open SDS %s\n\n", sdsname); return FAILURE;}

  nx_ = GDALGetRasterXSize(fs);
  ny_ = GDALGetRasterYSize(fs);

  QAI = allocate_stack(1, nc, _DT_SHORT_);
  alloc((void**)&modqa_, nc, sizeof(ushort));

  band = GDALGetRasterBand(fs, 1);
  if (GDALRasterIO(band, GF_Read,  0, 0, nx_, ny_, 
        modqa_, nx, ny, GDT_UInt16, 0, 0) == CE_Failure){
    printf("could not read image %s. ", KeyName); return FAILURE;}
  CPLFree(sdsname);
  GDALClose(fs);

  set_meta_band(QAI, 0, 1, 1, sensor, "QAI", 1, "unknown", &date);

  GDALClose(fp);


  compile_qai(QAI, modqa_, boa_, nc, nb, sid, nodata);

  set_meta(BOA, &date, geotran, nx, ny, proj, tx, ty, sid, sensor, "BOA", dtile);
  set_meta(QAI, &date, geotran, nx, ny, proj, tx, ty, sid, sensor, "QAI", dtile);

  //print_stack_info(BOA);
  //print_stack_info(QAI);

  write_stack(BOA);
  write_stack(QAI);
  write_modcube(dout, proj);

  free_stack(BOA);
  free_stack(QAI);

  free((void*)modqa_);
  


  return SUCCESS;
}

