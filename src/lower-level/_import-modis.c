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

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../cross-level/const-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/brick-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/quality-cl.h"


/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points


typedef struct {
  int n;
  char fimg[NPOW_10];
  char dout[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] image output-dir\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'image':      input image (HDF)\n");
  printf("  - 'output-dir': output directory\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;


  opterr = 0;

  // optional parameters
  while ((opt = getopt(argc, argv, "hvi")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        printf("FORCE version: %s\n", _VERSION_);
        exit(SUCCESS);
      case 'i':
        printf("Import MODIS surface reflectance daily\n");
        exit(SUCCESS);
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }


  // non-optional parameters
  args->n = 2;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->fimg, NPOW_10, argv[optind++]);
      copy_string(args->dout, NPOW_10, argv[optind++]);
    } else if (argc-optind < args->n){
      fprintf(stderr, "some non-optional arguments are missing.\n");
      usage(argv[0], FAILURE);
    } else if (argc-optind > args->n){
      fprintf(stderr, "too many non-optional arguments.\n");
      usage(argv[0], FAILURE);
    }
  } else {
    fprintf(stderr, "non-optional arguments are missing.\n");
    usage(argv[0], FAILURE);
  }

  return;
}


int get_modqa(ushort *modqa_, int index, int p, int bitfields){
int i;
short val = 0;


  for (i=0; i<bitfields; i++) val |= (short)(1 << i);

  return (short)(modqa_[p] >> index) & val;
}


void set_meta(brick_t *BRICK, date_t *date, double geotran[6], int nx, int ny, const char *proj, int tx, int ty, int sid, const char *sensor, const char *prd, const char *dout){
char fname[NPOW_10];
gdalopt_t format;
int nchar;


  nchar = snprintf(fname, NPOW_10, "%04d%02d%02d_LEVEL2_%s_%s", date->year, date->month, date->day, sensor, prd);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling output name name\n"); exit(FAILURE);}

  default_gdaloptions(_FMT_GTIFF_, &format);
  update_gdaloptions_blocksize(_FMT_GTIFF_, &format, nx, ny/10);

  set_brick_open(BRICK, 1);
  set_brick_format(BRICK, &format);
  set_brick_explode(BRICK, 0);

  set_brick_res(BRICK, geotran[1]);
  set_brick_ulx(BRICK, geotran[0]);
  set_brick_uly(BRICK, geotran[3]);
  set_brick_ncols(BRICK, nx);
  set_brick_nrows(BRICK, ny);
  set_brick_chunkncols(BRICK, nx);
  set_brick_chunknrows(BRICK, ny/10);
  set_brick_nchunks(BRICK, 10);
  set_brick_proj(BRICK, proj);
  set_brick_tilex(BRICK, tx);
  set_brick_tiley(BRICK, ty);

  set_brick_product(BRICK, prd);
  set_brick_sensorid(BRICK, sid);

  set_brick_name(BRICK, "FORCE Level 2 MODIS Import");
  set_brick_dirname(BRICK, dout);
  set_brick_filename(BRICK, fname);

  return;
}


void set_meta_band(brick_t *BRICK, int b, int scale, short nodata, const char *sensor, const char *domain, float wvl, const char *unit, date_t *date){


  set_brick_scale(BRICK, b, scale);
  set_brick_nodata(BRICK, b, nodata);
  set_brick_sensor(BRICK, b, sensor);
  set_brick_domain(BRICK, b, domain);
  set_brick_wavelength(BRICK, b, wvl);
  set_brick_unit(BRICK, b, unit);
  set_brick_date(BRICK, b, *date);
  set_brick_save(BRICK, b, 1);
  set_brick_bandname(BRICK, b, domain);

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


void compile_qai(brick_t *QAI, ushort *modqa_, short **boa_, int nc, int nb, int sid, short nodata){
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
    if (get_modqa(modqa_,  6, p, 2) >  2) set_aerosol(QAI, p, 2); // high aerosol
    if (get_modqa(modqa_,  8, p, 2) >  0) set_cloud(QAI, p, 3);   // cirrus
    if (get_modqa(modqa_, 10, p, 1) >  0) set_cloud(QAI, p, 2);   // internal cloud algo
    if (get_modqa(modqa_, 12, p, 1) >  0) set_snow(QAI, p, 1);    // snow/ice
    if (get_modqa(modqa_, 13, p, 1) >  0) set_cloud(QAI, p, 1);   // adjacent to cloud
    if (get_modqa(modqa_, 15, p, 1) >  0) set_snow(QAI, p, 1);    // internal snow algo

    for (b=0; b<nb; b++){
      if (b == 5 && sid == _SEN_MOD02_) continue;
      if (boa_[b][p] < 0){     set_subzero(QAI, p, 1);    break;}
      if (boa_[b][p] > 10000){ set_saturation(QAI, p, 1); break;}
    }

  }

  return;
}


int main ( int argc, char *argv[] ){
args_t args;

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
brick_t *BOA = NULL;
short **boa_ = NULL;
brick_t *QAI = NULL;
ushort *modqa_ = NULL;

char domain[7][NPOW_10] = { "BLUE", "GREEN", "RED", "NIR", "SWIR0", "SWIR1", "SWIR2" };
float wvl[7] = { 0.469, 0.555, 0.645, 0.858, 1.240, 1.640, 2.130 };
double geotran[6];
const char *proj_;
char proj[NPOW_10];
short nodata = -9999;


  parse_args(argc, argv, &args);


  // basename and extension
  basename_with_ext(args.fimg, binp, NPOW_10);
  extension2(args.fimg, ext, NPOW_10);

  // some tests on input parameters
  if (!fileexist(args.fimg)){
    printf("input file %s does not exist.\n\n", args.fimg); return FAILURE;}

  if (!fileexist(args.dout)){
    printf("output directory %s does not exist.\n\n", args.dout); return FAILURE;}

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
  
  nchar = snprintf(dtile, NPOW_10, "%s/X%04d_Y%04d", args.dout, tx, ty);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling output directory\n");return FAILURE;}

  if (!fileexist(dtile)) createdir(dtile);

  // open input dataset
  if ((fp = GDALOpen(args.fimg, GA_ReadOnly)) == NULL){
    printf("unable to open %s\n\n", args.fimg); return FAILURE;}

  // get SDS listing
  sds = GDALGetMetadata(fp, "SUBDATASETS");
  if (CSLCount(sds) == 0){
    printf("unable to retrieve SDS list from %s\n\n", args.fimg);return FAILURE;}
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
      BOA = allocate_brick(nb, nc, _DT_SHORT_);
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

  QAI = allocate_brick(1, nc, _DT_SHORT_);
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

  //print_brick_info(BOA);
  //print_brick_info(QAI);

  write_brick(BOA);
  write_brick(QAI);
  write_modcube(args.dout, proj);

  free_brick(BOA);
  free_brick(QAI);

  free((void*)modqa_);
  


  return SUCCESS;
}

