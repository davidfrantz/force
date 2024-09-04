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
This program inflates QAI layers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../../modules/cross-level/const-cl.h"
#include "../../modules/cross-level/utils-cl.h"
#include "../../modules/cross-level/string-cl.h"
#include "../../modules/cross-level/konami-cl.h"
#include "../../modules/cross-level/quality-cl.h"
#include "../../modules/higher-level/read-ard-hl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points


typedef struct {
  int n;
  char finp[NPOW_10];
  char dout[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] input-file output-dir\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'input-file': QAI file\n");
  printf("  - 'output-dir': Output directory for QIM files.'\n");
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
        get_version(NULL, 0);
        exit(SUCCESS);
      case 'i':
        printf("Inflate QAI bit layers\n");
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
      copy_string(args->finp, NPOW_10, argv[optind++]);
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


int main(int argc, char *argv[]){
args_t args;
double geotran[6];
char *pch   = NULL;
char oname[NPOW_10];
const char *proj;
GDALDatasetH fp;
brick_t *QAI = NULL;
brick_t *QIM = NULL;
small **qim_ = NULL;
int b, p;
cube_t *cube = NULL;


  parse_args(argc, argv, &args);

  GDALAllRegister();

  
  if ((cube = allocate_datacube()) == NULL){
    printf("unable to init cube\n"); return FAILURE;}

  if ((fp = GDALOpen(args.finp, GA_ReadOnly)) == NULL){
    printf("unable to open %s. ", args.finp); return FAILURE;}

  cube->nx = cube->cx = GDALGetRasterXSize(fp);
  cube->ny = cube->cy = GDALGetRasterYSize(fp);
  cube->nc = cube->nx*cube->ny;
  cube->cn = cube->cx*cube->cy;

  GDALGetGeoTransform(fp, geotran);
  cube->res = geotran[1];
  
  cube->tilesize  = cube->nx*cube->res;
  cube->chunksize = cube->ny*cube->res;

  proj = GDALGetProjectionRef(fp);
  copy_string(cube->proj, NPOW_10, proj);

  GDALClose(fp);
  

  if ((QAI = read_block(args.finp, _ARD_AUX_, NULL, 1, 1, 1, _DT_SHORT_, 0, 0, 0, cube, false, 0, 0)) == NULL){
      printf("Error reading QAI product %s\n", args.finp); return FAILURE;}


  QIM  = copy_brick(QAI, _QAI_FLAG_LENGTH_, _DT_SMALL_);
  if ((qim_ = get_bands_small(QIM)) == NULL){
    printf("Error getting QIM bands\n"); return FAILURE;}

  // output filename
  basename_without_ext(args.finp, oname, NPOW_10);
  if (strstr(oname, "_QAI")  != NULL) pch = strstr(oname, "_QAI");
  if (strstr(oname, "_INF")  != NULL) pch = strstr(oname, "_INF");
  if (pch == NULL){
    printf("Wrong product given. Give QAI or INF product\n"); return FAILURE;}
  strncpy(pch, "_QIM", 4);
  
  set_brick_dirname(QIM, args.dout);
  set_brick_filename(QIM, oname);
  set_brick_open(QIM, OPEN_CREATE);
  for (b=0; b<_QAI_FLAG_LENGTH_; b++) set_brick_save(QIM, b, true);

  set_brick_bandname(QIM, _QAI_FLAG_OFF_, "valid data");           
  set_brick_bandname(QIM, _QAI_FLAG_CLD_, "Cloud state");
  set_brick_bandname(QIM, _QAI_FLAG_SHD_, "Cloud shadow flag");    
  set_brick_bandname(QIM, _QAI_FLAG_SNW_, "Snow flag");
  set_brick_bandname(QIM, _QAI_FLAG_WTR_, "Water flag");           
  set_brick_bandname(QIM, _QAI_FLAG_AOD_, "Aerosol state");
  set_brick_bandname(QIM, _QAI_FLAG_SUB_, "Subzero flag");         
  set_brick_bandname(QIM, _QAI_FLAG_SAT_, "Saturation flag");
  set_brick_bandname(QIM, _QAI_FLAG_SUN_, "High sun zenith flag"); 
  set_brick_bandname(QIM, _QAI_FLAG_ILL_, "Illumination state");
  set_brick_bandname(QIM, _QAI_FLAG_SLP_, "Slope flag");           
  set_brick_bandname(QIM, _QAI_FLAG_WVP_, "Water vapor flag");
  
  for (b=0; b<_QAI_FLAG_LENGTH_; b++) set_brick_nodata(QIM, b, 255);

  #ifdef FORCE_DEBUG
  printf("valid data %d %d\n",           _QAI_FLAG_OFF_,  _QAI_BIT_OFF_);
  printf("Cloud state %d %d\n",          _QAI_FLAG_CLD_,  _QAI_BIT_CLD_);
  printf("Cloud shadow flag %d %d\n",    _QAI_FLAG_SHD_,  _QAI_BIT_SHD_);
  printf("Snow flag %d %d\n",            _QAI_FLAG_SNW_,  _QAI_BIT_SNW_);
  printf("Water flag %d %d\n",           _QAI_FLAG_WTR_,  _QAI_BIT_WTR_);
  printf("Aerosol state %d %d\n",        _QAI_FLAG_AOD_,  _QAI_BIT_AOD_);
  printf("Subzero flag %d %d\n",         _QAI_FLAG_SUB_,  _QAI_BIT_SUB_);
  printf("Saturation flag %d %d\n",      _QAI_FLAG_SAT_,  _QAI_BIT_SAT_);
  printf("High sun zenith flag %d %d\n", _QAI_FLAG_SUN_,  _QAI_BIT_SUN_);
  printf("Illumination state %d %d\n",   _QAI_FLAG_ILL_,  _QAI_BIT_ILL_);
  printf("Slope flag %d %d\n",           _QAI_FLAG_SLP_,  _QAI_BIT_SLP_);
  printf("Water vapor flag %d %d\n",     _QAI_FLAG_WVP_,  _QAI_BIT_WVP_);
  #endif

  
  #pragma omp parallel shared(cube, qim_, QAI) default(none)
  {
  
    #pragma omp for
    for (p=0; p<cube->nc; p++){
      qim_[_QAI_FLAG_OFF_][p] = get_off(QAI, p);
      qim_[_QAI_FLAG_CLD_][p] = get_cloud(QAI, p);
      qim_[_QAI_FLAG_SHD_][p] = get_shadow(QAI, p);
      qim_[_QAI_FLAG_SNW_][p] = get_snow(QAI, p);
      qim_[_QAI_FLAG_WTR_][p] = get_water(QAI, p);
      qim_[_QAI_FLAG_AOD_][p] = get_aerosol(QAI, p);
      qim_[_QAI_FLAG_SUB_][p] = get_subzero(QAI, p);
      qim_[_QAI_FLAG_SAT_][p] = get_saturation(QAI, p);
      qim_[_QAI_FLAG_SUN_][p] = get_lowsun(QAI, p);
      qim_[_QAI_FLAG_ILL_][p] = get_illumination(QAI, p);
      qim_[_QAI_FLAG_SLP_][p] = get_slope(QAI, p);
      qim_[_QAI_FLAG_WVP_][p] = get_vaporfill(QAI, p);
    }
    
  }


  write_brick(QIM);

  free_brick(QAI);
  free_brick(QIM);
  free_datacube(cube);


  return SUCCESS;
}

  