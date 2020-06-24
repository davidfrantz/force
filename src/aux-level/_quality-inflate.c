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
This program inflates QAI layers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/quality-cl.h"
#include "../higher-level/read-ard-hl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points


int main( int argc, char *argv[] ){
double geotran[6];
char iname[NPOW_10];
char oname[NPOW_10];
char d_out[NPOW_10];
char *pch = NULL;
const char *proj;
GDALDatasetH fp;
stack_t *QAI = NULL;
stack_t *QIM = NULL;
small **qim_ = NULL;
int b, p;
cube_t *cube = NULL;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 3){ printf("Usage: %s QAI dir\n\n", argv[0]); exit(1);}

  // parse arguments
  copy_string(iname, NPOW_10, argv[1]);
  copy_string(d_out, NPOW_10, argv[2]);

  GDALAllRegister();

  
  if ((cube = allocate_datacube()) == NULL){
    printf("unable to init cube\n"); return FAILURE;}

  if ((fp = GDALOpen(iname, GA_ReadOnly)) == NULL){
    printf("unable to open %s. ", iname); return FAILURE;}

  cube->nx = cube->cx = GDALGetRasterXSize(fp);
  cube->ny = cube->cy = GDALGetRasterYSize(fp);
  cube->nc = cube->nx*cube->ny;
  cube->cn = cube->cx*cube->cy;

  GDALGetGeoTransform(fp, geotran);
  cube->res = geotran[1];
  
  cube->tilesize  = cube->nx*cube->res;
  cube->chunksize = cube->ny*cube->res;

  proj = GDALGetProjectionRef(fp);
  if (strlen(proj) > NPOW_10-1){
    printf("cannot copy, string too long.\n"); return FAILURE;
  } else { strncpy(cube->proj, proj, strlen(proj)); cube->proj[strlen(proj)] = '\0';}

  GDALClose(fp);
  

  if ((QAI = read_block(iname, _ARD_AUX_, NULL, 1, 1, 1, _DT_SHORT_, 0, 0, 0, cube, false, 0, 0)) == NULL){
      printf("Error reading QAI product %s\n", iname); return FAILURE;}


  QIM  = copy_stack(QAI, _QAI_FLAG_LENGTH_, _DT_SMALL_);
  if ((qim_ = get_bands_small(QIM)) == NULL){
    printf("Error getting QIM bands\n"); return FAILURE;}

  // output filename
  basename_without_ext(iname, oname, NPOW_10);
  if (strstr(oname, "_QAI")  != NULL) pch = strstr(oname, "_QAI");
  if (strstr(oname, "_INF")  != NULL) pch = strstr(oname, "_INF");
  if (pch == NULL){
    printf("Wrong product given. Give QAI or INF product\n"); return FAILURE;}
  strncpy(pch, "_QIM", 4);
  
  set_stack_dirname(QIM, d_out);
  set_stack_filename(QIM, oname);
  set_stack_open(QIM, OPEN_CREATE);
  for (b=0; b<_QAI_FLAG_LENGTH_; b++) set_stack_save(QIM, b, true);

  set_stack_bandname(QIM, _QAI_FLAG_OFF_, "valid data");           
  set_stack_bandname(QIM, _QAI_FLAG_CLD_, "Cloud state");
  set_stack_bandname(QIM, _QAI_FLAG_SHD_, "Cloud shadow flag");    
  set_stack_bandname(QIM, _QAI_FLAG_SNW_, "Snow flag");
  set_stack_bandname(QIM, _QAI_FLAG_WTR_, "Water flag");           
  set_stack_bandname(QIM, _QAI_FLAG_AOD_, "Aerosol state");
  set_stack_bandname(QIM, _QAI_FLAG_SUB_, "Subzero flag");         
  set_stack_bandname(QIM, _QAI_FLAG_SAT_, "Saturation flag");
  set_stack_bandname(QIM, _QAI_FLAG_SUN_, "High sun zenith flag"); 
  set_stack_bandname(QIM, _QAI_FLAG_ILL_, "Illumination state");
  set_stack_bandname(QIM, _QAI_FLAG_SLP_, "Slope flag");           
  set_stack_bandname(QIM, _QAI_FLAG_WVP_, "Water vapor flag");
  
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


  write_stack(QIM);

  free_stack(QAI);
  free_stack(QIM);
  free_datacube(cube);


  return SUCCESS;
}

  