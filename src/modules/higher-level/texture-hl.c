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
This file contains functions for computing texture
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "texture-hl.h"

/** OpenCV **/
#include <opencv2/imgproc.hpp>
using namespace cv;


brick_t **compile_txt(ard_t *features, txt_t *txt, par_hl_t *phl, cube_t *cube, int *nproduct);
brick_t *compile_txt_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the bricks, in which TXT results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- features: input features
--- txt:      pointer to instantly useable TXT image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for TXT results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_txt(ard_t *features, txt_t *txt, par_hl_t *phl, cube_t *cube, int *nproduct){
brick_t **TXT = NULL;
int b, o, nprod = 7;
int error = 0;
int nchar;
char bname[NPOW_10];
char domain[NPOW_10];
enum { _ero_, _dil_, _opn_, _cls_, _grd_, _tht_, _bht_ };
int prodlen[7] = { phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature };
char prodname[7][NPOW_02] = { "ERO", "DIL", "OPN", "CLS", "GRD", "THT", "BHT" };
                             
int prodtype[7] = { _ero_, _dil_, _opn_, _cls_, _grd_, _tht_, _bht_ };

bool enable[7] = {  phl->txt.oero, phl->txt.odil, phl->txt.oopn, phl->txt.ocls, 
                    phl->txt.ogrd, phl->txt.otht, phl->txt.obht };

bool write[7]  = { phl->txt.oero, phl->txt.odil, phl->txt.oopn, phl->txt.ocls, 
                   phl->txt.ogrd, phl->txt.otht, phl->txt.obht };

short ***ptr[7] = { &txt->ero_, &txt->dil_, &txt->opn_, &txt->cls_, &txt->grd_, &txt->tht_, &txt->bht_ };


  alloc((void**)&TXT, nprod, sizeof(brick_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((TXT[o] = compile_txt_brick(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(TXT[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      } else {
        for (b=0; b<prodlen[o]; b++){
          basename_without_ext(phl->ftr.bname[b], bname, NPOW_10);
          if (strlen(bname) > NPOW_10-1){
            nchar = snprintf(domain, NPOW_10, "FEATURE-%04d", b+1);
            if (nchar < 0 || nchar >= NPOW_10){ 
              printf("Buffer Overflow in assembling domain\n"); error++;}
          } else { 
            nchar = snprintf(domain, NPOW_10, "%s_B%04d", bname, phl->ftr.band[b]);
            if (nchar < 0 || nchar >= NPOW_10){ 
              printf("Buffer Overflow in assembling domain\n"); error++;}
          }
          set_brick_domain(TXT[o],   b, domain);
          set_brick_bandname(TXT[o], b, domain);
        }
      }
    } else {
      TXT[o]  = NULL;
      *ptr[o] = NULL;
    }
  }


  if (error > 0){
    printf("%d compiling TXT product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(TXT[o]);
    free((void*)TXT);
    return NULL;
  }

  *nproduct = nprod;
  return TXT;
}


/** This function compiles a TXT brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for TXT result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_txt_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
char fname[NPOW_10];
char dname[NPOW_10];
char subname[NPOW_03];
int nchar;


  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Texture");
  set_brick_product(brick, prodname);

  if (phl->subfolders){
    copy_string(subname, NPOW_03, prodname);
  } else {
    subname[0] = '\0';
  }

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick), subname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%s_HL_TXT_%s", phl->txt.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);
  
  
  if (write){
    set_brick_open(brick, OPEN_BLOCK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, &phl->gdalopt);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
  }

  return brick;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the texture module
--- features:  input features
--- mask:      mask image
--- nf:        number of features
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with TXT results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **texture(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct){
txt_t txt;
brick_t **TXT;
small *mask_ = NULL;
int nprod = 0;
int f, i, j, p, nx, ny;
short nodata;


  // import bricks
  nx = get_brick_chunkncols(features[0].DAT);
  ny = get_brick_chunknrows(features[0].DAT);
  nodata = get_brick_nodata(features[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); 
      *nproduct = 0;
      return NULL;}
  }
  

  // compile products + bricks
  if ((TXT = compile_txt(features, &txt, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile TXT products!\n"); 
    *nproduct = 0;
    return NULL;
  }

  
  // structuring element (kernel)
  Mat morph_elem = getStructuringElement(MORPH_ELLIPSE,
                     Size(2*phl->txt.radius+1, 2*phl->txt.radius+1),
                     Point(phl->txt.radius, phl->txt.radius));


  #pragma omp parallel private(i,j,p) shared(nf,nx,ny,morph_elem,features,txt,phl,nodata) default(none)
  {

    #pragma omp for
    for (f=0; f<nf; f++){

      Mat src(ny, nx, CV_16S, features[f].dat[0], Mat::AUTO_STEP);
      Mat ero_dst(ny, nx, CV_16S);
      Mat dil_dst(ny, nx, CV_16S);
      Mat opn_dst(ny, nx, CV_16S);
      Mat cls_dst(ny, nx, CV_16S);
      Mat grd_dst(ny, nx, CV_16S);
      Mat tht_dst(ny, nx, CV_16S);
      Mat bht_dst(ny, nx, CV_16S);

      if (phl->txt.oero) morphologyEx(src, ero_dst, MORPH_ERODE,    morph_elem, Point(-1,-1), phl->txt.iter);
      if (phl->txt.odil) morphologyEx(src, dil_dst, MORPH_DILATE,   morph_elem, Point(-1,-1), phl->txt.iter);
      if (phl->txt.oopn) morphologyEx(src, opn_dst, MORPH_OPEN,     morph_elem, Point(-1,-1), phl->txt.iter);
      if (phl->txt.ocls) morphologyEx(src, cls_dst, MORPH_CLOSE,    morph_elem, Point(-1,-1), phl->txt.iter);
      if (phl->txt.ogrd) morphologyEx(src, grd_dst, MORPH_GRADIENT, morph_elem, Point(-1,-1), phl->txt.iter);
      if (phl->txt.otht) morphologyEx(src, tht_dst, MORPH_TOPHAT,   morph_elem, Point(-1,-1), phl->txt.iter);
      if (phl->txt.obht) morphologyEx(src, bht_dst, MORPH_BLACKHAT, morph_elem, Point(-1,-1), phl->txt.iter);

      for (i=0, p=0; i<ny; i++){
      for (j=0; j<nx; j++, p++){
        
        if (!features[f].msk[p] && phl->ftr.exclude){
          
          if (phl->txt.oero) txt.ero_[f][p] = nodata;
          if (phl->txt.odil) txt.dil_[f][p] = nodata;
          if (phl->txt.oopn) txt.opn_[f][p] = nodata;
          if (phl->txt.ocls) txt.cls_[f][p] = nodata;
          if (phl->txt.ogrd) txt.grd_[f][p] = nodata;
          if (phl->txt.otht) txt.tht_[f][p] = nodata;
          if (phl->txt.obht) txt.bht_[f][p] = nodata;
          
        } else {
          
          if (phl->txt.oero) txt.ero_[f][p] = ero_dst.at<short>(i,j);
          if (phl->txt.odil) txt.dil_[f][p] = dil_dst.at<short>(i,j);
          if (phl->txt.oopn) txt.opn_[f][p] = opn_dst.at<short>(i,j);
          if (phl->txt.ocls) txt.cls_[f][p] = cls_dst.at<short>(i,j);
          if (phl->txt.ogrd) txt.grd_[f][p] = grd_dst.at<short>(i,j);
          if (phl->txt.otht) txt.tht_[f][p] = tht_dst.at<short>(i,j);
          if (phl->txt.obht) txt.bht_[f][p] = bht_dst.at<short>(i,j);
          
        }
        
      }
      }

      src.release();
      ero_dst.release();
      dil_dst.release();
      opn_dst.release();
      cls_dst.release();
      grd_dst.release();
      tht_dst.release();
      bht_dst.release();

    }

  }

  morph_elem.release();

  *nproduct = nprod;
  return TXT;
}

