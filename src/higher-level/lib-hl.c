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
This file contains functions for library completeness testing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "lib-hl.h"


brick_t **compile_lib(ard_t *features, lib_t *lib, par_hl_t *phl, aux_lib_t *library, cube_t *cube, int *nproduct);
brick_t *compile_lib_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the bricks, in which LIB results are stored.
+++ It also sets metadata and sets pointers to instantly useable image
+++ arrays.
--- features: input features
--- lib:      pointer to instantly useable LIB image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for LIB results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_lib(ard_t *features, lib_t *lib, par_hl_t *phl, aux_lib_t *library, cube_t *cube, int *nproduct){
brick_t **LIB = NULL;
int b, o, nprod = 1;
int error = 0;
int nchar;
char bname[NPOW_10];
char domain[NPOW_10];
enum{ _mae_ };
int prodlen[1] ={ library->n + 1 };
char prodname[1][NPOW_02] ={ "MAE" };

int prodtype[1] ={ _mae_ };

bool enable[1] ={ true };

bool write[1]  ={ true };

short ***ptr[1] ={ &lib->mae_ };


  alloc((void**)&LIB, nprod, sizeof(brick_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((LIB[o] = compile_lib_brick(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(LIB[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      } else {
        for (b=0; b<prodlen[o]; b++){
          if (b < library->n){
            basename_without_ext(phl->lib.f_lib[b], bname, NPOW_10);
            if (strlen(bname) > NPOW_10-1){
              nchar = snprintf(domain, NPOW_10, "LIBRARY-%02d", b+1);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
            } else { 
              copy_string(domain, NPOW_10, bname);
            }
          } else {
            copy_string(domain, NPOW_10, "LIBRARY-SUMMARY");
          }
          set_brick_domain(LIB[o],   b, domain);
          set_brick_bandname(LIB[o], b, domain);
        }
      }
    } else{
      LIB[o]  = NULL;
      *ptr[o] = NULL;
    }
  }

  if (error > 0){
    printf("%d compiling LIB product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(LIB[o]);
    free((void*)LIB);
    return NULL;
  }

  *nproduct = nprod;
  return LIB;
}


/** This function compiles a LIB brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for LIB result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_lib_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;

  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Landscape Metrics");
  set_brick_product(brick, prodname);

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher,
    get_brick_tilex(brick), get_brick_tiley(brick));
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%s_HL_LSM_%s", phl->lib.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);


  if (write){
    set_brick_open(brick, OPEN_BLOCK);
  } else{
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


/** This function is the entry point to the library completeness testing
+++ module
--- features:  input features
--- mask:      mask image
--- nf:        number of features
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with TXT results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **library_completeness(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, aux_lib_t *library, cube_t *cube, int *nproduct){
lib_t lib;
brick_t **LIB;
small *mask_ = NULL;
int nprod = 0;
int l, s, f, p, nc;
short nodata;
double *newfeatures = NULL;
bool valid;
double mae, min_mae, min_mae_all, k;


  // import bricks
  nc = get_brick_chunkncells(features[0].DAT);
  nodata = get_brick_nodata(features[0].DAT, 0);

  // number of features okay?
  if (library->nf != nf){
    printf("number of features in library files is different from given features.\n");
    *nproduct = 0;
    return NULL;}

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask.");
      *nproduct = 0;
      return NULL;}
  }

  // compile products + bricks
  if ((LIB = compile_lib(features, &lib, phl, library, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile LIB products!\n");
    *nproduct = 0;
    return NULL;
  }

  
  #pragma omp parallel private(newfeatures,valid,mae,min_mae,min_mae_all,k,f,l,s) shared(lib,phl,features,library,mask_,nc,nf,nodata) default(none)
  {

    alloc((void**)&newfeatures, nf, sizeof(double));
    
    #pragma omp for
    for (p=0; p<nc; p++){
      
      for (l=0; l<=library->n; l++) lib.mae_[l][p] = nodata;
    
      if (mask_ != NULL && !mask_[p]) continue;

      for (f=0, valid=true; f<nf; f++){      
        if (!features[f].msk[p] && phl->ftr.exclude) valid = false;
      }
      if (!valid) continue;


      
      min_mae_all = SHRT_MAX;
      
      for (l=0; l<library->n; l++){
        
        min_mae = SHRT_MAX;

        for (f=0; f<nf; f++){
          if (phl->lib.rescale){
            newfeatures[f] = ((double)features[f].dat[0][p] - library->mean[l][f]) / library->sd[l][f];
          } else {
            newfeatures[f] = (double)features[f].dat[0][p];
          }
        }

        for (s=0; s<library->ns[l]; s++){
          
          mae = k = 0;

          for (f=0; f<nf; f++){
            
            if (!features[f].msk[p]) continue;

            mae += fabs(library->tab[l][s][f] - newfeatures[f]);
            k++;

          }
          
          if (k == 0) continue; else mae /= k;
          
          if (mae < min_mae) min_mae = mae;

        }
        
        if (min_mae < min_mae_all) min_mae_all = min_mae;
        if (min_mae < SHRT_MAX && phl->lib.rescale) min_mae *= 1000;
        if (min_mae < SHRT_MAX) lib.mae_[l][p] = (short)min_mae;

      }

      if (min_mae_all < SHRT_MAX && phl->lib.rescale) min_mae_all *= 1000;
      if (min_mae_all < SHRT_MAX) lib.mae_[library->n][p] = (short)min_mae_all;

    }
    
    
    free((void*)newfeatures);
    
  }

  *nproduct = nprod;
  return LIB;
}

