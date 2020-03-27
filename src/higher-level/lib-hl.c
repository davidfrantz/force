/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric
Correction for Environmental monitoring.

Copyright (C) 2013-2019 David Frantz

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


stack_t **compile_lib(ard_t *features, lib_t *lib, par_hl_t *phl, aux_lib_t *library, cube_t *cube, int *nproduct);
stack_t *compile_lib_stack(stack_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);

int test_objects(small *cld_, int nx, int ny, int **OBJ, int **SIZE, int *nobj);


/** This function compiles the stacks, in which LIB results are stored.
+++ It also sets metadata and sets pointers to instantly useable image
+++ arrays.
--- features: input features
--- lib:      pointer to instantly useable LIB image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output stacks (returned)
+++ Return:   stacks for LIB results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **compile_lib(ard_t *features, lib_t *lib, par_hl_t *phl, aux_lib_t *library, cube_t *cube, int *nproduct){
stack_t **LIB = NULL;
int o, nprod = 1;
int error = 0;
enum{ _mae_ };
int prodlen[1] ={ library->n + 1 };
char prodname[1][NPOW_02] ={ "MAE" };

int prodtype[1] ={ _mae_ };

bool enable[1] ={ true };

bool write[1]  ={ true };

short ***ptr[1] ={ &lib->mae_ };


  alloc((void**)&LIB, nprod, sizeof(stack_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((LIB[o] = compile_lib_stack(features[0].DAT, prodlen[prodtype[o]], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(LIB[o])) == NULL){
        printf("Error compiling %s product. ", prodname[o]); error++;
      }
    } else{
      LIB[o]  = NULL;
      *ptr[o] = NULL;
    }
  }

  if (error > 0){
    printf("%d compiling LIB product errors.\n", error);
    for (o=0; o<nprod; o++) free_stack(LIB[o]);
    free((void*)LIB);
    return NULL;
  }

  *nproduct = nprod;
  return LIB;
}


/** This function compiles a LIB stack
--- from:      stack from which most attributes are copied
--- nb:        number of bands in stack
--- write:     should this stack be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    stack for LIB result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *compile_lib_stack(stack_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
stack_t *stack = NULL;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;

  if ((stack = copy_stack(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_stack_name(stack, "FORCE Landscape Metrics");
  set_stack_product(stack, prodname);

  //printf("dirname should be assemlbed in write_stack, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher,
    get_stack_tilex(stack), get_stack_tiley(stack));
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_stack_dirname(stack, dname);

  nchar = snprintf(fname, NPOW_10, "%s_HL_LSM_%s", phl->lib.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_stack_filename(stack, fname);


  if (write){
    set_stack_open(stack, OPEN_BLOCK);
  } else{
    set_stack_open(stack, OPEN_FALSE);
  }
  set_stack_format(stack, phl->format);
  set_stack_explode(stack, phl->explode);
  set_stack_par(stack, phl->params->log);

  for (b=0; b<nb; b++){
    set_stack_save(stack, b, true);
  }

  return stack;
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
--- nproduct:  number of output stacks (returned)
+++ Return:    stacks with TXT results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t **library_completeness(ard_t *features, stack_t *mask, int nf, par_hl_t *phl, aux_lib_t *library, cube_t *cube, int *nproduct){
lib_t lib;
stack_t **LIB;
small *mask_ = NULL;
int nprod = 0;
int l, s, f, p, nc;
short nodata;
double *newfeatures = NULL;
bool valid;
double mae, min_mae, min_mae_all, k;


  // import stacks
  nc = get_stack_chunkncells(features[0].DAT);
  nodata = get_stack_nodata(features[0].DAT, 0);

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

  // compile products + stacks
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


      
      min_mae_all = INT_MAX;
      
      for (l=0; l<library->n; l++){
        
        min_mae = INT_MAX;

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
        if (min_mae < INT_MAX){
          if (phl->lib.rescale) min_mae *= 1000;
          lib.mae_[l][p] = (short)min_mae;
        }
        
      }

      if (min_mae_all < INT_MAX){
        if (phl->lib.rescale) min_mae_all *= 1000;
        lib.mae_[library->n][p] = (short)min_mae_all;
      }

    }
    
    
    free((void*)newfeatures);
    
  }

  *nproduct = nprod;
  return LIB;
}
