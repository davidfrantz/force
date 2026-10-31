/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric
Correction for Environmental monitoring.

Copyright (C) 2013-2026 David Frantz

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
+++++++++++++++++++++++++++++++ ++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 Gray-Level Co-occurrence Matrix (GLCM) header
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Copyright (C) 2013-2026 Katarzyna Ewa Lewinska, David Frantz
Contact: lewinska@hu-berlin.de
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

#include "glcm-hl.h"

// GLCM quantisation (same idea as n_grey / min_x / max_x in azvoleff/glcm)
#define GLCM_NLEVELS 32
#define GLCM_MIN    -10000
#define GLCM_MAX     10000

static inline int glcm_quantize(short value){
  int level = (int)(((double)value - GLCM_MIN) / (GLCM_MAX - GLCM_MIN) * GLCM_NLEVELS);
  if (level < 0) level = 0;
  if (level > GLCM_NLEVELS-1) level = GLCM_NLEVELS-1;
  return level;
}

brick_t **compile_glcm(ard_t *features, glcm_t *glcm, par_hl_t *phl, cube_t *cube, int *nproduct);
brick_t *compile_glcm_brick(brick_t *ard, int nb, bool write, char *prodname, par_hl_t *phl);


/** This function compiles the bricks, in which GLCM results are stored.
+++ It also sets metadata and sets pointers to instantly useable image
+++ arrays.
--- features: input features
--- glcm:      pointer to instantly useable GLCM image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for GLCM results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_glcm(ard_t *features, glcm_t *glcm, par_hl_t *phl, cube_t *cube, int *nproduct){
brick_t **GLCM = NULL;
int b, o, nprod = 5;
int error = 0;
int nchar;
char bname[NPOW_10];
char domain[NPOW_10];
enum{ _dis_, _hom_, _con_, _ent_, _uni_ };
int prodlen[5] ={ phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature, phl->ftr.nfeature };
char prodname[5][NPOW_10] ={ "DIS", "HOM", "CON", "ENT", "UNI" };

bool enable[5] ={ phl->glcm.odis, phl->glcm.ohom, phl->glcm.ocon, phl->glcm.oent, phl->glcm.ouni };
bool write[5]  ={ phl->glcm.odis, phl->glcm.ohom, phl->glcm.ocon, phl->glcm.oent, phl->glcm.ouni };

short ***ptr[5] ={ &glcm->dis_, &glcm->hom_, &glcm->con_, &glcm->ent_, &glcm->uni_ };


  alloc((void**)&GLCM, nprod, sizeof(brick_t*));

  for (o=0; o<nprod; o++){
    if (enable[o]){
      if ((GLCM[o] = compile_glcm_brick(features[0].DAT, prodlen[o], write[o], prodname[o], phl)) == NULL || (*ptr[o] = get_bands_short(GLCM[o])) == NULL){
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
          set_brick_domain(GLCM[o],   b, domain);
          set_brick_bandname(GLCM[o], b, domain);
        }
      }
    } else{
      GLCM[o]  = NULL;
      *ptr[o] = NULL;
    }
  }

  if (error > 0){
    printf("%d compiling GLCM product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(GLCM[o]);
    free((void*)GLCM);
    return NULL;
  }

  *nproduct = nprod;
  return GLCM;
}


/** This function compiles a GLCM brick
--- from:     brick from which most attributes are copied
--- nb:       number of bands in brick
--- write:    should this brick be written, or only used internally?
--- prodname: product name
--- phl:      HL parameters
+++ Return:   brick for GLCM result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_glcm_brick(brick_t *from, int nb, bool write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
char fname[NPOW_10];
char dname[NPOW_10];
char subname[NPOW_10];
int nchar;

  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE GLCM statistics");
  set_brick_product(brick, prodname);

  if (phl->subfolders){
    copy_string(subname, NPOW_10, prodname);
  } else {
    subname[0] = '\0';
  }

  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_higher,
    get_brick_tilex(brick), get_brick_tiley(brick), subname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%s_HL_GLCM_%s", phl->glcm.base, prodname);
  if (nchar < 0 || nchar >= NPOW_10){
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);

  if (write){
    set_brick_open(brick, OPEN_CHUNK);
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


/** This function is the entry point to the GLCM statistics
--- features: input features
--- mask:     mask image
--- nf:       number of features
--- phl:      HL parameters
--- cube:     datacube definition
--- nproduct: number of output bricks (returned)
+++ Return:   bricks with GLCM results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **glcm_statistics(ard_t *features, brick_t *mask, int nf, par_hl_t *phl, cube_t *cube, int *nproduct){
glcm_t glcm;
brick_t **GLCM;
small *mask_ = NULL;
int nprod = 0;
int f, nx, ny;
short nodata;

  printf("GLCM: entered, nf=%d, winsize=%d\n", nf, phl->glcm.winsize); fflush(stdout);

  cite_me(_CITE_GLCM_);

  // guard: window size must be an odd integer >= 3
  if (phl->glcm.winsize < 3){
    printf("Window size must be at least 3 (got %d).\n", phl->glcm.winsize);
    *nproduct = 0;
    return NULL;
  }
  if (phl->glcm.winsize % 2 == 0){
    printf("Window size must be odd so the window has a centre pixel (got %d).\n", phl->glcm.winsize);
    *nproduct = 0;
    return NULL;
  }

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
  if ((GLCM = compile_glcm(features, &glcm, phl, cube, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile GLCM products!\n");
    *nproduct = 0;
    return NULL;
  }


  for (f=0; f<nf; f++){

    #pragma omp parallel shared(ny,nx,f,nodata,glcm,mask_,features,phl) default(none)
    {
      static const int dr[4] = {0, 1, 1, 1};   // directions: 0°, 90°, 45°, 135°
      static const int dc[4] = {1, 0, 1, -1};

      int i, j, p, ni, nj, np, d, ni2, nj2, np2;
      int k, cell, gi, gj, ntouched = 0, glcm_count;
      int ws = phl->glcm.winsize;           // window size in pixels, e.g. 3 = 3x3
      int top, bottom, left, right;        // window bounds around the current pixel
      int *comat = NULL, *touched = NULL;  // per-thread matrix + list of used cells
      double diff, prob;
      double glcm_dis_sum, glcm_hom_sum, glcm_con_sum, glcm_ent, glcm_uni;

      if (phl->glcm.oent || phl->glcm.ouni){
        alloc((void**)&comat,   GLCM_NLEVELS*GLCM_NLEVELS, sizeof(int));
        alloc((void**)&touched, GLCM_NLEVELS*GLCM_NLEVELS, sizeof(int));
        memset(comat, 0, GLCM_NLEVELS*GLCM_NLEVELS*sizeof(int));
      }

      #pragma omp for collapse(2) schedule(guided)
      for (i=0; i<ny; i++){
      for (j=0; j<nx; j++){

        p = i*nx+j;

        // initialise selected products with nodata
        if (phl->glcm.odis) glcm.dis_[f][p] = nodata;
        if (phl->glcm.ohom) glcm.hom_[f][p] = nodata;
        if (phl->glcm.ocon) glcm.con_[f][p] = nodata;
        if (phl->glcm.oent) glcm.ent_[f][p] = nodata;
        if (phl->glcm.ouni) glcm.uni_[f][p] = nodata;

        if (mask_ != NULL && !mask_[p]) continue;
        if (phl->ftr.exclude && !features[f].msk[p]) continue;
        if (features[f].dat[0][p] == nodata) continue;

        // ws x ws window centred on (i,j)
        top    = i - ws/2;
        bottom = top + ws - 1;
        left   = j - ws/2;
        right  = left + ws - 1;

        glcm_dis_sum = glcm_hom_sum = glcm_con_sum = 0.0;
        glcm_count = 0;

        for (ni=top; ni<=bottom; ni++){
        for (nj=left; nj<=right; nj++){

          if (ni < 0 || nj < 0 || ni >= ny || nj >= nx) continue;

          np = ni*nx+nj;

          if (mask_ != NULL && !mask_[np]) continue;
          if (phl->ftr.exclude && !features[f].msk[np]) continue;
          if (features[f].dat[0][np] == nodata) continue;

          for (d=0; d<4; d++){

            ni2 = ni + dr[d];
            nj2 = nj + dc[d];

            // neighbour must stay inside the window
            if (ni2 < top || ni2 > bottom || nj2 < left || nj2 > right) continue;

            if (ni2 < 0 || nj2 < 0 || ni2 >= ny || nj2 >= nx) continue;

            np2 = ni2*nx+nj2;

            if (mask_ != NULL && !mask_[np2]) continue;
            if (phl->ftr.exclude && !features[f].msk[np2]) continue;
            if (features[f].dat[0][np2] == nodata) continue;

            // grey levels, not raw values
            gi = glcm_quantize(features[f].dat[0][np]);
            gj = glcm_quantize(features[f].dat[0][np2]);
            diff = (double)(gi - gj);

            glcm_dis_sum += fabs(diff);
            glcm_con_sum += diff * diff;
            glcm_hom_sum += 1.0 / (1.0 + diff * diff);

            // symmetric co-occurrence matrix (only touched cells are tracked)
            if (comat != NULL){
              cell = gi*GLCM_NLEVELS + gj;
              if (comat[cell] == 0) touched[ntouched++] = cell;
              comat[cell]++;
              cell = gj*GLCM_NLEVELS + gi;
              if (comat[cell] == 0) touched[ntouched++] = cell;
              comat[cell]++;
            }

            glcm_count++;
          }
        }
        }

        if (glcm_count > 0){

          // grey-level units: DIS 0..31, CON 0..961, HOM 0..1
          if (phl->glcm.odis) glcm.dis_[f][p] = (short)(glcm_dis_sum / glcm_count * 100.0);
          if (phl->glcm.ocon) glcm.con_[f][p] = (short)(glcm_con_sum / glcm_count * 10.0);
          if (phl->glcm.ohom) glcm.hom_[f][p] = (short)(glcm_hom_sum / glcm_count * 10000.0);

          if (comat != NULL){
            glcm_ent = glcm_uni = 0.0;

            for (k=0; k<ntouched; k++){
              cell = touched[k];
              prob = (double)comat[cell] / (2.0 * glcm_count);
              glcm_ent -= prob * log(prob);
              glcm_uni += prob * prob;
              comat[cell] = 0;            // reset for the next pixel
            }
            ntouched = 0;

            // max entropy = ln(32*32) = 6.93
            if (phl->glcm.oent) glcm.ent_[f][p] = (short)(glcm_ent * 1000.0);
            if (phl->glcm.ouni) glcm.uni_[f][p] = (short)(glcm_uni * 10000.0);
          }
        }
      }
      }

      free((void*)comat);      // free(NULL) is fine
      free((void*)touched);
    }

  }
  

  *nproduct = nprod;
  return GLCM;
}
