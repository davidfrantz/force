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
This file contains functions for Level 3 processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "plugin-hl.h"


typedef struct {
  int  prodlen;
  char **bandname;
  char prodname[NPOW_03];
  int  prodtype;
  int  enable;
  int  write;
  short ***ptr;
} brick_compile_info_t;

enum { _pyp_, _rpp_ };

brick_t *compile_plg_brick(brick_t *ard, brick_compile_info_t *info, par_hl_t *phl);
brick_t **compile_plg(ard_t *ard, plg_t *plg, par_hl_t *phl, cube_t *cube, int nt, int *nproduct);


int info_plg_pyp(brick_compile_info_t *info, int o, plg_t *plg, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "PYP");
  info[o].prodlen  = phl->plg.pyp.nb;
  info[o].bandname = phl->plg.pyp.bandname;
  info[o].prodtype = _pyp_;
  info[o].enable   = phl->plg.pyp.out;
  info[o].write    = phl->plg.pyp.out;
  info[o].ptr      = &plg->pyp_;

  return o+1;
}

int info_plg_rpp(brick_compile_info_t *info, int o, plg_t *plg, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "RPP");
  info[o].prodlen  = phl->plg.rpp.nb;
  info[o].bandname = phl->plg.rpp.bandname;
  info[o].prodtype = _rpp_;
  info[o].enable   = phl->plg.rpp.out;
  info[o].write    = phl->plg.rpp.out;
  info[o].ptr      = &plg->rpp_;

  return o+1;
}



/** This function compiles the bricks, in which PLG results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- plg:      pointer to instantly useable PLG image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nt:       number of ARD products over time
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for PLG results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_plg(ard_t *ard, plg_t *plg, par_hl_t *phl, cube_t *cube, int nt, int *nproduct){
brick_t **PLG = NULL;
int b;
date_t date;
int o = 0, nprod;
int error = 0;

brick_compile_info_t *info = NULL;


  nprod = 1 + // python plugin metrics
          1;  // R plugin metrics

  //printf("%d potential products.\n", nprod);

  alloc((void**)&info, nprod, sizeof(brick_compile_info_t));

  o = info_plg_pyp(info, o, plg, phl);
  o = info_plg_rpp(info, o, plg, phl);


  alloc((void**)&PLG, nprod, sizeof(brick_t*));

  //printf("scale, date, plg, bandnames, and sensor ID must be set in compile_plg!!!\n");

  
  for (o=0; o<nprod; o++){

    if (info[o].enable){

      if ((PLG[o] = compile_plg_brick(ard[0].DAT, &info[o], phl)) == NULL || (  *info[o].ptr = get_bands_short(PLG[o])) == NULL){
        printf("Error compiling %s product. ", info[o].prodname); error++;
      } else {

        init_date(&date);
        set_date(&date, 2000, 1, 1);

        for (b=0; b<info[o].prodlen; b++){
          set_brick_sensor(PLG[o],   b, "BLEND");
          set_brick_date(PLG[o], b, date);
        }

        print_brick_info(PLG[o]);

      }

    } else {
      PLG[o]  = NULL;
      *info[o].ptr = NULL;
    }

  }




  if (error > 0){
    printf("%d compiling PLG product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(PLG[o]);
    free((void*)PLG);
    return NULL;
  }

  free((void*)info);

  *nproduct = nprod;
  return PLG;
}


/** This function compiles a PLG brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for PLG result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_plg_brick(brick_t *from, brick_compile_info_t *info, par_hl_t *phl){
int b;
brick_t *brick = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
int nchar;


  if ((brick = copy_brick(from, info->prodlen, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Script Plugin");
  set_brick_product(brick, info->prodname);

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);

  nchar = snprintf(fname, NPOW_10, "%04d-%04d_%03d-%03d_HL_PLG_%s_%s", 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, 
    phl->doy_range[_MIN_], phl->doy_range[_MAX_], 
    phl->sen.target, info->prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);
  
  if (info->write){
    set_brick_open(brick, OPEN_BLOCK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, phl->format);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  for (b=0; b<info->prodlen; b++){
    set_brick_save(brick, b, true);
    set_brick_date(brick, b, date);
    set_brick_bandname(brick, b, info->bandname[b]);
    set_brick_domain(brick, b, info->bandname[b]);
  }

  return brick;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the plugin module
--- ard:       ARD
--- mask:      mask image
--- nt:        number of ARD products over time
--- phl:       HL parameters
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with PLG results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **script_plugin(ard_t *ard, brick_t *mask, int nt, par_hl_t *phl, cube_t *cube, int *nproduct){
plg_t plg;
brick_t **PLG;
small *mask_ = NULL;
int nprod = 0;
int nb, nx, ny, nc;
short nodata;


  // import bricks
  nx = get_brick_chunkncols(ard[0].DAT);
  ny = get_brick_chunknrows(ard[0].DAT);
  nc = get_brick_chunkncells(ard[0].DAT);
  nb = get_brick_nbands(ard[0].DAT);
  nodata = get_brick_nodata(ard[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return NULL;}
  }


  // compile products + bricks
  if ((PLG = compile_plg(ard, &plg, phl, cube, nt, &nprod)) == NULL || nprod == 0){
    printf("Unable to compile PLG products!\n"); 
    free((void*)PLG);
    *nproduct = 0;
    return NULL;
  }



  python_plugin(ard, NULL, &plg, mask_, nx, ny, nc, nb, nt, nodata, phl);
  //rstats_plugin(ard, NULL, &plg, mask_, nx, ny, nc, nb, nt, nodata, phl);


  *nproduct = nprod;
  return PLG;
}

