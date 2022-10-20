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
This file contains functions that define the datacubes
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "cube-ll.h"


int tile_level2(par_ll_t *pl2, cube_t *cube, brick_t **LEVEL2, int nprod);
int flush_level2(par_ll_t *pl2, meta_t *meta, brick_t **LEVEL2, int nprod);
multicube_t *start_datacube(par_ll_t *pl2, brick_t *brick);


/** This function tiles the image, computes tile cloud coverage and writes
+++ gridded images to disc.
--- pl2:    L2 parameters
--- cube:   data cube parameters
--- LEVEL2: L2 brick
--- nprod:  number of L2 products
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int tile_level2(par_ll_t *pl2, cube_t *cube, brick_t **LEVEL2, int nprod){
int i, j, p, np, nx, ny, b, nb, prod;
char dname[NPOW_10];
int nchar;
double geotran[6];
int tx, ty;  // tile id
double tulx, tuly; // ul of tile
double ulx, uly; // ul of image
int istart, jstart; // image-to-chip offset
int ntile = 0; // number of written tiles
int *tiles_x = NULL;
int *tiles_y = NULL;
int  tiles_k;
int err = 0;
bool empty;
short nodata;
brick_t **CUBED   = NULL;
short   **level2_ = NULL;
short   **cubed_  = NULL;
double ncld, ndata; // cloud cover
double scale, res;
int cube_nx, cube_ny, cube_nc;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  // get tile file and read it
  if (tile_readlist(pl2->f_tile, &tiles_x, &tiles_y, &tiles_k) != SUCCESS){
    printf("Reading tile file failed! "); return FAILURE;}

  // get dataset information
  get_brick_geotran(LEVEL2[0], geotran, 6);
  ulx = get_brick_ulx(LEVEL2[0]);
  uly = get_brick_uly(LEVEL2[0]);
  


  // intersect the image with the tile grid

  // initialize smaller cubed products
  alloc((void**)&CUBED, nprod, sizeof(brick_t*));

  for (prod=0; prod<(nprod); prod++){
    nb = get_brick_nbands(LEVEL2[prod]);
    res = get_brick_res(LEVEL2[prod]);
    scale = cube->res/res;
    CUBED[prod] = copy_brick(LEVEL2[prod], nb, _DT_NONE_);
    set_brick_geotran(CUBED[prod], geotran);
    set_brick_ncols(CUBED[prod], (int)(cube->nx*scale));
    set_brick_nrows(CUBED[prod], (int)(cube->ny*scale));
    set_brick_chunkncols(CUBED[prod], (int)(cube->cx*scale));
    set_brick_chunknrows(CUBED[prod], (int)(cube->cy*scale));
    allocate_brick_bands(CUBED[prod], nb, (int)(cube->nc*scale), _DT_SHORT_);
  }


//    #pragma omp for collapse(2) schedule(guided)
  for (ty=cube->tminy; ty<=cube->tmaxy; ty++){
  for (tx=cube->tminx; tx<=cube->tmaxx; tx++){

    // if tile is not allowlisted (if specified), skip
    if (tile_allowlisted(tiles_x, tiles_y, tiles_k, tx, ty) == FAILURE) continue;

    // upper left coordinate of current tile
    tulx = cube->origin_map.x + tx*cube->tilesize;
    tuly = cube->origin_map.y - ty*cube->tilesize;

    
    empty = true;
    ndata = ncld = 0.0;

    // copy to cubed products
    for (prod=0; prod<nprod; prod++){
      
      if (prod > 0 && empty) break;

      nb = get_brick_nbands(LEVEL2[prod]);
      nx  = get_brick_ncols(LEVEL2[prod]);
      ny  = get_brick_nrows(LEVEL2[prod]);
      res = get_brick_res(LEVEL2[prod]);
      scale = cube->res/res;
      cube_nx = (int)(cube->nx*scale);
      cube_ny = (int)(cube->ny*scale);
      cube_nc = (int)(cube->nc*scale);

      // image offset relative to current tile
      jstart = floor((tulx-ulx)/res); // changed from round to floor
      istart = floor((uly-tuly)/res); // changed from round to floor
      
      #ifdef FORCE_DEBUG
      printf("ul: %f/%f, offset: %d/%d\n", tulx, tuly, jstart, istart);
      #endif
      
      if ((level2_ = get_bands_short(LEVEL2[prod])) == NULL){ err++; continue;}
      if ((cubed_  = get_bands_short(CUBED[prod]))  == NULL){ err++; continue;}

      // init with nodata
      for (b=0; b<nb; b++){
        nodata = get_brick_nodata(LEVEL2[prod], b);
        for (p=0; p<cube_nc; p++) cubed_[b][p] = nodata;
      }

      // copy image to chip + compute tile cloud cover
      for (i=0, p=0; i<cube_ny; i++){
      for (j=0; j<cube_nx; j++, p++){

        np  = nx*(i+istart) + j+jstart; // image
        p = cube_nx*i+j;               // chip

        if (i+istart < 0 || i+istart >= ny || 
            j+jstart < 0 || j+jstart >= nx) continue;

        for (b=0; b<nb; b++) cubed_[b][p] = level2_[b][np];

        // count valid and cloudy pixels
        if (prod == 0){

          if (get_off(CUBED[prod], p)) continue;
          
          if (get_cloud(CUBED[prod], p) > 0 || get_shadow(CUBED[prod], p)) ncld++;
          ndata++;
          empty = false;

        }

      }
      }
      
    }


    // meteor cover of tile
    if (ndata > 0) ncld = ncld/ndata*100.0;


    #ifdef FORCE_DEBUG
    printf("tile X%04d_Y%04d: empty: %d, cloud cover: %03.0f%%\n", 
      tx, ty, empty, ncld);
    #endif

    if (ncld > pl2->maxtc) empty = true;


    // if there are data in tile -> output
    if (!empty){

      geotran[0] = tulx;
      geotran[3] = tuly;

      nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", cube->dname, tx, ty);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling dirname\n"); return FAILURE;}

      for (prod=0; prod<(nprod); prod++){
        set_brick_geotran(CUBED[prod], geotran);
        set_brick_parentname(CUBED[prod], cube->dname);
        set_brick_dirname(CUBED[prod], dname);
        if (write_brick(CUBED[prod]) == FAILURE){ err++; continue;}
      }

      ntile++;

    }

  }
  }
  

  if (err > 0){
    printf("error in cubing Level 2 products\n"); return FAILURE;}
  
  // print to stdout for logfile
  printf("%2d product(s) written. ", ntile);

  // clean
  free((void*)tiles_x); free((void*)tiles_y);
  for (prod=0; prod<nprod; prod++) free_brick(CUBED[prod]);
  free((void*)CUBED);


  #ifdef FORCE_CLOCK
  proctime_print("tiling output brick", TIME);
  #endif

  return SUCCESS;
}


/** This function is the alternative to tile_level2 and simply flushes the
+++ image to dics as it is.
--- pl2:    L2 parameters
--- meta:   metadata
--- LEVEL2: L2 brick
--- nprod:  number of L2 products
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int flush_level2(par_ll_t *pl2, meta_t *meta, brick_t **LEVEL2, int nprod){
int prod;
char dname[NPOW_10];
int nchar;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  nchar = snprintf(dname, NPOW_10, "%s/%s", pl2->d_level2, meta->refsys);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return FAILURE;}
  
  for (prod=0; prod<nprod; prod++){
    set_brick_parentname(LEVEL2[prod], pl2->d_level2);
    set_brick_dirname(LEVEL2[prod], dname);
    if (write_brick(LEVEL2[prod]) == FAILURE){
      printf("error flushing L2 products. \n"); return FAILURE;}
  }

  printf(" 1 product(s) written. ");


  #ifdef FORCE_CLOCK
  proctime_print("flushing output brick", TIME);
  #endif

  return SUCCESS;
}


/** This function compiles the data cube parameters from the information
+++ given in the parameter file.
--- pl2:    L2 parameters
--- brick:  input image brick
+++ Return: data cube parameters for one cube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
multicube_t *start_datacube(par_ll_t *pl2, brick_t *brick){
multicube_t *multicube = NULL;
cube_t *cube = NULL;
double tilex, tiley;
double utm_ulx, utm_uly, utm_lrx, utm_lry;
double ulx, uly, urx, ury, lrx, lry, llx, lly;
int t_ulx, t_uly, t_urx, t_ury, t_lrx, t_lry, t_llx, t_lly;
char utm_proj[NPOW_10];
double tol = 5e-3;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  
  
  //cube = allocate_datacube();
  multicube = allocate_multicube(1);
  multicube->cover[0] = true;
  cube = multicube->cube[0];

  copy_string(cube->dname, NPOW_10, pl2->d_level2);


  if (pl2->doreproj){

    cube->res = pl2->res;
    copy_string(cube->proj, NPOW_10, pl2->proj);
    
  } else {

    cube->res = get_brick_res(brick);
    get_brick_proj(brick, utm_proj, NPOW_10);
    copy_string(cube->proj, NPOW_10, utm_proj);
    
  }

  if (pl2->dotile){

    cube->tilesize  = pl2->tilesize;
    cube->chunksize = pl2->chunksize;

    if (fmod(cube->tilesize, cube->res) > tol){
      printf("TILE_SIZE must be a multiple of RESOLUTION. ");
      printf("If DO_REPROJ = FALSE, the image resolution overrides the RESOLUTION parameter. ");
      free_multicube(multicube);
      return NULL;
    }
    if (fmod(cube->chunksize, cube->res) > tol){
      printf("BLOCK_SIZE must be a multiple of RESOLUTION. ");
      printf("If DO_REPROJ = FALSE, the image resolution overrides the RESOLUTION parameter. ");
      free_multicube(multicube);
      return NULL;
    }
    if (fmod(cube->tilesize, cube->chunksize) > tol){
      printf("TILE_SIZE must be a multiple of BLOCK_SIZE. ");
      free_multicube(multicube);
      return NULL;
    }

    cube->nx = floor(cube->tilesize/cube->res);
    cube->ny = cube->nx;
    cube->nc = cube->nx*cube->ny;

    cube->cx = cube->nx;
    cube->cy = floor(cube->chunksize/cube->res);
    cube->cc = cube->cx*cube->cy;

    cube->origin_geo.x = pl2->orig_lon;
    cube->origin_geo.y = pl2->orig_lat;

    if ((warp_geo_to_any(cube->origin_geo.x,  cube->origin_geo.y,
                        &cube->origin_map.x, &cube->origin_map.y,
                         cube->proj)) == FAILURE){
      printf("Computing tile origin in dst_srs failed. ");
      free_multicube(multicube);
      return NULL;
    }

    if (brick != NULL){

      utm_ulx = get_brick_ulx(brick);
      utm_uly = get_brick_uly(brick);
      utm_lrx = utm_ulx + get_brick_width(brick);
      utm_lry = utm_uly - get_brick_height(brick);
      get_brick_proj(brick, utm_proj, NPOW_10);

      // UL
      if ((warp_any_to_any(utm_ulx, utm_uly, &ulx, &uly,
                          utm_proj, cube->proj)) == FAILURE){
        printf("computing upper left in dst_srs failed. ");
        free_multicube(multicube);
        return NULL;
      }

      // UR
      if ((warp_any_to_any(utm_lrx, utm_uly, &urx, &ury,
                          utm_proj, cube->proj)) == FAILURE){
        printf("computing upper left in dst_srs failed. ");
        free_multicube(multicube);
        return NULL;
      }
      
      // LR
      if ((warp_any_to_any(utm_lrx, utm_lry, &lrx, &lry,
                          utm_proj, cube->proj)) == FAILURE){
        printf("computing lower right in dst_srs failed. ");
        free_multicube(multicube);
        return NULL;
      }

      // LL
      if ((warp_any_to_any(utm_ulx, utm_lry, &llx, &lly,
                          utm_proj, cube->proj)) == FAILURE){
        printf("computing lower right in dst_srs failed. ");
        free_multicube(multicube);
        return NULL;
      }

      tile_find(ulx, uly, &tilex, &tiley, &t_ulx, &t_uly, cube);
      tile_find(urx, ury, &tilex, &tiley, &t_urx, &t_ury, cube);
      tile_find(lrx, lry, &tilex, &tiley, &t_lrx, &t_lry, cube);
      tile_find(llx, lly, &tilex, &tiley, &t_llx, &t_lly, cube);
      
      #ifdef FORCE_DEBUG
      printf("tile UL: %d %d\n", t_ulx, t_uly);
      printf("tile UR: %d %d\n", t_urx, t_ury);
      printf("tile LR: %d %d\n", t_lrx, t_lry);
      printf("tile LL: %d %d\n", t_llx, t_lly);
      #endif

      cube->tminx = INT_MAX;
      if (t_ulx < cube->tminx) cube->tminx = t_ulx;
      if (t_urx < cube->tminx) cube->tminx = t_urx;
      if (t_lrx < cube->tminx) cube->tminx = t_lrx;
      if (t_llx < cube->tminx) cube->tminx = t_llx;

      cube->tminy = INT_MAX;
      if (t_uly < cube->tminy) cube->tminy = t_uly;
      if (t_ury < cube->tminy) cube->tminy = t_ury;
      if (t_lry < cube->tminy) cube->tminy = t_lry;
      if (t_lly < cube->tminy) cube->tminy = t_lly;

      cube->tmaxx = INT_MIN;
      if (t_ulx > cube->tmaxx) cube->tmaxx = t_ulx;
      if (t_urx > cube->tmaxx) cube->tmaxx = t_urx;
      if (t_lrx > cube->tmaxx) cube->tmaxx = t_lrx;
      if (t_llx > cube->tmaxx) cube->tmaxx = t_llx;

      cube->tmaxy = INT_MIN;
      if (t_uly > cube->tmaxy) cube->tmaxy = t_uly;
      if (t_ury > cube->tmaxy) cube->tmaxy = t_ury;
      if (t_lry > cube->tmaxy) cube->tmaxy = t_lry;
      if (t_lly > cube->tmaxy) cube->tmaxy = t_lly;

      cube->tnx = cube->tmaxx-cube->tminx+1;
      cube->tny = cube->tmaxy-cube->tminy+1;
      cube->tnc = cube->tnx*cube->tny;

    }


    if (write_datacube_def(cube) == FAILURE){
      printf("Writing datacube definition failed. "); 
      free_multicube(multicube);
      return NULL;
    }

  } else {

    cube->tilesize  = 0;
    cube->chunksize = 0;

    cube->nx = cube->cx = 0;
    cube->ny = cube->cy = 0;
    cube->nc = cube->cc = 0;

    cube->origin_geo.x = 0;
    cube->origin_geo.y = 0;
    cube->origin_map.x = 0;
    cube->origin_map.y = 0;

    cube->tminx = 0;
    cube->tminy = 0;
    cube->tmaxx = 0;
    cube->tmaxy = 0;

    cube->tnx = 0;
    cube->tny = 0;
    cube->tnc = 0;

  }

  #ifdef FORCE_DEBUG
  print_datacube(cube);
  #endif


  #ifdef FORCE_CLOCK
  proctime_print("starting data cube", TIME);
  #endif

  return multicube;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function tiles/writes or writes-as-is the processed L2 image pro-
+++ ducts.
--- pl2:    L2 parameters
--- meta:   metadata
--- cube:   data cube parameters
--- LEVEL2: L2 brick
--- nprod:  number of L2 products
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int cube_level2(par_ll_t *pl2, meta_t *meta, cube_t *cube, brick_t **LEVEL2, int nprod){
int prod;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif
  

  /** tile or flush the data to disc
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  if (pl2->dotile){

    // tile the bricks
    if ((tile_level2(pl2, cube, LEVEL2, nprod)) != SUCCESS){
      printf("Tiling images failed! "); return FAILURE; }

  } else {

    // write bricks as they are
    if ((flush_level2(pl2, meta, LEVEL2, nprod)) != SUCCESS){
      printf("Flushing images to disc failed! "); return FAILURE; }

  }


  /** clean
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  for (prod=0; prod<nprod; prod++) free_brick(LEVEL2[prod]);
  free((void*)LEVEL2);


  #ifdef FORCE_CLOCK
  proctime_print("cubing Level 2 data", TIME);
  #endif

  return SUCCESS;
}


/** This function compiles the data cube parameters. It either compiles
+++ an individual datacube from the information given in the parameter 
+++ file - or predefined (EQUI7/GLANCE7) datacubes (can be multiple
+++ continental cubes) are compiled.
--- pl2:    L2 parameters
--- brick:  input image brick
+++ Return: data cube parameters for one cube
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
multicube_t *start_multicube(par_ll_t *pl2, brick_t *brick){
multicube_t *multicube = NULL;


  if (pl2->doreproj && 
     (strcmp(pl2->proj, "EQUI7")    == 0 ||
      strcmp(pl2->proj, "EQUI7-AF") == 0 ||
      strcmp(pl2->proj, "EQUI7-AN") == 0 ||
      strcmp(pl2->proj, "EQUI7-AS") == 0 ||
      strcmp(pl2->proj, "EQUI7-EU") == 0 ||
      strcmp(pl2->proj, "EQUI7-NA") == 0 ||
      strcmp(pl2->proj, "EQUI7-OC") == 0 ||
      strcmp(pl2->proj, "EQUI7-SA") == 0)){

    // initialize datacubes in EQUI7 specification
    if ((multicube = start_equi7cube(pl2, brick)) == NULL){
      printf("Starting EQUI7 datacubes failed.\n"); return NULL;}

  } else if (pl2->doreproj && 
     (strcmp(pl2->proj, "GLANCE7") == 0 ||
      strcmp(pl2->proj, "GLANCE7-AF") == 0 ||
      strcmp(pl2->proj, "GLANCE7-AN") == 0 ||
      strcmp(pl2->proj, "GLANCE7-AS") == 0 ||
      strcmp(pl2->proj, "GLANCE7-EU") == 0 ||
      strcmp(pl2->proj, "GLANCE7-NA") == 0 ||
      strcmp(pl2->proj, "GLANCE7-OC") == 0 ||
      strcmp(pl2->proj, "GLANCE7-SA") == 0)){
    
    // initialize datacubes in GLANCE7 specification
    if ((multicube = start_glance7cube(pl2, brick)) == NULL){
      printf("Starting GLANCE7 datacubes failed.\n"); return NULL;}

  } else {

    // initialize a single datacube (default)
    if ((multicube = start_datacube(pl2, brick)) == NULL){
      printf("Starting datacube failed.\n"); return NULL;}

  }
  
  return multicube;
}

