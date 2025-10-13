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
This file contains functions for organizing bricks in memory, and output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "brick_base-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_multiproc.h"  // CPL Multi-Threading
#include "gdalwarper.h"     // GDAL warper related entry points and defs

#ifdef __cplusplus
#include "ogr_spatialref.h" // coordinate systems services
#endif


/** This function allocates a brick
--- nb:       number of bands
--- nc:       number of cells
--- datatype: datatype
+++ Return:   brick (must be freed with free_brick)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *allocate_brick(int nb, int nc, int datatype){
brick_t *brick = NULL;


  if (nb < 1){
    printf("cannot allocate %d-band brick.\n", nb); 
    return NULL;}

  if (nc < 1 && datatype != _DT_NONE_){
    printf("cannot allocate %d-cell brick.\n", nc);
    return NULL;}

  alloc((void**)&brick, 1, sizeof(brick_t));

  init_brick(brick);    
  set_brick_nbands(brick, nb);

  alloc((void**)&brick->save,       nb, sizeof(bool));
  alloc((void**)&brick->nodata,     nb, sizeof(int));
  alloc((void**)&brick->scale,      nb, sizeof(float));
  alloc((void**)&brick->wavelength, nb, sizeof(float));
  alloc_2D((void***)&brick->unit, nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&brick->domain, nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&brick->bandname,   nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&brick->sensor,     nb, NPOW_10, sizeof(char));
  alloc((void**)&brick->date, nb, sizeof(date_t));
  
  init_brick_bands(brick);

  if (allocate_brick_bands(brick, nb, nc, datatype) == FAILURE){
    printf("couldn't allocate bands.\n"); return NULL;}

  return brick;
}


/** This function re-allocates a brick
--- brick:  brick (modified)
--- nb:     number of bands (new)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int reallocate_brick(brick_t *brick, int nb){
int b;
int nb0 = get_brick_nbands(brick);
int nc = get_brick_ncells(brick);
int datatype = get_brick_datatype(brick);

  if (nb == nb0) return SUCCESS;

  if (nb < 1){
    printf("cannot reallocate %d-band brick.\n", nb); 
    return FAILURE;}

  if (nc < 1){
    printf("cannot reallocate %d-cell brick.\n", nc);
    return FAILURE;}

  if (datatype == _DT_NONE_){
    printf("cannot reallocate brick with no datatype.\n");
    return FAILURE;}

  re_alloc((void**)&brick->save,         nb0, nb, sizeof(bool));
  re_alloc((void**)&brick->nodata,       nb0, nb, sizeof(int));
  re_alloc((void**)&brick->scale,        nb0, nb, sizeof(float));
  re_alloc((void**)&brick->wavelength,   nb0, nb, sizeof(float));
  re_alloc_2D((void***)&brick->unit,   nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&brick->domain,   nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&brick->bandname, nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&brick->sensor,   nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc((void**)&brick->date,         nb0, nb, sizeof(date_t));

  if (reallocate_brick_bands(brick, nb) == FAILURE){
    printf("couldn't reallocate bands.\n"); return FAILURE;}

  if (nb > nb0){
    for (b=nb0; b<nb; b++) copy_brick_band(brick, b, brick, 0);
  }

  set_brick_nbands(brick, nb);

  return SUCCESS;
}


/** This function copies a brick
--- from:     source brick
--- nb:       number of bands (new)
--- datatype: datatype (new)
+++ Return:   new brick (must be freed with free_brick)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *copy_brick(brick_t *from, int nb, int datatype){
brick_t *brick = NULL; 
int b, p;

  if (from->chunk[_X_] < 0 || from->chunk[_Y_] < 0){
    if ((brick = allocate_brick(nb, from->nc, datatype)) == NULL) return NULL;
  } else {
    if ((brick = allocate_brick(nb, from->cc, datatype)) == NULL) return NULL;
  }


  set_brick_name(brick, from->name);
  set_brick_product(brick, from->product);
  set_brick_dirname(brick, from->dname);
  set_brick_filename(brick, from->fname);
  set_brick_sensorid(brick, from->sid);

  set_brick_provdir(brick, from->provdir);
  set_brick_nprovenance(brick, from->nprovenance);
  for (p=0; p<from->nprovenance; p++){
    set_brick_provenance(brick, p, from->provenance[p]);
  }

  set_brick_geotran(brick, from->geotran);
  set_brick_nbands(brick, nb);
  set_brick_ncols(brick, from->nx);
  set_brick_nrows(brick, from->ny);
  set_brick_chunkncols(brick, from->cx);
  set_brick_chunknrows(brick, from->cy);
  set_brick_chunkwidth(brick, from->cwidth);
  set_brick_chunkheight(brick, from->cheight);
  set_brick_chunk_dim(brick, &from->dim_chunk);
  set_brick_chunkx(brick, from->chunk[_X_]);
  set_brick_chunky(brick, from->chunk[_Y_]);
  set_brick_tilex(brick, from->tile[_X_]);
  set_brick_tiley(brick, from->tile[_Y_]);
  set_brick_proj(brick, from->proj);
  set_brick_par(brick, from->par);

  set_brick_format(brick, &from->format);
  set_brick_open(brick, from->open);
  set_brick_explode(brick, from->explode);

  if (nb == from->nb){
    for (b=0; b<nb; b++) copy_brick_band(brick, b, from, b);
  } else {
    for (b=0; b<nb; b++) copy_brick_band(brick, b, from, 0);
  }

  return brick;
}


/** This function frees a brick
--- brick:  brick
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_brick(brick_t *brick){
int nb;

  if (brick == NULL) return;
  
  nb = get_brick_nbands(brick);
  
  if (brick->save        != NULL) free((void*)brick->save);
  if (brick->nodata      != NULL) free((void*)brick->nodata);
  if (brick->scale       != NULL) free((void*)brick->scale);
  if (brick->wavelength  != NULL) free((void*)brick->wavelength);
  if (brick->date        != NULL) free((void*)brick->date);
  brick->save        = NULL;
  brick->nodata      = NULL;
  brick->scale       = NULL;
  brick->wavelength  = NULL;
  brick->date        = NULL;

  if (brick->provenance != NULL) free_2D((void**)brick->provenance, brick->nprovenance);
  brick->provenance = NULL;

  if (brick->unit     != NULL) free_2D((void**)brick->unit,     nb);
  if (brick->domain   != NULL) free_2D((void**)brick->domain,   nb);
  if (brick->bandname != NULL) free_2D((void**)brick->bandname, nb);
  if (brick->sensor   != NULL) free_2D((void**)brick->sensor,   nb);
  brick->unit     = NULL;
  brick->domain   = NULL;
  brick->bandname = NULL;
  brick->sensor   = NULL;
  
  free_brick_bands(brick);

  free((void*)brick);
  brick = NULL;

  return;
}


/** This function allocates the bandwise information in a brick
--- brick:    brick (modified)
--- nb:       number of bands
--- nc:       number of cells
--- datatype: datatype
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int allocate_brick_bands(brick_t *brick, int nb, int nc, int datatype){
int nbyte;


  switch (datatype){
    case _DT_NONE_:
      set_brick_datatype(brick, _DT_NONE_);
      set_brick_byte(brick, (nbyte = 0));
      return SUCCESS;
    case _DT_SHORT_:
      set_brick_datatype(brick, _DT_SHORT_);
      set_brick_byte(brick, (nbyte = sizeof(short)));
      alloc_2D((void***)&brick->vshort, nb, nc, nbyte);
      break;
    case _DT_SMALL_:
      set_brick_datatype(brick, _DT_SMALL_);
      set_brick_byte(brick, (nbyte = sizeof(small)));
      alloc_2D((void***)&brick->vsmall, nb, nc, nbyte);
      break;
    case _DT_FLOAT_:
      set_brick_datatype(brick, _DT_FLOAT_);
      set_brick_byte(brick, (nbyte = sizeof(float)));
      alloc_2D((void***)&brick->vfloat, nb, nc, nbyte);
      break;
    case _DT_INT_:
      set_brick_datatype(brick, _DT_INT_);
      set_brick_byte(brick, (nbyte = sizeof(int)));
      alloc_2D((void***)&brick->vint, nb, nc, nbyte);
      break;
    case _DT_USHORT_:
      set_brick_datatype(brick, _DT_USHORT_);
      set_brick_byte(brick, (nbyte = sizeof(ushort)));
      alloc_2D((void***)&brick->vushort, nb, nc, nbyte);
      break;
    default:
      printf("unknown datatype for allocating brick. ");
      return FAILURE;
  }

  return SUCCESS;
}


/** This function re-allocates the bandwise information in a brick
--- brick:    brick (modified)
--- nb:       number of bands (new)
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int reallocate_brick_bands(brick_t *brick, int nb){
int nbyte = get_brick_byte(brick);
int nb0 = get_brick_nbands(brick);
int nc0 = get_brick_ncells(brick);
int nc  = get_brick_ncells(brick);
int datatype = get_brick_datatype(brick);


  switch (datatype){
    case _DT_SHORT_:
      re_alloc_2D((void***)&brick->vshort, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_SMALL_:
      re_alloc_2D((void***)&brick->vsmall, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_FLOAT_:
      re_alloc_2D((void***)&brick->vfloat, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_INT_:
      re_alloc_2D((void***)&brick->vint, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_USHORT_:
      re_alloc_2D((void***)&brick->vushort, nb0, nc0, nb, nc, nbyte);
      break;
    default:
      printf("unknown datatype for allocating brick. ");
      return FAILURE;
  }

  return SUCCESS;
}


/** This function copies a bandwise in a brick
--- brick:  target brick (modified)
--- b:      target band
--- from:   source brick
--- b_from: source band
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void copy_brick_band(brick_t *brick, int b, brick_t *from, int b_from){
  
  
  set_brick_save(brick, b, from->save[b_from]);
  set_brick_nodata(brick, b, from->nodata[b_from]);
  set_brick_scale(brick, b, from->scale[b_from]);
  set_brick_wavelength(brick, b, from->wavelength[b_from]);
  set_brick_unit(brick, b, from->unit[b_from]);
  set_brick_domain(brick, b, from->domain[b_from]);
  set_brick_bandname(brick, b, from->bandname[b_from]);
  set_brick_sensor(brick, b, from->sensor[b_from]);
  set_brick_date(brick, b, from->date[b_from]);

  return;
}


/** This function frees the bandwise information in a brick
--- brick:  brick
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_brick_bands(brick_t *brick){
int nb;

  if (brick == NULL) return;
  
  nb = get_brick_nbands(brick);

  if (brick->vshort  != NULL) free_2D((void**)brick->vshort,  nb); 
  if (brick->vsmall  != NULL) free_2D((void**)brick->vsmall,  nb); 
  if (brick->vfloat  != NULL) free_2D((void**)brick->vfloat,  nb); 
  if (brick->vint    != NULL) free_2D((void**)brick->vint,    nb); 
  if (brick->vushort != NULL) free_2D((void**)brick->vushort, nb); 
  
  brick->vshort  = NULL;
  brick->vsmall  = NULL;  
  brick->vfloat  = NULL;  
  brick->vint    = NULL;  
  brick->vushort = NULL;  

  return;
}


/** This function crops a brick. The cropping radius is given in projection
+++ units, and the corresponding number of pixels is removed from each side
+++ of the image. The input brick is freed within. The cropped brick is 
+++ returned.
--- from:     source brick (freed)
--- radius:   cropping radius in projection units
+++ Return:   cropped brick (must be freed with free_brick)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *crop_brick(brick_t *from, double radius){
brick_t *brick = NULL; 
int b, nb;
int pix;
double resolution;
int nx,  ny;
int nx_, ny_, nc_;
int i, j, p, p_;
int datatype;


  if (from == NULL) return NULL;
  
  if (radius <= 0){
    printf("negative radius. cannot crop.");
    free_brick(from);
    return NULL;
  }

  nb = get_brick_nbands(from);
  resolution = get_brick_res(from);
  datatype = get_brick_datatype(from);
  
  brick = copy_brick(from, nb, _DT_NONE_);

  if (from->chunk[_X_] < 0){
    nx = get_brick_ncols(from);
  } else {
    nx = get_brick_chunkncols(from);
  }

  if (from->chunk[_Y_] < 0){
    ny = get_brick_nrows(from);
  } else {
    ny = get_brick_chunknrows(from);
  }

  pix = (int)(radius/resolution);
  nx_ = nx - 2*pix;
  ny_ = ny - 2*pix;
  nc_ = nx_*ny_;

  if (from->chunk[_X_] < 0){
    set_brick_ncols(brick, nx_);
  } else {
    set_brick_chunkncols(brick, nx_);
  }

  if (from->chunk[_Y_] < 0){
    set_brick_nrows(brick, ny_);
  } else {
    set_brick_chunknrows(brick, ny_);
  }
  
  allocate_brick_bands(brick, nb, nc_, datatype);
  
  #ifdef FORCE_DEBUG
  int nc;
  if (from->chunk[_X_] < 0){
    nc = get_brick_ncols(from);
  } else {
    nc = get_brick_chunkncols(from);
  }
  if (from->chunk[_Y_] < 0){
    nc *= get_brick_nrows(from);
  } else {
    nc *= get_brick_chunknrows(from);
  }
  printf("cropping %d -> %d cols\n", nx, nx_);
  printf("cropping %d -> %d rows\n", ny, ny_);
  printf("cropping %d -> %d pixels\n", nc, nc_);
  #endif

  switch (datatype){
    case _DT_NONE_:
      free_brick(from);
      return brick;
    case _DT_SHORT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p_ = i*nx_+j;
        p  = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) brick->vshort[b][p_] = from->vshort[b][p];
      }
      }
      break;
    case _DT_SMALL_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) brick->vsmall[b][p_] = from->vsmall[b][p];
      }
      }
      break;
    case _DT_FLOAT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) brick->vfloat[b][p_] = from->vfloat[b][p];
      }
      }
      break;
    case _DT_INT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) brick->vint[b][p_] = from->vint[b][p];
      }
      }
      break;
    case _DT_USHORT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) brick->vushort[b][p_] = from->vushort[b][p];
      }
      }
      break;
    default:
      printf("unknown datatype. ");
      free_brick(brick);
      free_brick(from);
      return NULL;
  }

  free_brick(from);

  return brick;
}


/** This function initializes all values in a brick
--- brick:  brick
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_brick(brick_t *brick){


  copy_string(brick->name,      NPOW_10, "NA");
  copy_string(brick->product,   NPOW_10, "NA");
  copy_string(brick->dname,     NPOW_10, "NA");
  copy_string(brick->fname,     NPOW_10, "NA");
  copy_string(brick->provdir,   NPOW_10, "NA");

  brick->nprovenance = 0;
  brick->provenance  = NULL;

  brick->sid = -1;
  default_gdaloptions(_FMT_GTIFF_, &brick->format);
  brick->open = OPEN_FALSE;
  brick->explode = 0;
  brick->datatype = _DT_NONE_;
  brick->byte = 0;

  brick->nb =  0;
  brick->nx =  0;
  brick->ny =  0;
  brick->nc =  0;
  brick->cx =  0;
  brick->cy =  0;
  brick->cc =  0;
  memset(&brick->geotran, 0, _GT_LEN_*sizeof(double));
  brick->width  = 0;
  brick->height = 0;
  brick->cwidth  = 0;
  brick->cheight = 0;
  brick->chunk[_X_] = -1;
  brick->chunk[_Y_] = -1;
  memset(&brick->dim_chunk, 0, sizeof(dim_t));
  brick->tile[_X_] = 0;
  brick->tile[_Y_] = 0;

  copy_string(brick->proj,NPOW_10, "NA");
  copy_string(brick->par, NPOW_14, "NA");

  brick->save   = NULL;
  brick->nodata = NULL;
  brick->scale  = NULL;

  brick->wavelength = NULL;
  brick->unit = NULL;
  brick->domain = NULL;
  brick->bandname   = NULL;
  brick->sensor     = NULL;
  brick->date       = NULL;

  brick->vshort  = NULL;
  brick->vfloat  = NULL;
  brick->vint    = NULL;
  brick->vushort = NULL;
  brick->vsmall   = NULL;

  return;  
}


/** This function initializes all bandwise information in a brick
--- brick:  brick
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_brick_bands(brick_t *brick){
int b;

  for (b=0; b<brick->nb; b++){
    brick->save[b] = false;
    brick->nodata[b] = 0;
    brick->scale[b] = 0;
    brick->wavelength[b] = 0;
    copy_string(brick->unit[b],     NPOW_10, "NA");
    copy_string(brick->domain[b],   NPOW_10, "NA");
    copy_string(brick->bandname[b], NPOW_10, "NA");
    copy_string(brick->sensor[b],   NPOW_10, "NA");
    init_date(&brick->date[b]);
  }

  return;
}


/** This function prints a brick
--- brick:  brick
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_brick_info(brick_t *brick){
int b, p;


  printf("\nbrick info for %s - %s - SID %d\n", brick->name, brick->product, brick->sid);
  printf("open: %d, explode %d\n", 
    brick->open, brick->explode);
  print_gdaloptions(&brick->format);
  printf("datatype %d with %d bytes\n", 
    brick->datatype, brick->byte);
  printf("filename: %s/%s\n", brick->dname, brick->fname);
  for (p=0; p<brick->nprovenance; p++) printf("input #%04d: %s\n", brick->nprovenance+1, brick->provenance[p]);
  printf("nx: %d, ny: %d, nc: %d, res: %.3f, nb: %d\n", 
    brick->nx, brick->ny, brick->nc, 
    brick->geotran[_GT_RES_], brick->nb);
  printf("width: %.1f, height: %.1f\n", 
    brick->width, brick->height);
  printf("chunking: nx: %d, ny: %d, nc: %d, width: %.1f, height: %.1f, #: %d x %d = %d\n", 
    brick->cx, brick->cy, brick->cc, brick->cwidth, brick->cheight, 
    brick->dim_chunk.cols, brick->dim_chunk.rows, brick->dim_chunk.cells);
  printf("active chunk: X:%d, Y:%d, tile X%04d_Y%04d\n", 
    brick->chunk[_X_], brick->chunk[_Y_], brick->tile[_X_], brick->tile[_Y_]);
  printf("ulx: %.3f, uly: %.3f\n", 
    brick->geotran[_GT_ULX_], brick->geotran[_GT_ULY_]);
  printf("geotran: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", 
    brick->geotran[_GT_ULX_], brick->geotran[_GT_XRES_],
    brick->geotran[_GT_XROT_], brick->geotran[_GT_ULY_],
    brick->geotran[_GT_YROT_], brick->geotran[_GT_YRES_]);
  printf("proj: %s\n", brick->proj);
  printf("par: %s\n", brick->par);

  for (b=0; b<brick->nb; b++) print_brick_band_info(brick, b);

  printf("\n");

  return;
}


/** This function prints bandwise information in a brick
--- brick:  brick
--- b:      band
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_brick_band_info(brick_t *brick, int b){
  
  printf("++band # %d - save %d, nodata: %d, scale: %f\n", 
    b, brick->save[b], brick->nodata[b], brick->scale[b]);
  printf("wvl: %f, domain: %s, band name: %s, sensor ID: %s\n", 
    brick->wavelength[b], brick->domain[b], brick->bandname[b], brick->sensor[b]);
  print_date(&brick->date[b]);
    
  return;
}


/** This function convertes the pixel location from one brick to another.
+++ The bricks differ in spatial resolution.
--- from:   source brick
--- to:     target brick
--- p_from: source pixel
+++ Return: target pixel
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_brick_p2p(brick_t *from, brick_t *to, int p_from){
int i_from, i_to;
int j_from, j_to;

  i_from = floor(p_from/from->nx);
  i_to = floor(i_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  j_from = p_from-i_from*from->nx;
  j_to = floor(j_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  return i_to*to->nx+j_to;
}


/** This function convertes the pixel location from one brick to another.
+++ The bricks differ in spatial resolution.
--- from:   source brick
--- to:     target brick
--- p_from: source pixel
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_brick_p2ji(brick_t *from, brick_t *to, int p_from, int *i_to, int *j_to){
int i_from;
int j_from;

  i_from = floor(p_from/from->nx);
  *i_to = floor(i_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  j_from = p_from-i_from*from->nx;
  *j_to = floor(j_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  return;
}


/** This function convertes the pixel location from one brick to another.
+++ The bricks differ in spatial resolution.
--- from:   source brick
--- to:     target brick
--- p_from: source pixel
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
--- p_to:   target pixel  (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_brick_p2jip(brick_t *from, brick_t *to, int p_from, int *i_to, int *j_to, int *p_to){
int i_from;
int j_from;

  i_from = floor(p_from/from->nx);
  *i_to = floor(i_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  j_from = p_from-i_from*from->nx;
  *j_to = floor(j_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  *p_to = (*i_to)*to->nx+(*j_to);

  return;
}


/** This function convertes the pixel location from one brick to another.
+++ The bricks differ in spatial resolution.
--- from:   source brick
--- to:     target brick
--- i_from: source row
--- j_from: source column
+++ Return: target pixel
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_brick_ji2p(brick_t *from, brick_t *to, int i_from, int j_from){
int i_to;
int j_to;

  i_to = floor(i_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);
  j_to = floor(j_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  return i_to*to->nx+j_to;
}


/** This function convertes the pixel location from one brick to another.
+++ The bricks differ in spatial resolution.
--- from:   source brick
--- to:     target brick
--- i_from: source row
--- j_from: source column
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_brick_ji2ji(brick_t *from, brick_t *to, int i_from, int j_from, int *i_to, int *j_to){

  *i_to = floor(i_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);
  *j_to = floor(j_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);

  return;
}


/** This function convertes the pixel location from one brick to another.
+++ The bricks differ in spatial resolution.
--- from:   source brick
--- to:     target brick
--- i_from: source row
--- j_from: source column
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
--- p_to:   target pixel  (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_brick_ji2jip(brick_t *from, brick_t *to, int i_from, int j_from, int *i_to, int *j_to, int *p_to){

  *i_to = floor(i_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);
  *j_to = floor(j_from*from->geotran[_GT_RES_]/to->geotran[_GT_RES_]);
  *p_to = (*i_to)*to->nx+(*j_to);

  return;
}


/** This function returns the band that matches the given domain, e.g.
+++ spectral band. If the domain is not present in the brick, -1 is 
+++ returned.
--- brick:  brick
--- domain: domain
+++ Return: band that holds the given domain
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int find_domain(brick_t *brick, const char *domain){
int b, n = get_brick_nbands(brick);
char domain_[NPOW_10];

  for (b=0; b<n; b++){
    get_brick_domain(brick, b, domain_, NPOW_10);
    if (strcmp(domain_, domain) == 0) return b;
  }
  
  return -1;
}


/** This function sets the name of a brick
--- brick:  brick
--- name:   name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_name(brick_t *brick, const char *name){


  copy_string(brick->name, NPOW_10, name);

  return;
}


/** This function gets the name of a brick
--- brick:  brick
--- name:   name (modified)
--- size:   length of the buffer for name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_name(brick_t *brick, char name[], size_t size){


  copy_string(name, size, brick->name);

  return;
}


/** This function sets the product of a brick
--- brick:   brick
--- product: product
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_product(brick_t *brick, const char *product){
  

  copy_string(brick->product, NPOW_10, product);

  return;
}


/** This function gets the product of a brick
--- brick:   brick
--- product: product (modified)
--- size:    length of the buffer for product
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_product(brick_t *brick, char product[], size_t size){


  copy_string(product, size, brick->product);

  return;
}


/** This function sets the provenance directory of a brick
--- brick:   brick
--- provdir: provenance directory
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_provdir(brick_t *brick, const char *provdir){


  copy_string(brick->provdir, NPOW_10, provdir);

  return;
}


/** This function gets the provenance directory of a brick
--- brick:   brick
--- provdir: provenance directory (modified)
--- size:    length of the buffer for provdir
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_provdir(brick_t *brick, char provdir[], size_t size){


  copy_string(provdir, size, brick->provdir);

  return;
}


/** This function sets the directory-name of a brick
--- brick:  brick
--- dname:  directory-name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_dirname(brick_t *brick, const char *dname){


  copy_string(brick->dname, NPOW_10, dname);

  return;
}


/** This function gets the directory-name of a brick
--- brick:  brick
--- dname:  directory-name (modified)
--- size:   length of the buffer for dname
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_dirname(brick_t *brick, char dname[], size_t size){


  copy_string(dname, size, brick->dname);

  return;
}


/** This function sets the filename of a brick
--- brick:  brick
--- fname:  filename
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_filename(brick_t *brick, const char *fname){


  copy_string(brick->fname, NPOW_10, fname);

  return;
}


/** This function gets the filename of a brick
--- brick:  brick
--- fname:  filename (modified)
--- size:   length of the buffer for fname
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_filename(brick_t *brick, char fname[], size_t size){


  copy_string(fname, size, brick->fname);

  return;
}


/** This function sets the number of input images for provenance, and 
+++ allocates provenance array
--- brick:  brick
--- n:      number
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_nprovenance(brick_t *brick, int n){
  

  if (brick->nprovenance > 0 && brick->nprovenance != n && brick->provenance != NULL){
    free_2D((void**)brick->provenance, brick->nprovenance);
    brick->provenance = NULL;
  }

  if (brick->provenance == NULL){
    alloc_2D((void***)&brick->provenance, n, NPOW_10, sizeof(char));
  }

  brick->nprovenance = n;

  return;
}


/** This function gets the number of input images for provenance
--- brick:  brick
+++ Return: number
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_nprovenance(brick_t *brick){
  
  return brick->nprovenance;
}


/** This function sets the input provenance of a brick
--- brick:  brick
--- id:     input number
--- pname:  input name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_provenance(brick_t *brick, int id, const char *pname){


  copy_string(brick->provenance[id], NPOW_10, pname);

  return;
}


/** This function gets the input provenance of a brick
--- brick:  brick
--- id:     input number
--- pname:  input name (modified)
--- size:   length of the buffer for pname
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_provenance(brick_t *brick, int id, char pname[], size_t size){


  copy_string(pname, size, brick->provenance[id]);

  return;
}


/** This function sets the sensor ID of a brick
--- brick:  brick
--- sid:    sensor ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_sensorid(brick_t *brick, int sid){

  brick->sid = sid;

  return;
}


/** This function gets the sensor ID of a brick
--- brick:  brick
+++ Return: sensor ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_sensorid(brick_t *brick){
  
  return brick->sid;
}



/** This function sets the format of a brick
--- brick:   brick
--- format:  format
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_format(brick_t *brick, gdalopt_t *format){


  brick->format = *format;

  return;
}

/** This function gets the format of a brick
--- brick:  brick
+++ Return: format
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
gdalopt_t get_brick_format(brick_t *brick){
  
  return brick->format;
}


/** This function sets the opening option of a brick
--- brick:  brick
--- open:   opening option
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_open(brick_t *brick, int open){

  brick->open = open;

  return;
}


/** This function gets the opening option of a brick
--- brick:  brick
+++ Return: opening option
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool get_brick_open(brick_t *brick){
  
  return brick->open;
}


/** This function sets the explode-bands option of a brick
--- brick:   brick
--- explode: explode bands?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_explode(brick_t *brick, int explode){

  brick->explode = explode;

  return;
}


/** This function gets the explode-bands option of a brick
--- brick:  brick
+++ Return: explode bands?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool get_brick_explode(brick_t *brick){
  
  return brick->explode;
}


/** This function sets the datatype of a brick
--- brick:    brick
--- datatype: datatype
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_datatype(brick_t *brick, int datatype){

  if (datatype != _DT_SHORT_ && datatype != _DT_SMALL_ &&
      datatype != _DT_FLOAT_ && datatype != _DT_INT_  && 
      datatype != _DT_USHORT_ && datatype != _DT_NONE_){
    printf("unknown datatype %d.\n", datatype);}

  if (brick->datatype != _DT_NONE_ && brick->datatype != datatype){
    printf("WARNING: re-setting datatype.\n");
    printf("This might result in double-allocations.\n");}

  brick->datatype  = datatype;

  return;
}


/** This function gets the datatype of a brick
--- brick:  brick
+++ Return: datatype
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_datatype(brick_t *brick){
  
  return brick->datatype;
}


/** This function sets the bytesize of a brick
--- brick:  brick
--- byte:   bytesize
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_byte(brick_t *brick, size_t byte){

  brick->byte = (int)byte;

  return;
}


/** This function gets the bytesize of a brick
--- brick:  brick
+++ Return: bytesize
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_byte(brick_t *brick){
  
  return brick->byte;
}


/** This function sets the number of bands of a brick
--- brick:  brick
--- nb:     number of bands
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_nbands(brick_t *brick, int nb){

  if (nb <= 0) printf("number of bands must be > 0.\n");

  brick->nb = nb;

  return;
}


/** This function gets the number of bands of a brick
--- brick:  brick
+++ Return: number of bands
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_nbands(brick_t *brick){
  
  return brick->nb;
}


/** This function sets the number of columns of a brick
--- brick:  brick
--- nx:     number of columns
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_ncols(brick_t *brick, int nx){

  if (nx <= 0) printf("number of cols must be > 0.\n");

  brick->nx = nx;
  brick->nc = brick->nx*brick->ny;
  brick->width = brick->nx*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the number of columns of a brick
--- brick:  brick
+++ Return: number of columns
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_ncols(brick_t *brick){
  
  return brick->nx;
}


/** This function sets the number of rows of a brick
--- brick:  brick
--- ny:     number of rows
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_nrows(brick_t *brick, int ny){

  if (ny <= 0) printf("number of rows must be > 0.\n");

  brick->ny = ny;
  brick->nc = brick->nx*brick->ny;
  brick->height = brick->ny*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the number of rows of a brick
--- brick:  brick
+++ Return: number of rows
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_nrows(brick_t *brick){
  
  return brick->ny;
}


/** This function sets the number of cells of a brick
--- brick:  brick
--- nc:     number of cells
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_ncells(brick_t *brick, int nc){
  
  if (nc != brick->nx*brick->ny) printf("number of cells do not match with nx*ny.\n");

  brick->nc = brick->nx*brick->ny;
  
  return;
}


/** This function gets the number of cells of a brick
--- brick:  brick
+++ Return: number of cells
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_ncells(brick_t *brick){
  
  return brick->nc;
}


/** This function gets the total bytesize of a brick
--- brick:  brick
+++ Return: total bytesize
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
size_t get_brick_size(brick_t *brick){
int b = 0;
size_t size = 0;

  for (b=0; b<brick->nb; b++){
    size += brick->nc*brick->byte;
  }

  return size;
}


/** This function sets the number of columns in chunk of a brick
--- brick:  brick
--- cx:     number of columns in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


void set_brick_chunkncols(brick_t *brick, int cx){

  if (cx < 0) printf("number of chunking cols must be >= 0.\n");

  brick->cx = cx;
  brick->cc = brick->cx*brick->cy;
  brick->cwidth = brick->cx*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the number of columns in chunk of a brick
--- brick:  brick
+++ Return: number of columns in chunk
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunkncols(brick_t *brick){
  
  return brick->cx;
}


/** This function sets the number of rows in chunk of a brick
--- brick:  brick
--- cy:     number of rows in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunknrows(brick_t *brick, int cy){

  if (cy < 0) printf("number of chunking rows must be >= 0.\n");

  brick->cy = cy;
  brick->cc = brick->cx*brick->cy;
  brick->cheight = brick->cy*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the number of rows in chunk of a brick
--- brick:  brick
+++ Return: number of rows in chunk
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunknrows(brick_t *brick){
  
  return brick->cy;
}


/** This function sets the number of cells in chunk of a brick
--- brick:  brick
--- cc:     number of cells in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunkncells(brick_t *brick, int cc){
  
  if (cc != brick->cx*brick->cy) printf("number of chunking cells do not match with cx*cy.\n");

  brick->cc = brick->cx*brick->cy;
  
  return;
}


/** This function gets the number of cells in chunk of a brick
--- brick:  brick
+++ Return: number of cells in chunk
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunkncells(brick_t *brick){
  
  return brick->cc;
}


/** This function sets the chunk X-ID of a brick
--- brick:  brick
--- chunk:  chunk x-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunkx(brick_t *brick, int chunkx){

  if (chunkx >= 0 && chunkx >= brick->dim_chunk.cols){
    printf("current chunk %d is higher than chunks in X-direction %d.\n", 
      chunkx, brick->dim_chunk.cols);
  }

  brick->chunk[_X_] = chunkx;

  return;
}


/** This function gets the chunk X-ID of a brick
--- brick:  brick
+++ Return: chunk X-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunkx(brick_t *brick){

  return brick->chunk[_X_];
}


/** This function sets the chunk Y-ID of a brick
--- brick:  brick
--- chunk:  chunk y-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunky(brick_t *brick, int chunky){

  if (chunky >= 0 && chunky >= brick->dim_chunk.rows){
    printf("current chunk %d is higher than chunks in Y-direction %d.\n", 
      chunky, brick->dim_chunk.rows);
  }

  brick->chunk[_Y_] = chunky;

  return;
}


/** This function gets the chunk Y-ID of a brick
--- brick:  brick
+++ Return: chunk Y-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunky(brick_t *brick){
  
  return brick->chunk[_Y_];
}


/** This function sets the chunk dimensions in X-direction of a brick
--- brick:  brick
--- ncol:   number of columns in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunk_dim_x(brick_t *brick, int ncol){
dim_t dim;

  dim.cols = ncol;
  dim.rows = brick->dim_chunk.rows;
  dim.cells = dim.cols * dim.rows;

  set_brick_chunk_dim(brick, &dim);

}


/** This function sets the chunk dimensions in X-direction of a brick
--- brick:  brick
--- ncol:   number of columns in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunk_dim_y(brick_t *brick, int nrow){
dim_t dim;

  dim.cols = brick->dim_chunk.cols;
  dim.rows = nrow;
  dim.cells = dim.cols * dim.rows;

  set_brick_chunk_dim(brick, &dim);

}


/** This function gets the chunk dimensions in X-direction of a brick
--- brick:  brick
+++ Return: number of chunk columns in image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunk_dim_x(brick_t *brick){
  return brick->dim_chunk.cols;
}


/** This function gets the chunk dimensions in Y-direction of a brick
--- brick:  brick
+++ Return: number of chunk rows in image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunk_dim_y(brick_t *brick){
  return brick->dim_chunk.rows;
}

/** This function gets the number of chunks in a brick
--- brick:  brick
+++ Return: number of chunks in image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunk_dim_n(brick_t *brick){
  return brick->dim_chunk.cells;
}


/** This function sets the chunk dimensions of a brick
--- brick:  brick
--- dim:    chunk dimensions
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunk_dim(brick_t *brick, dim_t *dim){

  if (dim->cols < 0 || dim->rows < 0) printf("chunk dimensions %d||%d < 0.\n", dim->cols, dim->rows);
  if (dim->cols * dim->rows != dim->cells) printf("chunk X/Y dimensions do not match with # of chunk  cells.\n");

  brick->dim_chunk = *dim;

  return;
}


/** This function gets the chunk dimensions of a brick
--- brick:  brick
+++ Return: chunk dimensions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
dim_t get_brick_chunk_dim(brick_t *brick){

  return brick->dim_chunk;
}


/** This function sets the tile X-ID of a brick
--- brick:  brick
--- tx:     tile X-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_tilex(brick_t *brick, int tx){
  
  if (tx >= 9999 || tx < -999) printf("tile-x is out of bounds.\n");

  brick->tile[_X_] = tx;
  
  return;
}


/** This function gets the tile X-ID of a brick
--- brick:  brick
+++ Return: tile X-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_tilex(brick_t *brick){
  
  return brick->tile[_X_];
}


/** This function sets the tile Y-ID of a brick
--- brick:  brick
--- ty:     tile Y-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_tiley(brick_t *brick, int ty){
  
  if (ty >= 9999 || ty < -999) printf("tile-y is out of bounds.\n");

  brick->tile[_Y_] = ty;

  return;
}


/** This function gets the tile Y-ID of a brick
--- brick:  brick
+++ Return: tile Y-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_tiley(brick_t *brick){

  return brick->tile[_Y_];
}


/** This function sets the resolution of a brick
--- brick:  brick
--- res:    resolution
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_res(brick_t *brick, double res){

  if (res <= 0) printf("resolution must be > 0.\n");

  brick->geotran[_GT_XRES_] = res;
  brick->geotran[_GT_YRES_] = res * -1;
  brick->width  = brick->nx * res;
  brick->height = brick->ny * res;

  return;
}


/** This function gets the resolution of a brick
--- brick:  brick
+++ Return: resolution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_res(brick_t *brick){

  return brick->geotran[_GT_RES_];
}


/** This function sets the UL-X coordinate of a brick
--- brick:   brick
--- ulx: UL-X coordinate
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_ulx(brick_t *brick, double ulx){

  brick->geotran[_GT_ULX_] = ulx;

  return;
}


/** This function gets the UL-X coordinate of a brick
--- brick:  brick
+++ Return: UL-X coordinate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_ulx(brick_t *brick){

  return brick->geotran[_GT_ULX_];
}


/** This function gets the X coordinate of a column of a brick
--- brick:  brick
--- j:      column
+++ Return: X coordinate of a column
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_x(brick_t *brick, int j){

  return (brick->geotran[_GT_ULX_] + j*brick->geotran[_GT_RES_]);
}


/** This function sets the UL-Y coordinate of a brick
--- brick:  brick
--- uly:    UL-Y coordinate
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_uly(brick_t *brick, double uly){

  brick->geotran[_GT_ULY_] = uly;

  return;
}


/** This function gets the UL-Y coordinate of a brick
--- brick:  brick
+++ Return: UL-Y coordinate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_uly(brick_t *brick){

  return brick->geotran[_GT_ULY_];
}


/** This function gets the Y coordinate of a row of a brick
--- brick:  brick
--- i:      row
+++ Return: Y coordinate of a row 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_y(brick_t *brick, int i){

  return (brick->geotran[_GT_ULY_] + i*brick->geotran[_GT_RES_]);
}


/** This function gets the geographic coordinates of a column/row of a brick
--- brick:  brick
--- j:      column
--- i:      row
+++ lon:    longitude (returned)
+++ lat:    latiitude (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_geo(brick_t *brick, int j, int i, double *lon, double *lat){
double mapx, mapy;
double geox, geoy;

  // map coordinate
  mapx = get_brick_x(brick, j);
  mapy = get_brick_y(brick, i);

  // geo coordinate
  warp_any_to_geo(mapx, mapy, &geox, &geoy, brick->proj);
  
  *lon = geox;
  *lat = geoy;
  return ;
}


/** This function sets the geotransformation of a brick
--- brick:   brick
--- geotran: geotransformation
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_geotran(brick_t *brick, double *geotran){

  brick->geotran[_GT_XRES_] = geotran[_GT_XRES_];
  brick->geotran[_GT_YRES_] = geotran[_GT_YRES_];
  brick->geotran[_GT_ULX_]  = geotran[_GT_ULX_];
  brick->geotran[_GT_ULY_]  = geotran[_GT_ULY_];
  brick->geotran[_GT_XROT_] = geotran[_GT_XROT_];
  brick->geotran[_GT_YROT_] = geotran[_GT_YROT_];

  return;
}


/** This function gets the geotransformation of a brick
--- brick:   brick
--- geotran: geotransformation (modified)
--- size:    length of the buffer for geotran
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_geotran(brick_t *brick, double geotran[], size_t size){

  if (size != _GT_LEN_){
    printf("array is not compatible for getting geotran.\n"); return;}

  geotran[_GT_ULX_] = brick->geotran[_GT_ULX_];
  geotran[_GT_ULY_] = brick->geotran[_GT_ULY_];
  geotran[_GT_XRES_] = brick->geotran[_GT_XRES_];
  geotran[_GT_YRES_] = brick->geotran[_GT_YRES_];
  geotran[_GT_XROT_] = brick->geotran[_GT_XROT_];
  geotran[_GT_YROT_] = brick->geotran[_GT_YROT_];
  
  return;
}


/** This function sets the width of a brick
--- brick:  brick
--- width:  width
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_width(brick_t *brick, double width){

  if (width != brick->nx*brick->geotran[_GT_RES_]) printf("width does not match with nx*res.\n");

  brick->width = brick->nx*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the width of a brick
--- brick:  brick
+++ Return: width
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_width(brick_t *brick){
  
  return brick->width;
}


/** This function sets the height of a brick
--- brick:  brick
--- height: height
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_height(brick_t *brick, double height){

  if (height != brick->ny*brick->geotran[_GT_RES_]) printf("height does not match with ny*res.\n");

  brick->height = brick->ny*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the height of a brick
--- brick:  brick
+++ Return: height
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_height(brick_t *brick){
  
  return brick->height;
}


/** This function sets the chunk width of a brick
--- brick:   brick
--- cwidth: chunk width
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunkwidth(brick_t *brick, double cwidth){

  if (cwidth != brick->cx*brick->geotran[_GT_RES_]) printf("chunking width does not match with cx*res.\n");

  brick->cwidth = brick->cx*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the chunk width of a brick
--- brick:  brick
+++ Return: chunk width
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_chunkwidth(brick_t *brick){
  
  return brick->cwidth;
}


/** This function sets the chunk height of a brick
--- brick:   brick
--- cheight: chunk height
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunkheight(brick_t *brick, double cheight){

  if (cheight != brick->cy*brick->geotran[_GT_RES_]) printf("chunking height does not match with cy*res.\n");

  brick->cheight = brick->cy*brick->geotran[_GT_RES_];

  return;
}


/** This function gets the chunk height of a brick
--- brick:  brick
+++ Return: chunk height
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_chunkheight(brick_t *brick){
  
  return brick->cheight;
}


/** This function sets the projection of a brick
--- brick:  brick
--- proj:   projection
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_proj(brick_t *brick, const char *proj){


  copy_string(brick->proj, NPOW_10, proj);
  
  return;
}


/** This function gets the projection of a brick
--- brick:  brick
--- proj:   projection (modified)
--- size:   length of the buffer for proj
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_proj(brick_t *brick, char proj[], size_t size){


  copy_string(proj, size, brick->proj);
  
  return;
}


/** This function sets the parameters of a brick
--- brick:  brick
--- par:    parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_par(brick_t *brick, const char *par){


  copy_string(brick->par, NPOW_14, par);

  return;
}


/** This function gets the parameters of a brick
--- brick:  brick
--- par:    parameters (modified)
--- size:   length of the buffer for par
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_par(brick_t *brick, char par[], size_t size){


  copy_string(par, size, brick->par);

  return;
}


/** This function sets the write flag of a brick band
--- brick:  brick
--- b:      band
--- save:   write flag
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_save(brick_t *brick, int b, bool save){

  brick->save[b] = save;

  return;
}


/** This function gets the write flag of a brick band
--- brick:  brick
--- b:      band
+++ Return: write flag
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool get_brick_save(brick_t *brick, int b){

  return brick->save[b];
}


/** This function sets the nodata value of a brick band
--- brick:  brick
--- b:      band
--- nodata: nodata value
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_nodata(brick_t *brick, int b, int nodata){

  brick->nodata[b] = nodata;

  return;
}


/** This function gets the nodata value of a brick band
--- brick:  brick
--- b:      band
+++ Return: nodata value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_nodata(brick_t *brick, int b){

  return brick->nodata[b];
}


/** This function tests if pixel is nodata
--- brick:  brick
--- b:      band
--- p:      pixel
+++ Return: nodata?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool is_brick_nodata(brick_t *brick, int b, int p){
  
  return fequal(get_brick(brick, b, p), (float)brick->nodata[b]);
}


/** This function sets the scale of a brick band
--- brick:  brick
--- b:      band
--- scale:  scale
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_scale(brick_t *brick, int b, float scale){

  brick->scale[b] = scale;

  return;
}


/** This function gets the scale of a brick band
--- brick:  brick
--- b:      band
+++ Return: scale
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_brick_scale(brick_t *brick, int b){
  
  return brick->scale[b];
}


/** This function sets the wavelength of a brick band
--- brick:      brick
--- b:          band
--- wavelength: wavelength
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_wavelength(brick_t *brick, int b, float wavelength){

  brick->wavelength[b] = wavelength;

  return;
}


/** This function gets the wavelength of a brick band
--- brick:  brick
--- b:      band
+++ Return: wavelength
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_brick_wavelength(brick_t *brick, int b){
  
  return brick->wavelength[b];
}


/** This function sets the unit of a brick band
--- brick:  brick
--- b:      band
--- unit:   unit
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_unit(brick_t *brick, int b, const char *unit){


  copy_string(brick->unit[b], NPOW_10, unit);

  return;
}


/** This function gets the unit of a brick band
--- brick:  brick
--- b:      band
--- unit:   unit (modified)
--- size:   length of the buffer for unit
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_unit(brick_t *brick, int b, char unit[], size_t size){


  copy_string(unit, size, brick->unit[b]);

  return;
}


/** This function sets the domain of a brick band
--- brick:  brick
--- b:      band
--- domain: domain
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_domain(brick_t *brick, int b, const char *domain){


  copy_string(brick->domain[b], NPOW_10, domain);

  return;
}


/** This function gets the domain of a brick band
--- brick:  brick
--- b:      band
--- domain: domain (modified)
--- size:   length of the buffer for domain
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_domain(brick_t *brick, int b, char domain[], size_t size){


  copy_string(domain, size, brick->domain[b]);

  return;
}


/** This function sets the bandname of a brick band
--- brick:    brick
--- b:        band
--- bandname: bandname
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_bandname(brick_t *brick, int b, const char *bandname){


  copy_string(brick->bandname[b], NPOW_10, bandname);

  return;
}


/** This function gets the bandname of a brick band
--- brick:    brick
--- b:        band
--- bandname: bandname (modified)
--- size:     length of the buffer for bandname
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_bandname(brick_t *brick, int b, char bandname[], size_t size){

  
  copy_string(bandname, size, brick->bandname[b]);

  return;
}


/** This function sets the sensor of a brick band
--- brick:  brick
--- b:      band
--- sensor: sensor
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_sensor(brick_t *brick, int b, const char *sensor){


  copy_string(brick->sensor[b], NPOW_10, sensor);

  return;
}


/** This function gets the sensor of a brick band
--- brick:  brick
--- b:      band
--- sensor: sensor (modified)
--- size:   length of the buffer for sensor
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_sensor(brick_t *brick, int b, char sensor[], size_t size){


  copy_string(sensor, size, brick->sensor[b]);

  return;
}


/** This function sets the date of a brick band
--- brick:  brick
--- b:      band
--- date:   date
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_date(brick_t *brick, int b, date_t date){

  copy_date(&date, &brick->date[b]);

  return;
}


/** This function gets the date of a brick band
--- brick:  brick
--- b:      band
+++ Return: date
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
date_t get_brick_date(brick_t *brick, int b){
  
  return brick->date[b];
}


/** This function sets the days since current era of a brick band
--- brick:  brick
--- b:      band
--- ce:     days since current era
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_ce(brick_t *brick, int b, int ce){
  
  brick->date[b].ce = ce;
  
  return;
}


/** This function gets the days since current era of a brick band
--- brick:  brick
--- b:      band
+++ Return: days since current era
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_ce(brick_t *brick, int b){
  
  return brick->date[b].ce;
}


/** This function sets the day of a brick band band
--- brick:  brick
--- day:    day
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_day(brick_t *brick, int b, int day){
  
  brick->date[b].day = day;
  
  return;
}


/** This function gets the day of a brick band
--- brick:  brick
--- b:      band
+++ Return: day
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_day(brick_t *brick, int b){
  
  return brick->date[b].day;
}


/** This function sets the doy of a brick band
--- brick:  brick
--- doy:    doy
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_doy(brick_t *brick, int b, int doy){
  
  brick->date[b].doy = doy;
  
  return;
}


/** This function gets the doy of a brick band
--- brick:  brick
--- b:      band
+++ Return: doy
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_doy(brick_t *brick, int b){
  
  return brick->date[b].doy;
}


/** This function sets the week of a brick band
--- brick:  brick
--- b:      band
--- week:   week
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_week(brick_t *brick, int b, int week){
  
  brick->date[b].week = week;
  
  return;
}


/** This function gets the week of a brick band
--- brick:  brick
--- b:      band
+++ Return: week
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_week(brick_t *brick, int b){
  
  return brick->date[b].week;
}


/** This function sets the month of a brick band
--- brick:  brick
--- b:      band
--- month:  month
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_month(brick_t *brick, int b, int month){
  
  brick->date[b].month = month;
  
  return;
}


/** This function gets the month of a brick band
--- brick:  brick
--- b:      band
+++ Return: month
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_month(brick_t *brick, int b){
  
  return brick->date[b].month;
}


/** This function sets the year of a brick band
--- brick:  brick
--- b:      band
--- year:   year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_year(brick_t *brick, int b, int year){
  
  brick->date[b].year = year;
  
  return;
}


/** This function gets the year of a brick band
--- brick:  brick
--- b:      band
+++ Return: year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_year(brick_t *brick, int b){
  
  return brick->date[b].year;
}


/** This function sets the hour of a brick band
--- brick:  brick
--- b:      band
--- hour:   hour
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_hour(brick_t *brick, int b, int hour){
  
  brick->date[b].hh = hour;
  
  return;
}


/** This function gets the hour of a brick band
--- brick:  brick
--- b:      band
+++ Return: hour
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_hour(brick_t *brick, int b){
  
  return brick->date[b].hh;
}


/** This function sets the minute of a brick band
--- brick:  brick
--- b:      band
--- minute: minute
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_minute(brick_t *brick, int b, int minute){
  
  brick->date[b].mm = minute;
  
  return;
}


/** This function gets the minute of a brick band
--- brick:  brick
--- b:      band
+++ Return: minute
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_minute(brick_t *brick, int b){
  
  return brick->date[b].mm;
}


/** This function sets the second of a brick band
--- brick:  brick
--- b:      band
--- second: second
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_second(brick_t *brick, int b, int second){
  
  brick->date[b].ss = second;
  
  return;
}


/** This function gets the second of a brick band
--- brick:  brick
--- b:      band
+++ Return: second
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_second(brick_t *brick, int b){
  
  return brick->date[b].ss;
}


/** This function sets the timezone of a brick band
--- brick:    brick
--- b:        band
--- timezone: timezone
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_timezone(brick_t *brick, int b, int timezone){
  
  brick->date[b].tz = timezone;
  
  return;
}


/** This function gets the timezone of a brick band
--- brick:  brick
--- b:      band
+++ Return: timezone
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_timezone(brick_t *brick, int b){
  
  return brick->date[b].tz;
}


/** This function gets a formatted date of a brick band
--- brick:     brick
--- b:         band
--- formatted: formatted date (modified)
--- size:      length of the buffer for formatted
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_compactdate(brick_t *brick, int b, char formatted[], size_t size){

  compact_date(brick->date[b].year, brick->date[b].month, brick->date[b].day, formatted, size);

  return;
}


/** This function gets a formatted date of a brick band
--- brick:     brick
--- b:         band
--- formatted: formatted date (modified)
--- size:      length of the buffer for formatted
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_longdate(brick_t *brick, int b, char formatted[], size_t size){

  long_date(brick->date[b].year, brick->date[b].month, brick->date[b].day,
    brick->date[b].hh, brick->date[b].mm, brick->date[b].ss, brick->date[b].tz, 
    formatted, size);

  return;
}


/** This function sets an image value. This is slower than direct access
+++ to the memory, but probably more convenient as the correct datatype
+++ etc is chosen
--- brick:  brick
--- b:      band
--- p:      pixel
--- val:    value
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick(brick_t *brick, int b, int p, float val){

  switch (brick->datatype){
    case _DT_SHORT_:
      brick->vshort[b][p] = (short)val;
      break;
    case _DT_SMALL_:
      brick->vsmall[b][p] = (small)val;
      break;
    case _DT_FLOAT_:
      brick->vfloat[b][p] = (float)val;
      break;
    case _DT_INT_:
      brick->vint[b][p] = (int)val;
      break;
    case _DT_USHORT_:
      brick->vushort[b][p] = (ushort)val;
      break;
    default:
      printf("unknown datatype %d, (no value was set)\n", brick->datatype);
      return;
  }

}


/** This function gets an image value. This is slower than direct access
+++ to the memory, but probably more convenient as the correct datatype
+++ etc is chosen
--- brick:  brick
--- b:      band
--- p:      pixel
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_brick(brick_t *brick, int b, int p){

  switch (brick->datatype){
    case _DT_SHORT_:
      return (float)brick->vshort[b][p];
      break;
    case _DT_SMALL_:
      return (float)brick->vsmall[b][p];
      break;
    case _DT_FLOAT_:
      return (float)brick->vfloat[b][p];
      break;
    case _DT_INT_:
      return (float)brick->vint[b][p];
      break;
    case _DT_USHORT_:
      return (float)brick->vushort[b][p];
      break;
    default:
      printf("unknown datatype (return 0)");
      return 0.0;
  }

}


/** This function returns a pointer to the short image bands
--- brick:  brick
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short **get_bands_short(brick_t *brick){

  if (brick->vshort == NULL){
    printf("SHORT memory not available.\n"); return NULL;}

  return brick->vshort;
}


/** This function returns a pointer to the byte image bands
--- brick:  brick
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
small **get_bands_small(brick_t *brick){

  if (brick->vsmall == NULL){
    printf("SMALL memory not available.\n"); return NULL;}

  return brick->vsmall;
}


/** This function returns a pointer to the float image bands
--- brick:  brick
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float **get_bands_float(brick_t *brick){

  if (brick->vfloat == NULL){
    printf("FLOAT memory not available.\n"); return NULL;}

  return brick->vfloat;
}


/** This function returns a pointer to the integer image bands
--- brick:  brick
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int **get_bands_int(brick_t *brick){

  if (brick->vint == NULL){
    printf("INT memory not available.\n"); return NULL;}

  return brick->vint;
}


/** This function returns a pointer to the unsigned short image bands
--- brick:  brick
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort **get_bands_ushort(brick_t *brick){

  if (brick->vushort == NULL){
    printf("USHORT memory not available.\n"); return NULL;}

  return brick->vushort;
}


/** This function returns a pointer to a short image band
--- brick:  brick
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *get_band_short(brick_t *brick, int b){

  if (brick->vshort == NULL){
    printf("SHORT memory not available.\n"); return NULL;}
  
  return brick->vshort[b];
}


/** This function returns a pointer to a byte image band
--- brick:  brick
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
small *get_band_small(brick_t *brick, int b){

  if (brick->vsmall == NULL){
    printf("SMALL memory not available.\n"); return NULL;}
  
  return brick->vsmall[b];
}


/** This function returns a pointer to a float image band
--- brick:  brick
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *get_band_float(brick_t *brick, int b){

  if (brick->vfloat == NULL){
    printf("FLOAT memory not available.\n"); return NULL;}
  
  return brick->vfloat[b];
}


/** This function returns a pointer to an integer image band
--- brick:  brick
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int *get_band_int(brick_t *brick, int b){

  if (brick->vint == NULL){
    printf("INT memory not available.\n"); return NULL;}
  
  return brick->vint[b];
}


/** This function returns a pointer to an unsigned short image band
--- brick:  brick
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort *get_band_ushort(brick_t *brick, int b){
  
  if (brick->vushort == NULL){
    printf("USHORT memory not available.\n"); return NULL;}
  
  return brick->vushort[b];
}


/** This function returns a pointer to a short image band
--- brick:  brick
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *get_domain_short(brick_t *brick, const char *domain){
int b;

  if (brick->vshort == NULL){
    printf("SHORT memory not available.\n"); return NULL;}

  if ((b = find_domain(brick, domain)) < 0) return NULL;
  
  return brick->vshort[b];
}


/** This function returns a pointer to a byte image band
--- brick:  brick
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
small *get_domain_small(brick_t *brick, const char *domain){
int b;

  if (brick->vsmall == NULL){
    printf("SMALL memory not available.\n"); return NULL;}

  if ((b = find_domain(brick, domain)) < 0) return NULL;

  return brick->vsmall[b];
}


/** This function returns a pointer to a float image band
--- brick:  brick
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *get_domain_float(brick_t *brick, const char *domain){
int b;

  if (brick->vfloat == NULL){
    printf("FLOAT memory not available.\n"); return NULL;}

    if ((b = find_domain(brick, domain)) < 0) return NULL;

  return brick->vfloat[b];
}


/** This function returns a pointer to an integer image band
--- brick:  brick
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int *get_domain_int(brick_t *brick, const char *domain){
int b;

  if (brick->vint == NULL){
    printf("INT memory not available.\n"); return NULL;}

  if ((b = find_domain(brick, domain)) < 0) return NULL;

  return brick->vint[b];
}


/** This function returns a pointer to an unsigned short image band
--- brick:  brick
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort *get_domain_ushort(brick_t *brick, const char *domain){
int b;

  if (brick->vushort == NULL){
    printf("USHORT memory not available.\n"); return NULL;}

  if ((b = find_domain(brick, domain)) < 0) return NULL;

  return brick->vushort[b];
}


/** This function returns the minimum value of a band
--- brick:  brick
--- b:      band
+++ Return: minimum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_brick_min(brick_t *brick, int b){
int p;
float tmp, min = LONG_MAX;

   for (p=0; p<brick->nc; p++){
     if (is_brick_nodata(brick, b, p)) continue;
     if ((tmp = get_brick(brick, b, p)) < min) min = tmp;
   }

   return min;
}


/** This function returns the maximum value of a band
--- brick:  brick
--- b:      band
+++ Return: maximum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_brick_max(brick_t *brick, int b){
int p;
float tmp, max = LONG_MIN;

   for (p=0; p<brick->nc; p++){
     if (is_brick_nodata(brick, b, p)) continue;
     if ((tmp = get_brick(brick, b, p)) > max) max = tmp;
   }
   
   return max;
}


/** This function returns the minimum/maximum value of a band
--- brick:  brick
--- b:      band
--- min:    minimum
--- max:    maximum
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_range(brick_t *brick, int b, float *min, float *max){
int p;
float tmp, mx = LONG_MIN, mn = LONG_MAX;

   for (p=0; p<brick->nc; p++){
     if (is_brick_nodata(brick, b, p)) continue;
     tmp = get_brick(brick, b, p);
     if (tmp < mn) mn = tmp;
     if (tmp > mx) mx = tmp;
   }

   *min = mn;
   *max = mx;
   return;
}

