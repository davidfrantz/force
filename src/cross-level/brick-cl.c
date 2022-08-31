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
This file contains functions for organizing bricks in memory, and output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "brick-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_multiproc.h"  // CPL Multi-Threading
#include "gdalwarper.h"     // GDAL warper related entry points and defs
#include "ogr_spatialref.h" // coordinate systems services


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
  alloc_2D((void***)&brick->unit, nb, NPOW_04, sizeof(char));
  alloc_2D((void***)&brick->domain, nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&brick->bandname,   nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&brick->sensor,     nb, NPOW_04, sizeof(char));
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
  re_alloc_2D((void***)&brick->unit,   nb0, NPOW_04, nb, NPOW_04, sizeof(char));
  re_alloc_2D((void***)&brick->domain,   nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&brick->bandname, nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&brick->sensor,   nb0, NPOW_04, nb, NPOW_04, sizeof(char));
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
int b;

  if (from->chunk < 0){
    if ((brick = allocate_brick(nb, from->nc, datatype)) == NULL) return NULL;
  } else {
    if ((brick = allocate_brick(nb, from->cc, datatype)) == NULL) return NULL;
  }


  set_brick_name(brick, from->name);
  set_brick_product(brick, from->product);
  set_brick_parentname(brick, from->pname);
  set_brick_dirname(brick, from->dname);
  set_brick_filename(brick, from->fname);
  set_brick_sensorid(brick, from->sid);

  set_brick_geotran(brick, from->geotran);
  set_brick_nbands(brick, nb);
  set_brick_ncols(brick, from->nx);
  set_brick_nrows(brick, from->ny);
  set_brick_chunkncols(brick, from->cx);
  set_brick_chunknrows(brick, from->cy);
  set_brick_chunkwidth(brick, from->cwidth);
  set_brick_chunkheight(brick, from->cheight);
  set_brick_nchunks(brick, from->nchunk);
  set_brick_chunk(brick, from->chunk);
  set_brick_tilex(brick, from->tx);
  set_brick_tiley(brick, from->ty);
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
double res;
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
  res = get_brick_res(from);
  datatype = get_brick_datatype(from);
  
  brick = copy_brick(from, nb, _DT_NONE_);

  if (from->chunk < 0){
    nx = get_brick_ncols(from);
    ny = get_brick_nrows(from);
  } else {
    nx = get_brick_chunkncols(from);
    ny = get_brick_chunknrows(from);
  }

  pix = (int)(radius/res);
  nx_ = nx - 2*pix;
  ny_ = ny - 2*pix;
  nc_ = nx_*ny_;

  if (from->chunk < 0){
    set_brick_ncols(brick, nx_);
    set_brick_nrows(brick, ny_);
  } else {
    set_brick_chunkncols(brick, nx_);
    set_brick_chunknrows(brick, ny_);
  }
  allocate_brick_bands(brick, nb, nc_, datatype);
  
  #ifdef FORCE_DEBUG
  int nc;
  if (from->chunk < 0){
    nc = get_brick_ncells(from);
  } else {
    nc = get_brick_chunkncells(from);
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
int i;


  copy_string(brick->name,      NPOW_10, "NA");
  copy_string(brick->product,   NPOW_03, "NA");
  copy_string(brick->pname,     NPOW_10, "NA");
  copy_string(brick->dname,     NPOW_10, "NA");
  copy_string(brick->fname,     NPOW_10, "NA");

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
  brick->res = 0;
  for (i=0; i<6; i++) brick->geotran[i] = 0;
  brick->width  = 0;
  brick->height = 0;
  brick->cwidth  = 0;
  brick->cheight = 0;
  brick->chunk = -1;
  brick->nchunk = 0;
  brick->tx = 0;
  brick->ty = 0;

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
    copy_string(brick->unit[b],     NPOW_04, "NA");
    copy_string(brick->domain[b],   NPOW_10, "NA");
    copy_string(brick->bandname[b], NPOW_10, "NA");
    copy_string(brick->sensor[b],   NPOW_04, "NA");
    init_date(&brick->date[b]);
  }

  return;
}


/** This function prints a brick
--- brick:  brick
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_brick_info(brick_t *brick){
int b;


  printf("\nbrick info for %s - %s - SID %d\n", brick->name, brick->product, brick->sid);
  printf("open: %d, explode %d\n", 
    brick->open, brick->explode);
  print_gdaloptions(&brick->format);
  printf("datatype %d with %d bytes\n", 
    brick->datatype, brick->byte);
  printf("filename: %s/%s\n", brick->dname, brick->fname);
  printf("nx: %d, ny: %d, nc: %d, res: %.3f, nb: %d\n", 
    brick->nx, brick->ny, brick->nc, 
    brick->res, brick->nb);
  printf("width: %.1f, height: %.1f\n", 
    brick->width, brick->height);
  printf("chunking: nx: %d, ny: %d, nc: %d, width: %.1f, height: %.1f, #: %d\n", 
    brick->cx, brick->cy, brick->cc, brick->cwidth, brick->cheight, brick->nchunk);
  printf("active chunk: %d, tile X%04d_Y%04d\n", brick->chunk, brick->tx, brick->ty);
  printf("ulx: %.3f, uly: %.3f\n", 
    brick->geotran[0], brick->geotran[3]);
  printf("geotran: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", 
    brick->geotran[0], brick->geotran[1],
    brick->geotran[2], brick->geotran[3],
    brick->geotran[4], brick->geotran[5]);
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


/** This function outputs a brick
--- brick:  brick
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int write_brick(brick_t *brick){
int f, b, b_, p, o;
int b_brick, b_file, nbands, nfiles;
int ***bands = NULL;
char *lock = NULL;
double timeout;
GDALDatasetH fp_physical = NULL;
GDALDatasetH fp = NULL;
GDALDatasetH fo = NULL;
GDALRasterBandH band = NULL;
GDALDriverH driver_physical = NULL;
GDALDriverH driver = NULL;
char **options = NULL;
float *buf = NULL;
float now, old;
int xoff_write, yoff_write, nx_write, ny_write;

char bname[NPOW_10];
char fname[NPOW_10];
int nchar;

char ldate[NPOW_05];


GDALDataType file_datatype;

char **fp_meta = NULL;
char **band_meta = NULL;
char **sys_meta = NULL;
int n_fp_meta = NPOW_10; // adjust later
int n_band_meta = NPOW_10; // adjust later
int n_sys_meta = 0;
int i = 0;


  if (brick == NULL || brick->open == OPEN_FALSE) return SUCCESS;

  #ifdef FORCE_DEBUG
  print_brick_info(brick);
  #endif
  

  //CPLPushErrorHandler(CPLQuietErrorHandler);

  alloc_2DC((void***)&fp_meta,   n_fp_meta,   NPOW_14, sizeof(char));
  alloc_2DC((void***)&band_meta, n_band_meta, NPOW_14, sizeof(char));
  sys_meta = system_info(&n_sys_meta);


  copy_string(fp_meta[i++], NPOW_14, "FORCE_version");
  copy_string(fp_meta[i++], NPOW_14, _VERSION_);
  
  copy_string(fp_meta[i++], NPOW_14, "FORCE_description");
  copy_string(fp_meta[i++], NPOW_14, brick->name);
  
  copy_string(fp_meta[i++], NPOW_14, "FORCE_product");
  copy_string(fp_meta[i++], NPOW_14, brick->product);
  
  copy_string(fp_meta[i++], NPOW_14, "FORCE_param");
  copy_string(fp_meta[i++], NPOW_14, brick->par);


  // how many bands to output?
  for (b=0, b_=0; b<brick->nb; b++) b_ += brick->save[b];

  if (brick->explode){
    nfiles = b_;
    nbands = 1;
  } else {
    nfiles = 1;
    nbands = b_;
  }

  enum { _brick_, _FILE_};
  alloc_3D((void****)&bands, NPOW_01, nfiles, nbands, sizeof(int));
  // dim 1: 2 slots - brick and file
  // dim 2: output files
  // dim 3: band numbers

  for (b=0, b_=0; b<brick->nb; b++){
    
    if (!brick->save[b]) continue;
    
    if (brick->explode){
      bands[_brick_][b_][0] = b;
      bands[_FILE_][b_][0]  = 1;
    } else {
      bands[_brick_][0][b_] = b;
      bands[_FILE_][0][b_]  = b_+1;
    }

    b_++;
    
  }
  
  
  //CPLSetConfigOption("GDAL_PAM_ENABLED", "YES");
  
  // get driver
  if ((driver_physical = GDALGetDriverByName(brick->format.driver)) == NULL){
    printf("%s driver not found\n", brick->format.driver); return FAILURE;}
  if ((driver = GDALGetDriverByName("MEM")) == NULL){
    printf("%s driver not found\n", "MEM"); return FAILURE;}

  // set GDAL output options
  for (o=0; o<brick->format.n; o+=2){
    #ifdef FORCE_DEBUG
    printf("setting options %s = %s\n",  brick->format.option[o], brick->format.option[o+1]);
    #endif
    options = CSLSetNameValue(options, brick->format.option[o], brick->format.option[o+1]);
  }

  switch (brick->datatype){
    case _DT_SHORT_:
      file_datatype = GDT_Int16;
      break;
    case _DT_SMALL_:
      file_datatype = GDT_Byte;
      break;
    case _DT_FLOAT_:
      file_datatype = GDT_Float32;
      break;
    case _DT_INT_:
      file_datatype = GDT_Int32;
      break;
    case _DT_USHORT_:
      file_datatype = GDT_UInt16;
      break;
    default:
      printf("unknown datatype for writing brick. ");
      return FAILURE;
  }


  // output path
  if ((lock = (char*)CPLLockFile(brick->dname, 60)) == NULL){
    printf("Unable to lock directory %s (timeout: %ds). ", brick->dname, 60);
    return FAILURE;}
  createdir(brick->dname);
  CPLUnlockFile(lock);
  lock = NULL;


  for (f=0; f<nfiles; f++){
    
    if (brick->explode){
      nchar = snprintf(bname, NPOW_10, "_%s", brick->bandname[bands[_brick_][f][0]]);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling band ID\n"); return FAILURE;}      
    } else bname[0] = '\0';
  
    nchar = snprintf(fname, NPOW_10, "%s/%s%s.%s", brick->dname, 
      brick->fname, bname, brick->format.extension);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

    timeout = lock_timeout(get_brick_size(brick));

    if ((lock = (char*)CPLLockFile(fname, timeout)) == NULL){
      printf("Unable to lock file %s (timeout: %fs, nx/ny: %d/%d). ", fname, timeout, brick->nx, brick->ny);
      return FAILURE;}


    // mosaicking into existing file
    // read and rewrite brick (safer when using compression)
    if (brick->open != OPEN_CREATE && brick->open != OPEN_BLOCK && fileexist(fname)){

      // read brick
      #ifdef FORCE_DEBUG
      printf("reading existing file.\n");
      #endif

      if ((fo = GDALOpen(fname, GA_ReadOnly)) == NULL){
        printf("Unable to open %s. ", fname); return FAILURE;}

      if (GDALGetRasterCount(fo) != nbands){
        printf("Number of bands %d do not match for UPDATE/MERGE mode (file: %d). ", 
          nbands, GDALGetRasterCount(fo)); 
        return FAILURE;}
      if (GDALGetRasterXSize(fo) != brick->nx){
        printf("Number of cols %d do not match for UPDATE/MERGE mode (file: %d). ", 
          brick->nx, GDALGetRasterXSize(fo)); 
        return FAILURE;}
      if (GDALGetRasterYSize(fo) != brick->ny){
        printf("Number of rows %d do not match for UPDATE/MERGE mode (file: %d). ", 
          brick->ny, GDALGetRasterYSize(fo)); 
        return FAILURE;}

      alloc((void**)&buf, brick->nc, sizeof(float));

      for (b=0; b<nbands; b++){

        b_brick = bands[_brick_][f][b];
        b_file  = bands[_FILE_][f][b];
        
        band = GDALGetRasterBand(fo, b_file);

        if (GDALRasterIO(band, GF_Read, 0, 0, brick->nx, brick->ny, buf, 
          brick->nx, brick->ny, GDT_Float32, 0, 0) == CE_Failure){
          printf("Unable to read %s. ", fname); return FAILURE;} 


        for (p=0; p<brick->nc; p++){

          now = get_brick(brick, b_brick, p);
          old = buf[p];

          // if both old and now are valid: keep now or merge now and old
          if (now != brick->nodata[b_brick] && old != brick->nodata[b_brick]){
            if (brick->open == OPEN_MERGE) set_brick(brick, b_brick, p, (now+old)/2.0);
          // if only old is valid, take old value
          } else if (now == brick->nodata[b_brick] && old != brick->nodata[b_brick]){
            set_brick(brick, b_brick, p, old);
          }
          // if only now is valid, nothing to do

        }

      }

      GDALClose(fo);

      free((void*)buf);

    }


    // open for block mode or write from scratch
    if (brick->open == OPEN_BLOCK && fileexist(fname) && brick->chunk > 0){
      if ((fp = GDALOpen(fname, GA_Update)) == NULL){
        printf("Unable to open %s. ", fname); return FAILURE;}
    } else {
      if ((fp = GDALCreate(driver, fname, brick->nx, brick->ny, nbands, file_datatype, options)) == NULL){
        printf("Error creating memory file %s. ", fname); return FAILURE;}
    }
      
    if (brick->open == OPEN_BLOCK){
      if (brick->chunk < 0){
        printf("attempting to write invalid chunk\n");
        return FAILURE;
      }
      nx_write     = brick->cx;
      ny_write     = brick->cy;
      xoff_write   = 0;
      yoff_write   = brick->chunk*brick->cy;
    } else {
      nx_write     = brick->nx;
      ny_write     = brick->ny;
      xoff_write   = 0;
      yoff_write   = 0;
    }


    for (b=0; b<nbands; b++){

      b_brick = bands[_brick_][f][b];
      b_file  = bands[_FILE_][f][b];

      band = GDALGetRasterBand(fp, b_file);

      switch (brick->datatype){
        case _DT_SHORT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, brick->vshort[b_brick], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;}
          break;
        case _DT_SMALL_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, brick->vsmall[b_brick], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;
        case _DT_FLOAT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, brick->vfloat[b_brick], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;
        case _DT_INT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, brick->vint[b_brick], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;
        case _DT_USHORT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, brick->vushort[b_brick], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;

        default:
          printf("unknown datatype for writing brick. ");
          return FAILURE;
      }

      GDALSetDescription(band, brick->bandname[b_brick]);
      GDALSetRasterNoDataValue(band, brick->nodata[b_brick]);

    }

    // write essential geo-metadata
    #pragma omp critical
    {
      GDALSetGeoTransform(fp, brick->geotran);
      GDALSetProjection(fp,   brick->proj);
    }

    // in case of ENVI, update description
    //if (format == _FMT_ENVI_) 
    //GDALSetDescription(fp, brick->name);

    
    // copy to physical file. This is needed for drivers that do not support CREATE
    if ((fp_physical = GDALCreateCopy(driver_physical, fname, fp, FALSE, options, NULL, NULL)) == NULL){
        printf("Error creating file %s. ", fname); return FAILURE;}

    for (i=0; i<n_sys_meta; i+=2) GDALSetMetadataItem(fp_physical, sys_meta[i], sys_meta[i+1], "FORCE");
    for (i=0; i<n_fp_meta;  i+=2) GDALSetMetadataItem(fp_physical, fp_meta[i],  fp_meta[i+1],  "FORCE");

    for (b=0; b<nbands; b++){

      b_brick = bands[_brick_][f][b];
      b_file  = bands[_FILE_][f][b];

      i = 0;

      copy_string(band_meta[i++], NPOW_14, "Domain");
      copy_string(band_meta[i++], NPOW_14, brick->domain[b_brick]);

      copy_string(band_meta[i++], NPOW_14, "Wavelength");
      nchar = snprintf(band_meta[i], NPOW_14, "%.3f", brick->wavelength[b_brick]); i++;
      if (nchar < 0 || nchar >= NPOW_14){ 
        printf("Buffer Overflow in assembling band metadata\n"); return FAILURE;}

      copy_string(band_meta[i++], NPOW_14, "Wavelength_unit");
      copy_string(band_meta[i++], NPOW_14, brick->unit[b_brick]);

      copy_string(band_meta[i++], NPOW_14, "Scale");
      nchar = snprintf(band_meta[i], NPOW_14, "%.3f", brick->scale[b_brick]); i++;
      if (nchar < 0 || nchar >= NPOW_14){ 
        printf("Buffer Overflow in assembling band metadata\n"); return FAILURE;}

      copy_string(band_meta[i++], NPOW_14, "Sensor");
      copy_string(band_meta[i++], NPOW_14, brick->sensor[b_brick]);

      get_brick_longdate(brick, b_brick, ldate, NPOW_05-1);
      copy_string(band_meta[i++], NPOW_14, "Date");
      copy_string(band_meta[i++], NPOW_14, ldate);


      band = GDALGetRasterBand(fp_physical, b_file);

      for (i=0; i<n_band_meta; i+=2) GDALSetMetadataItem(band, band_meta[i], band_meta[i+1], "FORCE");

    }
    
    GDALClose(fp_physical);
    GDALClose(fp);

  
    CPLUnlockFile(lock);
    
  }

  if (options   != NULL){ CSLDestroy(options);                      options   = NULL;}
  if (fp_meta   != NULL){ free_2DC((void**)fp_meta);                fp_meta   = NULL;}
  if (band_meta != NULL){ free_2DC((void**)band_meta);              band_meta = NULL;}
  if (sys_meta  != NULL){ free_2DC((void**)sys_meta);               sys_meta  = NULL;}
  if (bands     != NULL){ free_3D((void***)bands, NPOW_01, nfiles); bands     = NULL;}

  //CPLPopErrorHandler();

  return SUCCESS;
}


/** This function reprojects a brick into any other projection. The extent
+++ of the warped image is unknown, thus it needs to be estimated first.
+++ The reprojection might be performed in chunks if the number of pixels
+++ is too large to do it in one step.
--- tile:    will the warped image be tiled? If yes, the extent is aligned
             with the tiling scheme
--- rsm:     resampling method
--- threads: number of threads to perform warping
--- from:    source brick (modified)
--- cube:    datacube definition (holds all projection parameters)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_from_brick_to_unknown_brick(bool tile, int rsm, int threads, brick_t *src, cube_t *cube){
int i, j, p, k, b, b_, chunk_nb, nb;
GDALDriverH driver;
GDALDatasetH src_dataset;
GDALDatasetH dst_dataset;
GDALRasterBandH src_band;
GDALDataType dt = GDT_Int16;
GDALWarpOptions *wopt = NULL;
GDALWarpOperation woper;
GDALResampleAlg resample[3] = { GRA_NearestNeighbour, GRA_Bilinear, GRA_Cubic };
void *transformer = NULL;
char src_proj[NPOW_10];
double src_geotran[6];
double dst_geotran[6];
short *src_ = NULL;
short **buf_ = NULL;
short nodata;
int src_nx, src_ny, src_nc;
int dst_nx, dst_ny, dst_nc;
double tmpx, tmpy;
size_t src_dim, chunk_nx, chunk_ny;
//size_t max_mem = 1342177280; // 1.25GB
//size_t max_mem = 1073741824; // 1GB
size_t max_mem = 805306368; // 0.75GB
char nthread[NPOW_04];
int nchar;
char **papszWarpOptions = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  // register drivers and fetch in-memory driver
  if ((driver = GDALGetDriverByName("MEM")) == NULL){
    printf("could not fetch in-memory driver. "); return FAILURE;}



  nb = get_brick_nbands(src);
  src_nx = get_brick_ncols(src);
  src_ny = get_brick_nrows(src);
  src_nc = get_brick_ncells(src);
  get_brick_geotran(src, src_geotran, 6);
  get_brick_proj(src, src_proj, NPOW_10);

  // create source image
  if ((src_dataset = GDALCreate(driver, "mem", src_nx, src_ny, nb, dt, NULL)) == NULL){
    printf("could not create src image. "); return FAILURE;}
  if (GDALSetProjection(src_dataset, src_proj) == CE_Failure){
    printf("could not set src projection. "); return FAILURE;}
  if (GDALSetGeoTransform(src_dataset, src_geotran) == CE_Failure){
    printf("could not set src geotransformation. "); return FAILURE;}
    
    // "copy" data to source image
  for (b=0; b<nb; b++){
    src_band = GDALGetRasterBand(src_dataset, b+1);
    if ((src_ = get_band_short(src, b)) == NULL) return FAILURE;
    if (GDALRasterIO(src_band, GF_Write, 0, 0, src_nx, src_ny, 
      src_, src_nx, src_ny, dt, 0, 0 ) == CE_Failure){
      printf("could not 'copy' src data. "); return FAILURE;}
  }

  #ifdef FORCE_DEBUG
  printf("WKT of brick: %s\n", src_proj);
  #endif


  /** approx. extent of destination dataset, align with datacube
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  // create transformer between source and destination
  if ((transformer = GDALCreateGenImgProjTransformer(src_dataset, src_proj,
    NULL, cube->proj, false, 0, 2)) == NULL){
    printf("could not create image to image transformer. "); return FAILURE;}

  // estimate approximate extent
  if (GDALSuggestedWarpOutput(src_dataset, GDALGenImgProjTransform, 
    transformer, dst_geotran, &dst_nx, &dst_ny) == CE_Failure){
    printf("could not suggest dst extent. "); return FAILURE;}

  // align with output grid of data cube
  if (tile){

    if (tile_align(cube, dst_geotran[0], dst_geotran[3], &tmpx, &tmpy) == SUCCESS){
      dst_geotran[0] = tmpx;
      dst_geotran[3] = tmpy;
    } else {
      printf("could not align with datacube. "); return FAILURE;
    }

  }
  
  // convert computed resolution to actual dst resolution
  // add some rows and columns, just to be sure
  dst_nx = dst_nx * dst_geotran[1] / cube->res + 10;
  dst_ny = dst_ny * dst_geotran[5] / cube->res * -1 + 10;
  dst_nc = dst_nx*dst_ny;
  dst_geotran[1] = cube->res; dst_geotran[5] = -1 * cube->res; 

  // destroy transformer
  GDALDestroyGenImgProjTransformer(transformer);

  #ifdef FORCE_DEBUG
  printf("src nx/ny: %d/%d\n",  src_nx, src_ny);
  print_dvector(src_geotran, "src geotransf.", 6, 1, 2);
  printf("dst nx/ny: %d/%d\n", dst_nx, dst_ny);
  print_dvector(dst_geotran, "dst geotransf.", 6, 1, 2);
  #endif



  /** check how many bands to do simultaneously
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/
  if (nb == 1){
    chunk_nb = 1;
  } else {
    chunk_nb = nb;
    while ((size_t)src_nc*(size_t)chunk_nb*sizeof(short) > max_mem) chunk_nb--;
    if (chunk_nb < 1) chunk_nb = 1;
  }

  #ifdef FORCE_DEBUG
  printf("warp %d bands of %d bands simultaneously\n", chunk_nb, nb);
  #endif


  // iterate over chunks of bands (this is more expensive than warping all bands at once,
  // but way less expensive than each band at once. it helps to stay below RAM limit of 8GB
  for (b=0; b<nb; b+=chunk_nb){

    if (b+chunk_nb > nb) chunk_nb = nb-b;

 
    /** "create" the destination dataset
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

    if ((dst_dataset = GDALCreate(driver, "mem", dst_nx, dst_ny, chunk_nb, dt, NULL)) == NULL){
      printf("could not create dst image. "); return FAILURE;}
    if (GDALSetProjection(dst_dataset, cube->proj) == CE_Failure){
      printf("could not set dst projection. "); return FAILURE;}
    if (GDALSetGeoTransform(dst_dataset, dst_geotran) == CE_Failure){
      printf("could not set dst geotransformation. "); return FAILURE;}

      
    /** create accurate transformer between source and destination
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/
    if ((transformer = GDALCreateGenImgProjTransformer(src_dataset, src_proj,
      dst_dataset, cube->proj, false, 0, 2)) == NULL){
      printf("could not create image to image transformer. "); return FAILURE;}


    // buffer to hold warped image
    alloc_2DC((void***)&buf_, chunk_nb, dst_nc, sizeof(short));
    for (b_=0; b_<chunk_nb; b_++){
      if ((nodata = get_brick_nodata(src, b+b_)) != 0){
        #pragma omp parallel shared(b_, dst_nc, buf_, nodata) default(none) 
        {
          #pragma omp for schedule(static)
          for (p=0; p<dst_nc; p++) buf_[b_][p] = nodata;
        }
      }
    }

    
    /** set warping options
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

    wopt = GDALCreateWarpOptions();
    wopt->hSrcDS = src_dataset;
    wopt->hDstDS = NULL;
    wopt->dfWarpMemoryLimit = max_mem;
    wopt->eResampleAlg = resample[rsm];
    wopt->nBandCount = chunk_nb;
    wopt->panSrcBands = (int*)CPLMalloc(sizeof(int)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->panSrcBands[b_] = b_+b+1;
    wopt->panDstBands = (int*)CPLMalloc(sizeof(int)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->panDstBands[b_] = b_+b+1;
    wopt->pTransformerArg = transformer;
    wopt->pfnTransformer = GDALGenImgProjTransform;
    wopt->eWorkingDataType = dt;

    wopt->padfSrcNoDataReal = (double*)CPLMalloc(sizeof(double)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->padfSrcNoDataReal[b_] = get_brick_nodata(src, b_+b);
    wopt->padfSrcNoDataImag = (double*)CPLMalloc(sizeof(double)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->padfSrcNoDataImag[b_] = 0;

    wopt->padfDstNoDataReal = (double*)CPLMalloc(sizeof(double)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->padfDstNoDataReal[b_] = get_brick_nodata(src, b_+b);
    wopt->padfDstNoDataImag = (double*)CPLMalloc(sizeof(double)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->padfDstNoDataImag[b_] = 0;

    nchar = snprintf(nthread, NPOW_04, "%d", threads);
    if (nchar < 0 || nchar >= NPOW_04){ 
      printf("Buffer Overflow in assembling threads\n"); return FAILURE;}

    papszWarpOptions = CSLSetNameValue(papszWarpOptions, "NUM_THREADS", nthread);
    papszWarpOptions = CSLSetNameValue(papszWarpOptions, "INIT_DEST", "-9999");
    wopt->papszWarpOptions = CSLDuplicate(papszWarpOptions);


    /** check if we can warp the image in one operation or need chunks
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/
    chunk_nx = src_nx; chunk_ny = src_ny; k = 0;
    //while ((src_dim = chunk_nx*chunk_ny*(size_t)nb*sizeof(short)) > INT_MAX){
    while ((src_dim = (size_t)chunk_nx*(size_t)chunk_ny*(size_t)chunk_nb*sizeof(short)) > max_mem){
      if (k % 2 == 0) chunk_nx/=2; else chunk_ny/=2;
      k++;}
    
    #ifdef FORCE_DEBUG
    printf("\nImage brick is warped in %d chunks.\n", k+1);
    #endif

    /** warp the image, use chunks if necessary
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/
    if (woper.Initialize(wopt) == CE_Failure){
        printf("could not initialize warper. "); return FAILURE;}

    for (i=0; i<src_ny; i+=chunk_ny){
    for (j=0; j<src_nx; j+=chunk_nx){
      if (woper.WarpRegionToBuffer(0, 0, dst_nx, dst_ny, buf_[0],
        dt, j, i, chunk_nx, chunk_ny) == CE_Failure){
        printf("could not warp. "); return FAILURE;}
    }
    }

    GDALDestroyGenImgProjTransformer(wopt->pTransformerArg);
    GDALDestroyWarpOptions(wopt);
  
    for (b_=0; b_<chunk_nb; b_++){
      re_alloc((void**)&src->vshort[b_+b], src_nc, dst_nc, sizeof(short));
      memmove(src->vshort[b_+b], buf_[b_], dst_nc*sizeof(short));
    }

    free((void*)buf_[0]); free((void*)buf_); buf_ = NULL;

    GDALClose(dst_dataset);

    #ifdef FORCE_DEBUG
    printf("\n%d bands were warped in %d chunks.\n", chunk_nb, k+1);
    #endif
    
  }

  GDALClose(src_dataset);


  // update geo metadata
  set_brick_geotran(src, dst_geotran);
  set_brick_ncols(src, dst_nx);
  set_brick_nrows(src, dst_ny);
  set_brick_proj(src, cube->proj);


  #ifdef FORCE_DEBUG
  print_brick_info(src);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("warping brick to brick", TIME);
  #endif

  return SUCCESS;
}


/** This function reprojects an image from disc into any other projection. 
+++ The extent of the warped image is known, and a target brick needs to 
+++ be given, which defines extent, projection etc. 
+++ The reprojection might be performed in chunks if the number of pixels
+++ is too large to do it in one step.
--- rsm:         resampling method
--- threads:     number of threads to perform warping
--- fname:       filename
--- dst:         destination brick (modified)
--- src_b:       which band to warp?    (band in file)
--- dst_b:       which band to warp to? (band in destination brick)
--- src_nodata:  nodata value of band in file
+++ Return:      SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_from_disc_to_known_brick(int rsm, int threads, const char *fname, brick_t *dst, int src_b, int dst_b, int src_nodata){
GDALDatasetH src_dataset;
GDALDatasetH dst_dataset;
//GDALRasterBandH src_band;
GDALDriverH driver;
GDALDataType dt = GDT_Float32;
GDALWarpOptions *wopt;
GDALWarpOperation woper;
GDALResampleAlg resample[3] = { GRA_NearestNeighbour, GRA_Bilinear, GRA_Cubic };
CPLErr eErr = CE_Failure;
void *transformer = NULL;
float *buf = NULL;
const char *src_proj = NULL;
char dst_proj[NPOW_10];
double dst_geotran[6];
int i, j, p, np, k, k_do;
int dst_nodata;
int nc_done_, nc_done = 0;
int src_nb, dst_nb;
int dst_nx, dst_ny;
int chunk_nx, chunk_ny;
int chunk_xoff, chunk_yoff;
float tmp;
char nthread[NPOW_04];
char initdata[NPOW_04];
int nchar;
char **papszWarpOptions = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

#ifdef FORCE_DEBUG
printf("warp_from_disc_to_known_brick should handle multiband src and dst images\n");
#endif
  
  // register drivers and fetch in-memory driver
  if ((driver = GDALGetDriverByName("MEM")) == NULL){
    printf("could not fetch in-memory driver. "); return FAILURE;}


  /** "create" source dataset
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  if ((src_dataset = GDALOpen(fname, GA_ReadOnly)) == NULL){
    printf("unable to open image for warping: %s\n", fname); return FAILURE;}

 
  #ifdef FORCE_DEBUG
  printf("src nodata is %d, band is %d, dataset is %s\n", src_nodata, src_b+1, fname);
  #endif

  src_proj = GDALGetProjectionRef(src_dataset);
  CPLAssert(src_proj != NULL && strlen(src_proj) > 0);
  src_nb = GDALGetRasterCount(src_dataset);
  
  if (src_b >= src_nb){
    printf("Requested band %d is out of bounds %d (disc)! ", src_b, src_nb); return FAILURE;}

  #ifdef FORCE_DEBUG
  printf("WKT of image on disc: %s\n", src_proj);
  #endif
  

  /** "create" the destination dataset
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  get_brick_geotran(dst, dst_geotran, 6);
  get_brick_proj(dst, dst_proj, NPOW_10);
  dst_nodata = get_brick_nodata(dst, 0);
  dst_nb = get_brick_nbands(dst);
  dst_nx = get_brick_ncols(dst);
  dst_ny = get_brick_nrows(dst);

  if (dst_b >= dst_nb){
    printf("Requested band %d is out of bounds %d (brick)! ", dst_b, dst_nb); return FAILURE;}

  if ((dst_dataset = GDALCreate(driver, "mem", dst_nx, dst_ny, 1, dt, NULL)) == NULL){
    printf("could not create dst image. "); return FAILURE;}
  if (GDALSetProjection(dst_dataset, dst_proj)){
    printf("could not set dst projection. "); return FAILURE;}
  if (GDALSetGeoTransform(dst_dataset, dst_geotran)){
    printf("could not set dst geotransformation. "); return FAILURE;}

  #ifdef FORCE_DEBUG
  printf("warp to UL-X: %.0f / UL-Y: %.0f @ res: %.0f\n", dst_geotran[0], dst_geotran[3], dst_geotran[1]);
  #endif
  

  /** create accurate transformer between source and destination
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  if ((transformer = GDALCreateGenImgProjTransformer(src_dataset, src_proj,
    dst_dataset, dst_proj, false, 0, 2)) == NULL){
    printf("could not create image to image transformer. "); return FAILURE;}
  

  /** set warping options
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  wopt = GDALCreateWarpOptions();
  wopt->hSrcDS = src_dataset;
  wopt->hDstDS = NULL;
  wopt->eResampleAlg = resample[rsm];
  wopt->nBandCount = 1;
  wopt->panSrcBands = 
    (int *) CPLMalloc(sizeof(int)*wopt->nBandCount);
  wopt->panSrcBands[0] = src_b+1;
  wopt->panDstBands = 
    (int *) CPLMalloc(sizeof(int)*wopt->nBandCount);
  wopt->panDstBands[0] = src_b+1;
  wopt->pTransformerArg = transformer;
  wopt->pfnTransformer = GDALGenImgProjTransform;
  wopt->eWorkingDataType = dt;

  wopt->padfSrcNoDataReal =
    (double*) CPLMalloc(sizeof(double)*wopt->nBandCount);
  wopt->padfSrcNoDataReal[0] = src_nodata;
  wopt->padfSrcNoDataImag =
    (double*) CPLMalloc(sizeof(double)*wopt->nBandCount);
  wopt->padfSrcNoDataImag[0] = 0.0;

  wopt->padfDstNoDataReal =
    (double*) CPLMalloc(sizeof(double)*wopt->nBandCount);
  wopt->padfDstNoDataReal[0] = dst_nodata;
  wopt->padfDstNoDataImag =
    (double*) CPLMalloc(sizeof(double)*wopt->nBandCount);
  wopt->padfDstNoDataImag[0] = 0.0;

  nchar = snprintf(nthread, NPOW_04, "%d", threads);
  if (nchar < 0 || nchar >= NPOW_04){ 
    printf("Buffer Overflow in assembling threads\n"); return FAILURE;}
    
  nchar = snprintf(initdata, NPOW_04, "%d", dst_nodata);
  if (nchar < 0 || nchar >= NPOW_04){ 
    printf("Buffer Overflow in assembling nodata\n"); return FAILURE;}
   
  papszWarpOptions = CSLSetNameValue(papszWarpOptions, "NUM_THREADS", nthread);
  papszWarpOptions = CSLSetNameValue(papszWarpOptions, "INIT_DEST", initdata);
  wopt->papszWarpOptions = CSLDuplicate(papszWarpOptions);

  // set nodata in destination image
  if (dst_nodata != 0){
    #pragma omp parallel shared(dst_nx, dst_ny, dst, dst_b, dst_nodata) default(none) 
    {
      #pragma omp for schedule(static)
      for (p=0; p<dst_nx*dst_ny; p++) set_brick(dst, dst_b, p, dst_nodata);
    }
  }

  // warp
  woper.Initialize(wopt);

  chunk_xoff = 0; chunk_yoff = 0;
  chunk_nx = dst_nx; chunk_ny = dst_ny;
  k = 0, k_do = 0;

  // start with whole image, if unsuccessful, warp in chunks
  while (eErr != CE_None && chunk_ny > 1){

    #ifdef FORCE_DEBUG
    printf("try to warp %d x %d pix, x / y offset %d / %d. ", 
      chunk_nx, chunk_ny, chunk_xoff, chunk_yoff);
    #endif

    alloc((void**)&buf, chunk_nx*chunk_ny, sizeof(float));
    if (dst_nodata != 0){
      #pragma omp parallel shared(chunk_nx, chunk_ny, buf, dst_nodata) default(none) 
      {
        #pragma omp for schedule(static)
        for (np=0; np<chunk_nx*chunk_ny; np++) buf[np] = dst_nodata;
      }
    }

    // warp
    eErr = woper.WarpRegionToBuffer(chunk_xoff, chunk_yoff, chunk_nx, chunk_ny, buf, dt, 0, 0, 0, 0);
 
    if (eErr == CE_Failure){

      #ifdef FORCE_DEBUG
      printf("Failed. Try smaller chunks.\n");
      #endif

      // decrease size
      if (k++ % 2 == 0){
        tmp = ceil(chunk_ny/2.0); chunk_ny = tmp;
      } else {
        tmp = ceil(chunk_nx/2.0); chunk_nx = tmp;
      }

    } else {

      #ifdef FORCE_DEBUG
      printf("OK.\n");
      #endif

      // copy buffer to image
      nc_done_ = 0;
      
      #pragma omp parallel private(j, p, np) shared(dst_nx, dst_ny, chunk_nx, chunk_ny, chunk_xoff, chunk_yoff, buf, dst, dst_b, dst_nodata) reduction(+: nc_done_) default(none) 
      {

        #pragma omp for schedule(static)
        for (i=chunk_yoff; i<chunk_yoff+chunk_ny; i++){
        for (j=chunk_xoff; j<chunk_xoff+chunk_nx; j++){
          if (i >= dst_ny || j >= dst_nx) continue;
          p  = i*dst_nx+j;
          np = (i-chunk_yoff)*chunk_nx + (j-chunk_xoff);
          set_brick(dst, dst_b, p, buf[np]);
          buf[np] = dst_nodata;
          nc_done_++;
        }
        }

      }
      
      nc_done += nc_done_;

      // if part of image is still missing, increase offsets
      if (nc_done < dst_nx*dst_ny){
        if (chunk_yoff < dst_ny && chunk_ny < dst_ny){
          chunk_yoff+=chunk_ny;
        } else if (chunk_xoff < dst_nx && chunk_nx < dst_nx){
          chunk_xoff+=chunk_nx;
        }
        eErr = CE_Failure;
      } else {
        eErr = CE_None;
      }

      k_do++;

    }

    free((void*)buf);

  }

  #ifdef FORCE_DEBUG
  printf("\nImage was warped in %d chunks.\n", k_do);
  #endif


  /** close & clean
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/
  GDALDestroyGenImgProjTransformer(wopt->pTransformerArg);
  GDALDestroyWarpOptions(wopt);
  GDALClose(src_dataset);
  GDALClose(dst_dataset);

  #ifdef FORCE_CLOCK
  proctime_print("warping disc to brick", TIME);
  #endif

  return SUCCESS;
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
  i_to = floor(i_from*from->geotran[1]/to->geotran[1]);

  j_from = p_from-i_from*from->nx;
  j_to = floor(j_from*from->geotran[1]/to->geotran[1]);

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
  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);

  j_from = p_from-i_from*from->nx;
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);

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
  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);

  j_from = p_from-i_from*from->nx;
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);
  
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

  i_to = floor(i_from*from->geotran[1]/to->geotran[1]);
  j_to = floor(j_from*from->geotran[1]/to->geotran[1]);


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

  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);

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

  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);
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
  

  copy_string(brick->product, NPOW_03, product);

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


/** This function sets the parent directory-name of a brick
--- brick:  brick
--- pname:  parent directory-name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_parentname(brick_t *brick, const char *pname){


  copy_string(brick->pname, NPOW_10, pname);

  return;
}


/** This function gets the parent directory-name of a brick
--- brick:  brick
--- pname:  parent directory-name (modified)
--- size:   length of the buffer for pname
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_parentname(brick_t *brick, char pname[], size_t size){


  copy_string(pname, size, brick->pname);

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
  brick->width = brick->nx*brick->res;

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
  brick->height = brick->ny*brick->res;

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
    if (get_brick_save(brick, b)) size += brick->nc*brick->byte;
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
  brick->cwidth = brick->cx*brick->res;

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
  brick->cheight = brick->cy*brick->res;

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


/** This function sets the chunk ID of a brick
--- brick:  brick
--- chunk:  chunk ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_chunk(brick_t *brick, int chunk){
  
  if (chunk >= 0 && chunk >= brick->nchunk) printf("current chunk %d is higher than max chunks %d.\n", chunk, brick->nchunk);

  brick->chunk = chunk;
  
  return;
}


/** This function gets the chunk ID of a brick
--- brick:  brick
+++ Return: chunk ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_chunk(brick_t *brick){
  
  return brick->chunk;
}


/** This function sets the number of chunks of a brick
--- brick:  brick
--- nchunk: number of chunks
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_nchunks(brick_t *brick, int nchunk){
  
  if (nchunk < 0) printf("nchunks %d < 0.\n", nchunk);

  brick->nchunk = nchunk;
  
  return;
}


/** This function gets the number of chunks of a brick
--- brick:  brick
+++ Return: number of chunks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_nchunks(brick_t *brick){
  
  return brick->nchunk;
}


/** This function sets the tile X-ID of a brick
--- brick:  brick
--- tx:     tile X-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_tilex(brick_t *brick, int tx){
  
  if (tx >= 9999 || tx < -999) printf("tile-x is out of bounds.\n");

  brick->tx = tx;
  
  return;
}


/** This function gets the tile X-ID of a brick
--- brick:  brick
+++ Return: tile X-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_tilex(brick_t *brick){
  
  return brick->tx;
}


/** This function sets the tile Y-ID of a brick
--- brick:  brick
--- ty:     tile Y-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_tiley(brick_t *brick, int ty){
  
  if (ty >= 9999 || ty < -999) printf("tile-y is out of bounds.\n");

  brick->ty = ty;
  
  return;
}


/** This function gets the tile Y-ID of a brick
--- brick:  brick
+++ Return: tile Y-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_brick_tiley(brick_t *brick){
  
  return brick->ty;
}


/** This function sets the resolution of a brick
--- brick:  brick
--- res:    resolution
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_res(brick_t *brick, double res){

  if (res <= 0) printf("resolution must be > 0.\n");

  brick->res = res;
  brick->geotran[1] = res;
  brick->geotran[5] = res*-1;
  brick->width  = brick->nx*brick->res;
  brick->height = brick->ny*brick->res;

  return;
}


/** This function gets the resolution of a brick
--- brick:  brick
+++ Return: resolution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_res(brick_t *brick){
  
  return brick->res;
}


/** This function sets the UL-X coordinate of a brick
--- brick:   brick
--- ulx: UL-X coordinate
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_ulx(brick_t *brick, double ulx){

  brick->geotran[0] = ulx;

  return;
}


/** This function gets the UL-X coordinate of a brick
--- brick:  brick
+++ Return: UL-X coordinate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_ulx(brick_t *brick){
  
  return brick->geotran[0];
}


/** This function gets the X coordinate of a column of a brick
--- brick:  brick
--- j:      column
+++ Return: X coordinate of a column
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_x(brick_t *brick, int j){
  
  return (brick->geotran[0] + j*brick->geotran[1]);
}


/** This function sets the UL-Y coordinate of a brick
--- brick:  brick
--- uly:    UL-Y coordinate
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_uly(brick_t *brick, double uly){

  brick->geotran[3] = uly;

  return;
}


/** This function gets the UL-Y coordinate of a brick
--- brick:  brick
+++ Return: UL-Y coordinate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_uly(brick_t *brick){
  
  return brick->geotran[3];
}


/** This function gets the Y coordinate of a row of a brick
--- brick:  brick
--- i:      row
+++ Return: Y coordinate of a row 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_brick_y(brick_t *brick, int i){
  
  return (brick->geotran[3] + i*brick->geotran[5]);
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

  brick->res = geotran[1];
  brick->geotran[0] = geotran[0];
  brick->geotran[1] = geotran[1];
  brick->geotran[2] = geotran[2];
  brick->geotran[3] = geotran[3];
  brick->geotran[4] = geotran[4];
  brick->geotran[5] = geotran[5];

  return;
}


/** This function gets the geotransformation of a brick
--- brick:   brick
--- geotran: geotransformation (modified)
--- size:    length of the buffer for geotran
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_brick_geotran(brick_t *brick, double geotran[], size_t size){

  if (size != 6){
    printf("array is not compatible for getting geotran.\n"); return;}

  geotran[0] = brick->geotran[0];
  geotran[1] = brick->geotran[1];
  geotran[2] = brick->geotran[2];
  geotran[3] = brick->geotran[3];
  geotran[4] = brick->geotran[4];
  geotran[5] = brick->geotran[5];
  
  return;
}


/** This function sets the width of a brick
--- brick:  brick
--- width:  width
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_brick_width(brick_t *brick, double width){
  
  if (width != brick->nx*brick->res) printf("width does not match with nx*res.\n");

  brick->width = brick->nx*brick->res;

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
  
  if (height != brick->ny*brick->res) printf("height does not match with ny*res.\n");

  brick->height = brick->ny*brick->res;
  
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
  
  if (cwidth != brick->cx*brick->res) printf("chunking width does not match with cx*res.\n");

  brick->cwidth = brick->cx*brick->res;

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
  
  if (cheight != brick->cy*brick->res) printf("chunking height does not match with cy*res.\n");

  brick->cheight = brick->cy*brick->res;
  
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


  copy_string(brick->unit[b], NPOW_04, unit);

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


  copy_string(brick->sensor[b], NPOW_04, sensor);

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

