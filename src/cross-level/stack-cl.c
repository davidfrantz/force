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
This file contains functions for organizing stacks in memory, and output
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "stack-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_multiproc.h"  // CPL Multi-Threading
#include "gdalwarper.h"     // GDAL warper related entry points and defs
#include "ogr_spatialref.h" // coordinate systems services


/** This function allocates a stack
--- nb:       number of bands
--- nc:       number of cells
--- datatype: datatype
+++ Return:   stack (must be freed with free_stack)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *allocate_stack(int nb, int nc, int datatype){
stack_t *stack = NULL;


  if (nb < 1){
    printf("cannot allocate %d-band stack.\n", nb); 
    return NULL;}

  if (nc < 1 && datatype != _DT_NONE_){
    printf("cannot allocate %d-cell stack.\n", nc);
    return NULL;}

  alloc((void**)&stack, 1, sizeof(stack_t));

  init_stack(stack);    
  set_stack_nbands(stack, nb);

  alloc((void**)&stack->save,       nb, sizeof(bool));
  alloc((void**)&stack->nodata,     nb, sizeof(int));
  alloc((void**)&stack->scale,      nb, sizeof(float));
  alloc((void**)&stack->wavelength, nb, sizeof(float));
  alloc_2D((void***)&stack->unit, nb, NPOW_04, sizeof(char));
  alloc_2D((void***)&stack->domain, nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&stack->bandname,   nb, NPOW_10, sizeof(char));
  alloc_2D((void***)&stack->sensor,     nb, NPOW_04, sizeof(char));
  alloc((void**)&stack->date, nb, sizeof(date_t));
  
  init_stack_bands(stack);

  if (allocate_stack_bands(stack, nb, nc, datatype) == FAILURE){
    printf("couldn't allocate bands.\n"); return NULL;}

  return stack;
}


/** This function re-allocates a stack
--- stack:  stack (modified)
--- nb:     number of bands (new)
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int reallocate_stack(stack_t *stack, int nb){
int b;
int nb0 = get_stack_nbands(stack);
int nc = get_stack_ncells(stack);
int datatype = get_stack_datatype(stack);

  if (nb == nb0) return SUCCESS;

  if (nb < 1){
    printf("cannot reallocate %d-band stack.\n", nb); 
    return FAILURE;}

  if (nc < 1){
    printf("cannot reallocate %d-cell stack.\n", nc);
    return FAILURE;}

  if (datatype == _DT_NONE_){
    printf("cannot reallocate stack with no datatype.\n");
    return FAILURE;}

  re_alloc((void**)&stack->save,         nb0, nb, sizeof(bool));
  re_alloc((void**)&stack->nodata,       nb0, nb, sizeof(int));
  re_alloc((void**)&stack->scale,        nb0, nb, sizeof(float));
  re_alloc((void**)&stack->wavelength,   nb0, nb, sizeof(float));
  re_alloc_2D((void***)&stack->unit,   nb0, NPOW_04, nb, NPOW_04, sizeof(char));
  re_alloc_2D((void***)&stack->domain,   nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&stack->bandname, nb0, NPOW_10, nb, NPOW_10, sizeof(char));
  re_alloc_2D((void***)&stack->sensor,   nb0, NPOW_04, nb, NPOW_04, sizeof(char));
  re_alloc((void**)&stack->date,         nb0, nb, sizeof(date_t));

  if (reallocate_stack_bands(stack, nb) == FAILURE){
    printf("couldn't reallocate bands.\n"); return FAILURE;}

  if (nb > nb0){
    for (b=nb0; b<nb; b++) copy_stack_band(stack, b, stack, 0);
  }

  set_stack_nbands(stack, nb);

  return SUCCESS;
}


/** This function copies a stack
--- from:     source stack
--- nb:       number of bands (new)
--- datatype: datatype (new)
+++ Return:   new stack (must be freed with free_stack)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *copy_stack(stack_t *from, int nb, int datatype){
stack_t *stack = NULL; 
int  b;

  if (from->chunk < 0){
    if ((stack = allocate_stack(nb, from->nc, datatype)) == NULL) return NULL;
  } else {
    if ((stack = allocate_stack(nb, from->cc, datatype)) == NULL) return NULL;
  }


  set_stack_name(stack, from->name);
  set_stack_product(stack, from->product);
  set_stack_dirname(stack, from->dname);
  set_stack_filename(stack, from->fname);
  set_stack_sensorid(stack, from->sid);
  set_stack_format(stack, from->format);
  set_stack_open(stack, from->open);
  set_stack_explode(stack, from->explode);

  set_stack_geotran(stack, from->geotran);
  set_stack_nbands(stack, nb);
  set_stack_ncols(stack, from->nx);
  set_stack_nrows(stack, from->ny);
  set_stack_chunkncols(stack, from->cx);
  set_stack_chunknrows(stack, from->cy);
  set_stack_chunkwidth(stack, from->cwidth);
  set_stack_chunkheight(stack, from->cheight);
  set_stack_nchunks(stack, from->nchunk);
  set_stack_chunk(stack, from->chunk);
  set_stack_tilex(stack, from->tx);
  set_stack_tiley(stack, from->ty);
  set_stack_proj(stack, from->proj);
  set_stack_par(stack, from->par);

  if (nb == from->nb){
    for (b=0; b<nb; b++) copy_stack_band(stack, b, from, b);
  } else {
    for (b=0; b<nb; b++) copy_stack_band(stack, b, from, 0);
  }

  return stack;
}


/** This function frees a stack
--- stack:  stack
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_stack(stack_t *stack){
int nb;

  if (stack == NULL) return;
  
  nb = get_stack_nbands(stack);
  
  if (stack->save        != NULL) free((void*)stack->save);
  if (stack->nodata      != NULL) free((void*)stack->nodata);
  if (stack->scale       != NULL) free((void*)stack->scale);
  if (stack->wavelength  != NULL) free((void*)stack->wavelength);
  if (stack->date        != NULL) free((void*)stack->date);
  stack->save        = NULL;
  stack->nodata      = NULL;
  stack->scale       = NULL;
  stack->wavelength  = NULL;
  stack->date        = NULL;
  
  if (stack->unit     != NULL) free_2D((void**)stack->unit,     nb);
  if (stack->domain   != NULL) free_2D((void**)stack->domain,   nb);
  if (stack->bandname != NULL) free_2D((void**)stack->bandname, nb);
  if (stack->sensor   != NULL) free_2D((void**)stack->sensor,   nb);
  stack->unit     = NULL;
  stack->domain   = NULL;
  stack->bandname = NULL;
  stack->sensor   = NULL;
  
  free_stack_bands(stack);

  free((void*)stack);
  stack = NULL;

  return;
}


/** This function allocates the bandwise information in a stack
--- stack:    stack (modified)
--- nb:       number of bands
--- nc:       number of cells
--- datatype: datatype
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int allocate_stack_bands(stack_t *stack, int nb, int nc, int datatype){
int nbyte;


  switch (datatype){
    case _DT_NONE_:
      set_stack_datatype(stack, _DT_NONE_);
      set_stack_byte(stack, (nbyte = 0));
      return SUCCESS;
    case _DT_SHORT_:
      set_stack_datatype(stack, _DT_SHORT_);
      set_stack_byte(stack, (nbyte = sizeof(short)));
      alloc_2D((void***)&stack->vshort, nb, nc, nbyte);
      break;
    case _DT_SMALL_:
      set_stack_datatype(stack, _DT_SMALL_);
      set_stack_byte(stack, (nbyte = sizeof(small)));
      alloc_2D((void***)&stack->vsmall, nb, nc, nbyte);
      break;
    case _DT_FLOAT_:
      set_stack_datatype(stack, _DT_FLOAT_);
      set_stack_byte(stack, (nbyte = sizeof(float)));
      alloc_2D((void***)&stack->vfloat, nb, nc, nbyte);
      break;
    case _DT_INT_:
      set_stack_datatype(stack, _DT_INT_);
      set_stack_byte(stack, (nbyte = sizeof(int)));
      alloc_2D((void***)&stack->vint, nb, nc, nbyte);
      break;
    case _DT_USHORT_:
      set_stack_datatype(stack, _DT_USHORT_);
      set_stack_byte(stack, (nbyte = sizeof(ushort)));
      alloc_2D((void***)&stack->vushort, nb, nc, nbyte);
      break;
    default:
      printf("unknown datatype for allocating stack. ");
      return FAILURE;
  }

  return SUCCESS;
}


/** This function re-allocates the bandwise information in a stack
--- stack:    stack (modified)
--- nb:       number of bands (new)
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int reallocate_stack_bands(stack_t *stack, int nb){
int nbyte = get_stack_byte(stack);
int nb0 = get_stack_nbands(stack);
int nc0 = get_stack_ncells(stack);
int nc  = get_stack_ncells(stack);
int datatype = get_stack_datatype(stack);


  switch (datatype){
    case _DT_SHORT_:
      re_alloc_2D((void***)&stack->vshort, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_SMALL_:
      re_alloc_2D((void***)&stack->vsmall, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_FLOAT_:
      re_alloc_2D((void***)&stack->vfloat, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_INT_:
      re_alloc_2D((void***)&stack->vint, nb0, nc0, nb, nc, nbyte);
      break;
    case _DT_USHORT_:
      re_alloc_2D((void***)&stack->vushort, nb0, nc0, nb, nc, nbyte);
      break;
    default:
      printf("unknown datatype for allocating stack. ");
      return FAILURE;
  }

  return SUCCESS;
}


/** This function copies a bandwise in a stack
--- stack:  target stack (modified)
--- b:      target band
--- from:   source stack
--- b_from: source band
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void copy_stack_band(stack_t *stack, int b, stack_t *from, int b_from){
  
  
  set_stack_save(stack, b, from->save[b_from]);
  set_stack_nodata(stack, b, from->nodata[b_from]);
  set_stack_scale(stack, b, from->scale[b_from]);
  set_stack_wavelength(stack, b, from->wavelength[b_from]);
  set_stack_unit(stack, b, from->unit[b_from]);
  set_stack_domain(stack, b, from->domain[b_from]);
  set_stack_bandname(stack, b, from->bandname[b_from]);
  set_stack_sensor(stack, b, from->sensor[b_from]);
  set_stack_date(stack, b, from->date[b_from]);

  return;
}


/** This function frees the bandwise information in a stack
--- stack:  stack
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_stack_bands(stack_t *stack){
int nb;

  if (stack == NULL) return;
  
  nb = get_stack_nbands(stack);

  if (stack->vshort  != NULL) free_2D((void**)stack->vshort,  nb); 
  if (stack->vsmall  != NULL) free_2D((void**)stack->vsmall,  nb); 
  if (stack->vfloat  != NULL) free_2D((void**)stack->vfloat,  nb); 
  if (stack->vint    != NULL) free_2D((void**)stack->vint,    nb); 
  if (stack->vushort != NULL) free_2D((void**)stack->vushort, nb); 
  
  stack->vshort  = NULL;
  stack->vsmall  = NULL;  
  stack->vfloat  = NULL;  
  stack->vint    = NULL;  
  stack->vushort = NULL;  

  return;
}


/** This function crops a stack. The cropping radius is given in projection
+++ units, and the corresponding number of pixels is removed from each side
+++ of the image. The input stack is freed within. The cropped stack is 
+++ returned.
--- from:     source stack (freed)
--- radius:   cropping radius in projection units
+++ Return:   cropped stack (must be freed with free_stack)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
stack_t *crop_stack(stack_t *from, double radius){
stack_t *stack = NULL; 
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
    free_stack(from);
    return NULL;
  }

  nb = get_stack_nbands(from);
  res = get_stack_res(from);
  datatype = get_stack_datatype(from);
  
  stack = copy_stack(from, nb, _DT_NONE_);

  if (from->chunk < 0){
    nx = get_stack_ncols(from);
    ny = get_stack_nrows(from);
  } else {
    nx = get_stack_chunkncols(from);
    ny = get_stack_chunknrows(from);
  }

  pix = (int)(radius/res);
  nx_ = nx - 2*pix;
  ny_ = ny - 2*pix;
  nc_ = nx_*ny_;

  if (from->chunk < 0){
    set_stack_ncols(stack, nx_);
    set_stack_nrows(stack, ny_);
  } else {
    set_stack_chunkncols(stack, nx_);
    set_stack_chunknrows(stack, ny_);
  }
  allocate_stack_bands(stack, nb, nc_, datatype);
  
  #ifdef FORCE_DEBUG
  int nc;
  if (from->chunk < 0){
    nc = get_stack_ncells(from);
  } else {
    nc = get_stack_chunkncells(from);
  }
  printf("cropping %d -> %d cols\n", nx, nx_);
  printf("cropping %d -> %d rows\n", ny, ny_);
  printf("cropping %d -> %d pixels\n", nc, nc_);
  #endif

  switch (datatype){
    case _DT_NONE_:
      free_stack(from);
      return stack;
    case _DT_SHORT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p_ = i*nx_+j;
        p  = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) stack->vshort[b][p_] = from->vshort[b][p];
      }
      }
      break;
    case _DT_SMALL_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) stack->vsmall[b][p_] = from->vsmall[b][p];
      }
      }
      break;
    case _DT_FLOAT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) stack->vfloat[b][p_] = from->vfloat[b][p];
      }
      }
      break;
    case _DT_INT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) stack->vint[b][p_] = from->vint[b][p];
      }
      }
      break;
    case _DT_USHORT_:
      for (i=0; i<ny_; i++){
      for (j=0; j<nx_; j++){
        p  = i*nx_+j;
        p_ = (i+pix)*nx + (j+pix);
        for (b=0; b<nb; b++) stack->vushort[b][p_] = from->vushort[b][p];
      }
      }
      break;
    default:
      printf("unknown datatype. ");
      free_stack(stack);
      free_stack(from);
      return NULL;
  }

  free_stack(from);

  return stack;
}


/** This function initializes all values in a stack
--- stack:  stack
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_stack(stack_t *stack){
int i;


  copy_string(stack->name,      NPOW_10, "NA");
  copy_string(stack->product,   NPOW_03, "NA");
  copy_string(stack->dname,     NPOW_10, "NA");
  copy_string(stack->fname,     NPOW_10, "NA");
  copy_string(stack->extension, NPOW_02, "NA");

  stack->sid = -1;
  stack->format = 0;
  stack->open = OPEN_FALSE;
  stack->explode = 0;
  stack->datatype = _DT_NONE_;
  stack->byte = 0;

  stack->nb =  0;
  stack->nx =  0;
  stack->ny =  0;
  stack->nc =  0;
  stack->cx =  0;
  stack->cy =  0;
  stack->cc =  0;
  stack->res = 0;
  for (i=0; i<6; i++) stack->geotran[i] = 0;
  stack->width  = 0;
  stack->height = 0;
  stack->cwidth  = 0;
  stack->cheight = 0;
  stack->chunk = -1;
  stack->nchunk = 0;
  stack->tx = 0;
  stack->ty = 0;

  copy_string(stack->proj,NPOW_10, "NA");
  copy_string(stack->par, NPOW_14, "NA");

  stack->save   = NULL;
  stack->nodata = NULL;
  stack->scale  = NULL;

  stack->wavelength = NULL;
  stack->unit = NULL;
  stack->domain = NULL;
  stack->bandname   = NULL;
  stack->sensor     = NULL;
  stack->date       = NULL;

  stack->vshort  = NULL;
  stack->vfloat  = NULL;
  stack->vint    = NULL;
  stack->vushort = NULL;
  stack->vsmall   = NULL;

  return;  
}


/** This function initializes all bandwise information in a stack
--- stack:  stack
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_stack_bands(stack_t *stack){
int b;

  for (b=0; b<stack->nb; b++){
    stack->save[b] = false;
    stack->nodata[b] = 0;
    stack->scale[b] = 0;
    stack->wavelength[b] = 0;
    copy_string(stack->unit[b],     NPOW_04, "NA");
    copy_string(stack->domain[b],   NPOW_10, "NA");
    copy_string(stack->bandname[b], NPOW_10, "NA");
    copy_string(stack->sensor[b],   NPOW_04, "NA");
    init_date(&stack->date[b]);
  }

  return;
}


/** This function prints a stack
--- stack:  stack
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_stack_info(stack_t *stack){
int b;


  printf("\nstack info for %s - %s - SID %d\n", stack->name, stack->product, stack->sid);
  printf("open: %d, format %d, explode %d\n", 
    stack->open, stack->format, stack->explode);
  printf("datatype %d with %d bytes\n", 
    stack->datatype, stack->byte);
  printf("filename: %s/%s.%s\n", stack->dname, stack->fname, stack->extension);
  printf("nx: %d, ny: %d, nc: %d, res: %.3f, nb: %d\n", 
    stack->nx, stack->ny, stack->nc, 
    stack->res, stack->nb);
  printf("width: %.1f, height: %.1f\n", 
    stack->width, stack->height);
  printf("chunking: nx: %d, ny: %d, nc: %d, width: %.1f, height: %.1f, #: %d\n", 
    stack->cx, stack->cy, stack->cc, stack->cwidth, stack->cheight, stack->nchunk);
  printf("active chunk: %d, tile X%04d_Y%04d\n", stack->chunk, stack->tx, stack->ty);
  printf("ulx: %.3f, uly: %.3f\n", 
    stack->geotran[0], stack->geotran[3]);
  printf("proj: %s\n", stack->proj);
  printf("par: %s\n", stack->par);

  for (b=0; b<stack->nb; b++) print_stack_band_info(stack, b);

  printf("\n");

  return;
}


/** This function prints bandwise information in a stack
--- stack:  stack
--- b:      band
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_stack_band_info(stack_t *stack, int b){
  
  printf("++band # %d - save %d, nodata: %d, scale: %f\n", 
    b, stack->save[b], stack->nodata[b], stack->scale[b]);
  printf("wvl: %f, domain: %s, band name: %s, sensor ID: %s\n", 
    stack->wavelength[b], stack->domain[b], stack->bandname[b], stack->sensor[b]);
  print_date(&stack->date[b]);
    
  return;
}


/** This function outputs a stack
--- stack:  stack
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int write_stack(stack_t *stack){
int f, b, b_, p;
int b_stack, b_file, nbands, nfiles;
int ***bands = NULL;
char *lock = NULL;
double timeout;
GDALDatasetH fp_cpy = NULL;
GDALDatasetH fp = NULL;
GDALDatasetH fo = NULL;
GDALRasterBandH band = NULL;
GDALDriverH driver = NULL;
GDALDriverH driver_cpy = NULL;
char **options = NULL;
float *buf = NULL;
float now, old;
char xchunk[NPOW_08];
char ychunk[NPOW_08];
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


  if (stack == NULL || stack->open == OPEN_FALSE) return SUCCESS;

  #ifdef FORCE_DEBUG
  print_stack_info(stack);
  #endif
  

  //CPLPushErrorHandler(CPLQuietErrorHandler);

  alloc_2DC((void***)&fp_meta,   n_fp_meta,   NPOW_14, sizeof(char));
  alloc_2DC((void***)&band_meta, n_band_meta, NPOW_14, sizeof(char));
  sys_meta = system_info(&n_sys_meta);


  copy_string(fp_meta[i++], NPOW_14, "FORCE_version");
  copy_string(fp_meta[i++], NPOW_14, _VERSION_);
  
  copy_string(fp_meta[i++], NPOW_14, "FORCE_description");
  copy_string(fp_meta[i++], NPOW_14, stack->name);
  
  copy_string(fp_meta[i++], NPOW_14, "FORCE_product");
  copy_string(fp_meta[i++], NPOW_14, stack->product);
  
  copy_string(fp_meta[i++], NPOW_14, "FORCE_param");
  copy_string(fp_meta[i++], NPOW_14, stack->par);


  // how many bands to output?
  for (b=0, b_=0; b<stack->nb; b++) b_ += stack->save[b];

  if (stack->explode){
    nfiles = b_;
    nbands = 1;
  } else {
    nfiles = 1;
    nbands = b_;
  }

  enum { _STACK_, _FILE_};
  alloc_3D((void****)&bands, NPOW_01, nfiles, nbands, sizeof(int));
  // dim 1: 2 slots - stack and file
  // dim 2: output files
  // dim 3: band numbers

  for (b=0, b_=0; b<stack->nb; b++){
    
    if (!stack->save[b]) continue;
    
    if (stack->explode){
      bands[_STACK_][b_][0] = b;
      bands[_FILE_][b_][0]  = 1;
    } else {
      bands[_STACK_][0][b_] = b;
      bands[_FILE_][0][b_]  = b_+1;
    }

    b_++;
    
  }
  
  
  //CPLSetConfigOption("GDAL_PAM_ENABLED", "YES");
  

  // choose between formats
  switch (stack->format){
    case _FMT_ENVI_:
      driver = GDALGetDriverByName("ENVI");
      break;
    case _FMT_GTIFF_:
      driver = GDALGetDriverByName("GTiff");
      options = CSLSetNameValue(options, "COMPRESS", "LZW");
      options = CSLSetNameValue(options, "PREDICTOR", "2");
      options = CSLSetNameValue(options, "INTERLEAVE", "BAND");
      options = CSLSetNameValue(options, "BIGTIFF", "YES");
      if (stack->cx > 0){
        nchar = snprintf(xchunk, NPOW_08, "%d", stack->cx);
        if (nchar < 0 || nchar >= NPOW_08){ 
          printf("Buffer Overflow in assembling BLOCKXSIZE\n"); return FAILURE;}
        options = CSLSetNameValue(options, "BLOCKXSIZE", xchunk);
      }
      if (stack->cy > 0){
        nchar = snprintf(ychunk, NPOW_08, "%d", stack->cy);
        if (nchar < 0 || nchar >= NPOW_08){ 
          printf("Buffer Overflow in assembling BLOCKYSIZE\n"); return FAILURE;}
        options = CSLSetNameValue(options, "BLOCKYSIZE", ychunk);
      }
      break;
    case _FMT_JPEG_:
      driver = GDALGetDriverByName("MEM");
      driver_cpy = GDALGetDriverByName("JPEG");
      break;
    default:
      printf("unknown format. ");
      return FAILURE;
  }

  switch (stack->datatype){
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
      printf("unknown datatype for writing stack. ");
      return FAILURE;
  }


  // output path
  if ((lock = (char*)CPLLockFile(stack->dname, 60)) == NULL){
    printf("Unable to lock directory %s (timeout: %ds). ", stack->dname, 60);
    return FAILURE;}
  createdir(stack->dname);
  CPLUnlockFile(lock);
  lock = NULL;

  
  for (f=0; f<nfiles; f++){
    
    if (stack->explode){
      nchar = snprintf(bname, NPOW_10, "_%s", stack->bandname[bands[_STACK_][f][0]]);
      if (nchar < 0 || nchar >= NPOW_10){ 
        printf("Buffer Overflow in assembling band ID\n"); return FAILURE;}      
    } else bname[0] = '\0';
  
    nchar = snprintf(fname, NPOW_10, "%s/%s%s.%s", stack->dname, 
      stack->fname, bname, stack->extension);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

    timeout = lock_timeout(get_stack_size(stack));

    if ((lock = (char*)CPLLockFile(fname, timeout)) == NULL){
      printf("Unable to lock file %s (timeout: %fs, nx/ny: %d/%d). ", fname, timeout, stack->nx, stack->ny);
      return FAILURE;}


    // mosaicking into existing file
    // read and rewrite stack (safer when using compression)
    if (stack->open != OPEN_CREATE && stack->open != OPEN_BLOCK && fileexist(fname)){

      // read stack
      #ifdef FORCE_DEBUG
      printf("reading existing file.\n");
      #endif

      if ((fo = GDALOpen(fname, GA_ReadOnly)) == NULL){
        printf("Unable to open %s. ", fname); return FAILURE;}

      if (GDALGetRasterCount(fo) != nbands){
        printf("Number of bands %d do not match for UPDATE/MERGE mode (file: %d). ", 
          nbands, GDALGetRasterCount(fo)); 
        return FAILURE;}
      if (GDALGetRasterXSize(fo) != stack->nx){
        printf("Number of cols %d do not match for UPDATE/MERGE mode (file: %d). ", 
          stack->nx, GDALGetRasterXSize(fo)); 
        return FAILURE;}
      if (GDALGetRasterYSize(fo) != stack->ny){
        printf("Number of rows %d do not match for UPDATE/MERGE mode (file: %d). ", 
          stack->ny, GDALGetRasterYSize(fo)); 
        return FAILURE;}

      alloc((void**)&buf, stack->nc, sizeof(float));

      for (b=0; b<nbands; b++){

        b_stack = bands[_STACK_][f][b];
        b_file  = bands[_FILE_][f][b];
        
        band = GDALGetRasterBand(fo, b_file);

        if (GDALRasterIO(band, GF_Read, 0, 0, stack->nx, stack->ny, buf, 
          stack->nx, stack->ny, GDT_Float32, 0, 0) == CE_Failure){
          printf("Unable to read %s. ", fname); return FAILURE;} 


        for (p=0; p<stack->nc; p++){

          now = get_stack(stack, b_stack, p);
          old = buf[p];

          // if both old and now are valid: keep now or merge now and old
          if (now != stack->nodata[b_stack] && old != stack->nodata[b_stack]){
            if (stack->open == OPEN_MERGE) set_stack(stack, b_stack, p, (now+old)/2.0);
          // if only old is valid, take old value
          } else if (now == stack->nodata[b_stack] && old != stack->nodata[b_stack]){
            set_stack(stack, b_stack, p, old);
          }
          // if only now is valid, nothing to do

        }

      }

      GDALClose(fo);

      free((void*)buf);

    }


    // open for block mode or write from scratch
    if (stack->open == OPEN_BLOCK && fileexist(fname) && stack->chunk > 0){
      if ((fp = GDALOpen(fname, GA_Update)) == NULL){
        printf("Unable to open %s. ", fname); return FAILURE;}
    } else {
      if ((fp = GDALCreate(driver, fname, stack->nx, stack->ny, nbands, file_datatype, options)) == NULL){
        printf("Error creating file %s. ", fname); return FAILURE;}
    }
      
    if (stack->open == OPEN_BLOCK){
      if (stack->chunk < 0){
        printf("attempting to write invalid chunk\n");
        return FAILURE;
      }
      nx_write     = stack->cx;
      ny_write     = stack->cy;
      xoff_write   = 0;
      yoff_write   = stack->chunk*stack->cy;
    } else {
      nx_write     = stack->nx;
      ny_write     = stack->ny;
      xoff_write   = 0;
      yoff_write   = 0;
    }


    for (b=0; b<nbands; b++){

      b_stack = bands[_STACK_][f][b];
      b_file  = bands[_FILE_][f][b];

      i = 0;


      copy_string(band_meta[i++], NPOW_14, "Domain");
      copy_string(band_meta[i++], NPOW_14, stack->domain[b_stack]);

      copy_string(band_meta[i++], NPOW_14, "Wavelength");
      nchar = snprintf(band_meta[i], NPOW_14, "%.3f", stack->wavelength[b_stack]); i++;
      if (nchar < 0 || nchar >= NPOW_14){ 
        printf("Buffer Overflow in assembling band metadata\n"); return FAILURE;}

      copy_string(band_meta[i++], NPOW_14, "Wavelength_unit");
      copy_string(band_meta[i++], NPOW_14, stack->unit[b_stack]);

      copy_string(band_meta[i++], NPOW_14, "Scale");
      nchar = snprintf(band_meta[i], NPOW_14, "%.3f", stack->scale[b_stack]); i++;
      if (nchar < 0 || nchar >= NPOW_14){ 
        printf("Buffer Overflow in assembling band metadata\n"); return FAILURE;}

      copy_string(band_meta[i++], NPOW_14, "Sensor");
      copy_string(band_meta[i++], NPOW_14, stack->sensor[b_stack]);

      get_stack_longdate(stack, b_stack, ldate, NPOW_05-1);
      copy_string(band_meta[i++], NPOW_14, "Date");
      copy_string(band_meta[i++], NPOW_14, ldate);


      band = GDALGetRasterBand(fp, b_file);

      switch (stack->datatype){
        case _DT_SHORT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, stack->vshort[b_stack], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;}
          break;
        case _DT_SMALL_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, stack->vsmall[b_stack], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;
        case _DT_FLOAT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, stack->vfloat[b_stack], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;
        case _DT_INT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, stack->vint[b_stack], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;
        case _DT_USHORT_:
          if (GDALRasterIO(band, GF_Write, xoff_write, yoff_write, 
            nx_write, ny_write, stack->vushort[b_stack], 
            nx_write, ny_write, file_datatype, 0, 0) == CE_Failure){
            printf("Unable to write %s. ", fname); return FAILURE;} 
          break;

        default:
          printf("unknown datatype for writing stack. ");
          return FAILURE;
      }

      GDALSetDescription(band, stack->bandname[b_stack]);
      GDALSetRasterNoDataValue(band, stack->nodata[b_stack]);
      for (i=0; i<n_band_meta; i+=2) GDALSetMetadataItem(band, band_meta[i], band_meta[i+1], "FORCE");

    }

    // write essential geo-metadata
    #pragma omp critical
    {
      GDALSetGeoTransform(fp, stack->geotran);
      GDALSetProjection(fp,   stack->proj);
    }

    // in case of ENVI, update description
    //if (format == _FMT_ENVI_) 
    //GDALSetDescription(fp, stack->name);


    for (i=0; i<n_sys_meta; i+=2) GDALSetMetadataItem(fp, sys_meta[i], sys_meta[i+1], "FORCE");
    for (i=0; i<n_fp_meta;  i+=2) GDALSetMetadataItem(fp, fp_meta[i],  fp_meta[i+1],  "FORCE");
    
    
    if (stack->format == _FMT_JPEG_){
      if ((fp_cpy = GDALCreateCopy(driver_cpy, fname, fp, FALSE, NULL, NULL, NULL)) == NULL){
          printf("Error creating file %s. ", fname); return FAILURE;}
      GDALClose(fp_cpy);
    }
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


/** This function reprojects a stack into any other projection. The extent
+++ of the warped image is unknown, thus it needs to be estimated first.
+++ The reprojection might be performed in chunks if the number of pixels
+++ is too large to do it in one step.
--- tile:    will the warped image be tiled? If yes, the extent is aligned
             with the tiling scheme
--- rsm:     resampling method
--- threads: number of threads to perform warping
--- from:    source stack (modified)
--- cube:    datacube definition (holds all projection parameters)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_from_stack_to_unknown_stack(bool tile, int rsm, int threads, stack_t *src, cube_t *cube){
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



  nb = get_stack_nbands(src);
  src_nx = get_stack_ncols(src);
  src_ny = get_stack_nrows(src);
  src_nc = get_stack_ncells(src);
  get_stack_geotran(src, src_geotran, 6);
  get_stack_proj(src, src_proj, NPOW_10);

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
  printf("WKT of stack: %s\n", src_proj);
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
      if ((nodata = get_stack_nodata(src, b+b_)) != 0){
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
    for (b_=0; b_<chunk_nb; b_++) wopt->padfSrcNoDataReal[b_] = get_stack_nodata(src, b_+b);
    wopt->padfSrcNoDataImag = (double*)CPLMalloc(sizeof(double)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->padfSrcNoDataImag[b_] = 0;

    wopt->padfDstNoDataReal = (double*)CPLMalloc(sizeof(double)*chunk_nb);
    for (b_=0; b_<chunk_nb; b_++) wopt->padfDstNoDataReal[b_] = get_stack_nodata(src, b_+b);
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
    printf("\nImage stack is warped in %d chunks.\n", k+1);
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
  set_stack_geotran(src, dst_geotran);
  set_stack_ncols(src, dst_nx);
  set_stack_nrows(src, dst_ny);
  set_stack_proj(src, cube->proj);


  #ifdef FORCE_DEBUG
  print_stack_info(src);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("warping stack to stack", TIME);
  #endif

  return SUCCESS;
}


/** This function reprojects an image from disc into any other projection. 
+++ The extent of the warped image is known, and a target stack needs to 
+++ be given, which defines extent, projection etc. 
+++ The reprojection might be performed in chunks if the number of pixels
+++ is too large to do it in one step.
--- rsm:         resampling method
--- threads:     number of threads to perform warping
--- fname:       filename
--- dst:         destination stack (modified)
--- src_b:       which band to warp?    (band in file)
--- dst_b:       which band to warp to? (band in destination stack)
--- src_nodata:  nodata value of band in file
+++ Return:      SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_from_disc_to_known_stack(int rsm, int threads, const char *fname, stack_t *dst, int src_b, int dst_b, int src_nodata){
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
printf("warp_from_disc_to_known_stack should handle multiband src and dst images\n");
#endif
  
  // register drivers and fetch in-memory driver
  if ((driver = GDALGetDriverByName("MEM")) == NULL){
    printf("could not fetch in-memory driver. "); return FAILURE;}


  /** "create" source dataset
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  if ((src_dataset = GDALOpen(fname, GA_ReadOnly)) == NULL){
    printf("unable to open image for warping: %s\n", fname); return FAILURE;}
    
  //src_band = GDALGetRasterBand(src_dataset, src_b+1);
  //src_nodata = (int)GDALGetRasterNoDataValue(src_band, NULL); doesn't work...
  
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

  get_stack_geotran(dst, dst_geotran, 6);
  get_stack_proj(dst, dst_proj, NPOW_10);
  dst_nodata = get_stack_nodata(dst, 0);
  dst_nb = get_stack_nbands(dst);
  dst_nx = get_stack_ncols(dst);
  dst_ny = get_stack_nrows(dst);

  if (dst_b >= dst_nb){
    printf("Requested band %d is out of bounds %d (stack)! ", dst_b, dst_nb); return FAILURE;}

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
      for (p=0; p<dst_nx*dst_ny; p++) set_stack(dst, dst_b, p, dst_nodata);
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
          set_stack(dst, dst_b, p, buf[np]);
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
  proctime_print("warping disc to stack", TIME);
  #endif

  return SUCCESS;
}


/** This function convertes the pixel location from one stack to another.
+++ The stacks differ in spatial resolution.
--- from:   source stack
--- to:     target stack
--- p_from: source pixel
+++ Return: target pixel
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_stack_p2p(stack_t *from, stack_t *to, int p_from){
int i_from, i_to;
int j_from, j_to;

  i_from = floor(p_from/from->nx);
  i_to = floor(i_from*from->geotran[1]/to->geotran[1]);

  j_from = p_from-i_from*from->nx;
  j_to = floor(j_from*from->geotran[1]/to->geotran[1]);

  return i_to*to->nx+j_to;
}


/** This function convertes the pixel location from one stack to another.
+++ The stacks differ in spatial resolution.
--- from:   source stack
--- to:     target stack
--- p_from: source pixel
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_stack_p2ji(stack_t *from, stack_t *to, int p_from, int *i_to, int *j_to){
int i_from;
int j_from;

  i_from = floor(p_from/from->nx);
  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);

  j_from = p_from-i_from*from->nx;
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);

  return;
}


/** This function convertes the pixel location from one stack to another.
+++ The stacks differ in spatial resolution.
--- from:   source stack
--- to:     target stack
--- p_from: source pixel
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
--- p_to:   target pixel  (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_stack_p2jip(stack_t *from, stack_t *to, int p_from, int *i_to, int *j_to, int *p_to){
int i_from;
int j_from;

  i_from = floor(p_from/from->nx);
  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);

  j_from = p_from-i_from*from->nx;
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);
  
  *p_to = (*i_to)*to->nx+(*j_to);

  return;
}


/** This function convertes the pixel location from one stack to another.
+++ The stacks differ in spatial resolution.
--- from:   source stack
--- to:     target stack
--- i_from: source row
--- j_from: source column
+++ Return: target pixel
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_stack_ji2p(stack_t *from, stack_t *to, int i_from, int j_from){
int i_to;
int j_to;

  i_to = floor(i_from*from->geotran[1]/to->geotran[1]);
  j_to = floor(j_from*from->geotran[1]/to->geotran[1]);


  return i_to*to->nx+j_to;
}


/** This function convertes the pixel location from one stack to another.
+++ The stacks differ in spatial resolution.
--- from:   source stack
--- to:     target stack
--- i_from: source row
--- j_from: source column
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_stack_ji2ji(stack_t *from, stack_t *to, int i_from, int j_from, int *i_to, int *j_to){

  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);

  return;
}


/** This function convertes the pixel location from one stack to another.
+++ The stacks differ in spatial resolution.
--- from:   source stack
--- to:     target stack
--- i_from: source row
--- j_from: source column
--- i_to:   target row    (returned)
--- j_to:   target column (returned)
--- p_to:   target pixel  (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void convert_stack_ji2jip(stack_t *from, stack_t *to, int i_from, int j_from, int *i_to, int *j_to, int *p_to){

  *i_to = floor(i_from*from->geotran[1]/to->geotran[1]);
  *j_to = floor(j_from*from->geotran[1]/to->geotran[1]);
  *p_to = (*i_to)*to->nx+(*j_to);

  return;
}


/** This function returns the band that matches the given domain, e.g.
+++ spectral band. If the domain is not present in the stack, -1 is 
+++ returned.
--- stack:  stack
--- domain: domain
+++ Return: band that holds the given domain
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int find_domain(stack_t *stack, const char *domain){
int b, n = get_stack_nbands(stack);
char domain_[NPOW_10];

  for (b=0; b<n; b++){
    get_stack_domain(stack, b, domain_, NPOW_10);
    if (strcmp(domain_, domain) == 0) return b;
  }
  
  return -1;
}


/** This function sets the name of a stack
--- stack:  stack
--- name:   name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_name(stack_t *stack, const char *name){


  copy_string(stack->name, NPOW_10, name);

  return;
}


/** This function gets the name of a stack
--- stack:  stack
--- name:   name (modified)
--- size:   length of the buffer for name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_name(stack_t *stack, char name[], size_t size){


  copy_string(name, size, stack->name);

  return;
}


/** This function sets the product of a stack
--- stack:   stack
--- product: product
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_product(stack_t *stack, const char *product){
  

  copy_string(stack->product, NPOW_03, product);

  return;
}


/** This function gets the product of a stack
--- stack:   stack
--- product: product (modified)
--- size:    length of the buffer for product
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_product(stack_t *stack, char product[], size_t size){


  copy_string(product, size, stack->product);

  return;
}


/** This function sets the directory-name of a stack
--- stack:  stack
--- dname:  directory-name
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_dirname(stack_t *stack, const char *dname){


  copy_string(stack->dname, NPOW_10, dname);

  return;
}


/** This function gets the directory-name of a stack
--- stack:  stack
--- dname:  directory-name (modified)
--- size:   length of the buffer for dname
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_dirname(stack_t *stack, char dname[], size_t size){


  copy_string(dname, size, stack->dname);

  return;
}


/** This function sets the filename of a stack
--- stack:  stack
--- fname:  filename
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_filename(stack_t *stack, const char *fname){


  copy_string(stack->fname, NPOW_10, fname);

  return;
}


/** This function gets the filename of a stack
--- stack:  stack
--- fname:  filename (modified)
--- size:   length of the buffer for fname
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_filename(stack_t *stack, char fname[], size_t size){


  copy_string(fname, size, stack->fname);

  return;
}


/** This function sets the extension of a stack
--- stack:     stack
--- extension: extension
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_extension(stack_t *stack, const char *extension){
  
  
  if (get_stack_format(stack) == _FMT_ENVI_ && strcmp(extension, "dat") != 0){
    printf("extension does not match with format.\n");}
  if (get_stack_format(stack) == _FMT_GTIFF_ && strcmp(extension, "tif") != 0){
    printf("extension does not match with format.\n");}
  if (get_stack_format(stack) == _FMT_JPEG_ && strcmp(extension, "jpg") != 0){
    printf("extension does not match with format.\n");}

  copy_string(stack->extension, NPOW_02, extension);

  return;
}


/** This function gets the extension of a stack
--- stack:     stack
--- extension: extension (modified)
--- size:      length of the buffer for extension
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_extension(stack_t *stack, char extension[], size_t size){


  copy_string(extension, size, stack->extension);

  return;
}


/** This function sets the sensor ID of a stack
--- stack:  stack
--- sid:    sensor ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_sensorid(stack_t *stack, int sid){

  stack->sid = sid;

  return;
}


/** This function gets the sensor ID of a stack
--- stack:  stack
+++ Return: sensor ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_sensorid(stack_t *stack){
  
  return stack->sid;
}


/** This function sets the format of a stack
--- stack:   stack
--- format:  format
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_format(stack_t *stack, int format){

  stack->format = format;

  if (format == _FMT_ENVI_){
    set_stack_extension(stack, "dat");
  } else if (format == _FMT_GTIFF_){
    set_stack_extension(stack, "tif");
  } else if (format == _FMT_JPEG_){
    set_stack_extension(stack, "jpg");
  } else {
    set_stack_extension(stack, "xxx");
    printf("unknown format.\n");
  }

  return;
}

/** This function gets the format of a stack
--- stack:  stack
+++ Return: format
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_format(stack_t *stack){
  
  return stack->format;
}


/** This function sets the opening option of a stack
--- stack:  stack
--- open:   opening option
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_open(stack_t *stack, int open){

  stack->open = open;

  return;
}


/** This function gets the opening option of a stack
--- stack:  stack
+++ Return: opening option
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool get_stack_open(stack_t *stack){
  
  return stack->open;
}


/** This function sets the explode-bands option of a stack
--- stack:   stack
--- explode: explode bands?
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_explode(stack_t *stack, int explode){

  stack->explode = explode;

  return;
}


/** This function gets the explode-bands option of a stack
--- stack:  stack
+++ Return: explode bands?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool get_stack_explode(stack_t *stack){
  
  return stack->explode;
}


/** This function sets the datatype of a stack
--- stack:    stack
--- datatype: datatype
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_datatype(stack_t *stack, int datatype){

  if (datatype != _DT_SHORT_ && datatype != _DT_SMALL_ &&
      datatype != _DT_FLOAT_ && datatype != _DT_INT_  && 
      datatype != _DT_USHORT_ && datatype != _DT_NONE_){
    printf("unknown datatype %d.\n", datatype);}

  if (stack->datatype != _DT_NONE_ && stack->datatype != datatype){
    printf("WARNING: re-setting datatype.\n");
    printf("This might result in double-allocations.\n");}

  stack->datatype  = datatype;

  return;
}


/** This function gets the datatype of a stack
--- stack:  stack
+++ Return: datatype
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_datatype(stack_t *stack){
  
  return stack->datatype;
}


/** This function sets the bytesize of a stack
--- stack:  stack
--- byte:   bytesize
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_byte(stack_t *stack, size_t byte){

  stack->byte = (int)byte;

  return;
}


/** This function gets the bytesize of a stack
--- stack:  stack
+++ Return: bytesize
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_byte(stack_t *stack){
  
  return stack->byte;
}


/** This function sets the number of bands of a stack
--- stack:  stack
--- nb:     number of bands
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_nbands(stack_t *stack, int nb){

  if (nb <= 0) printf("number of bands must be > 0.\n");

  stack->nb = nb;

  return;
}


/** This function gets the number of bands of a stack
--- stack:  stack
+++ Return: number of bands
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_nbands(stack_t *stack){
  
  return stack->nb;
}


/** This function sets the number of columns of a stack
--- stack:  stack
--- nx:     number of columns
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_ncols(stack_t *stack, int nx){

  if (nx <= 0) printf("number of cols must be > 0.\n");

  stack->nx = nx;
  stack->nc = stack->nx*stack->ny;
  stack->width = stack->nx*stack->res;

  return;
}


/** This function gets the number of columns of a stack
--- stack:  stack
+++ Return: number of columns
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_ncols(stack_t *stack){
  
  return stack->nx;
}


/** This function sets the number of rows of a stack
--- stack:  stack
--- ny:     number of rows
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_nrows(stack_t *stack, int ny){

  if (ny <= 0) printf("number of rows must be > 0.\n");

  stack->ny = ny;
  stack->nc = stack->nx*stack->ny;
  stack->height = stack->ny*stack->res;

  return;
}


/** This function gets the number of rows of a stack
--- stack:  stack
+++ Return: number of rows
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_nrows(stack_t *stack){
  
  return stack->ny;
}


/** This function sets the number of cells of a stack
--- stack:  stack
--- nc:     number of cells
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_ncells(stack_t *stack, int nc){
  
  if (nc != stack->nx*stack->ny) printf("number of cells do not match with nx*ny.\n");

  stack->nc = stack->nx*stack->ny;
  
  return;
}


/** This function gets the number of cells of a stack
--- stack:  stack
+++ Return: number of cells
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_ncells(stack_t *stack){
  
  return stack->nc;
}


/** This function gets the total bytesize of a stack
--- stack:  stack
+++ Return: total bytesize
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
size_t get_stack_size(stack_t *stack){
int b = 0;
size_t size = 0;

  for (b=0; b<stack->nb; b++){
    if (get_stack_save(stack, b)) size += stack->nc*stack->byte;
  }

  return size;
}


/** This function sets the number of columns in chunk of a stack
--- stack:  stack
--- cx:     number of columns in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


void set_stack_chunkncols(stack_t *stack, int cx){

  if (cx < 0) printf("number of chunking cols must be >= 0.\n");

  stack->cx = cx;
  stack->cc = stack->cx*stack->cy;
  stack->cwidth = stack->cx*stack->res;

  return;
}


/** This function gets the number of columns in chunk of a stack
--- stack:  stack
+++ Return: number of columns in chunk
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_chunkncols(stack_t *stack){
  
  return stack->cx;
}


/** This function sets the number of rows in chunk of a stack
--- stack:  stack
--- cy:     number of rows in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_chunknrows(stack_t *stack, int cy){

  if (cy < 0) printf("number of chunking rows must be >= 0.\n");

  stack->cy = cy;
  stack->cc = stack->cx*stack->cy;
  stack->cheight = stack->cy*stack->res;

  return;
}


/** This function gets the number of rows in chunk of a stack
--- stack:  stack
+++ Return: number of rows in chunk
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_chunknrows(stack_t *stack){
  
  return stack->cy;
}


/** This function sets the number of cells in chunk of a stack
--- stack:  stack
--- cc:     number of cells in chunk
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_chunkncells(stack_t *stack, int cc){
  
  if (cc != stack->cx*stack->cy) printf("number of chunking cells do not match with cx*cy.\n");

  stack->cc = stack->cx*stack->cy;
  
  return;
}


/** This function gets the number of cells in chunk of a stack
--- stack:  stack
+++ Return: number of cells in chunk
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_chunkncells(stack_t *stack){
  
  return stack->cc;
}


/** This function sets the chunk ID of a stack
--- stack:  stack
--- chunk:  chunk ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_chunk(stack_t *stack, int chunk){
  
  if (chunk >= 0 && chunk >= stack->nchunk) printf("current chunk %d is higher than max chunks %d.\n", chunk, stack->nchunk);

  stack->chunk = chunk;
  
  return;
}


/** This function gets the chunk ID of a stack
--- stack:  stack
+++ Return: chunk ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_chunk(stack_t *stack){
  
  return stack->chunk;
}


/** This function sets the number of chunks of a stack
--- stack:  stack
--- nchunk: number of chunks
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_nchunks(stack_t *stack, int nchunk){
  
  if (nchunk < 0) printf("nchunks %d < 0.\n", nchunk);

  stack->nchunk = nchunk;
  
  return;
}


/** This function gets the number of chunks of a stack
--- stack:  stack
+++ Return: number of chunks
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_nchunks(stack_t *stack){
  
  return stack->nchunk;
}


/** This function sets the tile X-ID of a stack
--- stack:  stack
--- tx:     tile X-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_tilex(stack_t *stack, int tx){
  
  if (tx >= 9999 || tx < -999) printf("tile-x is out of bounds.\n");

  stack->tx = tx;
  
  return;
}


/** This function gets the tile X-ID of a stack
--- stack:  stack
+++ Return: tile X-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_tilex(stack_t *stack){
  
  return stack->tx;
}


/** This function sets the tile Y-ID of a stack
--- stack:  stack
--- ty:     tile Y-ID
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_tiley(stack_t *stack, int ty){
  
  if (ty >= 9999 || ty < -999) printf("tile-y is out of bounds.\n");

  stack->ty = ty;
  
  return;
}


/** This function gets the tile Y-ID of a stack
--- stack:  stack
+++ Return: tile Y-ID
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_tiley(stack_t *stack){
  
  return stack->ty;
}


/** This function sets the resolution of a stack
--- stack:  stack
--- res:    resolution
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_res(stack_t *stack, double res){

  if (res <= 0) printf("resolution must be > 0.\n");

  stack->res = res;
  stack->geotran[1] = res;
  stack->geotran[5] = res*-1;
  stack->width  = stack->nx*stack->res;
  stack->height = stack->ny*stack->res;

  return;
}


/** This function gets the resolution of a stack
--- stack:  stack
+++ Return: resolution
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_res(stack_t *stack){
  
  return stack->res;
}


/** This function sets the UL-X coordinate of a stack
--- stack:   stack
--- ulx: UL-X coordinate
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_ulx(stack_t *stack, double ulx){

  stack->geotran[0] = ulx;

  return;
}


/** This function gets the UL-X coordinate of a stack
--- stack:  stack
+++ Return: UL-X coordinate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_ulx(stack_t *stack){
  
  return stack->geotran[0];
}


/** This function gets the X coordinate of a column of a stack
--- stack:  stack
--- j:      column
+++ Return: X coordinate of a column
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_x(stack_t *stack, int j){
  
  return (stack->geotran[0] + j*stack->geotran[1]);
}


/** This function sets the UL-Y coordinate of a stack
--- stack:  stack
--- uly:    UL-Y coordinate
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_uly(stack_t *stack, double uly){

  stack->geotran[3] = uly;

  return;
}


/** This function gets the UL-Y coordinate of a stack
--- stack:  stack
+++ Return: UL-Y coordinate
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_uly(stack_t *stack){
  
  return stack->geotran[3];
}


/** This function gets the Y coordinate of a row of a stack
--- stack:  stack
--- i:      row
+++ Return: Y coordinate of a row 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_y(stack_t *stack, int i){
  
  return (stack->geotran[3] + i*stack->geotran[5]);
}


/** This function gets the geographic coordinates of a column/row of a stack
--- stack:  stack
--- j:      column
--- i:      row
+++ lon:    longitude (returned)
+++ lat:    latiitude (returned)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_geo(stack_t *stack, int j, int i, double *lon, double *lat){
double mapx, mapy;
double geox, geoy;

  // map coordinate
  mapx = get_stack_x(stack, j);
  mapy = get_stack_y(stack, i);

  // geo coordinate
  warp_any_to_geo(mapx, mapy, &geox, &geoy, stack->proj);
  
  *lon = geox;
  *lat = geoy;
  return ;
}


/** This function sets the geotransformation of a stack
--- stack:   stack
--- geotran: geotransformation
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_geotran(stack_t *stack, double *geotran){

  stack->res = geotran[1];
  stack->geotran[0] = geotran[0];
  stack->geotran[1] = geotran[1];
  stack->geotran[2] = geotran[2];
  stack->geotran[3] = geotran[3];
  stack->geotran[4] = geotran[4];
  stack->geotran[5] = geotran[5];

  return;
}


/** This function gets the geotransformation of a stack
--- stack:   stack
--- geotran: geotransformation (modified)
--- size:    length of the buffer for geotran
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_geotran(stack_t *stack, double geotran[], size_t size){

  if (size != 6){
    printf("array is not compatible for getting geotran.\n"); return;}

  geotran[0] = stack->geotran[0];
  geotran[1] = stack->geotran[1];
  geotran[2] = stack->geotran[2];
  geotran[3] = stack->geotran[3];
  geotran[4] = stack->geotran[4];
  geotran[5] = stack->geotran[5];
  
  return;
}


/** This function sets the width of a stack
--- stack:  stack
--- width:  width
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_width(stack_t *stack, double width){
  
  if (width != stack->nx*stack->res) printf("width does not match with nx*res.\n");

  stack->width = stack->nx*stack->res;

  return;
}


/** This function gets the width of a stack
--- stack:  stack
+++ Return: width
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_width(stack_t *stack){
  
  return stack->width;
}


/** This function sets the height of a stack
--- stack:  stack
--- height: height
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_height(stack_t *stack, double height){
  
  if (height != stack->ny*stack->res) printf("height does not match with ny*res.\n");

  stack->height = stack->ny*stack->res;
  
  return;
}


/** This function gets the height of a stack
--- stack:  stack
+++ Return: height
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_height(stack_t *stack){
  
  return stack->height;
}


/** This function sets the chunk width of a stack
--- stack:   stack
--- cwidth: chunk width
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_chunkwidth(stack_t *stack, double cwidth){
  
  if (cwidth != stack->cx*stack->res) printf("chunking width does not match with cx*res.\n");

  stack->cwidth = stack->cx*stack->res;

  return;
}


/** This function gets the chunk width of a stack
--- stack:  stack
+++ Return: chunk width
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_chunkwidth(stack_t *stack){
  
  return stack->cwidth;
}


/** This function sets the chunk height of a stack
--- stack:   stack
--- cheight: chunk height
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_chunkheight(stack_t *stack, double cheight){
  
  if (cheight != stack->cy*stack->res) printf("chunking height does not match with cy*res.\n");

  stack->cheight = stack->cy*stack->res;
  
  return;
}


/** This function gets the chunk height of a stack
--- stack:  stack
+++ Return: chunk height
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double get_stack_chunkheight(stack_t *stack){
  
  return stack->cheight;
}


/** This function sets the projection of a stack
--- stack:  stack
--- proj:   projection
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_proj(stack_t *stack, const char *proj){


  copy_string(stack->proj, NPOW_10, proj);
  
  return;
}


/** This function gets the projection of a stack
--- stack:  stack
--- proj:   projection (modified)
--- size:   length of the buffer for proj
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_proj(stack_t *stack, char proj[], size_t size){


  copy_string(proj, size, stack->proj);
  
  return;
}


/** This function sets the parameters of a stack
--- stack:  stack
--- par:    parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_par(stack_t *stack, const char *par){


  copy_string(stack->par, NPOW_14, par);

  return;
}


/** This function gets the parameters of a stack
--- stack:  stack
--- par:    parameters (modified)
--- size:   length of the buffer for par
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_par(stack_t *stack, char par[], size_t size){


  copy_string(par, size, stack->par);

  return;
}


/** This function sets the write flag of a stack band
--- stack:  stack
--- b:      band
--- save:   write flag
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_save(stack_t *stack, int b, bool save){

  stack->save[b] = save;

  return;
}


/** This function gets the write flag of a stack band
--- stack:  stack
--- b:      band
+++ Return: write flag
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool get_stack_save(stack_t *stack, int b){

  return stack->save[b];
}


/** This function sets the nodata value of a stack band
--- stack:  stack
--- b:      band
--- nodata: nodata value
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_nodata(stack_t *stack, int b, int nodata){

  stack->nodata[b] = nodata;

  return;
}


/** This function gets the nodata value of a stack band
--- stack:  stack
--- b:      band
+++ Return: nodata value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_nodata(stack_t *stack, int b){

  return stack->nodata[b];
}


/** This function tests if pixel is nodata
--- stack:  stack
--- b:      band
--- p:      pixel
+++ Return: nodata?
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool is_stack_nodata(stack_t *stack, int b, int p){
  
  return fequal(get_stack(stack, b, p), (float)stack->nodata[b]);
}


/** This function sets the scale of a stack band
--- stack:  stack
--- b:      band
--- scale:  scale
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_scale(stack_t *stack, int b, float scale){

  stack->scale[b] = scale;

  return;
}


/** This function gets the scale of a stack band
--- stack:  stack
--- b:      band
+++ Return: scale
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_stack_scale(stack_t *stack, int b){
  
  return stack->scale[b];
}


/** This function sets the wavelength of a stack band
--- stack:      stack
--- b:          band
--- wavelength: wavelength
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_wavelength(stack_t *stack, int b, float wavelength){

  stack->wavelength[b] = wavelength;

  return;
}


/** This function gets the wavelength of a stack band
--- stack:  stack
--- b:      band
+++ Return: wavelength
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_stack_wavelength(stack_t *stack, int b){
  
  return stack->wavelength[b];
}


/** This function sets the unit of a stack band
--- stack:  stack
--- b:      band
--- unit:   unit
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_unit(stack_t *stack, int b, const char *unit){


  copy_string(stack->unit[b], NPOW_04, unit);

  return;
}


/** This function gets the unit of a stack band
--- stack:  stack
--- b:      band
--- unit:   unit (modified)
--- size:   length of the buffer for unit
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_unit(stack_t *stack, int b, char unit[], size_t size){


  copy_string(unit, size, stack->unit[b]);

  return;
}


/** This function sets the domain of a stack band
--- stack:  stack
--- b:      band
--- domain: domain
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_domain(stack_t *stack, int b, const char *domain){


  copy_string(stack->domain[b], NPOW_10, domain);

  return;
}


/** This function gets the domain of a stack band
--- stack:  stack
--- b:      band
--- domain: domain (modified)
--- size:   length of the buffer for domain
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_domain(stack_t *stack, int b, char domain[], size_t size){


  copy_string(domain, size, stack->domain[b]);

  return;
}


/** This function sets the bandname of a stack band
--- stack:    stack
--- b:        band
--- bandname: bandname
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_bandname(stack_t *stack, int b, const char *bandname){


  copy_string(stack->bandname[b], NPOW_10, bandname);

  return;
}


/** This function gets the bandname of a stack band
--- stack:    stack
--- b:        band
--- bandname: bandname (modified)
--- size:     length of the buffer for bandname
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_bandname(stack_t *stack, int b, char bandname[], size_t size){

  
  copy_string(bandname, size, stack->bandname[b]);

  return;
}


/** This function sets the sensor of a stack band
--- stack:  stack
--- b:      band
--- sensor: sensor
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_sensor(stack_t *stack, int b, const char *sensor){


  copy_string(stack->sensor[b], NPOW_04, sensor);

  return;
}


/** This function gets the sensor of a stack band
--- stack:  stack
--- b:      band
--- sensor: sensor (modified)
--- size:   length of the buffer for sensor
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_sensor(stack_t *stack, int b, char sensor[], size_t size){


  copy_string(sensor, size, stack->sensor[b]);

  return;
}


/** This function sets the date of a stack band
--- stack:  stack
--- b:      band
--- date:   date
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_date(stack_t *stack, int b, date_t date){

  copy_date(&date, &stack->date[b]);

  return;
}


/** This function gets the date of a stack band
--- stack:  stack
--- b:      band
+++ Return: date
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
date_t get_stack_date(stack_t *stack, int b){
  
  return stack->date[b];
}


/** This function sets the days since current era of a stack band
--- stack:  stack
--- b:      band
--- ce:     days since current era
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_ce(stack_t *stack, int b, int ce){
  
  stack->date[b].ce = ce;
  
  return;
}


/** This function gets the days since current era of a stack band
--- stack:  stack
--- b:      band
+++ Return: days since current era
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_ce(stack_t *stack, int b){
  
  return stack->date[b].ce;
}


/** This function sets the day of a stack band band
--- stack:  stack
--- day:    day
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_day(stack_t *stack, int b, int day){
  
  stack->date[b].day = day;
  
  return;
}


/** This function gets the day of a stack band
--- stack:  stack
--- b:      band
+++ Return: day
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_day(stack_t *stack, int b){
  
  return stack->date[b].day;
}


/** This function sets the doy of a stack band
--- stack:  stack
--- doy:    doy
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_doy(stack_t *stack, int b, int doy){
  
  stack->date[b].doy = doy;
  
  return;
}


/** This function gets the doy of a stack band
--- stack:  stack
--- b:      band
+++ Return: doy
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_doy(stack_t *stack, int b){
  
  return stack->date[b].doy;
}


/** This function sets the week of a stack band
--- stack:  stack
--- b:      band
--- week:   week
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_week(stack_t *stack, int b, int week){
  
  stack->date[b].week = week;
  
  return;
}


/** This function gets the week of a stack band
--- stack:  stack
--- b:      band
+++ Return: week
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_week(stack_t *stack, int b){
  
  return stack->date[b].week;
}


/** This function sets the month of a stack band
--- stack:  stack
--- b:      band
--- month:  month
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_month(stack_t *stack, int b, int month){
  
  stack->date[b].month = month;
  
  return;
}


/** This function gets the month of a stack band
--- stack:  stack
--- b:      band
+++ Return: month
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_month(stack_t *stack, int b){
  
  return stack->date[b].month;
}


/** This function sets the year of a stack band
--- stack:  stack
--- b:      band
--- year:   year
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_year(stack_t *stack, int b, int year){
  
  stack->date[b].year = year;
  
  return;
}


/** This function gets the year of a stack band
--- stack:  stack
--- b:      band
+++ Return: year
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_year(stack_t *stack, int b){
  
  return stack->date[b].year;
}


/** This function sets the hour of a stack band
--- stack:  stack
--- b:      band
--- hour:   hour
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_hour(stack_t *stack, int b, int hour){
  
  stack->date[b].hh = hour;
  
  return;
}


/** This function gets the hour of a stack band
--- stack:  stack
--- b:      band
+++ Return: hour
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_hour(stack_t *stack, int b){
  
  return stack->date[b].hh;
}


/** This function sets the minute of a stack band
--- stack:  stack
--- b:      band
--- minute: minute
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_minute(stack_t *stack, int b, int minute){
  
  stack->date[b].mm = minute;
  
  return;
}


/** This function gets the minute of a stack band
--- stack:  stack
--- b:      band
+++ Return: minute
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_minute(stack_t *stack, int b){
  
  return stack->date[b].mm;
}


/** This function sets the second of a stack band
--- stack:  stack
--- b:      band
--- second: second
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_second(stack_t *stack, int b, int second){
  
  stack->date[b].ss = second;
  
  return;
}


/** This function gets the second of a stack band
--- stack:  stack
--- b:      band
+++ Return: second
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_second(stack_t *stack, int b){
  
  return stack->date[b].ss;
}


/** This function sets the timezone of a stack band
--- stack:    stack
--- b:        band
--- timezone: timezone
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack_timezone(stack_t *stack, int b, int timezone){
  
  stack->date[b].tz = timezone;
  
  return;
}


/** This function gets the timezone of a stack band
--- stack:  stack
--- b:      band
+++ Return: timezone
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_stack_timezone(stack_t *stack, int b){
  
  return stack->date[b].tz;
}


/** This function gets a formatted date of a stack band
--- stack:     stack
--- b:         band
--- formatted: formatted date (modified)
--- size:      length of the buffer for formatted
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_compactdate(stack_t *stack, int b, char formatted[], size_t size){

  compact_date(stack->date[b].year, stack->date[b].month, stack->date[b].day, formatted, size);

  return;
}


/** This function gets a formatted date of a stack band
--- stack:     stack
--- b:         band
--- formatted: formatted date (modified)
--- size:      length of the buffer for formatted
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_longdate(stack_t *stack, int b, char formatted[], size_t size){

  long_date(stack->date[b].year, stack->date[b].month, stack->date[b].day,
    stack->date[b].hh, stack->date[b].mm, stack->date[b].ss, stack->date[b].tz, 
    formatted, size);

  return;
}


/** This function sets an image value. This is slower than direct access
+++ to the memory, but probably more convenient as the correct datatype
+++ etc is chosen
--- stack:  stack
--- b:      band
--- p:      pixel
--- val:    value
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void set_stack(stack_t *stack, int b, int p, float val){

  switch (stack->datatype){
    case _DT_SHORT_:
      stack->vshort[b][p] = (short)val;
      break;
    case _DT_SMALL_:
      stack->vsmall[b][p] = (small)val;
      break;
    case _DT_FLOAT_:
      stack->vfloat[b][p] = (float)val;
      break;
    case _DT_INT_:
      stack->vint[b][p] = (int)val;
      break;
    case _DT_USHORT_:
      stack->vushort[b][p] = (ushort)val;
      break;
    default:
      printf("unknown datatype %d, (no value was set)\n", stack->datatype);
      return;
  }

}


/** This function gets an image value. This is slower than direct access
+++ to the memory, but probably more convenient as the correct datatype
+++ etc is chosen
--- stack:  stack
--- b:      band
--- p:      pixel
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_stack(stack_t *stack, int b, int p){

  switch (stack->datatype){
    case _DT_SHORT_:
      return (float)stack->vshort[b][p];
      break;
    case _DT_SMALL_:
      return (float)stack->vsmall[b][p];
      break;
    case _DT_FLOAT_:
      return (float)stack->vfloat[b][p];
      break;
    case _DT_INT_:
      return (float)stack->vint[b][p];
      break;
    case _DT_USHORT_:
      return (float)stack->vushort[b][p];
      break;
    default:
      printf("unknown datatype (return 0)");
      return 0.0;
  }

}


/** This function returns a pointer to the short image bands
--- stack:  stack
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short **get_bands_short(stack_t *stack){

  if (stack->vshort == NULL){
    printf("SHORT memory not available.\n"); return NULL;}

  return stack->vshort;
}


/** This function returns a pointer to the byte image bands
--- stack:  stack
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
small **get_bands_small(stack_t *stack){

  if (stack->vsmall == NULL){
    printf("SMALL memory not available.\n"); return NULL;}

  return stack->vsmall;
}


/** This function returns a pointer to the float image bands
--- stack:  stack
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float **get_bands_float(stack_t *stack){

  if (stack->vfloat == NULL){
    printf("FLOAT memory not available.\n"); return NULL;}

  return stack->vfloat;
}


/** This function returns a pointer to the integer image bands
--- stack:  stack
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int **get_bands_int(stack_t *stack){

  if (stack->vint == NULL){
    printf("INT memory not available.\n"); return NULL;}

  return stack->vint;
}


/** This function returns a pointer to the unsigned short image bands
--- stack:  stack
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort **get_bands_ushort(stack_t *stack){

  if (stack->vushort == NULL){
    printf("USHORT memory not available.\n"); return NULL;}

  return stack->vushort;
}


/** This function returns a pointer to a short image band
--- stack:  stack
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *get_band_short(stack_t *stack, int b){

  if (stack->vshort == NULL){
    printf("SHORT memory not available.\n"); return NULL;}
  
  return stack->vshort[b];
}


/** This function returns a pointer to a byte image band
--- stack:  stack
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
small *get_band_small(stack_t *stack, int b){

  if (stack->vsmall == NULL){
    printf("SMALL memory not available.\n"); return NULL;}
  
  return stack->vsmall[b];
}


/** This function returns a pointer to a float image band
--- stack:  stack
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *get_band_float(stack_t *stack, int b){

  if (stack->vfloat == NULL){
    printf("FLOAT memory not available.\n"); return NULL;}
  
  return stack->vfloat[b];
}


/** This function returns a pointer to an integer image band
--- stack:  stack
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int *get_band_int(stack_t *stack, int b){

  if (stack->vint == NULL){
    printf("INT memory not available.\n"); return NULL;}
  
  return stack->vint[b];
}


/** This function returns a pointer to an unsigned short image band
--- stack:  stack
--- b:      band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort *get_band_ushort(stack_t *stack, int b){
  
  if (stack->vushort == NULL){
    printf("USHORT memory not available.\n"); return NULL;}
  
  return stack->vushort[b];
}


/** This function returns a pointer to a short image band
--- stack:  stack
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
short *get_domain_short(stack_t *stack, const char *domain){
int b;

  if (stack->vshort == NULL){
    printf("SHORT memory not available.\n"); return NULL;}

  if ((b = find_domain(stack, domain)) < 0) return NULL;
  
  return stack->vshort[b];
}


/** This function returns a pointer to a byte image band
--- stack:  stack
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
small *get_domain_small(stack_t *stack, const char *domain){
int b;

  if (stack->vsmall == NULL){
    printf("SMALL memory not available.\n"); return NULL;}

  if ((b = find_domain(stack, domain)) < 0) return NULL;

  return stack->vsmall[b];
}


/** This function returns a pointer to a float image band
--- stack:  stack
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float *get_domain_float(stack_t *stack, const char *domain){
int b;

  if (stack->vfloat == NULL){
    printf("FLOAT memory not available.\n"); return NULL;}

    if ((b = find_domain(stack, domain)) < 0) return NULL;

  return stack->vfloat[b];
}


/** This function returns a pointer to an integer image band
--- stack:  stack
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int *get_domain_int(stack_t *stack, const char *domain){
int b;

  if (stack->vint == NULL){
    printf("INT memory not available.\n"); return NULL;}

  if ((b = find_domain(stack, domain)) < 0) return NULL;

  return stack->vint[b];
}


/** This function returns a pointer to an unsigned short image band
--- stack:  stack
--- domain: domain of the band
+++ Return: value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
ushort *get_domain_ushort(stack_t *stack, const char *domain){
int b;

  if (stack->vushort == NULL){
    printf("USHORT memory not available.\n"); return NULL;}

  if ((b = find_domain(stack, domain)) < 0) return NULL;

  return stack->vushort[b];
}


/** This function returns the minimum value of a band
--- stack:  stack
--- b:      band
+++ Return: minimum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_stack_min(stack_t *stack, int b){
int p;
float tmp, min = LONG_MAX;

   for (p=0; p<stack->nc; p++){
     if (is_stack_nodata(stack, b, p)) continue;
     if ((tmp = get_stack(stack, b, p)) < min) min = tmp;
   }

   return min;
}


/** This function returns the maximum value of a band
--- stack:  stack
--- b:      band
+++ Return: maximum
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float get_stack_max(stack_t *stack, int b){
int p;
float tmp, max = LONG_MIN;

   for (p=0; p<stack->nc; p++){
     if (is_stack_nodata(stack, b, p)) continue;
     if ((tmp = get_stack(stack, b, p)) > max) max = tmp;
   }
   
   return max;
}


/** This function returns the minimum/maximum value of a band
--- stack:  stack
--- b:      band
--- min:    minimum
--- max:    maximum
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_stack_range(stack_t *stack, int b, float *min, float *max){
int p;
float tmp, mx = LONG_MIN, mn = LONG_MAX;

   for (p=0; p<stack->nc; p++){
     if (is_stack_nodata(stack, b, p)) continue;
     tmp = get_stack(stack, b, p);
     if (tmp < mn) mn = tmp;
     if (tmp > mx) mx = tmp;
   }

   *min = mn;
   *max = mx;
   return;
}

