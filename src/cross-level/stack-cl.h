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
Image header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef IMAGE_CL_H
#define IMAGE_CL_H

#include <stdio.h>   // core input and output functions
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/warp-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/lock-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/sys-cl.h"
#include "../cross-level/utils-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char name[NPOW_10];      // name of stack
  char product[NPOW_03];   // product short name
  char dname[NPOW_10];     // dirpath  for product
  char fname[NPOW_10];     // filename for product
  char extension[NPOW_02]; // file extension
  int sid;               // sensor ID
  int format;            // output format
  int open;              // open mode
  int explode;           // explode to single-bands?
  int datatype;          // datatype
  int byte;              // number of bytes

  int nb;                // number of images
  int nx;                // number of columns
  int ny;                // number of rows
  int nc;                // number of cells
  int cx;                // number of chunking columns
  int cy;                // number of chunking rows
  int cc;                // number of chunking cells
  double res;             // resolution
  double geotran[6];     // geotransformation
  double width;           // width of image (nx*res)
  double height;          // height of image (ny*res)
  double cwidth;          // chunking width of image (nx*res)
  double cheight;         // chunking height of image (ny*res)
  int chunk;             // ID of chunk
  int nchunk;             // # of chunks
  int tx, ty;            // ID of tile
  char proj[NPOW_10];      // projection

  char par[NPOW_14];       // parameterization

  bool  *save;           // save band?
  int   *nodata;         // nodata value
  float *scale;          // scale factor

  float *wavelength;     // wavelength
  char  **unit;          // wavelength unit
  char  **domain;        // band domain (e.g. NIR)
  char  **bandname;      // band name
  char  **sensor;        // sensor ID
  date_t *date;          // date
                         
  short  **vshort;       // data (z/xy flattened by line)
  float  **vfloat;       // data (z/xy flattened by line)
  int    **vint;         // data (z/xy flattened by line)
  ushort **vushort;      // data (z/xy flattened by line)
  small  **vsmall;       // data (z/xy flattened by line)
    
} stack_t;

int      allocate_stack_bands(stack_t *stack, int nb, int nc, int datatype);
int      reallocate_stack_bands(stack_t *stack, int nb);
stack_t *allocate_stack(int nb, int nc, int datatype);
int      reallocate_stack(stack_t *stack, int nb);
void     free_stack_bands(stack_t *stack);
void     free_stack(stack_t *stack);
void     copy_stack_band(stack_t *stack, int b, stack_t *from, int b_from);
stack_t *copy_stack(stack_t *from, int nb, int datatype);
stack_t *crop_stack(stack_t *from, double radius);
int      warp_from_stack_to_unknown_stack(bool tile, int rsm, int threads, stack_t *src, cube_t *cube);
int      warp_from_disc_to_known_stack(int rsm, int threads, const char *fname, stack_t *dst, int src_b, int dst_b, int src_nodata);
void     init_stack_bands(stack_t *stack);
void     init_stack(stack_t *stack);
int      write_stack(stack_t *stack);
void     print_stack_band_info(stack_t *stack, int b);
void     print_stack_info(stack_t *stack);
int      convert_stack_p2p(stack_t *from, stack_t *to, int p_from);
void     convert_stack_p2ji(stack_t *from, stack_t *to, int p_from, int *i_to, int *j_to);
void     convert_stack_p2jip(stack_t *from, stack_t *to, int p_from, int *i_to, int *j_to, int *p_to);
int      convert_stack_ji2p(stack_t *from, stack_t *to, int i_from, int j_from);
void     convert_stack_ji2ji(stack_t *from, stack_t *to, int i_from, int j_from, int *i_to, int *j_to);
void     convert_stack_ji2jip(stack_t *from, stack_t *to, int i_from, int j_from, int *i_to, int *j_to, int *p_to);
int      find_domain(stack_t *stack, const char *domain);
void     set_stack_name(stack_t *stack, const char *name);
void     get_stack_name(stack_t *stack, char name[], size_t size);
void     set_stack_product(stack_t *stack, const char *product);
void     get_stack_product(stack_t *stack, char product[], size_t size);
void     set_stack_dirname(stack_t *stack, const char *dname);
void     get_stack_dirname(stack_t *stack, char dname[], size_t size);
void     set_stack_filename(stack_t *stack, const char *fname);
void     get_stack_filename(stack_t *stack, char fname[], size_t size);
void     set_stack_extension(stack_t *stack, const char *extension);
void     get_stack_extension(stack_t *stack, char extension[], size_t size);
void     set_stack_sensorid(stack_t *stack, int sid);
int      get_stack_sensorid(stack_t *stack);
void     set_stack_format(stack_t *stack, int format);
int      get_stack_format(stack_t *stack);
void     set_stack_open(stack_t *stack, int open);
bool     get_stack_open(stack_t *stack);
void     set_stack_explode(stack_t *stack, int explode);
bool     get_stack_explode(stack_t *stack);
void     set_stack_datatype(stack_t *stack, int datatype);
int      get_stack_datatype(stack_t *stack);
void     set_stack_output_datatype(stack_t *stack, int datatype);
int      get_stack_output_datatype(stack_t *stack);
void     set_stack_byte(stack_t *stack, size_t byte);
int      get_stack_byte(stack_t *stack);
void     set_stack_nbands(stack_t *stack, int nb);
int      get_stack_nbands(stack_t *stack);
void     set_stack_ncols(stack_t *stack, int nx);
int      get_stack_ncols(stack_t *stack);
void     set_stack_nrows(stack_t *stack, int ny);
int      get_stack_nrows(stack_t *stack);
void     set_stack_ncells(stack_t *stack, int nc);
int      get_stack_ncells(stack_t *stack);
size_t   get_stack_size(stack_t *stack);
void     set_stack_chunkncols(stack_t *stack, int cx);
int      get_stack_chunkncols(stack_t *stack);
void     set_stack_chunknrows(stack_t *stack, int cy);
int      get_stack_chunknrows(stack_t *stack);
void     set_stack_chunkncells(stack_t *stack, int cc);
int      get_stack_chunkncells(stack_t *stack);
void     set_stack_chunk(stack_t *stack, int chunk);
int      get_stack_chunk(stack_t *stack);
void     set_stack_nchunks(stack_t *stack, int nchunk);
int      get_stack_nchunks(stack_t *stack);
void     set_stack_tilex(stack_t *stack, int tx);
int      get_stack_tilex(stack_t *stack);
void     set_stack_tiley(stack_t *stack, int ty);
int      get_stack_tiley(stack_t *stack);
void     set_stack_res(stack_t *stack, double res);
double   get_stack_res(stack_t *stack);
void     set_stack_ulx(stack_t *stack, double ulx);
double   get_stack_ulx(stack_t *stack);
double   get_stack_x(stack_t *stack, int j);
void     set_stack_uly(stack_t *stack, double uly);
double   get_stack_uly(stack_t *stack);
double   get_stack_y(stack_t *stack, int i);
void     get_stack_geo(stack_t *stack, int j, int i, double *lon, double *lat);
void     set_stack_geotran(stack_t *stack, double *geotran);
void     get_stack_geotran(stack_t *stack, double geotran[], size_t size);
void     set_stack_width(stack_t *stack, double width);
double   get_stack_width(stack_t *stack);
void     set_stack_height(stack_t *stack, double height);
double   get_stack_height(stack_t *stack);
void     set_stack_chunkwidth(stack_t *stack, double cwidth);
double   get_stack_chunkwidth(stack_t *stack);
void     set_stack_chunkheight(stack_t *stack, double cheight);
double   get_stack_chunkheight(stack_t *stack);
void     set_stack_proj(stack_t *stack, const char *proj);
void     get_stack_proj(stack_t *stack, char proj[], size_t size);
void     set_stack_par(stack_t *stack, const char *par);
void     get_stack_par(stack_t *stack, char par[], size_t size);
void     set_stack_save(stack_t *stack, int b, bool save);
bool     get_stack_save(stack_t *stack, int b);
void     set_stack_nodata(stack_t *stack, int b, int nodata);
int      get_stack_nodata(stack_t *stack, int b);
bool     is_stack_nodata(stack_t *stack, int b, int p);
void     set_stack_scale(stack_t *stack, int b, float scale);
float    get_stack_scale(stack_t *stack, int b);
void     set_stack_wavelength(stack_t *stack, int b, float wavelength);
float    get_stack_wavelength(stack_t *stack, int b);
void     set_stack_unit(stack_t *stack, int b, const char *unit);
void     get_stack_unit(stack_t *stack, int b, char unit[], size_t size);
void     set_stack_domain(stack_t *stack, int b, const char *wvl_domain);
void     get_stack_domain(stack_t *stack, int b, char wvl_domain[], size_t size);
void     set_stack_bandname(stack_t *stack, int b, const char *bandname);
void     get_stack_bandname(stack_t *stack, int b, char bandname[], size_t size);
void     set_stack_sensor(stack_t *stack, int b, const char *sensor);
void     get_stack_sensor(stack_t *stack, int b, char sensor[], size_t size);
void     set_stack_date(stack_t *stack, int b, date_t date);
date_t   get_stack_date(stack_t *stack, int b);
void     set_stack_ce(stack_t *stack, int b, int ce);
int      get_stack_ce(stack_t *stack, int b);
void     set_stack_day(stack_t *stack, int b, int day);
int      get_stack_day(stack_t *stack, int b);
void     set_stack_doy(stack_t *stack, int b, int doy);
int      get_stack_doy(stack_t *stack, int b);
void     set_stack_week(stack_t *stack, int b, int week);
int      get_stack_week(stack_t *stack, int b);
void     set_stack_month(stack_t *stack, int b, int month);
int      get_stack_month(stack_t *stack, int b);
void     set_stack_year(stack_t *stack, int b, int year);
int      get_stack_year(stack_t *stack, int b);
void     set_stack_hour(stack_t *stack, int b, int hour);
int      get_stack_hour(stack_t *stack, int b);
void     set_stack_minute(stack_t *stack, int b, int minute);
int      get_stack_minute(stack_t *stack, int b);
void     set_stack_second(stack_t *stack, int b, int second);
int      get_stack_second(stack_t *stack, int b);
void     set_stack_timezone(stack_t *stack, int b, int timezone);
int      get_stack_timezone(stack_t *stack, int b);
int      get_stack_timezone(stack_t *stack, int b);
void     get_stack_compactdate(stack_t *stack, int b, char formatted[], size_t size);
void     get_stack_longdate(stack_t *stack, int b, char formatted[], size_t size);
void     set_stack(stack_t *stack, int b, int p, float val);
float    get_stack(stack_t *stack, int b, int p);
short  **get_bands_short(stack_t *stack);
small  **get_bands_small(stack_t *stack);
float  **get_bands_float(stack_t *stack);
int    **get_bands_int(stack_t *stack);
ushort **get_bands_ushort(stack_t *stack);
short   *get_band_short(stack_t *stack, int b);
small   *get_band_small(stack_t *stack, int b);
float   *get_band_float(stack_t *stack, int b);
int     *get_band_int(stack_t *stack, int b);
ushort  *get_band_ushort(stack_t *stack, int b);
short   *get_domain_short(stack_t *stack, const char *domain);
small   *get_domain_small(stack_t *stack, const char *domain);
float   *get_domain_float(stack_t *stack, const char *domain);
int     *get_domain_int(stack_t *stack, const char *domain);
ushort  *get_domain_ushort(stack_t *stack, const char *domain);
float    get_stack_min(stack_t *stack, int b);
float    get_stack_max(stack_t *stack, int b);
void     get_stack_range(stack_t *stack, int b, float *min, float *max);

#ifdef __cplusplus
}
#endif

#endif

