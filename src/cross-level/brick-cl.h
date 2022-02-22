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
  char name[NPOW_10];             // name of brick
  char product[NPOW_03];          // product short name
  char dname[NPOW_10];            // dirpath  for product
  char fname[NPOW_10];            // filename for product
  char extension[NPOW_02];        // file extension
  int sid;                        // sensor ID
  int format;                     // output format
  char driver[NPOW_04];           // GDAL driver short name
  char gdalopt[NPOW_06][NPOW_10]; // GDAL output options
  int ngdalopt;                   // number of GDAL output options
  int open;                       // open mode
  int explode;                    // explode to single-bands?
  int datatype;                   // datatype
  int byte;                       // number of bytes

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
    
} brick_t;

int      allocate_brick_bands(brick_t *brick, int nb, int nc, int datatype);
int      reallocate_brick_bands(brick_t *brick, int nb);
brick_t *allocate_brick(int nb, int nc, int datatype);
int      reallocate_brick(brick_t *brick, int nb);
void     free_brick_bands(brick_t *brick);
void     free_brick(brick_t *brick);
void     copy_brick_band(brick_t *brick, int b, brick_t *from, int b_from);
brick_t *copy_brick(brick_t *from, int nb, int datatype);
brick_t *crop_brick(brick_t *from, double radius);
int      warp_from_brick_to_unknown_brick(bool tile, int rsm, int threads, brick_t *src, cube_t *cube);
int      warp_from_disc_to_known_brick(int rsm, int threads, const char *fname, brick_t *dst, int src_b, int dst_b, int src_nodata);
void     init_brick_bands(brick_t *brick);
void     init_brick(brick_t *brick);
int      write_brick(brick_t *brick);
void     print_brick_band_info(brick_t *brick, int b);
void     print_brick_info(brick_t *brick);
int      convert_brick_p2p(brick_t *from, brick_t *to, int p_from);
void     convert_brick_p2ji(brick_t *from, brick_t *to, int p_from, int *i_to, int *j_to);
void     convert_brick_p2jip(brick_t *from, brick_t *to, int p_from, int *i_to, int *j_to, int *p_to);
int      convert_brick_ji2p(brick_t *from, brick_t *to, int i_from, int j_from);
void     convert_brick_ji2ji(brick_t *from, brick_t *to, int i_from, int j_from, int *i_to, int *j_to);
void     convert_brick_ji2jip(brick_t *from, brick_t *to, int i_from, int j_from, int *i_to, int *j_to, int *p_to);
int      find_domain(brick_t *brick, const char *domain);
void     set_brick_name(brick_t *brick, const char *name);
void     get_brick_name(brick_t *brick, char name[], size_t size);
void     set_brick_product(brick_t *brick, const char *product);
void     get_brick_product(brick_t *brick, char product[], size_t size);
void     set_brick_dirname(brick_t *brick, const char *dname);
void     get_brick_dirname(brick_t *brick, char dname[], size_t size);
void     set_brick_filename(brick_t *brick, const char *fname);
void     get_brick_filename(brick_t *brick, char fname[], size_t size);
void     set_brick_extension(brick_t *brick, const char *extension);
void     get_brick_extension(brick_t *brick, char extension[], size_t size);
void     set_brick_sensorid(brick_t *brick, int sid);
int      get_brick_sensorid(brick_t *brick);
void     set_brick_driver(brick_t *brick, const char *driver);
void     get_brick_driver(brick_t *brick, char driver[], size_t size);
void     set_brick_gdaloptions(brick_t *brick, int o, const char *gdalopt);
void     get_brick_gdaloptions(brick_t *brick, int o, char gdalopt[], size_t size);
void     set_brick_format(brick_t *brick, int format);
int      get_brick_format(brick_t *brick);
void     set_brick_open(brick_t *brick, int open);
bool     get_brick_open(brick_t *brick);
void     set_brick_explode(brick_t *brick, int explode);
bool     get_brick_explode(brick_t *brick);
void     set_brick_datatype(brick_t *brick, int datatype);
int      get_brick_datatype(brick_t *brick);
void     set_brick_output_datatype(brick_t *brick, int datatype);
int      get_brick_output_datatype(brick_t *brick);
void     set_brick_byte(brick_t *brick, size_t byte);
int      get_brick_byte(brick_t *brick);
void     set_brick_nbands(brick_t *brick, int nb);
int      get_brick_nbands(brick_t *brick);
void     set_brick_ncols(brick_t *brick, int nx);
int      get_brick_ncols(brick_t *brick);
void     set_brick_nrows(brick_t *brick, int ny);
int      get_brick_nrows(brick_t *brick);
void     set_brick_ncells(brick_t *brick, int nc);
int      get_brick_ncells(brick_t *brick);
size_t   get_brick_size(brick_t *brick);
void     set_brick_chunkncols(brick_t *brick, int cx);
int      get_brick_chunkncols(brick_t *brick);
void     set_brick_chunknrows(brick_t *brick, int cy);
int      get_brick_chunknrows(brick_t *brick);
void     set_brick_chunkncells(brick_t *brick, int cc);
int      get_brick_chunkncells(brick_t *brick);
void     set_brick_chunk(brick_t *brick, int chunk);
int      get_brick_chunk(brick_t *brick);
void     set_brick_nchunks(brick_t *brick, int nchunk);
int      get_brick_nchunks(brick_t *brick);
void     set_brick_tilex(brick_t *brick, int tx);
int      get_brick_tilex(brick_t *brick);
void     set_brick_tiley(brick_t *brick, int ty);
int      get_brick_tiley(brick_t *brick);
void     set_brick_res(brick_t *brick, double res);
double   get_brick_res(brick_t *brick);
void     set_brick_ulx(brick_t *brick, double ulx);
double   get_brick_ulx(brick_t *brick);
double   get_brick_x(brick_t *brick, int j);
void     set_brick_uly(brick_t *brick, double uly);
double   get_brick_uly(brick_t *brick);
double   get_brick_y(brick_t *brick, int i);
void     get_brick_geo(brick_t *brick, int j, int i, double *lon, double *lat);
void     set_brick_geotran(brick_t *brick, double *geotran);
void     get_brick_geotran(brick_t *brick, double geotran[], size_t size);
void     set_brick_width(brick_t *brick, double width);
double   get_brick_width(brick_t *brick);
void     set_brick_height(brick_t *brick, double height);
double   get_brick_height(brick_t *brick);
void     set_brick_chunkwidth(brick_t *brick, double cwidth);
double   get_brick_chunkwidth(brick_t *brick);
void     set_brick_chunkheight(brick_t *brick, double cheight);
double   get_brick_chunkheight(brick_t *brick);
void     set_brick_proj(brick_t *brick, const char *proj);
void     get_brick_proj(brick_t *brick, char proj[], size_t size);
void     set_brick_par(brick_t *brick, const char *par);
void     get_brick_par(brick_t *brick, char par[], size_t size);
void     set_brick_save(brick_t *brick, int b, bool save);
bool     get_brick_save(brick_t *brick, int b);
void     set_brick_nodata(brick_t *brick, int b, int nodata);
int      get_brick_nodata(brick_t *brick, int b);
bool     is_brick_nodata(brick_t *brick, int b, int p);
void     set_brick_scale(brick_t *brick, int b, float scale);
float    get_brick_scale(brick_t *brick, int b);
void     set_brick_wavelength(brick_t *brick, int b, float wavelength);
float    get_brick_wavelength(brick_t *brick, int b);
void     set_brick_unit(brick_t *brick, int b, const char *unit);
void     get_brick_unit(brick_t *brick, int b, char unit[], size_t size);
void     set_brick_domain(brick_t *brick, int b, const char *wvl_domain);
void     get_brick_domain(brick_t *brick, int b, char wvl_domain[], size_t size);
void     set_brick_bandname(brick_t *brick, int b, const char *bandname);
void     get_brick_bandname(brick_t *brick, int b, char bandname[], size_t size);
void     set_brick_sensor(brick_t *brick, int b, const char *sensor);
void     get_brick_sensor(brick_t *brick, int b, char sensor[], size_t size);
void     set_brick_date(brick_t *brick, int b, date_t date);
date_t   get_brick_date(brick_t *brick, int b);
void     set_brick_ce(brick_t *brick, int b, int ce);
int      get_brick_ce(brick_t *brick, int b);
void     set_brick_day(brick_t *brick, int b, int day);
int      get_brick_day(brick_t *brick, int b);
void     set_brick_doy(brick_t *brick, int b, int doy);
int      get_brick_doy(brick_t *brick, int b);
void     set_brick_week(brick_t *brick, int b, int week);
int      get_brick_week(brick_t *brick, int b);
void     set_brick_month(brick_t *brick, int b, int month);
int      get_brick_month(brick_t *brick, int b);
void     set_brick_year(brick_t *brick, int b, int year);
int      get_brick_year(brick_t *brick, int b);
void     set_brick_hour(brick_t *brick, int b, int hour);
int      get_brick_hour(brick_t *brick, int b);
void     set_brick_minute(brick_t *brick, int b, int minute);
int      get_brick_minute(brick_t *brick, int b);
void     set_brick_second(brick_t *brick, int b, int second);
int      get_brick_second(brick_t *brick, int b);
void     set_brick_timezone(brick_t *brick, int b, int timezone);
int      get_brick_timezone(brick_t *brick, int b);
int      get_brick_timezone(brick_t *brick, int b);
void     get_brick_compactdate(brick_t *brick, int b, char formatted[], size_t size);
void     get_brick_longdate(brick_t *brick, int b, char formatted[], size_t size);
void     set_brick(brick_t *brick, int b, int p, float val);
float    get_brick(brick_t *brick, int b, int p);
short  **get_bands_short(brick_t *brick);
small  **get_bands_small(brick_t *brick);
float  **get_bands_float(brick_t *brick);
int    **get_bands_int(brick_t *brick);
ushort **get_bands_ushort(brick_t *brick);
short   *get_band_short(brick_t *brick, int b);
small   *get_band_small(brick_t *brick, int b);
float   *get_band_float(brick_t *brick, int b);
int     *get_band_int(brick_t *brick, int b);
ushort  *get_band_ushort(brick_t *brick, int b);
short   *get_domain_short(brick_t *brick, const char *domain);
small   *get_domain_small(brick_t *brick, const char *domain);
float   *get_domain_float(brick_t *brick, const char *domain);
int     *get_domain_int(brick_t *brick, const char *domain);
ushort  *get_domain_ushort(brick_t *brick, const char *domain);
float    get_brick_min(brick_t *brick, int b);
float    get_brick_max(brick_t *brick, int b);
void     get_brick_range(brick_t *brick, int b, float *min, float *max);

#ifdef __cplusplus
}
#endif

#endif

