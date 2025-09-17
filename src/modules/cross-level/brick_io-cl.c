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


#include "brick_io-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "cpl_string.h"     // various convenience functions for strings
#include "gdal.h"           // public (C callable) GDAL entry points
#include "cpl_multiproc.h"  // CPL Multi-Threading
#include "gdalwarper.h"     // GDAL warper related entry points and defs

#ifdef __cplusplus
#include "ogr_spatialref.h" // coordinate systems services
#endif


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
  GDALDatasetH fp_copy = NULL;
  GDALDatasetH fp_finish = NULL;
  GDALDatasetH fp = NULL;
  GDALDatasetH fo = NULL;
  GDALRasterBandH band = NULL;
  GDALDriverH driver_physical = NULL;
  GDALDriverH driver_memory   = NULL;
  GDALDriverH driver_create   = NULL;
  GDALDataType file_datatype;
  int create;
  char **driver_metadata = NULL;
  char **options = NULL;
  float *buf = NULL;
  float now, old;
  int xoff_write, yoff_write, nx_write, ny_write;
  
  FILE *fprov = NULL;
  char provname[NPOW_10];
  
  char bname[NPOW_10];
  char fname[NPOW_10];
  int nchar;
  
  char version[NPOW_10];
  
  char ldate[NPOW_05];
  char lwritetime[NPOW_05];
  date_t today;
  
  char c_update[2][NPOW_04] = { "create", "update" };
  bool update;
  
  
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
  
    get_version(version, NPOW_10);
    copy_string(fp_meta[i++], NPOW_14, "FORCE_version");
    copy_string(fp_meta[i++], NPOW_14, version);
    
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
    if ((driver_memory   = GDALGetDriverByName("MEM")) == NULL){
      printf("%s driver not found\n", "MEM"); return FAILURE;}
  
    driver_metadata = GDALGetMetadata(driver_physical, NULL);
    //CSLPrint(driver_metadata, NULL);
  
    create = CSLFetchBoolean(driver_metadata, GDAL_DCAP_CREATE, false);
    if (!create && !CSLFetchBoolean(driver_metadata, GDAL_DCAP_CREATECOPY, false)){
      printf("%s driver does not support creating, nor create-copying datasets\n", brick->format.driver);
      return FAILURE;
    }
  
    if (create){
      driver_create = driver_physical;
    } else {
      driver_create = driver_memory;
    }
  
    //CSLDestroy(driver_metadata);
    driver_metadata   = NULL;
  
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
  
    // provenance file
    current_date(&today);
    nchar = snprintf(provname, NPOW_10, "%s/provenance_%04d%02d%02d.csv", 
      brick->provdir, today.year, today.month, today.day);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling provenance file\n"); return FAILURE;}     
  
  
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
      if (brick->open != OPEN_CREATE && brick->open != OPEN_CHUNK && fileexist(fname)){
  
        update = true;
  
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
              if (brick->open == OPEN_QAI_MERGE) merge_qai_from_values(brick, p, (short)now, (short)old);
            // if only old is valid, take old value
            } else if (now == brick->nodata[b_brick] && old != brick->nodata[b_brick]){
              set_brick(brick, b_brick, p, old);
            }
            // if only now is valid, nothing to do
  
          }
  
        }
  
        GDALClose(fo);
  
        free((void*)buf);
  
      } else {
        update = false;
      }
  
  
      // open for chunk mode or write from scratch
      if (brick->open == OPEN_CHUNK && fileexist(fname) && 
         (brick->chunk[_X_] > 0 || brick->chunk[_Y_] > 0)){
        if ((fp = GDALOpen(fname, GA_Update)) == NULL){
          printf("Unable to open %s. ", fname); return FAILURE;}
      } else {
        if ((fp = GDALCreate(driver_create, fname, brick->nx, brick->ny, nbands, file_datatype, options)) == NULL){
          printf("Error creating file %s. ", fname); return FAILURE;}
      }
        
      if (brick->open == OPEN_CHUNK){
        if (brick->chunk[_X_] < 0 || 
            brick->chunk[_Y_] < 0 || 
            brick->chunk[_X_] >= brick->dim_chunk.cols || 
            brick->chunk[_Y_] >= brick->dim_chunk.rows){
          printf("attempting to write invalid chunk\n");
          return FAILURE;
        }
        nx_write     = brick->cx;
        ny_write     = brick->cy;
        xoff_write   = brick->chunk[_X_] * brick->cx;
        yoff_write   = brick->chunk[_Y_] * brick->cy;
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
  
  
      // copy to physical file. This is needed for drivers that do not support CREATE
      if (!create){
        if ((fp_copy = GDALCreateCopy(driver_physical, fname, fp, FALSE, options, NULL, NULL)) == NULL){
          printf("Error creating file %s. ", fname); return FAILURE;}
        fp_finish = fp_copy;
      } else {
        fp_finish = fp;
      }
  
      for (i=0; i<n_sys_meta; i+=2) GDALSetMetadataItem(fp_finish, sys_meta[i], sys_meta[i+1], "FORCE");
      for (i=0; i<n_fp_meta;  i+=2) GDALSetMetadataItem(fp_finish, fp_meta[i],  fp_meta[i+1],  "FORCE");
  
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
  
  
        band = GDALGetRasterBand(fp_finish, b_file);
  
        for (i=0; i<n_band_meta; i+=2) GDALSetMetadataItem(band, band_meta[i], band_meta[i+1], "FORCE");
  
      }
   
      if (!create) GDALClose(fp_copy);
      GDALClose(fp);
  
    
      CPLUnlockFile(lock);
    
      // write provenance info
      if (brick->nprovenance > 0 && 
          brick->chunk[_X_] <= 0 && 
          brick->chunk[_Y_] <= 0){
  
        if ((lock = (char*)CPLLockFile(provname, timeout)) == NULL){
          printf("Unable to lock file %s (timeout: %fs). ", provname, timeout);
          return FAILURE;}
  
        if (fileexist(provname)){
  
          if ((fprov = fopen(provname, "a")) == NULL){
            printf("Unable to re-open provenance file!\n"); 
            return FAILURE;}
  
        } else {
  
          if ((fprov = fopen(provname, "w")) == NULL){
            printf("Unable to create provenance file!\n"); 
            return FAILURE;}
  
          fprintf(fprov, "%s,%s,%s,%s\n", "file", "origin", "mode", "creation");
  
        }
  
        current_date(&today);
        long_date(today.year, today.month, today.day, today.hh, today.mm, today.ss, today.tz, lwritetime, NPOW_05);
  
        fprintf(fprov, "%s,", fname);
        for (p=0; p<(brick->nprovenance-1); p++) fprintf(fprov, "%s;", brick->provenance[p]);
        fprintf(fprov, "%s,%s,%s\n", brick->provenance[p], c_update[update], lwritetime);
  
        fclose(fprov);
  
        CPLUnlockFile(lock);
  
      }
    
  
    }
  
    if (options   != NULL){ CSLDestroy(options);                      options   = NULL;}
    if (fp_meta   != NULL){ free_2DC((void**)fp_meta);                fp_meta   = NULL;}
    if (band_meta != NULL){ free_2DC((void**)band_meta);              band_meta = NULL;}
    if (sys_meta  != NULL){ free_2DC((void**)sys_meta);               sys_meta  = NULL;}
    if (bands     != NULL){ free_3D((void***)bands, NPOW_01, nfiles); bands     = NULL;}
  
    //CPLPopErrorHandler();
  
    return SUCCESS;
  }
  

/** This function reads a full brick as it is stored on disk.
+++ Int16 datatype is assumed.
--- file:       filename
+++ Return:    image brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *read_brick(char *file){
brick_t *brick  = NULL;
short   *brick_short_ = NULL;
GDALDatasetH dataset;
GDALRasterBandH band;
gdalopt_t format;
int b, nbands;
dim_t dim;
double geotran[_GT_LEN_];
const char *projection = NULL;


  dataset = GDALOpenEx(file, GDAL_OF_READONLY, NULL, NULL, NULL);
  if (dataset == NULL){
    printf("unable to open %s.\n", file); 
    exit(FAILURE);
  }

  nbands = GDALGetRasterCount(dataset);
  if (nbands <= 0){
    printf("no bands found in %s.\n", file); 
    GDALClose(dataset); 
    exit(FAILURE);
  } 

  dim.cols = (int)GDALGetRasterXSize(dataset);
  dim.rows = (int)GDALGetRasterYSize(dataset);
  dim.cells = dim.cols*dim.rows;
  GDALGetGeoTransform(dataset, geotran); 
  projection = GDALGetProjectionRef(dataset);

  #ifdef FORCE_DEBUG
  print_dvector(geotran, "Geotransformation", _GT_LEN_, 10, 2);
  #endif


  brick = allocate_brick(nbands, dim.cells, _DT_SHORT_);

  for (b=0; b<nbands; b++){


    if ((brick_short_ = get_band_short(brick, b)) == NULL){
      printf("unable to get short band from brick.\n"); 
      exit(FAILURE);
    }

    band = GDALGetRasterBand(dataset, b+1);
    if (GDALRasterIO(band, GF_Read, 
      0, 0, dim.cols, dim.rows, 
      brick_short_, dim.cols, dim.rows, GDT_Int16, 0, 0) == CE_Failure){
      printf("could not read image.\n"); return NULL;}

  }

  GDALClose(dataset);

  //CSLDestroy(open_options);

  // compile brick correctly
  set_brick_geotran(brick,    geotran);
  set_brick_res(brick,        geotran[_GT_RES_]);
  set_brick_proj(brick,       projection);
  set_brick_ncols(brick,      dim.cols);
  set_brick_nrows(brick,      dim.rows);

  set_brick_filename(brick, "DONOTOUTPUT");
  set_brick_dirname(brick, "DONOTOUTPUT");
  set_brick_provdir(brick, "DONOTOUTPUT");
  set_brick_name(brick, "GENERIC BRICK");

  set_brick_nprovenance(brick, 1);
  set_brick_provenance(brick, 0, file);

  default_gdaloptions(_FMT_GTIFF_, &format);

  set_brick_open(brick,   OPEN_FALSE);
  set_brick_format(brick, &format);
  
  return brick;
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
  double src_geotran[_GT_LEN_];
  double dst_geotran[_GT_LEN_];
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
    get_brick_geotran(src, src_geotran, _GT_LEN_);
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
      NULL, cube->projection, false, 0, 2)) == NULL){
      printf("could not create image to image transformer. "); return FAILURE;}
  
    // estimate approximate extent
    if (GDALSuggestedWarpOutput(src_dataset, GDALGenImgProjTransform, 
      transformer, dst_geotran, &dst_nx, &dst_ny) == CE_Failure){
      printf("could not suggest dst extent. "); return FAILURE;}
  
    // align with output grid of data cube
    if (tile){
  
      if (tile_align(cube, dst_geotran[_GT_ULX_], dst_geotran[_GT_ULY_], &tmpx, &tmpy) == SUCCESS){
        dst_geotran[_GT_ULX_] = tmpx;
        dst_geotran[_GT_ULY_] = tmpy;
      } else {
        printf("could not align with datacube. "); return FAILURE;
      }
  
    }
    
    // convert computed resolution to actual dst resolution
    // add some rows and columns, just to be sure
    dst_nx = dst_nx * dst_geotran[_GT_XRES_] / cube->resolution + 10;
    dst_ny = dst_ny * dst_geotran[_GT_YRES_] / cube->resolution * -1 + 10;
    dst_nc = dst_nx*dst_ny;
    dst_geotran[_GT_XRES_] = cube->resolution; 
    dst_geotran[_GT_YRES_] = cube->resolution * -1;

    // destroy transformer
    GDALDestroyGenImgProjTransformer(transformer);
  
    #ifdef FORCE_DEBUG
    printf("src nx/ny: %d/%d\n",  src_nx, src_ny);
    print_dvector(src_geotran, "src geotransf.", _GT_LEN_, 1, 2);
    printf("dst nx/ny: %d/%d\n", dst_nx, dst_ny);
    print_dvector(dst_geotran, "dst geotransf.", _GT_LEN_, 1, 2);
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
      if (GDALSetProjection(dst_dataset, cube->projection) == CE_Failure){
        printf("could not set dst projection. "); return FAILURE;}
      if (GDALSetGeoTransform(dst_dataset, dst_geotran) == CE_Failure){
        printf("could not set dst geotransformation. "); return FAILURE;}
  
        
      /** create accurate transformer between source and destination
      ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/
      if ((transformer = GDALCreateGenImgProjTransformer(src_dataset, src_proj,
        dst_dataset, cube->projection, false, 0, 2)) == NULL){
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
    set_brick_proj(src, cube->projection);
  
  
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
  double dst_geotran[_GT_LEN_];
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
  
    get_brick_geotran(dst, dst_geotran, _GT_LEN_);
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
  
  