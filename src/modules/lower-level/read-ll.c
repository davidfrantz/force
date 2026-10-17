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
This file contains functions for reading Level 1 data
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "read-ll.h"



int init_level1_brick(par_ll_t *pl2, rtd_t *rtd, meta_t *meta, brick_t **dn){

  if (pl2 == NULL || rtd == NULL || meta == NULL || dn == NULL || *dn != NULL){
    RETURN_ERROR("Invalid input.");
  }


  /** initialize Digital Number brick
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ **/

  brick_t *DN = NULL;

  DN = allocate_brick(meta->band_number, 0, _DT_NONE_);
  set_brick_dirname(DN, pl2->d_temp);
  set_brick_provdir(DN, pl2->d_temp);
  set_brick_filename(DN, "DIGITAL-NUMBERS");
  set_brick_name(DN, "FORCE Digital Number brick");
  set_brick_open(DN, false);
  set_brick_product(DN, "DN_");
  set_brick_par(DN, pl2->params->log);
  set_brick_format(DN, &pl2->gdalopt);
  set_brick_datatype(DN, _DT_USHORT_);

  set_brick_nprovenance(DN, 1);
  set_brick_provenance(DN, 0, pl2->d_level1);

  set_brick_res(DN, meta->res);
  set_brick_nrows(DN, meta->dim.rows);
  set_brick_ncols(DN, meta->dim.cols);
  set_brick_ulx(DN, meta->ul.x);
  set_brick_uly(DN, meta->ul.y);

  char wkt[NPOW_10];

  if (meta->processing_level == _LEVEL_1C_){
    copy_string(wkt, NPOW_10, meta->projection);
  } else if (meta->processing_level == _LEVEL_1B_){
    if (!pl2->doreproj || !pl2->dotile){
      RETURN_ERROR("L1B cannot be used without reprojection and tiling.");
    }
    copy_string(wkt, NPOW_10, pl2->proj);
  } else {
    RETURN_ERROR("Invalid processing level.");
  }

  set_brick_proj(DN, wkt);

  for (int b=0; b<meta->band_number; b++){

    set_brick_sensor(DN, b, rtd->sensor_mapping.l2_sensor);

    set_brick_save(DN, b, true);
    set_brick_nodata(DN, b, meta->nodata);
    set_brick_date(DN, b, meta->date);
    set_brick_unit(DN, b, "micrometers");

    set_brick_domain(DN, b, rtd->band_mapping.domains[b]);

    char bandname[NPOW_10];
    concat_string_2(bandname, NPOW_10, 
      rtd->band_mapping.domains[b], rtd->band_mapping.l1_bands[b], " - B");
    set_brick_bandname(DN, b, bandname);

    set_brick_wavelength(DN, b, meta->wavelength[b] / 1000.0);

  }

  #ifdef CMIX_FAS
  set_brick_dirname(DN, pl2->d_level1);
  #endif

  #ifdef FORCE_DEBUG
  print_brick_info(DN);
  #endif

  *dn = DN;
  return SUCCESS;
}


int read_level1(rtd_t *rtd, meta_t *meta, cube_t *cube, brick_t *DN, par_ll_t *pl2){

  if (meta == NULL || cube == NULL || DN == NULL || pl2 == NULL){
    RETURN_ERROR("Invalid input.");
  }

  // orthorectified Level 1 data (L1C) - read directly into DN brick
  if (meta->processing_level == _LEVEL_1C_){

    if (read_level1c(meta, DN, pl2) != SUCCESS){
      RETURN_ERROR("reading Level 1C data failed.");
    }
    
  // non-orthorectified Level 1 data (L1B) - warp into DN brick
  } else if (meta->processing_level == _LEVEL_1B_){
    
    if (read_level1b(rtd, meta, cube, DN, pl2) != SUCCESS){
      RETURN_ERROR("reading Level 1B data failed.");
    }

  } else {
    RETURN_ERROR("Invalid processing level.");
  }

  return SUCCESS;
}

/** This function reads all necessary or available Level 1C data
--- meta:    metadata
--- DN:      Digital Numbers
--- pl2:     L2 parameters
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_level1c(meta_t *meta, brick_t *DN, par_ll_t *pl2){


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  int nb  = get_brick_nbands(DN);
  int nx  = get_brick_ncols(DN);
  int ny  = get_brick_nrows(DN);
  int nc  = get_brick_ncells(DN);
  double res = get_brick_res(DN);
  allocate_brick_bands(DN, nb, nc, _DT_USHORT_);

  ushort **dn_ = NULL;
  if ((dn_ = get_bands_ushort(DN)) == NULL) return FAILURE;

  CPLSetConfigOption("GDAL_PAM_ENABLED", "NO");
  CPLSetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS");
  //CPLPushErrorHandler(CPLQuietErrorHandler);

  int threads;

  if (pl2->ithread){
    if (pl2->nthread > nb){
      threads = nb;
    } else {
      threads = pl2->nthread;
    }
  } else {
    threads = 1;
  }

  int error = 0;

  #pragma omp parallel num_threads(threads) shared(dn_,nb,meta,nx,ny,res) reduction(+: error) default(none)
  {
 
    #pragma omp for
    for (int b=0; b<nb; b++){


      GDALDatasetH dataset;
      if ((dataset = GDALOpen(meta->image_path.string[b], GA_ReadOnly)) == NULL){
        printf("unable to open %s", meta->image_path.string[b]); 
        error++;
        continue;
      }// else {
        //CPLPopErrorHandler();
      //}
      
      #ifdef FORCE_DEBUG
      GDALDriverH driver = GDALGetDatasetDriver(dataset);
      printf("Driver: %s/%s\n", GDALGetDriverShortName(driver), GDALGetDriverLongName(driver));
      #endif

      // get image resolution
      double geotran[_GT_LEN_];
      GDALGetGeoTransform(dataset, geotran);
      double res_image = geotran[_GT_RES_];

      int xoff_disc_access = floor(meta->col_offset*res/res_image);
      int yoff_disc_access = floor(meta->row_offset*res/res_image);
      int nx_disc_access = floor(nx*res/res_image);
      int ny_disc_access = floor(ny*res/res_image);

      #ifdef FORCE_DEBUG
      printf("reading %d/%d pixels with offset %d/%d into buffer with %d/%d pixels\n", 
        nx_disc_access, ny_disc_access, xoff_disc_access, yoff_disc_access, nx, ny);
      #endif


      GDALRasterBandH band = GDALGetRasterBand(dataset, 1);
      if (GDALRasterIO(band, GF_Read, xoff_disc_access, yoff_disc_access, 
        nx_disc_access, ny_disc_access, dn_[b], 
        nx, ny, GDT_UInt16, 0, 0) == CE_Failure){
        printf("could not read %s. ", meta->image_path.string[b]); 
        error++;
        GDALClose(dataset);
        continue;
      }

      GDALClose(dataset);

    }

  }
  
  if (error > 0){
    printf("reading error. "); return FAILURE;}


  #ifdef FORCE_DEBUG
  print_brick_info(DN); set_brick_open(DN, OPEN_CREATE); write_brick(DN);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("read Level 1", TIME);
  #endif

  return SUCCESS;
}


/** This function reads all necessary or available Level 1B data
+++ This function projects the Level 1B data to the output grid and writes
+++ it to the Digital Number brick.
--- rtd:     runtime data
--- meta:    metadata
--- cube:    datacube parameters
--- DN:      Digital Numbers
--- pl2:     L2 parameters
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int read_level1b(rtd_t *rtd, meta_t *meta, cube_t *cube, brick_t *DN, par_ll_t *pl2){

  if (meta == NULL || cube == NULL || DN == NULL || pl2 == NULL){
    RETURN_ERROR("Invalid input.");
  }

  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  if (!pl2->doreproj){
    RETURN_ERROR("L1B cannot be used without reprojection.");
  }

  if (!pl2->dotile){
    RETURN_ERROR("L1B cannot be used without tiling.");
  }


  // compute the view geometry, result is one row of view angles for each 
  // pixel of the detector. Number of columns is the same as a matching 
  // band of the input L1B data. The lowest resolution band is used to
  // save memory and processing speed. 
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  coord_t top, bottom;

  if (get_center_coordinates_l1b(meta, &top, &bottom) != SUCCESS){
    RETURN_ERROR("Getting center coordinates for Level 1B data failed.");
  }

  float **view_angles = NULL;
  dim_t view_angle_dim;
  if (view_angles_sentinel2_l1b(rtd, &top, &bottom, &view_angles, &view_angle_dim) != SUCCESS){
    free_2D((void**)view_angles, 2);
    RETURN_ERROR("Calculating view angles for Level 1B data failed.");
  }


  // create "orthorectification-ready" datasets for each band 
  // orthoready_datasets has meta->band_number elements
  // lowres_dim is the dimension of the lowest resolution band,
  // which we use as carrier for warping the view angles.
  //  lowres_band is the index of that band.
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  GDALDatasetH *orthoready_datasets = NULL;
  dim_t lowres_dim;
  int lowres_band;

  if (compile_orthoready_datasets(meta, &orthoready_datasets, &lowres_dim, &lowres_band) != SUCCESS){
    RETURN_ERROR("Compiling orthoready datasets failed.");
  }

  if (lowres_dim.cols != view_angle_dim.cols){
    for (int b=0; b<meta->band_number; b++){
      if (orthoready_datasets[b] != NULL) GDALClose(orthoready_datasets[b]);
    }
    free((void*)orthoready_datasets);
    free_2D((void**)view_angles, 2);
    RETURN_ERROR("No band has the same number of columns as the view angle array");
  }


  // create "orthorectification-ready" datasets for the view angles
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
  GDALDatasetH *orthoready_angle_datasets = NULL;

  if (compile_orthoready_angle_datasets(orthoready_datasets[lowres_band], &orthoready_angle_datasets, view_angles, &lowres_dim) != SUCCESS){
    for (int b=0; b<meta->band_number; b++){
      if (orthoready_datasets[b] != NULL) GDALClose(orthoready_datasets[b]);
    }
    free((void*)orthoready_datasets);
    free_2D((void**)view_angles, 2);
    RETURN_ERROR("Compiling orthoready datasets for view angles failed.");
  }

  free_2D((void**)view_angles, 2);


  // estimate the warped extent and align it to output grid
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  coord_t ul, lr;

  if (get_aligned_warped_extent(meta, cube, pl2, orthoready_datasets[0], &ul, &lr) != SUCCESS){
    for (int b=0; b<meta->band_number; b++){
      if (orthoready_datasets[b] != NULL) GDALClose(orthoready_datasets[b]);
    }
    free((void*)orthoready_datasets);
    RETURN_ERROR("Estimating warped extent failed.");
  }


  // update our metadata and initialized DN brick
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  meta->ul = ul;
  meta->dim.cols = (int)round((lr.x - ul.x) / meta->res);
  meta->dim.rows = (int)round((ul.y - lr.y) / meta->res);
  meta->dim.cells = meta->dim.cols * meta->dim.rows;

  set_brick_nrows(DN, meta->dim.rows);
  set_brick_ncols(DN, meta->dim.cols);
  set_brick_ulx(DN, meta->ul.x);
  set_brick_uly(DN, meta->ul.y);

  printf("maximum integer: %d\n", INT_MAX);
  printf("about to allocate DN brick with %d bands, %d rows, %d cols, %d cells\n", meta->band_number, meta->dim.rows, meta->dim.cols, meta->dim.cells);

  allocate_brick_bands(DN, meta->band_number, meta->dim.cells, _DT_USHORT_);

  printf("allocated DN brick with %d bands, %d rows, %d cols, %d cells\n", meta->band_number, meta->dim.rows, meta->dim.cols, meta->dim.cells);
  // allocate view angle brick
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  brick_t *view_angle_brick = copy_brick(DN, 2, _DT_FLOAT_);
  set_brick_domain(view_angle_brick, ZEN, "ZENITH");
  set_brick_domain(view_angle_brick, AZI, "AZIMUTH");
  set_brick_filename(view_angle_brick, "VIEW-ANGLES");
  set_brick_name(view_angle_brick, "FORCE View Angle brick");

  printf("allocated view angle brick with %d bands, %d rows, %d cols, %d cells\n", get_brick_nbands(view_angle_brick), get_brick_nrows(view_angle_brick), get_brick_ncols(view_angle_brick), get_brick_ncells(view_angle_brick));

  // Warp the images to the output grid and read them into the DN brick
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  int error = 0;

  #pragma omp parallel shared(meta, pl2, rtd, orthoready_datasets, ul, lr, DN, stderr) reduction(+:error) default(none) 
  {

    #pragma omp for schedule(static)

    for (int b=0; b<meta->band_number; b++){

      #ifdef FORCE_DEBUG
      printf("Start to warp band %d to output grid.\n", b);
      #endif

      if (warp_orthoready_dataset(meta, pl2, orthoready_datasets[b], 
        rtd->band_mapping.resolution[b], &ul, &lr, &meta->dim, meta->res, 
        GDT_UInt16, get_band_ushort(DN, b)) != SUCCESS){
        CONTINUE_ERROR("Warping orthoready dataset %d failed.", &error, b);
      }

      #ifdef FORCE_DEBUG
      printf("Warped band %d to output grid.\n", b);
      #endif

    }

  }

  for (int b=0; b<meta->band_number; b++){
    if (orthoready_datasets[b] != NULL) GDALClose(orthoready_datasets[b]);
  }
  free((void*)orthoready_datasets);

  // potentially adapt to directly warp to coarse grid resolution, would require
  // average resampling, and maybe adaption of LR coordinate
  // but leave for now
  // handled below by averaging of the view angles to the coarse grid
  for (int b=0; b<2; b++){

    if (warp_orthoready_dataset(meta, pl2, orthoready_angle_datasets[b], 
      rtd->band_mapping.resolution[lowres_band], &ul, &lr, &meta->dim, meta->res, 
      GDT_Float32, get_band_float(view_angle_brick, b)) != SUCCESS){
      CONTINUE_ERROR("Warping orthoready view angle dataset %d failed.", &error, b);
    }

    #ifdef FORCE_DEBUG
    printf("Warped angle band %d to output grid.\n", b);
    #endif

  }

  for (int b=0; b<2; b++){
    if (orthoready_angle_datasets[b] != NULL) GDALClose(orthoready_angle_datasets[b]);
  }
  free((void*)orthoready_angle_datasets);

  if (error > 0){
    free_brick(view_angle_brick);
    RETURN_ERROR("Warping error in %d datasets.", error);
  }

  #ifdef FORCE_DEBUG
  print_brick_info(DN); set_brick_open(DN, OPEN_CREATE); write_brick(DN);
  print_brick_info(view_angle_brick); set_brick_open(view_angle_brick, OPEN_CREATE); write_brick(view_angle_brick);
  #endif


  // construct a simple view grid for the coarse grid computations.
  // Mimick the layout of L1C data
  // Angles should be in degrees, so convert after averaging
  //+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  meta->view_grid.res = 5000.0;
  #ifdef FORCE_MAGIC
  fprintf(stderr, "Warn Magic Number: coarse grid in L1B construction initialized with resolution of %d m.\n", meta->view_grid.res);
  fprintf(stderr, "Warn Magic Number: This is redundant and should be refactored.\n");
  #endif

  meta->view_grid.dim.cols = ceil(meta->dim.cols * meta->res / meta->view_grid.res);
  meta->view_grid.dim.rows = ceil(meta->dim.rows * meta->res / meta->view_grid.res);
  meta->view_grid.dim.cells = meta->view_grid.dim.cols * meta->view_grid.dim.rows;
  meta->view_grid.nodata = FLT_MIN;

  alloc((void**)&meta->view_grid.zen, meta->view_grid.dim.cells, sizeof(float));
  alloc((void**)&meta->view_grid.azi, meta->view_grid.dim.cells, sizeof(float));

  int cell_size = floor(meta->view_grid.res/meta->res);

  float **view_angle_data = get_bands_float(view_angle_brick);

  for (int g_row=0, g_cell=0; g_row<meta->view_grid.dim.rows; g_row++){
  for (int g_col=0; g_col<meta->view_grid.dim.cols; g_col++, g_cell++){

    double sum[2] = {0.0, 0.0};
    double num[2] = {0.0, 0.0};

    for (int i_row=g_row*cell_size; i_row<(g_row*cell_size+cell_size); i_row++){
    for (int i_col=g_col*cell_size; i_col<(g_col*cell_size+cell_size); i_col++){

      if (i_row >= meta->dim.rows || i_col >= meta->dim.cols) continue;

      int i_cell = i_row*meta->dim.cols+i_col;

      for (int b=0; b<2; b++){
        if (!dequal(view_angle_data[b][i_cell], meta->nodata)){
          sum[b] += view_angle_data[b][i_cell];
          num[b]++;
        }
      }

    }
    }

    if (num[ZEN] > 0){
      meta->view_grid.zen[g_cell] = sum[ZEN] / num[ZEN] * _R2D_CONV_;
      //printf("g_cell %d: zenith = %.2f, num = %.2f\n", g_cell, meta->view_grid.zen[g_cell], num[ZEN]);
    } else {
      meta->view_grid.zen[g_cell] = meta->view_grid.nodata;
    }

    if (num[AZI] > 0){
      meta->view_grid.azi[g_cell] = sum[AZI] / num[AZI] * _R2D_CONV_;
      //printf("g_cell %d: azimuth = %.2f, num = %.2f\n", g_cell, meta->view_grid.azi[g_cell], num[AZI]);
    } else {
      meta->view_grid.azi[g_cell] = meta->view_grid.nodata;
    }

  }
  }

  #ifdef FORCE_DEBUG
  printf("View grid: %d rows, %d cols, %d cells, %.2f m resolution\n", 
    meta->view_grid.dim.rows, meta->view_grid.dim.cols, meta->view_grid.dim.cells, meta->view_grid.res);
  printf("View zenith:\n");
  for (int g_row=0, g_cell=0; g_row<meta->view_grid.dim.rows; g_row++){
  for (int g_col=0; g_col<meta->view_grid.dim.cols; g_col++, g_cell++){
    printf("%06.2f ", meta->view_grid.zen[g_cell]);
  }
  printf("\n");
  }
  printf("View azimuth:\n");
  for (int g_row=0, g_cell=0; g_row<meta->view_grid.dim.rows; g_row++){
  for (int g_col=0; g_col<meta->view_grid.dim.cols; g_col++, g_cell++){
    printf("%06.2f ", meta->view_grid.azi[g_cell]);
  }
  printf("\n");
  }
  #endif

  free_brick(view_angle_brick);

  #ifdef FORCE_CLOCK
  proctime_print("read Level 1", TIME);
  #endif

  return SUCCESS;
}


/** This function gets the center coordinates for a Level 1B dataset.
+++ The center coordinates are the coordinates of the center pixel of the geolocation dataset.
--- meta:    metadata
--- top:     coordinates of the top pixel
--- bottom:  coordinates of the bottom pixel
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
 int get_center_coordinates_l1b(meta_t *meta, coord_t *top, coord_t *bottom){

  if (meta == NULL || top == NULL || bottom == NULL){
    RETURN_ERROR("Invalid input.");
  }


  GDALDatasetH geolocation_dataset;
  if ((geolocation_dataset = GDALOpen(meta->geolocation_path.string[0], GA_ReadOnly)) == NULL){
    RETURN_ERROR("Unable to open %s", meta->geolocation_path.string[0]);
  }

  int ncol = GDALGetRasterXSize(geolocation_dataset);
  int nrow = GDALGetRasterYSize(geolocation_dataset);

  for (int i=0; i<2; i++){

    GDALRasterBandH band = GDALGetRasterBand(geolocation_dataset, i+1);
    if (band == NULL){
      GDALClose(geolocation_dataset);
      RETURN_ERROR("Unable to get band %d from geolocation dataset.", i+1);
    }

    double top_value, bottom_value;

    if (GDALRasterIO(band, GF_Read, ncol / 2, 0, 1, 1, 
        &top_value, 1, 1, GDT_Float64, 0, 0) == CE_Failure){
      GDALClose(geolocation_dataset);
      RETURN_ERROR("Could not read top coordinate from geolocation dataset.");
    }

    if (GDALRasterIO(band, GF_Read, ncol / 2, nrow - 1, 1, 1, 
        &bottom_value, 1, 1, GDT_Float64, 0, 0) == CE_Failure){
      GDALClose(geolocation_dataset);
      RETURN_ERROR("Could not read bottom coordinate from geolocation dataset.");
    }

    if (i == 0){
      top->x = top_value * _D2R_CONV_;
      bottom->x = bottom_value * _D2R_CONV_;
    } else if (i == 1){
      top->y = top_value * _D2R_CONV_;
      bottom->y = bottom_value * _D2R_CONV_;
    }

  }

  GDALClose(geolocation_dataset);

  #ifdef FORCE_DEBUG
  printf("Top-center coordinate (geo): %.4f (X), %.4f (Y)\n", 
    top->x * _R2D_CONV_, top->y * _R2D_CONV_);
  printf("Bottom-center coordinate (geo): %.4f (X), %.4f (Y)\n", 
    bottom->x * _R2D_CONV_, bottom->y * _R2D_CONV_);
  printf("Top-center coordinate (rad): %.4f (X), %.4f (Y)\n", 
    top->x, top->y);
  printf("Bottom-center coordinate (rad): %.4f (X), %.4f (Y)\n", 
    bottom->x, bottom->y);
  #endif

  return SUCCESS;
}


/** This function estimates the warped extent of a geolocation-array transformer
+++ This is a preflight warp to determine the output extent/resolution, not the
+++ actual warp that will be performed later. The output extent is aligned to the output grid.
+++ Either to the data cube tile grid if tiling is enabled, or to the output resolution of the 
+++ lowest-resolution band if tiling is disabled.
--- meta:    metadata
--- cube:    datacube parameters
--- pl2:     L2 parameters
--- orthoready_dataset:  a GDAL dataset that is ready for orthorectification
--- ul:      upper-left coordinates of the warped extent (output)
--- lr:      lower-right coordinates of the warped extent (output)
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int get_aligned_warped_extent(meta_t *meta, cube_t *cube, par_ll_t *pl2, 
  GDALDatasetH orthoready_dataset, coord_t *ul, coord_t *lr){

  if (meta == NULL || cube == NULL || pl2 == NULL || 
      orthoready_dataset == NULL || ul == NULL || lr == NULL){
    RETURN_ERROR("Invalid input.");
  }

  // dev comments
  // Calling GDALSuggestedWarpOutput2() on a geolocation-array
  // transformer crashes inside libgdal's backmap generation. gdalwarp -geoloc
  // (without -te) performs an unconstrained warp and lets GDAL determine the
  // output extent/resolution internally without that crash, so replicate
  // that here and read back the resulting geotransform/size.
  // This is a preflight warp to determine the output extent/resolution, not the
  // actual warp that will be performed later.
  // Downside is, we perform  a full warp of the first band just to get the 
  // output extent/resolution, which takes time
  char *warp_options[] = {
      "-t_srs", pl2->proj,
      "-r", "near",
      "-geoloc",
      "-of", "MEM",
      NULL
  };

  GDALWarpAppOptions *warp_app_options = GDALWarpAppOptionsNew(warp_options, NULL);

  int error = 0;
  GDALDatasetH warped_dataset = GDALWarp(
    "", NULL, 1, &orthoready_dataset, warp_app_options, &error);
  GDALWarpAppOptionsFree(warp_app_options);

  if (error != CE_None || warped_dataset == NULL){
    if (warped_dataset != NULL) GDALClose(warped_dataset);
    RETURN_ERROR("Suggesting warp output extent failed: %s", CPLGetLastErrorMsg());
  }

  double suggested_geotransform[_GT_LEN_];
  if (GDALGetGeoTransform(warped_dataset, suggested_geotransform) != CE_None){
    GDALClose(warped_dataset);
    RETURN_ERROR("Could not read suggested geotransform");
  }

  coord_t suggested_dim;
  suggested_dim.x = GDALGetRasterXSize(warped_dataset);
  suggested_dim.y = GDALGetRasterYSize(warped_dataset);
  GDALClose(warped_dataset);

  coord_t suggested_ul, suggested_lr;
  suggested_ul.x = suggested_geotransform[_GT_ULX_];
  suggested_ul.y = suggested_geotransform[_GT_ULY_];
  suggested_lr.x = suggested_ul.x + suggested_dim.x * suggested_geotransform[_GT_XRES_];
  suggested_lr.y = suggested_ul.y + suggested_dim.y * suggested_geotransform[_GT_YRES_];

  #ifdef FORCE_DEBUG
  printf("Suggested extent: \n");
  printf("  UL-X: %.9f\n", suggested_ul.x);
  printf("  UL-Y: %.9f\n", suggested_ul.y);
  printf("  LR-X: %.9f\n", suggested_lr.x);
  printf("  LR-Y: %.9f\n", suggested_lr.y);
  #endif


  coord_t aligned_ul, aligned_lr;

  if (tile_align_resolution(cube, meta->lowest_res, suggested_ul.x, suggested_ul.y, &aligned_ul.x, &aligned_ul.y) == FAILURE){
    RETURN_ERROR("Aligning pixel coordinates failed");
  }

  if (tile_align_resolution(cube, meta->lowest_res, suggested_lr.x, suggested_lr.y, &aligned_lr.x, &aligned_lr.y) == FAILURE){
    RETURN_ERROR("Aligning pixel coordinates failed");
  }

  aligned_lr.x += meta->lowest_res;
  aligned_lr.y -= meta->lowest_res;

  #ifdef FORCE_MAGIC
  fprintf(stderr, "WARNING: Using hard-coded extent adjustment for Sentinel-2 data. This is a temporary solution and should be replaced with a more robust approach.\n");
  #endif

  #ifdef FORCE_DEBUG
  printf("Adjusted extent: \n");
  printf("  UL-X: %.9f\n", aligned_ul.x);
  printf("  UL-Y: %.9f\n", aligned_ul.y);
  printf("  LR-X: %.9f\n", aligned_lr.x);
  printf("  LR-Y: %.9f\n", aligned_lr.y);
  printf("  Number of columns: %d\n", (int)round((aligned_lr.x - aligned_ul.x) / meta->res));
  printf("  Number of rows: %d\n",    (int)round((aligned_ul.y - aligned_lr.y) / meta->res));
  #endif

  *ul = aligned_ul;
  *lr = aligned_lr;

  return SUCCESS;
}


int compile_orthoready_datasets(meta_t *meta, GDALDatasetH **orthoready_datasets, 
  dim_t *lowres_dim, int *lowres_band){

  if (meta == NULL || orthoready_datasets == NULL ||
      lowres_dim == NULL || lowres_band == NULL){
    RETURN_ERROR("Invalid input.");
  }

  GDALDriverH memory_driver = GDALGetDriverByName("MEM");
  if (memory_driver == NULL){
    RETURN_ERROR("Unable to load MEM driver"); 
  }

  GDALDatasetH *orthoready;
  alloc((void**)&orthoready, meta->band_number, sizeof(GDALDatasetH));

  dim_t *dim;
  alloc((void**)&dim, meta->band_number, sizeof(dim_t));

  // keywords for geolocation metadata
  const char *geolocation_keys[] = {
    "SRS", "X_BAND", "Y_BAND", "PIXEL_OFFSET", "LINE_OFFSET", 
    "PIXEL_STEP", "LINE_STEP", "GEOREFERENCING_CONVENTION"
  };
  int num_geolocation_keys = sizeof(geolocation_keys) / sizeof(geolocation_keys[0]);

  int error = 0;
  
  // this will be within a parallel region, so errors will be counted and handled after the parallel region
  for (int b=0; b<meta->band_number; b++){

    GDALDatasetH physical_image_dataset;

    if ((physical_image_dataset = GDALOpen(meta->image_path.string[b], GA_ReadOnly)) == NULL){
      CONTINUE_ERROR("Unable to open %s", &error, meta->image_path.string[b]); 
    }

    GDALRasterBandH band = GDALGetRasterBand(physical_image_dataset, 1);
    if (band == NULL){
      GDALClose(physical_image_dataset);
      CONTINUE_ERROR("Unable to get band for %s", &error, meta->image_path.string[b]); 
    }
    dim[b].cols = GDALGetRasterBandXSize(band);
    dim[b].rows = GDALGetRasterBandYSize(band);

    orthoready[b] = GDALCreateCopy(
      memory_driver, "", physical_image_dataset, FALSE, NULL, NULL, NULL);
    GDALClose(physical_image_dataset);

    if (orthoready[b] == NULL){
      CONTINUE_ERROR("Unable to create orthoready dataset for band %d", &error, b);
    }

    // retrieve GEOLOCATION metadata from the physical geolocation dataset 
    // and set it on the orthoready dataset

    GDALDatasetH physical_geolocation_dataset;
    if ((physical_geolocation_dataset = GDALOpen(meta->geolocation_path.string[b], GA_ReadOnly)) == NULL){
      CONTINUE_ERROR("Unable to open %s", &error, meta->geolocation_path.string[b]);
    }

    for (int i=0; i<num_geolocation_keys; i++){
      const char *key = geolocation_keys[i];
      const char *value = GDALGetMetadataItem(physical_geolocation_dataset, key, NULL);
      if (value == NULL){
        GDALClose(physical_geolocation_dataset);
        CONTINUE_ERROR("Missing geolocation metadata key %s in %s", &error, key, meta->geolocation_path.string[b]);
      }
      // gdalwarp -geoloc fails to transform most points with a negative step
      if (strstr(key, "STEP") != NULL){
        double step_value = atof(value);
        if (step_value < 0) {
          step_value = -step_value;
          char step_str[32];
          snprintf(step_str, sizeof(step_str), "%.1f", step_value);
          value = step_str;
        }
      }
      // if the SRS is specified as an EPSG code, convert it to WKT for GDAL
      if (strings_equal(key, "SRS") && strncmp(value, "EPSG:", 5) == 0){
        char srs_wkt[NPOW_10];
        if (epsg_to_wkt(atoi(value + 5), srs_wkt) != SUCCESS){
          GDALClose(physical_geolocation_dataset);
          CONTINUE_ERROR("Unable to convert geolocation SRS %s to WKT", &error, value);
        }
        value = srs_wkt;
      }
      // set the metadata item on the orthoready dataset into GEOLOCATION domain
      GDALSetMetadataItem(orthoready[b], key, value, "GEOLOCATION");
    }

    // set the X_DATASET and Y_DATASET metadata items to point to the geolocation dataset
    GDALSetMetadataItem(orthoready[b], "X_DATASET", meta->geolocation_path.string[b], "GEOLOCATION");
    GDALSetMetadataItem(orthoready[b], "Y_DATASET", meta->geolocation_path.string[b], "GEOLOCATION");

    GDALClose(physical_geolocation_dataset);

  }

  if (error > 0){
    for (int b=0; b<meta->band_number; b++){
      if (orthoready[b] != NULL) GDALClose(orthoready[b]);
    }
    free((void*)orthoready);
    free((void*)dim);
    RETURN_ERROR("Creation of orthoready dataset failed for %d bands", error);
  }


  *lowres_dim = dim[0];
  *lowres_band = 0;

  for (int b=1; b<meta->band_number; b++){
    if (dim[b].cols < (*lowres_dim).cols){
      *lowres_dim = dim[b];
      *lowres_band = b;
    }
  }

  free((void*)dim);


  #ifdef FORCE_DEBUG
  for (int b=0; b<meta->band_number; b++){
    GDALInfoOptions *info_options = GDALInfoOptionsNew(NULL, NULL);
    char *info = GDALInfo(orthoready[b], info_options);

    if (info != NULL){
      printf("Orthoready dataset info for band %d:\n", b);
      printf("%s", info);
      CPLFree(info);
    }
    GDALInfoOptionsFree(info_options);
  }
  #endif

  *orthoready_datasets = orthoready;
  return SUCCESS;
}


int compile_orthoready_angle_datasets(GDALDatasetH orthoready_carrier, 
  GDALDatasetH **orthoready_angle_datasets, float **view_angle_row, dim_t *dim){

  if (orthoready_carrier == NULL || orthoready_angle_datasets == NULL ||
      view_angle_row == NULL || dim == NULL){
    RETURN_ERROR("Invalid input.");
  }

  GDALDriverH memory_driver = GDALGetDriverByName("MEM");
  if (memory_driver == NULL){
    RETURN_ERROR("Unable to load MEM driver"); 
  }

  int nb = 2;
  GDALDatasetH *orthoready;
  alloc((void**)&orthoready, nb, sizeof(GDALDatasetH));

  int error = 0;

  for (int b=0; b<nb; b++){

    orthoready[b] = GDALCreate(memory_driver, "", dim->cols, dim->rows, 1, GDT_Float32, NULL);
    if (orthoready[b] == NULL){
      CONTINUE_ERROR("Unable to create angle dataset for band %d", &error, b);
    }

    // same sensor grid as the matching band, so its GEOLOCATION domain applies as-is
    char **geolocation_metadata = GDALGetMetadata(orthoready_carrier, "GEOLOCATION");
    GDALSetMetadata(orthoready[b], geolocation_metadata, "GEOLOCATION");

    GDALRasterBandH band = GDALGetRasterBand(orthoready[b], 1);

    int write_error = 0;
    for (int i=0; i<dim->rows; i++){
      if (GDALRasterIO(band, GF_Write, 0, i, dim->cols, 1,
          view_angle_row[b], dim->cols, 1, GDT_Float32, 0, 0) == CE_Failure){
        write_error++;
        break;
      }
    }

    if (write_error > 0){
      CONTINUE_ERROR("Unable to write angle values to band %d", &error, b);
    }

  }

  if (error > 0){
    for (int b=0; b<nb; b++){
      if (orthoready[b] != NULL) GDALClose(orthoready[b]);
    }
    free((void*)orthoready);
    RETURN_ERROR("Creation of orthoready datasets failed for %d bands", error);
  }


  #ifdef FORCE_DEBUG
  for (int b=0; b<nb; b++){
    GDALInfoOptions *info_options = GDALInfoOptionsNew(NULL, NULL);
    char *info = GDALInfo(orthoready[b], info_options);

    if (info != NULL){
      printf("Orthoready dataset info for band %d:\n", b);
      printf("%s", info);
      CPLFree(info);
    }
    GDALInfoOptionsFree(info_options);
  }
  #endif

  *orthoready_angle_datasets = orthoready;
  return SUCCESS;
}


int warp_orthoready_dataset(meta_t *meta, par_ll_t *pl2, GDALDatasetH orthoready_dataset, 
  double original_resolution, coord_t *ul, coord_t *lr, dim_t *expected_dim, double expected_resolution,
  GDALDataType gdal_type, void *output_buffer){

  if (meta == NULL || pl2 == NULL || orthoready_dataset == NULL ||
      ul == NULL || lr == NULL || expected_dim == NULL || output_buffer == NULL){
    RETURN_ERROR("Invalid input.");
  }


  // prepare warp options
  char c_res[32] = {0};
  int nchar = snprintf(c_res, sizeof(c_res), "%.9f", original_resolution);
  if (nchar < 0 || nchar >= sizeof(c_res)){
    RETURN_ERROR("Could not format resolution (%.9f).", original_resolution);
  }

  char c_min_x[32];
  nchar = snprintf(c_min_x, sizeof(c_min_x), "%.9f", (ul->x < lr->x) ? ul->x : lr->x);
  if (nchar < 0 || nchar >= sizeof(c_min_x)){
    RETURN_ERROR("Could not format min x coordinate (%.9f).", (ul->x < lr->x) ? ul->x : lr->x);
  }

  char c_min_y[32];
  nchar = snprintf(c_min_y, sizeof(c_min_y), "%.9f", (ul->y < lr->y) ? ul->y : lr->y);
  if (nchar < 0 || nchar >= sizeof(c_min_y)){
    RETURN_ERROR("Could not format min y coordinate (%.9f).", (ul->y < lr->y) ? ul->y : lr->y);
  }

  char c_max_x[32];
  nchar = snprintf(c_max_x, sizeof(c_max_x), "%.9f", (lr->x > ul->x) ? lr->x : ul->x);
  if (nchar < 0 || nchar >= sizeof(c_max_x)){
    RETURN_ERROR("Could not format max x coordinate (%.9f).", (lr->x > ul->x) ? lr->x : ul->x);
  }

  char c_max_y[32];
  nchar = snprintf(c_max_y, sizeof(c_max_y), "%.9f", (lr->y > ul->y) ? lr->y : ul->y);
  if (nchar < 0 || nchar >= sizeof(c_max_y)){
    RETURN_ERROR("Could not format max y coordinate (%.9f).", (lr->y > ul->y) ? lr->y : ul->y);
  }

  char c_nodata[32] = {0};
  nchar = snprintf(c_nodata, sizeof(c_nodata), "%d", meta->nodata);
  if (nchar < 0 || nchar >= sizeof(c_nodata)){
    RETURN_ERROR("Could not format nodata value (%d).", meta->nodata);
  }

  char *warpoptions[] = {
      "-t_srs", pl2->proj,
      "-r", "cubic",
      "-geoloc",
      "-srcnodata", c_nodata,
      "-dstnodata", c_nodata,
      "-of", "MEM",
      "-te", c_min_x, c_min_y, c_max_x, c_max_y,
      "-tr", c_res, c_res,
      NULL
  };

  GDALWarpAppOptions *warp_app_options = GDALWarpAppOptionsNew(warpoptions, NULL);

  int gdal_error = 0;
  // GDAL warps in chunks; chunks outside the swath fail to transform and stay nodata
  CPLPushErrorHandler(CPLQuietErrorHandler);
  GDALDatasetH warped_dataset = GDALWarp("", NULL, 1, &orthoready_dataset, warp_app_options, &gdal_error);
  CPLPopErrorHandler();

  GDALWarpAppOptionsFree(warp_app_options);

  if (gdal_error != CE_None || warped_dataset == NULL){
    if (warped_dataset != NULL) GDALClose(warped_dataset);
    RETURN_ERROR("Warping failed.");
  }

  GDALRasterBandH warped_band = GDALGetRasterBand(warped_dataset, 1);
  if (warped_band == NULL){
    GDALClose(warped_dataset);
    RETURN_ERROR("Cannot open warped band.");
  }

  // just to be sure, check that the warped dataset has the expected size
  int warped_ncol = GDALGetRasterBandXSize(warped_band); 
  int warped_nrow = GDALGetRasterBandYSize(warped_band);
  if (expected_dim->cols != (warped_ncol * original_resolution / expected_resolution) || 
      expected_dim->rows != (warped_nrow * original_resolution / expected_resolution)){
    GDALClose(warped_dataset);
    RETURN_ERROR("Warped dataset size (%d x %d) does not match expected size (%d x %d) x resolution ratio %.2f", 
      warped_ncol, warped_nrow, expected_dim->cols, expected_dim->rows, original_resolution / expected_resolution);
  }

  // read into brick buffer
  if (GDALRasterIO(warped_band, GF_Read, 0, 0, warped_ncol, warped_nrow, 
    output_buffer, expected_dim->cols, expected_dim->rows, gdal_type, 0, 0) == CE_Failure){
    GDALClose(warped_dataset);
    RETURN_ERROR("Cannot read warped band.");
  }

  GDALClose(warped_dataset);
  
  return SUCCESS;
}


/** This function detects extreme values and builds the nodata and satu-
+++ ration masks.
--- meta:   metadata
--- DN:     digital numbers
--- QAI:    Quality Assurance Information
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int bounds_level1(meta_t *meta, brick_t *DN, brick_t **QAI, par_ll_t *pl2){
int b, b_temp, b_cirrus, nb, p, nx, ny, nc;
brick_t *qai  = NULL;
ushort **dn_  = NULL;
small   *off_ = NULL;
int tvalid = 0;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  
  nb = get_brick_nbands(DN);
  nx = get_brick_ncols(DN);
  ny = get_brick_nrows(DN);
  nc = get_brick_ncells(DN);
  b_temp   = find_domain(DN,  "TEMP");
  b_cirrus = find_domain(DN,  "CIRRUS");

  // initialize a brick with general metadata
  qai = copy_brick(DN, 1, _DT_SHORT_);

  // set brick metadata
  set_brick_name(qai, "FORCE QAI brick");
  set_brick_product(qai, "QAI");
  set_brick_filename(qai, "QAI");
  set_brick_nodata(qai, 0, 1); 
  set_brick_wavelength(qai, 0, 1);
  set_brick_domain(qai, 0, "QAI");
  set_brick_bandname(qai, 0, "Quality assurance information");


  // get DN and OFF arrays for faster computation
  if ((dn_  = get_bands_ushort(DN))   == NULL) return FAILURE;

  alloc((void**)&off_, nc, sizeof(small));


  #pragma omp parallel private(b) shared(b_temp, b_cirrus, nb, nc, dn_, off_, qai, meta) reduction(+:tvalid) default(none) 
  {

    #pragma omp for schedule(static)
    for (p=0; p<nc; p++){

      for (b=0; b<nb; b++){

        // if any layer void --> boundary
        if (b != b_cirrus && dn_[b][p] == 0){ off_[p] = true; break;}

        // if any (non-temp) layer saturated
        if (b != b_temp && dn_[b][p] >= meta->saturation){ set_saturation(qai, p, true); break;}

        // if temperature has any non-0 value
        if (b == b_temp) tvalid++;

      }
    }
  }


  // buffer one pixel (pixels are somehow contaminated. due to resampling?)
  if (pl2->bufnodata) buffer_(off_, nx, ny, 1);
  for (p=0; p<nc; p++) set_off(qai, p, off_[p]);
  free((void*)off_);

  if (b_temp >= 0 && tvalid == 0){
    printf("zero-filled temperature. "); return FAILURE;}

  if (impulse_noise_level1(meta, DN, qai, pl2) == FAILURE){
    printf("detecting impulse noise failed.\n"); return FAILURE;}


  #ifdef FORCE_DEBUG
  print_brick_info(qai); set_brick_open(qai, OPEN_CREATE); write_brick(qai);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("boundaries Level 1", TIME);
  #endif

  *QAI = qai;
  return SUCCESS;
}


/** This function attempts to detect and mask Impulse Noise, a phenomenon
+++ observed in 8bit Landsat data. The first three bands (RGB) are used
+++ to detect this. Note that IN is not confined to these bands and small
+++ IN won't be detected.. This function simply identifies the worst occu-
+++ rences.
--- meta:   metadata
--- DN:     digital numbers
--- QAI:    Quality Assurance Information
--- pl2:    L2 parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int impulse_noise_level1(meta_t *meta, brick_t *DN, brick_t *QAI, par_ll_t *pl2){
int k, count = 0, b, bands[3], nb = 3;
int i, j, ii, jj, p, q, nx, ny;
double mx[3],varx[3], sd[3];
double  maxsd, max2sd;
ushort **dn_  = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  // impact noise was not observed yet in 16bit data
  char sensor[NPOW_10]; 
  get_brick_sensor(DN, 0, sensor, NPOW_10);
  if (!strings_equal(sensor, "LND04") ||
      !strings_equal(sensor, "LND05") ||
      !strings_equal(sensor, "LND07")) return SUCCESS;

  if (!pl2->impulse) return SUCCESS;

  nx = get_brick_ncols(DN);
  ny = get_brick_nrows(DN);

  if ((bands[0] = find_domain(DN,  "BLUE"))  < 0){
    printf("no BLUE band available.\n"); return FAILURE;}
  if ((bands[1] = find_domain(DN,  "GREEN")) < 0){
    printf("no GREEN band available.\n"); return FAILURE;}
  if ((bands[2] = find_domain(DN,  "RED"))   < 0){
    printf("no RED band available.\n"); return FAILURE;}


  if ((dn_ = get_bands_ushort(DN)) == NULL) return FAILURE;

  for (i=1; i<(ny-1); i++){
  for (j=1; j<(nx-1); j++){
    
    p = i*nx+j;

    if (get_off(QAI, p)) continue;

    k = 0;
    for (b=0; b<nb; b++) mx[b] = varx[b] = 0;

    for (ii=-1; ii<=1; ii++){
    for (jj=-1; jj<=1; jj++){

      q = nx*(i+ii)+j+jj;

      if (get_off(QAI, q)) continue;

      k++;

      if (k == 1){
        for (b=0; b<nb; b++) mx[b] = dn_[bands[b]][q];
      } else {
        for (b=0; b<nb; b++) var_recurrence(dn_[bands[b]][q], &mx[b], &varx[b], k);
      }

    }
    }


    if (k>0){

      for (b=0; b<nb; b++) sd[b] = standdev(varx[b], k);

      max2sd = maxsd = sd[0];
      for (b=1; b<nb; b++){
        if (sd[b] > maxsd){ max2sd = maxsd; maxsd = sd[b];}
      }
      if ((maxsd-max2sd) > 15){ set_off(QAI, p, true); count++;}

    }


  }
  }


  #ifdef FORCE_CLOCK
  proctime_print("Impulse Noise Level 1", TIME);
  #endif

  return SUCCESS;
}


/** This function converts all the DN bands to TOA reflectance or bright-
+++ ness Temperature. In case of Sentinel-2, TOA reflectance is first re-
+++ transformed to DNs before it is converted to TOA reflectance again.
+++ This is done to maintain a constant calibration between sensors and
+++ to retain the flexibility to e.g. use another E0 spectrum.
--- meta:    metadata
--- atc:     atmospheric correction factors
--- DN:      digital numbers
--- TOA:     Top of Atmosphere reflectance and temperature
--- QAI:     Quality Assurance Information
+++ Return:  SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int convert_level1(meta_t *meta, atc_t *atc, brick_t *DN, brick_t **toa, brick_t *QAI){
brick_t  *TOA  = NULL;
ushort **dn_  = NULL;
short  **toa_ = NULL;
float    *sun_ = NULL;
int b, b_temp, b_cirrus, nb, nc, p, g;
short nodata;
float A, rad, dsun, pi_dsun2, tmp;
float toa_scale;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nb = get_brick_nbands(DN);
  nc = get_brick_ncells(DN);
  nodata = _FORCE_NO_DATA_;

  TOA = copy_brick(DN, nb, _DT_SHORT_);
  
  // temperature band?
  b_temp   = find_domain(TOA, "TEMP");
  b_cirrus = find_domain(DN,  "CIRRUS");

  // update metadata
  set_brick_name(TOA, "FORCE TOA brick");
  set_brick_product(TOA, "TOA");
  set_brick_filename(TOA, "TOA");

  for (b=0; b<nb; b++){

    set_brick_nodata(TOA, b, nodata);
    if (b != b_temp){
      set_brick_scale(TOA, b, 10000);
    } else {
      set_brick_scale(TOA, b, 100);
    }

  }


  // get brick arrays for faster computation
  if ((dn_  = get_bands_ushort(DN)) == NULL) return FAILURE;
  if ((toa_ = get_bands_short(TOA)) == NULL) return FAILURE;
  if ((sun_ = get_band_float(atc->xy_sun, cZEN)) == NULL) return FAILURE;


  /** TOA reflectance to TOA reflectance (Sentinel-2) 
  in early processing versions, scale factor was 1000, now 10000 **/
  if (meta->mission == SENTINEL2){

    for (b=0; b<nb; b++){

      toa_scale = get_brick_scale(TOA, b);
      
      #pragma omp parallel shared(b, nc, toa_scale, dn_, toa_, meta, QAI, nodata) default(none) 
      {

        #pragma omp for schedule(static)
        for (p=0; p<nc; p++){
          if (get_off(QAI, p)){ 
            toa_[b][p] = nodata; 
          } else {
            toa_[b][p] = (dn_[b][p] + meta->cal[b].reflectance.offset) / 
            meta->cal[b].reflectance.scale*toa_scale;
          }
        }
        
      }

    }

  /** digital numbers to TOA reflectance and brightness temperature (Landsat) **/
  } else {

    dsun = doy2dsun(get_brick_doy(TOA, 0));
    pi_dsun2  = M_PI*dsun*dsun;

    for (b=0; b<nb; b++){

      toa_scale = get_brick_scale(TOA, b);

      A = (meta->cal[b].radiance.lmax-meta->cal[b].radiance.lmin) / 
          (meta->cal[b].radiance.qmax-meta->cal[b].radiance.qmin);

      #pragma omp parallel private(rad, tmp, g) shared(b, b_temp, b_cirrus, nc, nodata, toa_scale, dn_, toa_, sun_, QAI, A, pi_dsun2, meta, atc) default(none) 
      {

        #pragma omp for schedule(guided)
        for (p=0; p<nc; p++){

          if (get_off(QAI, p)){ toa_[b][p] = nodata; continue;}

          // dn to radiance
          rad = A * (dn_[b][p]-meta->cal[b].radiance.qmin) + meta->cal[b].radiance.lmin;

          // radiance to brightness temperature in kelvin
          if (b == b_temp){

            tmp = meta->cal[b].temperature.k2/log((meta->cal[b].temperature.k1/rad)+1)*toa_scale;
            if (tmp < SHRT_MAX){
              toa_[b][p] = (short)tmp;
            } else {
              toa_[b][p] = SHRT_MAX;
            }

          // radiance to reflectance
          } else {

            g = convert_brick_p2p(QAI, atc->xy_sun, p);

            if (meta->cal[b].reflectance.scale <= 0){
              // old-style DN -> radiance -> reflectance conversion (should not happen anymore)
              tmp = rad*pi_dsun2 / (atc->E0[b]*sun_[g]);
            } else {
              // new DN -> reflectance conversion
              tmp = (meta->cal[b].reflectance.offset + meta->cal[b].reflectance.scale*dn_[b][p]) / sun_[g];
            }

            if (tmp < FLT_MIN){
              if (b == b_cirrus){
                toa_[b][p] = (short)0;
              } else {
                toa_[b][p] = (short)nodata;
                set_off(QAI, p, true);
              }
            } else {
              toa_[b][p] = (short)(tmp*toa_scale);
            }

          }

        }
        
      }

    }

  }


  #ifdef FORCE_DEBUG
  print_brick_info(TOA); set_brick_open(TOA, OPEN_CREATE); write_brick(TOA);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("Level 1 to TOA conversion", TIME);
  #endif

  *toa = TOA;
  return SUCCESS;
}

