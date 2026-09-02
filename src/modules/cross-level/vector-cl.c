/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2025 David Frantz

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
This file contains functions for handling vector files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "vector-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "gdal_utils.h"       // Public (C callable) GDAL Utilities entry points.
#include "cpl_string.h"     // various convenience functions for strings
#include "ogr_spatialref.h" // coordinate systems services


/** This function reprojects a vector from disc into memory. 
--- input_path:       path of vector file
--- destination_proj: destination projection (WKT)
+++ Return:           reprojected vector dataset handle
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
GDALDatasetH warp_vector_from_disc(char *input_path, const char *destination_proj){
GDALDatasetH input_dataset;
GDALDatasetH output_dataset;
char **options = NULL;
GDALVectorTranslateOptions *translate_options = NULL;


  // open input dataset
  if ((input_dataset = GDALOpenEx(input_path, GDAL_OF_VECTOR, NULL, NULL, NULL)) == NULL){
    fprintf(stderr, "Unable to open %s\n", input_path);
    exit(FAILURE);
  }

  // GDAL 3.10+ uses WAL (Write-Ahead Logging) for GPKG; the SQLite connection
  // requires at least one read per layer to fully initialize the WAL state before
  // the dataset is passed to GDALVectorTranslate — without this warm-up,
  // GDALVectorTranslate silently fails on GPKG in GDAL 3.13+
  for (int i=0; i<GDALDatasetGetLayerCount(input_dataset); i++){
    OGRLayerH warm_layer = GDALDatasetGetLayer(input_dataset, i);
    OGRFeatureH warm_feat = OGR_L_GetNextFeature(warm_layer);
    if (warm_feat != NULL) OGR_F_Destroy(warm_feat);
    OGR_L_ResetReading(warm_layer);
  }

  // prepare the translation options
  options = CSLAddString(options, "-f");
  options = CSLAddString(options, "Memory");
  options = CSLAddString(options, "-t_srs");
  options = CSLAddString(options, destination_proj);
  translate_options = GDALVectorTranslateOptionsNew(options, NULL);

  // warp the dataset into a new Memory dataset (pszDest owns the name, hDstDS is NULL)
  if ((output_dataset = GDALVectorTranslate("warped", NULL, 1, &input_dataset, translate_options, NULL)) == NULL){
    fprintf(stderr, "Failed to reproject vector dataset %s\n", input_path);
    fprintf(stderr, "GDAL error: %s\n", CPLGetLastErrorMsg());
    exit(FAILURE);
  }

  // cleanup
  CSLDestroy(options);
  GDALVectorTranslateOptionsFree(translate_options);
  GDALClose(input_dataset);

  return output_dataset;
}


/** This function rasterizes a vector, which is alread in memory (that is opened)
+++ The destination brick defines the spatial resolution and extent
--- vector_dataset:    vector dataset handle
--- destination_brick: destination brick (some dataset with the correct properties)
+++ Return:            output brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *rasterize_vector_from_memory(GDALDatasetH vector_dataset, brick_t *destination_brick){
GDALDatasetH raster_dataset;
GDALDatasetH output_dataset;
GDALRasterBandH raster_band;
GDALDriverH raster_driver;
char **options = NULL;
GDALRasterizeOptions *rasterize_options = NULL;
char destination_proj[NPOW_10];
double destination_geotran[6];
int destination_nx, destination_ny;
brick_t *output_brick = NULL;
small *brick_band = NULL;


  // can we do something?
  if (vector_dataset == NULL){
    fprintf(stderr, "Input vector dataset is NULL\n");
    exit(FAILURE);
  }

  // GDAL 3.10+ WAL warm-up: same issue as in warp_vector_from_disc — GDALRasterize
  // also requires at least one feature read per layer before use with a GPKG dataset
  for (int i=0; i<GDALDatasetGetLayerCount(vector_dataset); i++){
    OGRLayerH warm_layer = GDALDatasetGetLayer(vector_dataset, i);
    OGRFeatureH warm_feat = OGR_L_GetNextFeature(warm_layer);
    if (warm_feat != NULL) OGR_F_Destroy(warm_feat);
    OGR_L_ResetReading(warm_layer);
  }

  // get information for destination
  get_brick_geotran(destination_brick, destination_geotran, 6);
  get_brick_proj(destination_brick, destination_proj, NPOW_10);
  destination_nx = get_brick_ncols(destination_brick);
  destination_ny = get_brick_nrows(destination_brick);

  // initialize output brick
  output_brick = copy_brick(destination_brick, 1, _DT_SMALL_);
  if ((brick_band = get_band_small(output_brick, 0)) == NULL){
    fprintf(stderr, "Getting band from copied brick failed\n");
    exit(FAILURE);
  }

  // Use the MEM driver to create an in-memory raster dataset
  if ((raster_driver = GDALGetDriverByName("MEM")) == NULL) {
    fprintf(stderr, "MEM driver not available.\n");
    exit(FAILURE);
  }

  // create in-memory dataset
  // note: closing this dataset yields a segmentation fault. probably not close this one
  if ((raster_dataset = GDALCreate(raster_driver, "", destination_nx, destination_ny, 1, GDT_Byte, NULL)) == NULL) {
    fprintf(stderr, "Failed to create in-memory raster dataset.\n");
    exit(FAILURE);
  }

  // set output information
  GDALSetGeoTransform(raster_dataset, destination_geotran);
  GDALSetProjection(raster_dataset, destination_proj);

  // Prepare rasterization options
  options = CSLAddString(options, "-burn");
  options = CSLAddString(options, "1");
  rasterize_options = GDALRasterizeOptionsNew(options, NULL);

  // Perform rasterization
  if ((output_dataset = GDALRasterize(NULL, raster_dataset, vector_dataset, rasterize_options, NULL)) == NULL) {
    fprintf(stderr, "Rasterization failed\n");
    exit(FAILURE);
  }

  // Read data into output brick
  raster_band = GDALGetRasterBand(output_dataset, 1);
  if (GDALRasterIO(raster_band, GF_Read, 0, 0, destination_nx, destination_ny, 
      brick_band, destination_nx, destination_ny, GDT_Byte, 0, 0) != CE_None) {
    fprintf(stderr, "Reading raster data into brick failed.\n");
    exit(FAILURE);
  }

  // cleanup
  CSLDestroy(options);
  GDALRasterizeOptionsFree(rasterize_options);
  GDALClose(output_dataset);

  #ifdef FORCE_DEBUG
  set_brick_filename(output_brick, "AOI");
  set_brick_open(output_brick, OPEN_CREATE); 
  write_brick(output_brick);
  #endif

  return output_brick;
}



/** This function rasterizes a vector that is read from disc
+++ The destination brick defines the SRS, spatial resolution and extent.
+++ If the vector is in a different SRS than the destination, it is reprojected first.
--- input_path:       path of vector file
--- destination_brick: destination brick (some dataset with the correct properties)
+++ Return:            output brick
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *rasterize_vector_from_disc(char *input_path, brick_t *destination_brick){
GDALDatasetH vector_dataset;
OGRSpatialReferenceH vector_srs;
OGRSpatialReferenceH destination_srs;
OGRLayerH layer;
char destination_proj[NPOW_10];
brick_t *output_brick = NULL;


  // get WKT of destination
  get_brick_proj(destination_brick, destination_proj, NPOW_10);

  // open input dataset
  if ((vector_dataset = GDALOpenEx(input_path, GDAL_OF_VECTOR, NULL, NULL, NULL)) == NULL){
    fprintf(stderr, "Unable to open %s\n", input_path);
    exit(FAILURE);
  }

  // Access the first layer in the dataset
  if ((layer = GDALDatasetGetLayer(vector_dataset, 0)) == NULL) {
    fprintf(stderr, "Failed to get the first layer in %s\n", input_path);
    GDALClose(vector_dataset);
    exit(FAILURE);
  }

  // Get the SRS of the layer
  if ((vector_srs = OGR_L_GetSpatialRef(layer)) == NULL) {
    fprintf(stderr, "No spatial reference found for 1st layer in %s\n", input_path);
    GDALClose(vector_dataset);
    exit(FAILURE);
  }

  // compare SRS objects directly — WKT string comparison is fragile across GDAL versions
  // (GDAL 3.x uses WKT2 which produces different strings for the same CRS)
  if ((destination_srs = OSRNewSpatialReference(destination_proj)) == NULL){
    fprintf(stderr, "Failed to parse destination projection\n");
    GDALClose(vector_dataset);
    exit(FAILURE);
  }

  #ifdef FORCE_DEBUG
  {
    char *vector_proj_dbg = NULL;
    if (OSRExportToWkt(vector_srs, &vector_proj_dbg) == OGRERR_NONE){
      printf("WKT of AOI: \n%s\n", vector_proj_dbg);
      CPLFree(vector_proj_dbg);
    }
    printf("WKT of Destination: \n%s\n", destination_proj);
  }
  #endif

  // warp if SRS differs
  if (!OSRIsSame(destination_srs, vector_srs)){

    OSRDestroySpatialReference(destination_srs);
    // close dataset again
    GDALClose(vector_dataset);
    
    // warp the dataset to destination WKT
    if ((vector_dataset = warp_vector_from_disc(input_path, destination_proj)) == NULL){
      fprintf(stderr, "warping AOI failed\n");
      exit(FAILURE);
    }

  } else {
    OSRDestroySpatialReference(destination_srs);
  }

  output_brick = rasterize_vector_from_memory(vector_dataset, destination_brick);
  GDALClose(vector_dataset);

  return output_brick;
}
