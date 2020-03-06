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
This program creates a shapefile containing the tile IDs of the prepro-
cessing grid + the bounding box of each tile in projected coordinates
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/warp-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "ogr_spatialref.h" // coordinate systems services
#include "ogr_api.h"        // OGR geometry and feature definition


int main( int argc, char *argv[] ){
char *dir = NULL;
char *format = NULL;
char fname[NPOW_10];
char tile_id[NPOW_10];
int nchar;
int px, py, x0, y0, x1, y1, c;
double left, right, top, bottom;
char *wkt = NULL;
OGRSFDriverH driver;
OGRSpatialReferenceH srs;
GDALDatasetH fp;
OGRLayerH layer;
OGRFieldDefnH field;
OGRFeatureH feature;
OGRGeometryH poly, ring;
coord_t geo[4], map[4], min, max, tile;
cube_t *cube = NULL;


  if (argc >= 2) check_arg(argv[1]);
  if (argc != 7){ printf("usage: %s datacube bottom top left right format\n\n", argv[0]); 
                  printf("             bottom top left right in geographic coordinates\n");
                  printf("             format: shp or kml\n\n");
                  return FAILURE;}


  OGRRegisterAll();

  // get command line parameters
  dir      = argv[1];
  geo[0].x = atof(argv[4]); geo[0].y = atof(argv[2]); // LL
  geo[1].x = atof(argv[5]); geo[1].y = atof(argv[2]); // LR
  geo[2].x = atof(argv[4]); geo[2].y = atof(argv[3]); // UL
  geo[3].x = atof(argv[5]); geo[3].y = atof(argv[3]); // UR
  format   = argv[6];
  min.x = INT_MAX; min.y = INT_MAX; max.x = INT_MIN; max.y = INT_MIN; 


  // get driver
  if (strcmp(format, "shp") == 0){
    if ((driver = OGRGetDriverByName("ESRI Shapefile")) == NULL){
      printf("%s driver is not available\n", format); return FAILURE;}
  } else if (strcmp(format, "kml") == 0){
    if ((driver = OGRGetDriverByName("KML")) == NULL){
      printf("%s driver is not available\n", format); return FAILURE;}
  } else {
      printf("unknown format %s, use shp or kml\n", format); return FAILURE;
  }

  // read datacube definition
  if ((cube = read_datacube_def(dir)) == NULL){
    printf("Reading datacube definition failed.\n"); return FAILURE;}
  wkt = cube->proj;

  // Output name
  nchar = snprintf(fname, NPOW_10, "%s/%s", dir, format);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

  if (fileexist(fname)){
    printf("Grid already exists: %s.\n", fname); return FAILURE;}

  // get border coordinates in target css coordinates
  for (c=0; c<4; c++){
    if ((warp_geo_to_any(geo[c].x,  geo[c].y, &map[c].x, &map[c].y, 
                     cube->proj)) == FAILURE){
      printf("Computing bbox coordinates in dst_srs failed!\n"); return FAILURE;}
    if (map[c].x < min.x) min.x = map[c].x;
    if (map[c].y < min.y) min.y = map[c].y;
    if (map[c].x > max.x) max.x = map[c].x;
    if (map[c].y > max.y) max.y = map[c].y;
  }

  // find the UL and LR tile
  tile_find(min.x, max.y, &tile.x, &tile.y, &x0, &y0, cube);
  tile_find(max.x, min.y, &tile.x, &tile.y, &x1, &y1, cube);


  // create file
  if ((fp = OGR_Dr_CreateDataSource(driver, fname, NULL)) == NULL){
    printf("Error creating output file.\n"); return FAILURE;}
    

  // set output projection
  srs = OSRNewSpatialReference(NULL);
  OSRImportFromWkt(srs, &wkt);

  // create layer
  if ((layer = OGR_DS_CreateLayer(fp, "grid", srs, 
        wkbPolygon, NULL)) == NULL){
    printf("Error creating layer.\n"); return FAILURE;}

  // add field
  field = OGR_Fld_Create("Tile_ID", OFTString);
  OGR_Fld_SetWidth(field, 12);
  if (OGR_L_CreateField(layer, field, TRUE) != OGRERR_NONE){
      printf("Error creating field.\n"); return FAILURE;}
  OGR_Fld_Destroy(field);

  // add field
  field = OGR_Fld_Create("Tile_X", OFTInteger );
  OGR_Fld_SetWidth(field, 4);
  if (OGR_L_CreateField(layer, field, TRUE) != OGRERR_NONE){
      printf("Error creating field.\n"); return FAILURE;}
  OGR_Fld_Destroy(field);

  // add field
  field = OGR_Fld_Create("Tile_Y", OFTInteger );
  OGR_Fld_SetWidth(field, 4);
  if (OGR_L_CreateField(layer, field, TRUE) != OGRERR_NONE){
      printf("Error creating field.\n"); return FAILURE;}
  OGR_Fld_Destroy(field);


  for (py=y0; py<=y1; py++){
  for (px=x0; px<=x1; px++){

    // bounding box
    left   = cube->origin_map.x + cube->tilesize*px;
    right  = cube->origin_map.x + cube->tilesize*(px+1);
    top    = cube->origin_map.y - cube->tilesize*py;
    bottom = cube->origin_map.y - cube->tilesize*(py+1);

    // tile ID
    nchar = snprintf(tile_id, NPOW_10, "X%04d_Y%04d", px, py);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling tile ID\n"); return FAILURE;}


    // create feature
    feature = OGR_F_Create(OGR_L_GetLayerDefn(layer));

    // set fields
    OGR_F_SetFieldString(feature, OGR_F_GetFieldIndex(feature, "Tile_ID"), tile_id);
    OGR_F_SetFieldInteger(feature, OGR_F_GetFieldIndex(feature, "Tile_X"), px);
    OGR_F_SetFieldInteger(feature, OGR_F_GetFieldIndex(feature, "Tile_Y"), py);

    // create local geometry
    poly = OGR_G_CreateGeometry(wkbPolygon);
    ring = OGR_G_CreateGeometry(wkbLinearRing );
    OGR_G_SetPointCount(ring, 5);
    OGR_G_SetPoint_2D(ring, 0, left,  top);
    OGR_G_SetPoint_2D(ring, 1, right, top);
    OGR_G_SetPoint_2D(ring, 2, right, bottom);
    OGR_G_SetPoint_2D(ring, 3, left,  bottom);
    OGR_G_SetPoint_2D(ring, 4, left,  top);
    OGR_G_AddGeometry(poly, ring);
    OGR_F_SetGeometry(feature, poly);
    OGR_G_DestroyGeometry(poly);

    // create feature in the file
    if (OGR_L_CreateFeature(layer, feature) != OGRERR_NONE){
      printf("Error creating feature in file.\n"); return FAILURE;}
    OGR_F_Destroy(feature);

  }
  }

  //GDALClose(fp);
  OGR_DS_Destroy(fp);


  return SUCCESS;
}

