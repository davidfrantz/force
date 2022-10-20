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
This program creates a shapefile containing the tile IDs of the prepro-
cessing grid + the bounding box of each tile in projected coordinates
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include <ctype.h>   // testing and mapping characters
#include <unistd.h>  // standard symbolic constants and types 

#include "../cross-level/const-cl.h"
#include "../cross-level/konami-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/warp-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "gdal.h"           // public (C callable) GDAL entry points
#include "ogr_spatialref.h" // coordinate systems services
#include "ogr_api.h"        // OGR geometry and feature definition


enum { _bottom_, _top_, _left_, _right_ };
enum { _ll_, _lr_, _ul_, _ur_ };


typedef struct {
  int n;
  double bbox[4]; // bottom,top,left,right
  char dcube[NPOW_10];
  char format[NPOW_10];
} args_t;


void usage(char *exe, int exit_code){


  printf("Usage: %s [-h] [-v] [-i] [-b bottom,top,left,right] [-f format] datacube-dir\n", exe);
  printf("\n");
  printf("  -h  = show this help\n");
  printf("  -v  = show version\n");
  printf("  -i  = show program's purpose\n");
  printf("\n");
  printf("  -b bottom,top,left,right  = bounding box\n");
  printf("     use geographic coordinates! 4 comma-separated numbers\n");
  printf("\n");
  printf("  -f format  = output format: shp or kml (default)\n");
  printf("\n");
  printf("  Positional arguments:\n");
  printf("  - 'datacube-dir': directory of existing datacube\n");
  printf("\n");

  exit(exit_code);
  return;
}


void parse_args(int argc, char *argv[], args_t *args){
int opt;
char buffer[NPOW_10];
char *ptr = NULL;
const char *separator = "/,";
int i;


  opterr = 0;

  // default parameters
  args->bbox[_bottom_] =  -90;
  args->bbox[_top_]    =   90;
  args->bbox[_left_]   = -180;
  args->bbox[_right_]  =  180;
  copy_string(args->format, NPOW_10, "kml");

  // optional parameters
  while ((opt = getopt(argc, argv, "hvit:b:f:")) != -1){
    switch(opt){
      case 'h':
        usage(argv[0], SUCCESS);
      case 'v':
        printf("FORCE version: %s\n", _VERSION_);
        exit(SUCCESS);
      case 'i':
        printf("Extract the data cube grid to shapefile\n");
        exit(SUCCESS);
      case 'b':
        copy_string(buffer, NPOW_10, optarg);
        ptr = strtok(buffer, separator);
        i = 0;
        while (ptr != NULL){
          if (i < 4) args->bbox[i] = atof(ptr);
          ptr = strtok(NULL, separator);
          i++;
        }
        if (i != 4){
          fprintf(stderr, "Bounding box must have 4 numbers.\n");
          usage(argv[0], FAILURE);
        } 
        break;
      case 'f':
        copy_string(args->format, NPOW_10, optarg);
        break;
      case '?':
        if (isprint(optopt)){
          fprintf(stderr, "Unknown option `-%c'.\n", optopt);
        } else {
          fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
        }
        usage(argv[0], FAILURE);
      default:
        fprintf(stderr, "Error parsing arguments.\n");
        usage(argv[0], FAILURE);
    }
  }

  // non-optional parameters
  args->n = 1;

  if (optind < argc){
    konami_args(argv[optind]);
    if (argc-optind == args->n){
      copy_string(args->dcube, NPOW_10, argv[optind++]);
    } else if (argc-optind < args->n){
      fprintf(stderr, "some non-optional arguments are missing.\n");
      usage(argv[0], FAILURE);
    } else if (argc-optind > args->n){
      fprintf(stderr, "too many non-optional arguments.\n");
      usage(argv[0], FAILURE);
    }
  } else {
    fprintf(stderr, "non-optional arguments are missing.\n");
    usage(argv[0], FAILURE);
  }

  return;
}


int main(int argc, char *argv[]){
args_t args;
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


  parse_args(argc, argv, &args);

  OGRRegisterAll();

  // get geographic coords
  geo[_ll_].x = args.bbox[_left_];  geo[_ll_].y = args.bbox[_bottom_];
  geo[_lr_].x = args.bbox[_right_]; geo[_lr_].y = args.bbox[_bottom_];
  geo[_ul_].x = args.bbox[_left_];  geo[_ul_].y = args.bbox[_top_];
  geo[_ur_].x = args.bbox[_right_]; geo[_ur_].y = args.bbox[_top_];
  min.x = INT_MAX; min.y = INT_MAX; max.x = INT_MIN; max.y = INT_MIN; 


  // get driver
  if (strcmp(args.format, "shp") == 0){
    if ((driver = OGRGetDriverByName("ESRI Shapefile")) == NULL){
      fprintf(stderr, "%s driver is not available.\n", args.format); usage(argv[0], FAILURE);}
  } else if (strcmp(args.format, "kml") == 0){
    if ((driver = OGRGetDriverByName("KML")) == NULL){
      fprintf(stderr, "%s driver is not available.\n", args.format); usage(argv[0], FAILURE);}
  } else {
      fprintf(stderr, "Unknown format %s.\n", args.format); usage(argv[0], FAILURE);
  }

  // read datacube definition
  if ((cube = read_datacube_def(args.dcube)) == NULL){
    fprintf(stderr, "Reading datacube definition failed.\n"); usage(argv[0], FAILURE);}
  wkt = cube->proj;

  // Output name
  nchar = snprintf(fname, NPOW_10, "%s/%s", args.dcube, args.format);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return FAILURE;}

  if (fileexist(fname)){
    fprintf(stderr, "Grid already exists: %s.\n", fname); usage(argv[0], FAILURE);}

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

