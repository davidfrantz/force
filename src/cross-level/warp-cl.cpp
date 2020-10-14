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
This file contains functions that handle warping and reprojection requests
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "warp-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_conv.h"       // various convenience functions for CPL
#include "gdalwarper.h"     // GDAL warper related entry points and defs
#include "ogr_spatialref.h" // coordinate systems services


/** Reproject point from geographic to any projection
+++ This function reprojects a coordinate.
--- srs_x:    x-coordinate in source projection
--- srs_y:    y-coordinate in source projection
--- dst_x:    x-coordinate in target projection (returned)
--- dst_y:    y-coordinate in target projection (returned)
--- dst_wkt:  target projection
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_geo_to_any(double  srs_x, double  srs_y, 
               double *dst_x, double *dst_y, char *dst_wkt){
OGRSpatialReference oSrcSRS, oDstSRS;
OGRCoordinateTransformation *poCT = NULL;
char *wkt = dst_wkt;
double x, y;


  CPLSetConfigOption("OGR_CT_FORCE_TRADITIONAL_GIS_ORDER", "YES"); 

  if (srs_x < -180 || srs_x > 180){
    printf("Longitude is out of bounds.\n"); return FAILURE;}
  if (srs_y <  -90 || srs_y >  90){
    printf("Latitude  is out of bounds.\n"); return FAILURE;}

  x = srs_x; y = srs_y;

  // set coordinate systems
  oSrcSRS.SetWellKnownGeogCS("WGS84");
  oDstSRS.importFromWkt(&wkt);

  // create transformation
  poCT = OGRCreateCoordinateTransformation(&oSrcSRS, &oDstSRS);

  // transform
  if (poCT == NULL || !poCT->Transform(1, &x, &y)){
    printf( "Transformation failed.\n" ); return FAILURE;
  } else delete poCT;

  *dst_x = x; *dst_y = y;
  return SUCCESS;
}


/** Reproject point from any projection to geographic
+++ This function reprojects a coordinate.
--- fname:  filename
--- srs_x:    x-coordinate in source projection
--- srs_y:    y-coordinate in source projection
--- dst_x:    x-coordinate in target projection (returned)
--- dst_y:    y-coordinate in target projection (returned)
--- src_wkt:  source projection
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_any_to_geo(double  srs_x, double  srs_y, 
               double *dst_x, double *dst_y, char *src_wkt){
OGRSpatialReference oSrcSRS, oDstSRS;
OGRCoordinateTransformation *poCT = NULL;
char *wkt = src_wkt;
double x, y;


  CPLSetConfigOption("OGR_CT_FORCE_TRADITIONAL_GIS_ORDER", "YES"); 

  x = srs_x; y = srs_y;

  // set coordinate systems
  oSrcSRS.importFromWkt(&wkt);
  oDstSRS.SetWellKnownGeogCS("WGS84");

  // create transformation
  poCT = OGRCreateCoordinateTransformation(&oSrcSRS, &oDstSRS);

  // transform
  if (poCT == NULL || !poCT->Transform(1, &x, &y)){
    printf( "Transformation failed.\n" ); return FAILURE;
  } else delete poCT;

  if (x < -180 || x > 180){
    printf("Longitude is out of bounds.\n"); return FAILURE;}
  if (y <  -90 || y >  90){
    printf("Latitude  is out of bounds.\n"); return FAILURE;}

  *dst_x = x; *dst_y = y;
  return SUCCESS;
}


/** Reproject point from any to any other projection
+++ This function reprojects a coordinate.
--- srs_x:    x-coordinate in source projection
--- srs_y:    y-coordinate in source projection
--- dst_x:    x-coordinate in target projection (returned)
--- dst_y:    y-coordinate in target projection (returned)
--- src_wkt:  source projection
--- dst_wkt:  target projection
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int warp_any_to_any(double  srs_x, double  srs_y,
                 double *dst_x, double *dst_y,
                 char *src_wkt, char *dst_wkt){
OGRSpatialReference oSrcSRS, oDstSRS;
OGRCoordinateTransformation *poCT = NULL;
char *src_css = src_wkt;
char *dst_css = dst_wkt;
double x, y;


  CPLSetConfigOption("OGR_CT_FORCE_TRADITIONAL_GIS_ORDER", "YES"); 

  x = srs_x; y = srs_y;

  // set coordinate systems
  oSrcSRS.importFromWkt(&src_css);
  oDstSRS.importFromWkt(&dst_css);

  // create transformation
  poCT = OGRCreateCoordinateTransformation(&oSrcSRS, &oDstSRS);

  // transform
  if (poCT == NULL || !poCT->Transform(1, &x, &y)){
    printf( "Transformation failed.\n" ); return FAILURE;
  } else delete poCT;

  *dst_x = x; *dst_y = y;
  return SUCCESS;
}

