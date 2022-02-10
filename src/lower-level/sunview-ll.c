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
This file contains functions forcomputing sun/view geometry
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "sunview-ll.h"


date_t localtime2gmt(date_t *dmeta, double localtime, double lon);


/** Convert local overpass time to GMT time
+++ This function was adopted from code kindly provided by Hakui Zhang.
--- dmeta      date of the metadata
--- localtime: local time in decimal hours
--- lon:       longitude
+++ Return:    GMT date
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
date_t localtime2gmt(date_t *dmeta, double localtime, double lon){
int doy, hh, mm, ss;
double gmttime, left_mm, left_ss;;
date_t dgmt;


  gmttime = localtime-lon/15.0;

  doy = dmeta->doy;

  if (gmttime > 24){
    doy++;
    gmttime -= 24.0;
    if (doy > 365) doy = 365;
  }

  if (gmttime < 0){
    doy--;
    gmttime += 24.0;
    if (doy<1) doy = 1;
  }


  copy_date(dmeta, &dgmt);
  set_date_doy(&dgmt, doy);

  hh = (int)gmttime;
  left_mm = (gmttime-hh)*60.0;
  mm = (int)left_mm;
  left_ss = (left_mm-mm)*60.0;
  ss = (int)left_ss;

  set_time(&dgmt, hh, mm, ss);

  return dgmt;
}


/** Compute a standardized sun zenith for a given latitude to be used for
+++ a sun angle harmonization between Landsat and Sentinel-2 in the BRDF
+++ correction. The average sun zenith between a standard, latitude-depen-
+++ dent sun zenith of Landsat 8 and Sentinel-2 is computed.
+++ This function was adopted from code kindly provided by Hakui Zhang.
--- dmeta      date of the metadata
--- lat:       latitude
--- lon:       longitude
+++ Return:    sun zenith (in degrees)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
double standard_sunzenith(date_t *dmeta, double lat, double lon){
double inclination[_MISSION_LENGTH_];
double localtime_equator[_MISSION_LENGTH_];
double localtime = 0;
date_t dgmt;
double rlat = lat * _D2R_CONV_;
float szen, sazi;


  // orbital parameters 

  inclination[LANDSAT] = 98.2 * _D2R_CONV_;
  localtime_equator[LANDSAT] = 10.18333333333;

  inclination[SENTINEL2] = 98.62 * _D2R_CONV_;
  localtime_equator[SENTINEL2] = 10.5;


  // compute localtime using astronomical model

  localtime += localtime_equator[LANDSAT] - 
    asin(tan(rlat)/tan(inclination[LANDSAT])) * _R2D_CONV_ / 15.0;

  localtime += localtime_equator[SENTINEL2] - 
    asin(tan(rlat)/tan(inclination[SENTINEL2])) * _R2D_CONV_ / 15.0;

  localtime /= 2.0;

  dgmt = localtime2gmt(dmeta, localtime, lon);

  sunpos(lat, lon, dgmt, &szen, &sazi);

  #ifdef FORCE_DEBUG
  print_date(dmeta);
  printf("lat: %.2f, lon: %.2f\n", lat, lon);
  printf("localtime: %.2f\n", localtime);
  print_date(&dgmt);
  printf("standardized sun zenith: %.2f, azimuth: %.2f\n", szen, sazi);
  #endif

  return szen;
}


/** Compute sun positions and view geometry
+++ This function computes sun positions (+cos/sin/tan), and view angles
--- pl2:    L2 parameters
--- meta:   metadata
--- mission: mission ID
--- atc:    atmospheric correction factors
--- QAI:    Quality Assurance Information
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int sun_target_view(par_ll_t *pl2, meta_t *meta, int mission, atc_t *atc, brick_t *QAI){
double lat, lon;
float zen, azi;
int e, f, g, p;
int ne, nf, nc;
float *xy_szen = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  nf  = get_brick_ncols(atc->xy_sun);
  ne  = get_brick_nrows(atc->xy_sun);
  nc  = get_brick_ncells(QAI);

  // average satellite height (m)
  if (mission == LANDSAT){
    atc->view.H = 705000.0;
  } else if (mission == SENTINEL2){
    atc->view.H = 786000.0;
  } else { printf("unknown mission.\n"); return FAILURE;}
  atc->view.H2 = atc->view.H*atc->view.H;

  // compute approx. view geometry for Landsat
  if (mission == LANDSAT){
    if (viewgeo(pl2, QAI, atc) == FAILURE){
      printf("error in view geometry. "); return FAILURE;}
  }


  for (e=0, g=0; e<ne; e++){
  for (f=0; f<nf; f++, g++){

    // geo coordinate
    get_brick_geo(atc->xy_sun, f, e, &lon, &lat);

    // calculate sun angles
    sunpos(lat, lon, get_brick_date(atc->xy_sun, 0), &zen, &azi);

    // degree to radians
    zen *= _D2R_CONV_;
    azi *= _D2R_CONV_;

    set_brick(atc->xy_sun,  ZEN, g, zen);
    set_brick(atc->xy_sun,  AZI, g, azi);
    set_brick(atc->xy_sun, cZEN, g, cos(zen));
    set_brick(atc->xy_sun, cAZI, g, cos(azi));
    set_brick(atc->xy_sun, sZEN, g, sin(zen));
    set_brick(atc->xy_sun, sAZI, g, sin(azi));
    set_brick(atc->xy_sun, tZEN, g, tan(zen));
    set_brick(atc->xy_sun, tAZI, g, tan(azi));

    // satellite view geometry
    if (view_angle(meta, mission, atc, QAI, f, e, g) == FAILURE){
      printf("error in view geometry. "); return FAILURE;}

  }
  }

  // min/max of cos(szen) & cos(vzen)
  get_brick_range(atc->xy_sun,  cZEN, &atc->cosszen[0], &atc->cosszen[1]);
  get_brick_range(atc->xy_view, cZEN, &atc->cosvzen[0], &atc->cosvzen[1]);


  if ((xy_szen = get_band_float(atc->xy_sun, ZEN)) == NULL) return FAILURE;

  // low sun flag
  #pragma omp parallel private(g) shared(nc, QAI, xy_szen, atc) default(none) 
  {

    #pragma omp for schedule(guided)
    for (p=0; p<nc; p++){
      if (get_off(QAI, p)) continue;
      g = convert_brick_p2p(QAI, atc->xy_sun, p);
      if (xy_szen[g] > 1.308997) set_lowsun(QAI, p, true);
    }
  }

  printf("\nSun-Sensor-Geometry :::\n");
  printf("min cosszen: %.2f\n", atc->cosszen[0]);
  printf("max cosszen: %.2f\n", atc->cosszen[1]);
  printf("min cosvzen: %.2f\n", atc->cosvzen[0]);
  printf("max cosvzen: %.2f\n", atc->cosvzen[1]);
  printf("min szen(°): %.2f\n", acos(atc->cosszen[1]) * _R2D_CONV_);
  printf("max szen(°): %.2f\n", acos(atc->cosszen[0]) * _R2D_CONV_);
  printf("min vzen(°): %.2f\n", acos(atc->cosvzen[1]) * _R2D_CONV_);
  printf("max vzen(°): %.2f\n", acos(atc->cosvzen[0]) * _R2D_CONV_);


  #ifdef FORCE_DEBUG
  print_brick_info(atc->xy_sun);  set_brick_open(atc->xy_sun,  OPEN_CREATE); write_brick(atc->xy_sun);
  print_brick_info(atc->xy_view); set_brick_open(atc->xy_view, OPEN_CREATE); write_brick(atc->xy_view);
  printf("min/max cosszen: %.2f/%.2f\n", atc->cosszen[0], atc->cosszen[1]);
  printf("min/max cosvzen: %.2f/%.2f\n", atc->cosvzen[0], atc->cosvzen[1]);
  #endif

  #ifdef FORCE_CLOCK
  proctime_print("sun-target-view geometry", TIME);
  #endif

  return SUCCESS;
}


/** Compute approximate view geometry
+++ This function computes the approximate view geometry for Landsat
--- pl2:    L2 parameters
--- QAI:    Quality Assurance Information
--- atc:    atmospheric correction factors
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int viewgeo(par_ll_t *pl2, brick_t *QAI, atc_t *atc){
int i, j, p, nx, ny;
int uly, ury, lly, lry;
int ulx, urx, llx, lrx;
int uy, ly, ux, lx;
double guly, gury, glly, glry;
double gulx, gurx, gllx, glrx;
double guy, gly, gux, glx;
double dx, dy;


  uly = -1; lry = -1; urx = -1;
  llx = get_brick_ncells(QAI);
  lrx = lly = ury = ulx = 0;

  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);

  // find corners of image
  for (i=0, p=0; i<ny; i++){
  for (j=0; j<nx; j++, p++){
    if (get_off(QAI, p)) continue;
    if (uly == -1){ uly = i; ulx = j;}
    if (j > urx){ ury = i; urx = j;}
    if (j <= llx){ lly = i; llx = j;}
    if (i >= lry){ lry = i; lrx = j;}
  }
  }

  #ifdef ACIX2
  // in ACIX, images are cropped to subregions, thus 
  // image boundaries cannot be obtained from the image data
  // --> use information from angle metadata
  if (parse_angledata_landsat(pl2, &ulx, &uly, &urx, &ury, &lrx, &lry, &llx, &lly) == FAILURE) return FAILURE;
  #endif

  
  // midpoints of top and bottom
  ux = (ulx+urx)/2; uy = (uly+ury)/2;
  lx = (llx+lrx)/2; ly = (lly+lry)/2;


  // geo coordinate
  get_brick_geo(QAI, ulx, uly, &gulx, &guly);
  get_brick_geo(QAI, ux,  uy,  &gux,  &guy);
  get_brick_geo(QAI, urx, ury, &gurx, &gury);
  get_brick_geo(QAI, llx, lly, &gllx, &glly);
  get_brick_geo(QAI, lx,  ly,  &glx,  &gly);
  get_brick_geo(QAI, lrx, lry, &glrx, &glry);

  // difference from top to bottom
  dx = gux-glx; dy = gly-guy;

  // angle of nadir line, measured clockwise from north or south
  atc->view.geo_angle_nadir = atan2(dy,dx) + M_PI/2;


  // cartesian definition of nadir line: ax + by + c = 0
  atc->view.a = uy-ly;
  atc->view.b = lx-ux;
  atc->view.c = ly*ux-lx*uy;

  // the perpendicular distance of a point to the nadir line is given by
  // |ax+by+c|/sqrt(aa+bb)
  atc->view.ab = sqrt(atc->view.a*atc->view.a +
                      atc->view.b*atc->view.b);


  #ifdef FORCE_DEBUG
  printf("\nimage corners:\n");
  printf("UL (%d/%d), UM (%d/%d), UR (%d/%d)\n", ulx, uly, ux, uy, urx, ury);
  printf("LL (%d/%d), LM (%d/%d), LR (%d/%d)\n", llx, lly, lx, ly, lrx, lry);
  printf("UL (%05.2f/%05.2f), UM (%05.2f/%05.2f), UR (%05.2f/%05.2f)\n", gulx, guly, gux, guy, gurx, gury);
  printf("LL (%05.2f/%05.2f), LM (%05.2f/%05.2f), LR (%05.2f/%05.2f)\n", gllx, glly, glx, gly, glrx, glry);
  printf("\nestimated view geometry in image coordinates:\n");
  printf(" a: %.0f, b: %.0f, c: %.0f\n", atc->view.a, atc->view.b, atc->view.c);
  printf("\nestimated view geometry in geographic coordinates:\n");
  printf(" angle_nadir: %.2f\n", atc->view.geo_angle_nadir*_R2D_CONV_);
  printf("\nsensor height parameters:\n");
  printf(" H: %.0f, H2: %.0f\n", atc->view.H, atc->view.H2);
  #endif

  return SUCCESS;
}


/** Compute view angle
+++ This function computes the view angles
--- meta:   metadata
--- mission: mission ID
--- atc:    atmospheric correction factors
--- QAI:    Quality Assurance Information
--- f:      column in coarse grid
--- e:      row in coarse grid
--- g:      cell in coarse grid
+++ Return: SUCCESS/FAILURE
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int view_angle(meta_t *meta, int mission, atc_t *atc, brick_t *QAI, int f, int e, int g){
int i, j, p, nx, ny;
float dist, res, cres, rres, zen = 0, azi = 0;


  nx = get_brick_ncols(QAI);
  ny = get_brick_nrows(QAI);

  res  = get_brick_res(QAI);
  cres = get_brick_res(atc->xy_view);
  rres = res/cres;


  if (mission == LANDSAT){

    // distance from nadir line in pixels
    dist = (atc->view.a*f/rres + atc->view.b*e/rres + atc->view.c) / 
      atc->view.ab;

    // view zenith
    zen = acos(atc->view.H/sqrt(atc->view.H2+res*dist*res*dist));
    if (zen != zen) zen = 0;

    // view azimuth
    if (dist > 0){ // left of nadir, i.e. right looking
      azi = atc->view.geo_angle_nadir+M_PI/2;
    } else {       // right of nadir, i.e. left looking
      azi = atc->view.geo_angle_nadir-M_PI/2;
      while (azi < 0) azi += 2*M_PI;
    }

  } else if (mission == SENTINEL2){

    if (meta->s2.vzen[e][f] == meta->s2.nodata || meta->s2.vazi[e][f] == meta->s2.nodata){

      zen = atc->nodata;
      azi = atc->nodata;

      // set nodata for all pixels in this grid cell
      for (i=e/rres; i<((e+1)/rres); i++){
      for (j=f/rres; j<((f+1)/rres); j++){

      if (i > ny-1) continue;
      if (j > nx-1) continue;
        p = i*nx+j;
        set_off(QAI, p, true);
      }
      }

    } else {
      zen = meta->s2.vzen[e][f]*_D2R_CONV_;
      azi = meta->s2.vazi[e][f]*_D2R_CONV_;
    }

  }

  if (fequal(zen, atc->nodata)){
    set_brick(atc->xy_view,  ZEN, g, atc->nodata);
    set_brick(atc->xy_view,  AZI, g, atc->nodata);
    set_brick(atc->xy_view, cZEN, g, atc->nodata);
    set_brick(atc->xy_view, cAZI, g, atc->nodata);
    set_brick(atc->xy_view, sZEN, g, atc->nodata);
    set_brick(atc->xy_view, sAZI, g, atc->nodata);
    set_brick(atc->xy_view, tZEN, g, atc->nodata);
    set_brick(atc->xy_view, tAZI, g, atc->nodata);
  } else {
    set_brick(atc->xy_view,  ZEN, g, zen);
    set_brick(atc->xy_view,  AZI, g, azi);
    set_brick(atc->xy_view, cZEN, g, cos(zen));
    set_brick(atc->xy_view, cAZI, g, cos(azi));
    set_brick(atc->xy_view, sZEN, g, sin(zen));
    set_brick(atc->xy_view, sAZI, g, sin(azi));
    set_brick(atc->xy_view, tZEN, g, tan(zen));
    set_brick(atc->xy_view, tAZI, g, tan(azi));
  }
  

  return SUCCESS;
}

