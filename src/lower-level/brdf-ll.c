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

/** The following code was mainly adopted from the Ambrals forward model 
+++ code, openly shared by Crystal Schaaf, Univ. of Massachusetts Boston
+++ https://www.umb.edu/spectralmass/terra_aqua_modis/modis_user_tools
+++ Ambrals Forward Model Copyright (C) 1999, 2000 Crystal Schaaf
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for BRDF forward modelling
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "brdf-ll.h"


#define FMAX(X,Y) ((X)>(Y)?(X):(Y))
#define FMIN(X,Y) ((X)<(Y)?(X):(Y))


/** BRDF correction factors
+++ This function applies the BRDF forward model and computes a correction
+++ factor to convert measured BOA reflectance to NBAR, fixed to 45° sun
+++ zenith and nadir looking. The paramaters represent a global set of 
+++ parameters as tabulated by Roy et al. 2016-2017 for Landsat and Senti-
+++ nel-2. 
+++-----------------------------------------------------------------------
+++ Roy, D.P, Zhang, H.K., Gomez-Dans, J.L., Lewis, P.E., Schaaf, C.B., Su
+++ n, Q., Li, J., Huang, H., Kovalskyy, V. (2016). A general method to no
+++ rmalize Landsat reflectance data to nadir BRDF adjusted reflectance. R
+++ emote Sensing of Environment, 176, 255-271.
+++-----------------------------------------------------------------------
+++ Roy, D.P, Li, J., Zhang, H.K., Yan, L., Huang, H., Li, Z. (2017). Exam
+++ ination of Sentinel-2A multi-spectral instrument (MSI) reflectance ani
+++ sotropy and the suitability of a general method to normalize MSI refle
+++ ctance to nadir BRDF adjusted reflectance. Remote Sensing of Environme
+++ nt 199, 25-38.
+++-----------------------------------------------------------------------
+++ Roy, D.P. Li, Z., Zhang, H.K. (2017). Adjustment of Sentinel-2 Multi-S
+++ pectral Instrument (MSI) Red-Edge Band Reflectance to Nadir BRDF Adjus
+++ ted Reflectance (NBAR) and Quantification of Red-Edge Band BRDF Effect
+++ s. Remote Sensing 9 (12), 1325.
+++-----------------------------------------------------------------------
--- sun:    sun angle brick
--- view:   view angle brick
--- cor:    brdf correction factor brick
--- g:      Coarse grid cell
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int brdf_factor(brick_t *sun, brick_t *view, brick_t *cor, int g){
int e, f;
double lon, lat;
float szen, sazi, vzen, vazi, standard_szen;
float brdf_correction;
int b,  nb; // band ID of actual bands
int b_, nb_ = 10; // band ID and number of bands for which we have parameters
float iso[10] = { 0.0774, 0.1306, 0.1690, 0.2085, 0.2316, 0.2599, 0.3093, 0.3093, 0.3430, 0.2658 };
float vol[10] = { 0.0372, 0.0580, 0.0574, 0.0845, 0.1003, 0.1197, 0.1535, 0.1535, 0.1154, 0.0639 };
float geo[10] = { 0.0079, 0.0178, 0.0227, 0.0256, 0.0273, 0.0294, 0.0330, 0.0330, 0.0453, 0.0387 };
char domain[10][NPOW_10] = { "BLUE", "GREEN", "RED", 
                           "REDEDGE1", "REDEDGE2",
                           "REDEDGE3", "BROADNIR",
                           "NIR", "SWIR1", "SWIR2" };


  nb = get_brick_nbands(cor);
  szen = get_brick(sun,  ZEN, g);
  vzen = get_brick(view, ZEN, g);
  sazi = get_brick(sun,  AZI, g);
  vazi = get_brick(view, AZI, g);

  convert_brick_p2ji(sun, sun, g, &e, &f);
  get_brick_geo(sun, f, e, &lon, &lat);

  // initialize
  for (b=0; b<nb; b++) set_brick(cor, b, g, 1.0);

  #ifdef FORCE_DEBUG
  printf("BRDF: ");
  #endif


  standard_szen = standard_sunzenith(sun->date, lat, lon);
  standard_szen *= _D2R_CONV_;

  #ifdef FORCE_DEBUG
  printf("actual sun zenith: %.2f, azimuth: %.2f\n", 
    szen*_R2D_CONV_, sazi*_R2D_CONV_);
  #endif

  for (b_=0; b_<nb_; b_++){

    if ((b = find_domain(cor, domain[b_])) < 0) continue;

    brdf_correction = 
      brdf_forward(standard_szen, 0, 0,   iso[b_], vol[b_], geo[b_]) / 
      //brdf_forward(45*_D2R_CONV_, 0, 0,   iso[b_], vol[b_], geo[b_]) / 
      brdf_forward(szen, vzen, sazi-vazi, iso[b_], vol[b_], geo[b_]);

    set_brick(cor, b, g, brdf_correction);

    #ifdef FORCE_DEBUG
    printf(" %.3f", brdf_correction);
    #endif

  }

  #ifdef FORCE_DEBUG
  printf("\n");
  #endif

  return SUCCESS;
}


/** BRDF forward model
+++ This function runs the Ross-Thick-Li-Sparse-Reciprocal model in the 
+++ forward mode and returns the calculated reflectance.
--- ti:     sunzenith in radians
--- tv:     view zenith in radians
--- phi:    relative azimuth in radians
--- iso:    kernel weight for isotropic scattering
--- vol:    kernel weight for volumetric scattering
--- geo:    kernel weight for geometric scattering
+++ Return: reflectance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float brdf_forward(float ti, float tv, float phi, float iso, float vol, float geo){
float cosphi, sinphi, costv, sintv, costi, sinti;
float cosphaang, phaang, sinphaang, rosskernel, tantv, tanti;
float likernel, refl;

  cosphi = cos(phi); sinphi = sin(phi);
  costv =  cos(tv);  costi = cos(ti);
  sintv =  sin(tv);  sinti = sin(ti);

  GetPhaang(costv, costi, sintv, sinti, cosphi, &cosphaang, &phaang, &sinphaang);
  rosskernel = ((M_PI/2 - phaang) * cosphaang + sinphaang)/(costi + costv) - M_PI/4;

  tantv = sintv/costv;
  tanti = sinti/costi;

  LiKernel(2.0,1.0,tantv,tanti,sinphi,cosphi,&likernel);

  refl = iso + vol*rosskernel + geo*likernel;

  return refl;
}


/** Li-Sparse kernel
+++ This function calculates the Li-Sparse kernel
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void LiKernel(float hbratio, float brratio, float tantv, float tanti,
              float sinphi, float cosphi, float *result){
float sintvp, costvp, tantvp, sintip, costip, tantip;
float phaangp, cosphaangp, sinphaangp, distancep, overlap, temp;

  GetpAngles(brratio,tantv,&sintvp,&costvp,&tantvp);
  GetpAngles(brratio,tanti,&sintip,&costip,&tantip);
  GetPhaang(costvp,costip,sintvp,sintip,cosphi,&cosphaangp,&phaangp,&sinphaangp);
  GetDistance(tantvp,tantip,cosphi,&distancep);
  GetOverlap(hbratio,distancep,costvp,costip,tantvp,tantip,sinphi,&overlap,&temp);

  *result = overlap - temp + 0.5 * (1.0+cosphaangp)/costvp/costip;

  return;
}


/** Phase angle
+++ This function calculates the phase angle
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void GetPhaang(float cos1, float cos2, float sin1, float sin2, float cos3,
               float *cosres, float *res,float *sinres){

  *cosres = cos1*cos2 + sin1*sin2*cos3;
  *res = acos( FMAX(-1.0, FMIN(1.0,*cosres)) );
  *sinres = sin(*res);

  return;
}


/** Prime angles
+++ This function calculates the 'prime' angles
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void GetpAngles(float brratio, float tan1, float *sinp, float *cosp, float *tanp){
float angp;

  *tanp = brratio*tan1;
//  if(*tanp < 0) *tanp = 0.0;
  angp  = atan(*tanp);
  *sinp = sin(angp);
  *cosp = cos(angp);

  return;
}


/** D distance
+++ This function calculates the D distance
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void GetDistance(float tan1, float tan2, float cos3,float *res){
float temp;

  temp = tan1*tan1+tan2*tan2-2.0*tan1*tan2*cos3;
  *res = sqrt(FMAX(0.0,temp));

  return;
}


/** Overlap distance
+++ This function calculates the overlap distance
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void GetOverlap(float hbratio, float distance, float cos1, float cos2,
                float tan1, float tan2, float sin3, float *overlap, float *temp){
float cost, sint, tvar;

  *temp = 1.0/cos1 + 1.0/cos2;
   cost = hbratio*sqrt(distance*distance+tan1*tan1*tan2*tan2*sin3*sin3)/(*temp);
   cost = FMAX(-1.0, FMIN(1.0,cost));
   tvar = acos(cost);
   sint = sin(tvar);
  *overlap = 1.0/M_PI *(tvar-sint*cost)*(*temp);
  *overlap = FMAX(0.0,*overlap);

  return;
}

