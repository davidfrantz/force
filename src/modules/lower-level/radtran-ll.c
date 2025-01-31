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
This file contains functions for radiative transfer computations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "radtran-ll.h"


/** Compute molecular optical depth
+++ This function computes MOD @ sea-level.
--- wvl:    wavelength in µm
+++ Return: molecular optical depth
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float molecular_optical_depth(float wvl){
float mod;

  mod = 0.0088*pow(wvl, -4.15+0.2*wvl);

  return mod;
}


/** Elevation scaling factors for AOD
--- z:      elevation in km
--- Hp:     scale height
+++ Return: correction factor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float aod_elev_factor(float z, float Hp){

  return exp(-z/Hp);
}


/** Elevation scaling factors for MOD
--- z:      elevation in km
+++ Return: correction factor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float mod_elev_factor(float z){

  return exp(-0.1188*z - 0.0016*z*z);
}


/** Elevation scaling of MOD
+++ This function scales MOD to a given elevation. in this process, MOD is
+++ first scaled to sea level, then to the target elevation.
--- mod:    molecular optical depth @ a given elevation
--- Hr_is:  correction factor (corresponds to the mod's elevation)
---         If mod is reported at sea level, give 1.
--- Hr:     correction factor (corresponds to the target elevation)
+++ Return: scaled MOD
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float mod_elev_scale(float mod, float Hr_is, float Hr){

  return mod/Hr_is*Hr;
}


/** Elevation scaling of AOD
+++ This function scales AOD to a given elevation. in this process, AOD is
+++ first scaled to sea level, then to the target elevation.
--- aod:    aerosol optical depth @ a given elevation
--- Ha_is:  correction factor (corresponds to the aod's elevation)
---         If aod is reported at sea level, give 1.
--- Ha:     correction factor (corresponds to the target elevation)
+++ Return: scaled AOD
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float aod_elev_scale(float aod, float Ha_is, float Ha){

  return aod/Ha_is*Ha;
}


/** Compute optical depth
--- aod:   aerosol optical depth
--- mod:   molecular optical depth
+++ Return: Optical depth
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float optical_depth(float aod, float mod){

  return aod+mod;
}


/** Compute scattering transmittances
+++ This function computes the total, direct and diffuse portions of the
+++ scattering transmittance.
--- aod:   aerosol optical depth
--- mod:   molecular optical depth
--- od:    optical depth
--- ms:    air mass down
--- mv:    air mass up
--- Ts:    scattering transmittance down         (returned)
--- Tv:    scattering transmittance up           (returned)
--- tsd:   direct  scattering transmittance down (returned)
--- tss:   diffuse scattering transmittance down (returned)
--- tvd:   direct  scattering transmittance up   (returned)
--- tvs:   diffuse scattering transmittance up   (returned)
+++ Return: Total scattering transmittance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float scatt_transmitt(float aod, float mod, float od, float ms, float mv, float *Ts, float *Tv, float *tsd, float *tss, float *tvd, float *tvs){
float tmp;

  tmp = -1.0*(0.52*mod+0.167*aod);
  
  *Ts  = exp(tmp/ms);
  *Tv  = exp(tmp/mv);
  *tvd = exp(-1.0*od/mv);
  *tvs = (*Tv)-(*tvd);
  *tsd = exp(-1.0*od/ms);
  *tss = (*Ts)-(*tsd);

  return (*Ts)*(*Tv);
}


/** Compute path reflectance
+++ This function computes the path reflectance assuming the multiple or
+++ single scattering approximation.
--- multi:  multiple scattering?
--- sob:    assymetry
--- aod:    aerosol optical depth
--- mod:    molecular optical depth
--- od:     optical depth
--- Pa:     phase function aerosol
--- Pr:     phase function molecular
--- tsd:    direct scattering transmittance down
--- tvd:    direct scattering transmittance up
--- ms:     air mass down
--- mv:     air mass up
+++ Return: path reflectance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float path_ref(bool multi, float sob, float aod, float mod, float od, float Pa, float Pr, float tsd, float tvd, float ms, float mv){
float xav, phase, rv, rs, rho_p;

  if (multi){

    // multiple scattering (Sobolev)
    xav = (sob*aod)/od;
    phase = (aod/od)*Pa + (mod/od)*Pr;
    rv = 1.0 + 1.5*mv + (1.0-1.5*mv)*tvd;
    rs = 1.0 + 1.5*ms + (1.0-1.5*ms)*tsd;
    rho_p = 1.0 - rv*rs/(4.0 + (3.0 - xav)*od) +
       ((3.0 + xav)*mv*ms - 2.0*(mv + ms) + phase) * 
        (1.0 - exp(-od*(1.0/mv+1.0/ms))) / (mv+ms)/4.0;

  } else {

    // single scattering (Gordon)
    rho_p = aod*Pa/(4.0*mv*ms) + mod*Pr/(4.0*mv*ms);

  }

  return rho_p;
}


/** Compute spherical albedo
--- aod:    aerosol optical depth
--- mod:    molecular optical depth
--- od:     optical depth
+++ Return: spherical albedo
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sphere_albedo(float aod, float mod, float od){
float s;

  s = exp(-1.0*od) * (0.92*mod + 0.333*aod);
  
  return s;
}


/** Compute aerosol environmental weighting function
+++ This function computes the aerosol environmental weighting function 
+++ in dependence of target radius.
--- r:      target radius in km
+++ Return: aerosol environmental weighting function parameter
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float env_weight_aerosol(float r){
float Fa;

  Fa = 1.0-(0.375*exp(-0.2*r)+0.625*exp(-1.83*r));

  return Fa;
}


/** Compute molecular environmental weighting function
+++ This function computes the molecular environmental weighting function 
+++ in dependence of target radius.
--- r:      target radius in km
+++ Return: molecular environmental weighting function parameter
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float env_weight_molecular(float r){
float Fr;

  Fr = 1.0-(0.930*exp(-0.8*r)+0.070*exp(-1.10*r));

  return Fr;
}


/** Compute environmental weighting function
+++ This function computes the environmental weighting function in depen-
+++ dence of AOD and MOD.
--- aod:    aerosol optical depth
--- mod:    molecular optical depth
--- aod:    environmental weighting function, aerosol
--- mod:    environmental weighting function, molecular
+++ Return: environmental weighting function parameter
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float env_weight(float aod, float mod, float Fa, float Fr){
float F;

  F = (0.8333*aod*Fa + 0.5*mod*Fr) / (0.8333*aod + 0.5*mod);

  return F;
}


/** Compute backscattering angle
--- ms:     air mass down
--- mv:     air mass up
--- ms:     azimuth down
--- mv:     azimuth up
+++ Return: cosine of backscattering angle
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float backscatter(float ms, float mv, float sazi, float vazi){
float cospsi;

  cospsi = -1.0*mv*ms - sqrt((1-mv*mv)*(1-ms*ms)) * cos(vazi-sazi);

  return cospsi;
}


/** Compute molecular phase function
--- cospsi: cosine of backscattering angle
+++ Return: molecular phase function value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float phase_molecular(float cospsi){
float Pr;

  Pr = 0.75*(1.0+cospsi*cospsi);

  return Pr;
}


/** Compute aerosol phase function
--- cospsi: cosine of backscattering angle



+++ Return: aerosol phase function value
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float phase_aerosol(float cospsi, float *hg){
float Pa;
  
  Pa = hg[0]/pow(hg[2]-hg[4]*cospsi, 1.5) +
       hg[1]/pow(hg[3]+hg[5]*cospsi, 1.5);

  return Pa;
}


/** Compute water vapor transmittance
+++ This function computes the one-way water vapor transmittance based on
+++ the bands' relative spectral response and absorption coefficients
--- w:      total path water vapor in cm
--- m:      air mass one path
--- b_rsr:  ID in relative spectral response array
+++ Return: water vapor transmittance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float wvp_transmitt(float w, float m, int b_rsr){
int wvl;
float tmp, tmp2, Tw;
double a, s;

  tmp = w/m;

  for (wvl=0, a=0, s=0; wvl<_WVL_DIM_; wvl++){
    tmp2 = _AW_[wvl]*tmp;
    a += _RSR_[b_rsr][wvl]*
            exp((-1.2110662*tmp2)/pow(1+24.1229127*tmp2, 0.3669996));
    s += _RSR_[b_rsr][wvl];
  }
  Tw = a/s;

  return Tw;
}


/** Compute ozone transmittance
+++ This function computes the one-way ozone transmittance based on
+++ the bands' relative spectral response and absorption coefficients
--- o:      total ozone
--- m:      air mass one path
--- b_rsr:  ID in relative spectral response array
+++ Return: water vapor transmittance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float ozone_transmitt(float o, float m, int b_rsr){
int wvl;
float tmp, To;
double a, s;

  tmp = -o/m;

  for (wvl=0, a=0, s=0; wvl<_WVL_DIM_; wvl++){
    a += _RSR_[b_rsr][wvl]*exp(_AO_[wvl]*tmp);
    s += _RSR_[b_rsr][wvl];
  }
  To = a/s;

  return To;
}


/** Compute gas transmittance
+++ This function computes the total gas transmittance. Currently, only
+++ water vapor & ozone are considered.
--- Tsw:    water vapor transmittance down
--- Tvw:    water vapor transmittance up
--- Tso:    ozone transmittance down
--- Tvo:    ozone transmittance up
+++ Return: total gas transmittance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float gas_transmitt(float Tsw, float Tvw, float Tso, float Tvo){
  
  return Tsw*Tvw*Tso*Tvo;
}


/** Compute fresnel reflection
+++ This function computes the fresnel reflection, assuming the surface is
+++ flat. Kay et al. 2009, Remote Sensing.
--- i:      incidence angle
+++ Return: fresnel reflection
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float fresnel_reflection(float i){
float iphase, rhof;

  iphase = asin(sin(i)/1.3333);
  rhof = 0.5 * (
        (sin(i-iphase)/sin(i+iphase))*(sin(i-iphase)/sin(i+iphase)) +
        (tan(i-iphase)/tan(i+iphase))*(tan(i-iphase)/tan(i+iphase)));

  return rhof;
}


/** Compute cosine of illumination angle
--- csz:    cosine of sun zenith
--- ssz:    sine   of sun zenith
--- ctz:    cosine of terrain slope
--- stz:    sine   of terrain slope
--- sa:     sun azimuth
--- ta:     terrain aspect
+++ Return: cosine of illumination angle
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float illumin(float csz, float ssz, float ctz, float stz, float sa, float ta){
float ci;

  ci = csz*ctz + ssz*stz*cos(sa-ta);

  return ci;
}


/** Compute C-factor, empirical method
+++ This function computes the C-factor used for topographic correction.
+++ C is computd based on the regression coefficients between the cosine
+++ of the illumination angle and the TOA reflectance.
--- b:      intercept
--- m:      slope
+++ Return: C-factor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float c_factor_emp(double b, double m){
float c;

  if (b < 0 || m <= 0){
    c = 0.0;
  } else {
    c = (float)(b/m);
  }

  return c;
}


/** Compute C-factor, theoretical method
+++ This function computes the C-factor used for topographic correction.
+++ C is computd based on the proportion of diffuse and direct irradiance,
+++ the sky view factor @ cos i = 0, and the downwelling air mass. 
+++-----------------------------------------------------------------------
+++ Kobayashi, S. & Sanga-Ngoie, K. (2008): The integrated radiometric co-
+++ rection of optical remote sensing imageries, International Journal of 
+++ Remote Sensing, 29 (20). DOI: 10.1080/01431160701881889
+++-----------------------------------------------------------------------
--- h0:     sky view factor @ cos i = 0
--- f:      f-factor
--- ms:     air mass down
+++ Return: C-factor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float c_factor_com(float h0, float f, float ms){
float c;

  c = h0*f*ms;

  return c;
}


/** Compute f-factor
+++ This function computes the f-factor used for computing C, and for pro-
+++ pagating empirically derived C-factors through the spectrum. f is the
+++ proportion of diffuse and direct irradiance. 
+++-----------------------------------------------------------------------
+++ Kobayashi, S. & Sanga-Ngoie, K. (2008): The integrated radiometric co-
+++ rection of optical remote sensing imageries, International Journal of 
+++ Remote Sensing, 29 (20). DOI: 10.1080/01431160701881889
+++-----------------------------------------------------------------------
--- tss:    diffuse scattering transmittance down
--- tsd:    direct  scattering transmittance down
+++ Return: f-factor
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float f_factor(float tss, float tsd){
float f;

  f = tss/tsd;

  return f;
}

