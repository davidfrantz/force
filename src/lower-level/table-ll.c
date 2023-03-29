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
Tables for radiometric processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "table-ll.h"


/** This function computes the main wavelength based on the bands' 
+++ relative spectral response.
--- b_rsr:  ID in relative spectral response array
+++ Return: wavelength
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float wavelength(int b_rsr){
int w;
float wvl;
double v = 0, s = 0;


  for (w=0; w<_WVL_DIM_; w++){
    v += _RSR_[b_rsr][w]*_WVL_[w];
    s += _RSR_[b_rsr][w];
  }

  if (s > 0) wvl = (float)(v/s); else wvl = 0;

  return wvl;
}



/** This function computes Exoatmospheric solar irradianceE0 values based
+++ on the bands' relative spectral response and E0 spectrum.
--- b_rsr:  ID in relative spectral response array
+++ Return: E0
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float E0(int b_rsr){
int w;
float E0;
double e = 0, s = 0;


  for (w=0; w<_WVL_DIM_; w++){
    e += _RSR_[b_rsr][w]*_E0_[w];
    s += _RSR_[b_rsr][w];
  }

  if (s > 0) E0 = (float)(e/s); else E0 = 0;

  return E0;
}


/** Exoatmospheric irradiance
+++ Thuillier spectrum @1nm [410-2400] in W/m^2/µm
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** Water vapor absorption coefficients
+++ Water vapor absorption from HITRAN 2016 @1nm [240-2400] in 1/cm
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** Ozone absorption coefficients
+++ Bird & Riordan 1986 @1nm [410-2400] (interpolated to match other tables)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/




/** Water spectral library
+++ Spectra @1nm [410-900] obtained from WASI (Gege 2004)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
