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
Sun position header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef SUN_CL_H
#define SUN_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <math.h>    // common mathematical functions

#include "../cross-level/const-cl.h"
#include "../cross-level/date-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

float doy2dsun(int doy);
float sunJC(int year, int month, int day, float timenow);
float sunGeomMeanLong(float t);
float sunGeomMeanAnomaly(float t);
float sunEccentricityEarthOrbit(float t);
float sunEqOfCenter(float t);
float sunTrueLong(float t);
float sunTrueAnomaly(float t);
float sunRadVector(float t);
float sunApparentLong(float t);
float sunMeanObliquityOfEcliptic(float t);
float sunObliquityCorrection(float t);
float sunRtAscension(float t);
float sunDeclination(float t);
float sunEquationOfTime(float t);
float sunHourAngle(float time, float longitude, float eqtime);
void sunpos(float latitude, float longitude, date_t date, float *zen, float *azi);

#ifdef __cplusplus
}
#endif

#endif

