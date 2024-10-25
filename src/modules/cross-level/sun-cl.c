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

/** The solar position calculations were adpoted from the National Oceanic
+++ & Atmospheric Administration Earth System Research Lab Solar Position 
+++ Calculator, online available at:
+++ https://www.esrl.noaa.gov/gmd/grad/solcalc/azel.html
+++ Solar Position Calculator Copyright (C) 2017 Chris Cornwall, Aaron 
+++ Horiuchi, and Chris Lehman
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions for calculating sun positions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "sun-cl.h"


/** Earth-Sun distance
+++ This function computes dsun in astronomical units Spencer, 1971.
--- doy:    Day-of-Year
+++ Return: Earth-Sun distance
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float doy2dsun(int doy){
float day_angle, dsun;

  if (doy > 365) doy = 365;

  day_angle = 2*M_PI*(doy-1)/365;
  dsun = 1/sqrt(1.00011 + 0.034221*cos(day_angle) + 
                          0.001280*sin(day_angle) + 
                          0.000719*cos(2*day_angle) + 
                          0.000077*sin(2*day_angle));

  return dsun;
}


/** Calendar day to Julian centuries since J2000.0
--- year:    Year
--- month:   Month
--- day:     Day
--- timenow: Time in hours
+++ Return:  Julian centuries since J2000.0
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunJC(int year, int month, int day, float timenow){
float A, B, JD, JC;

  if (month <= 2){ year -= 1; month += 12; }
  A = floor(year/100.0);
  B = 2 - A + floor(A/4.0);

  JD = floor(365.25*(year + 4716)) + floor(30.6001*(month+1)) + day + B - 1524.5;
  JD += timenow/24.0;
  JC = (JD - 2451545.0)/36525.0;

  return JC;
}


/** Geometric mean longitude of the sun
--- t:      Time
+++ Return: Geometric mean longitude of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunGeomMeanLong(float t){
float L0;

  L0 = 280.46646 + t * (36000.76983 + 0.0003032 * t);
  while(L0 > 360.0) L0 -= 360.0;
  while(L0 < 0.0)   L0 += 360.0;

  return L0;
}


/** Geometric mean anomaly of the sun
--- t:      Time
+++ Return: Geometric mean anomaly of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunGeomMeanAnomaly(float t){
float M;

  M = 357.52911 + t * (35999.05029 - 0.0001537 * t);

  return M;
}


/** Eccentricity of earth's orbit
--- t:      Time
+++ Return: Eccentricity of earth's orbit
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunEccentricityEarthOrbit(float t){
float e;

  e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);

  return e;
}


/** Equation of center for the sun
--- t:      Time
+++ Return: Equation of center for the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunEqOfCenter(float t){
float m, mrad, sinm[3], C;

  m = sunGeomMeanAnomaly(t);
  mrad = m*_D2R_CONV_;
  sinm[0] = sin(mrad);
  sinm[1] = sin(mrad+mrad);
  sinm[2] = sin(mrad+mrad+mrad);
  C = sinm[0] * (1.914602 - t * (0.004817 + 0.000014 * t)) + sinm[1] * (0.019993 - 0.000101 * t) + sinm[2] * 0.000289;

  return C;
}


/** True longitude of the sun
--- t:      Time
+++ Return: True longitude of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunTrueLong(float t){
float l0, c, O;

  l0 = sunGeomMeanLong(t);
  c = sunEqOfCenter(t);
  O = l0 + c;

  return O;
}


/** True anamoly of the sun
--- t:      Time
+++ Return: anamoly of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunTrueAnomaly(float t){
float m, c, v;

  m = sunGeomMeanAnomaly(t);
  c = sunEqOfCenter(t);
  v = m + c;

  return v;
}


/** Distance to the sun in astronomical units
--- t:      Time
+++ Return: Distance to the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunRadVector(float t){
float v, e, R;

  v = sunTrueAnomaly(t);
  e = sunEccentricityEarthOrbit(t);
  R = (1.000001018 * (1 - e * e)) / (1 + e * cos(v*_D2R_CONV_));

  return R;
}


/** Apparent longitude of the sun
--- t:      Time
+++ Return: Apparent longitude of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunApparentLong(float t){
float o, omega, lambda;

  o = sunTrueLong(t);
  omega = 125.04 - 1934.136 * t;
  lambda = o - 0.00569 - 0.00478 * sin(omega*_D2R_CONV_);

  return lambda;
}


/** Mean obliquity of the ecliptic
--- t:      Time
+++ Return: Mean obliquity of the ecliptic
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunMeanObliquityOfEcliptic(float t){
float seconds, e0;

  seconds = 21.448 - t*(46.8150 + t*(0.00059 - t*(0.001813)));
  e0 = 23.0 + (26.0 + (seconds/60.0))/60.0;

  return e0;
}


/** Corrected obliquity of the ecliptic
--- t:      Time
+++ Return: Corrected obliquity of the ecliptic
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunObliquityCorrection(float t){
float e0, omega, e;

  e0 = sunMeanObliquityOfEcliptic(t);
  omega = 125.04 - 1934.136 * t;
  e = e0 + 0.00256 * cos(omega*_D2R_CONV_);

  return e;
}


/** Right ascension of the sun
--- t:      Time
+++ Return: Right ascension of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunRtAscension(float t){
float e, lambda, tananum, tanadenom, alpha;

  e = sunObliquityCorrection(t);
  lambda = sunApparentLong(t);
  tananum = cos(e*_D2R_CONV_) * sin(lambda*_D2R_CONV_);
  tanadenom = cos(lambda*_D2R_CONV_);
  alpha = atan2(tananum, tanadenom)*_R2D_CONV_;

  return alpha;
}


/** Declination of the sun
--- t:      Time
+++ Return: Declination of the sun
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunDeclination(float t){
float e, lambda, sint, theta;

  e = sunObliquityCorrection(t);
  lambda = sunApparentLong(t);
  sint = sin(e*_D2R_CONV_) * sin(lambda*_D2R_CONV_);
  theta = asin(sint)*_R2D_CONV_;

  return theta;
}


/** Difference between true solar time and mean
--- t:      Time
+++ Return: Difference between true solar time and mean
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunEquationOfTime(float t){
float epsilon, l0, e, m, y, sin2l0, cos2l0, sin4l0, sinm, sin2m, Etime;

  epsilon = sunObliquityCorrection(t);
  l0 = sunGeomMeanLong(t);
  e = sunEccentricityEarthOrbit(t);
  m = sunGeomMeanAnomaly(t);

  y = tan((epsilon*_D2R_CONV_)/2.0);
  y *= y;

  sin2l0 = sin(2.0 * l0*_D2R_CONV_);
  sinm   = sin(m*_D2R_CONV_);
  cos2l0 = cos(2.0 * l0*_D2R_CONV_);
  sin4l0 = sin(4.0 * l0*_D2R_CONV_);
  sin2m  = sin(2.0 * m*_D2R_CONV_);

  Etime = y * sin2l0 - 2.0 * e * sinm + 4.0 * e * y * sinm * cos2l0
         - 0.5 * y * y * sin4l0 - 1.25 * e * e * sin2m;

  return Etime*_R2D_CONV_*4.0;
}


/** Hour angle
--- time:      Time
--- longitude: Longitude
--- eqtime:    Equation of time
+++ Return:    Hour angle
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float sunHourAngle(float time, float longitude, float eqtime){
  return (15.0*(time - (longitude/15.0) - (eqtime/60.0)));
}


/** Sun zenith and azimuth
+++ This function calculates the sun zenith and azimuth in degrees.
--- latitude:  Latitude
--- longitude: Longitude
--- date:      Date
--- zen:       Sun zenith  (returned)
--- azi:       Sun azimuth (returned)
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void sunpos(float latitude, float longitude, date_t date, float *zen, float *azi){
float lat[4], lon, zone;
float timenow[2], JC;
float solarDec[4], eqTime;
float solarTimeFix, trueSolarTime, hourAngle;
float csz, azDenom, zenith[3], azimuth, azRad;
float exoelev, refrac, te;


  lat[0] = latitude;
  lon    = longitude;
  zone   = date.tz;

  timenow[0] = date.hh + date.mm/60.0 + date.ss/3600.0 - zone;  // in hours since 0Z := GMT
  timenow[1] = date.hh * 60.0 + date.mm + date.ss/60.0; // in minutes in timezone
  zone *= 60;
  JC = sunJC(date.year, date.month, date.day, timenow[0]);
  solarDec[0] = sunDeclination(JC);
  solarDec[1] = solarDec[0]*_D2R_CONV_;
  solarDec[2] = sin(solarDec[1]);
  solarDec[3] = cos(solarDec[1]);
  eqTime = sunEquationOfTime(JC);

  lat[1] = lat[0]*_D2R_CONV_;
  lat[2] = sin(lat[1]);
  lat[3] = cos(lat[1]);

  solarTimeFix = eqTime + 4.0 * lon - zone;
  trueSolarTime = timenow[1] + solarTimeFix;
  while (trueSolarTime > 1440) trueSolarTime -= 1440;

  hourAngle = trueSolarTime / 4.0 - 180.0;
  if (hourAngle < -180) hourAngle += 360.0;

  csz = lat[2]*solarDec[2] + 
    lat[3]*solarDec[3]*cos(hourAngle*_D2R_CONV_);
  if (csz > 1.0) csz = 1.0; else if (csz < -1.0) csz = -1.0; 
  zenith[0] = acos(csz);
  zenith[1] = zenith[0]*_R2D_CONV_;

  azDenom = lat[3]*sin(zenith[0]);
  if (fabs(azDenom) > 0.001){

    azRad = ( lat[2]*cos(zenith[0]) - solarDec[2] ) / azDenom;
    if (fabs(azRad) > 1.0){
      if (azRad < 0) azRad = -1.0; else azRad = 1.0;
    }

    azimuth = 180.0 - acos(azRad)*_R2D_CONV_;
    if (hourAngle > 0.0) azimuth = -1.0*azimuth;

  } else {
    if (lat[0] > 0.0) azimuth = 180.0; else azimuth = 0.0;
  }

  if (azimuth < 0.0) azimuth += 360.0;

  // refraction correction
  exoelev = 90.0 - zenith[1];
  if (exoelev > 85.0){
    refrac = 0.0;
  } else {
    te = tan(exoelev*_D2R_CONV_);
    if (exoelev > 5.0){
      refrac = 58.1 / te - 0.07 / (te*te*te) +
        0.000086 / (te*te*te*te*te);
    } else if (exoelev > -0.575){
      refrac = 1735.0  + exoelev *
          (-518.2  + exoelev * 
           (103.4  + exoelev * 
           (-12.79 + exoelev*0.711)));
    } else {
      refrac = -20.774 / te;
    }
    refrac = refrac / 3600.0;
  }
  zenith[2] = zenith[1] - refrac;

  *zen = zenith[2];
  *azi = azimuth;

  return;
}

