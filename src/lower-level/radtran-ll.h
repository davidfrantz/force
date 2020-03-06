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
Radiative transfer computations header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef RADTRAN_LL_H
#define RADTRAN_LL_H

#include <stdbool.h> // boolean data type
#include <math.h>    // common mathematical functions

#include "../lower-level/table-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

float molecular_optical_depth(float wvl);
float aod_elev_factor(float z, float Hp);
float mod_elev_factor(float z);
float mod_elev_scale(float mod, float Hr_is, float Hr);
float aod_elev_scale(float aod, float Ha_is, float Ha);
float optical_depth(float aod, float mod);
float scatt_transmitt(float aod, float mod, float od, float ms, float mv, float *Ts, float *Tv, float *tsd, float *tss, float *tvd, float *tvs);
float path_ref(bool multi, float sob, float aod, float mod, float od, float Pa, float Pr, float tsd, float tvd, float ms, float mv);
float sphere_albedo(float aod, float mod, float od);
float env_weight_aerosol(float r);
float env_weight_molecular(float r);
float env_weight(float aod, float mod, float Fa, float Fr);
float backscatter(float ms, float mv, float sazi, float vazi);
float phase_molecular(float cospsi);
float phase_aerosol(float cospsi, float *hg);
float wvp_transmitt(float w, float m, int b_rsr);
float ozone_transmitt(float o, float m, int b_rsr);
float gas_transmitt(float Tsw, float Tvw, float Tso, float Tvo);
float fresnel_reflection(float i);
float illumin(float csz, float ssz, float ctz, float stz, float sa, float ta);
float c_factor_emp(double b, double m);
float c_factor_com(float h0, float f, float ms);
float f_factor(float tss, float tsd);

#ifdef __cplusplus
}
#endif

#endif

