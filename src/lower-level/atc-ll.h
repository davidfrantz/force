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
Atm. correction parameter header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef ATC_LL_H
#define ATC_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick-cl.h"
#include "../lower-level/param-ll.h"
#include "../lower-level/meta-ll.h"


#ifdef __cplusplus
extern "C" {
#endif

// viewing geometry
typedef struct {
  float a, b, c, ab;     // cartesian definition of nadir line
  float geo_angle_nadir; // angle of nadir line
  float H, H2;           // satellite elevation, and square of elevation
} view_t;

// elevation stats/bins
typedef struct {
  float avg, min, max; // average/min/max elevation
  float step;
  //float cmin, cmax;    // min/max of binned elevation
  int cnum;            // number of elevation bins
} dem_t;

// two-term Henyey-Greenstein coefficicents
typedef struct {
  float g1, g2, alpha;
  float hg[6], sob;
} tthg_t;

// atmospheric parameters for image
typedef struct {
  int nx, ny, nc, res;   // number and resolution of coarse grid cells
  float nodata;
  view_t view;               // viewing geometry
  float *wvl, *lwvl, *lwvl2; // wavelengths, log of wvl, square of log of wvl
  float *E0;                 // solar irradiance
  float cc;                  // cloud cover
  dem_t dem;                 // elevation stats/bins
  float cosszen[2];          // min/max cos. of sun zenith
  float cosvzen[2];          // min/max cos. of view zenith
  float *od;                 // total optical depth
  float *mod;                // molecular optical depth
  float *aod;                // aerosol optical depth
  float Hr, Ha;              // elevation correction factors for AOD/MOD
  float Hp;                  // scale height for aerosols
  tthg_t tthg;               // two-term Henyey-Greenstein coefficicents
  bool *aod_bands;           // flag: use band to estimate AOD
  bool aodmap;               // flag: AOD map successful
  float Fr, Fa, km;          // environmental reflectance weighting function
  float wvp;                 // water vapor concentration
  float ***Tw_lut;           // water vapor transmittance LUT

  brick_t *xy_mod;     // molecular optical depth
  brick_t *xy_aod;     // aerosol optical depth
  brick_t *xy_Pr;      // phase function, molecular scatt.
  brick_t *xy_Pa;      // phase function, aerosol scatt.
  brick_t *xy_Hr;      // elevation correction factors for MOD
  brick_t *xy_Ha;      // elevation correction factors for AOD
  brick_t *xy_Tvw;     // water vapor transmittance up
  brick_t *xy_Tsw;     // water vapor transmittance down
  brick_t *xy_Tvo;     // ozone transmittance up 
  brick_t *xy_Tso;     // ozone transmittance down
  brick_t *xy_Tg;      // gaseous transmittance
  brick_t *xy_brdf;    // BRDF correction factor
  brick_t *xy_fresnel; // Fresnel reflection
  brick_t *xy_dem;     // elevation bin
  brick_t *xy_interp;  // aod interpolation flag
  brick_t *xy_view;    // view angles
  brick_t *xy_sun;     // sun angles
  brick_t *xy_psi;     // backscattering angle

//  brick_t xy_mv;      // air mass, up & down --> atc.cg[g].mv = atc.cg[g].cosvzen;
//  brick_t xy_ms;      // air mass, up & down --> atc.cg[g].ms = atc.cg[g].cosszen;

// wvp and ozone, both not in cg anymore, doesn*t make sense
  
  brick_t **xyz_od;    // total optical depth
  brick_t **xyz_mod;   // molecular optical depth
  brick_t **xyz_aod;   // aerosol optical depth
  brick_t **xyz_Hr;    // elevation correction factors for MOD
  brick_t **xyz_Ha;    // elevation correction factors for AOD
  brick_t **xyz_rho_p; // path reflectance
  brick_t **xyz_Ts;    // scattering transmittance total, down
  brick_t **xyz_tsd;   // scattering transmittance direct, down
  brick_t **xyz_tss;   // scattering transmittance diffuse, down
  brick_t **xyz_Tv;    // scattering transmittance total, up
  brick_t **xyz_tvd;   // scattering transmittance direct, up
  brick_t **xyz_tvs;   // scattering transmittance diffuse, up
  brick_t **xyz_T;     // total scattering transmittance
  brick_t **xyz_s;     // spherical albedo
  brick_t **xyz_F;     // environmental reflectance weighting function
  brick_t **xyz_Tg;    // gaseous transmittance

} atc_t;


atc_t *allocate_atc(par_ll_t *pl2, meta_t *meta, brick_t *DN);
void free_atc(atc_t *atc);
float **atc_get_band_reshaped(brick_t **xyz, int b);

#ifdef __cplusplus
}
#endif

#endif

