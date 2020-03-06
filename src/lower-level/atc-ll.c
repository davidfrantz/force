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
This file contains functions for handling atmosph. correction parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "atc-ll.h"


void init_atc(meta_t *meta, stack_t *DN, atc_t *atc);
void init_atc_xy(par_ll_t *pl2, stack_t *DN, atc_t *atc);
void init_atc_xyz(atc_t *atc);
void free_atc_xy(atc_t *atc);
void free_atc_xyz(atc_t *atc);


/** This function initializes parameters used for atmospheric correction.
+++ This includes precomputing TTHG parameters, environmental weighting,
+++ absorption coefficients, wavelengths and allocating memory.
--- meta:   metadata
--- DN:     Digital Numbers
--- atc:    atmospheric correction factors
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_atc(meta_t *meta, stack_t *DN, atc_t *atc){;
int r = 1, b, nb = get_stack_nbands(DN);


  /** nodata value **/
  atc->nodata = -9999;

    /** initialize Henyey-Greenstein parameters for faster computation
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  atc->tthg.g1 = 0.801; atc->tthg.g2 = 0.580; atc->tthg.alpha = 0.948;
  atc->tthg.hg[0] = (1.0-atc->tthg.g1*atc->tthg.g1)*atc->tthg.alpha;
  atc->tthg.hg[1] = (1.0-atc->tthg.g2*atc->tthg.g2)*(1.0-atc->tthg.alpha);
  atc->tthg.hg[2] = 1.0+atc->tthg.g1*atc->tthg.g1; 
  atc->tthg.hg[3] = 1.0+atc->tthg.g2*atc->tthg.g2;
  atc->tthg.hg[4] = 2.0*atc->tthg.g1; 
  atc->tthg.hg[5] = 2.0*atc->tthg.g2;
  atc->tthg.sob = 3.0*(atc->tthg.alpha*(atc->tthg.g1+atc->tthg.g2)-atc->tthg.g2);
 

  /** initialize environmental weighting functions for MOD/AOD
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  atc->km = r*(get_stack_res(DN)/1000.0) + 0.5*get_stack_res(DN)/1000.0;
  atc->Fr = 1.0-(0.930*exp(-0.8*atc->km)+0.070*exp(-1.10*atc->km));
  atc->Fa = 1.0-(0.375*exp(-0.2*atc->km)+0.625*exp(-1.83*atc->km));


  /** allocate band-dependent variables
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  alloc((void**)&atc->wvl,       nb, sizeof(float)); 
  alloc((void**)&atc->lwvl,      nb, sizeof(float));
  alloc((void**)&atc->lwvl2,     nb, sizeof(float));
  alloc((void**)&atc->E0,        nb, sizeof(float));
  alloc((void**)&atc->od,        nb, sizeof(float));
  alloc((void**)&atc->mod,       nb, sizeof(float));
  alloc((void**)&atc->aod,       nb, sizeof(float));
  alloc((void**)&atc->aod_bands, nb, sizeof(bool));


  /** set nodata or other init value
  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

  atc->Tw_lut = NULL;

  atc->xy_mod    = NULL;
  atc->xy_aod    = NULL;
  atc->xy_Pr     = NULL;
  atc->xy_Pa     = NULL;
  atc->xy_Hr     = NULL;
  atc->xy_Ha     = NULL;
  atc->xy_Tvw    = NULL;
  atc->xy_Tsw    = NULL;
  atc->xy_Tvo    = NULL;
  atc->xy_Tso    = NULL;
  atc->xy_Tg     = NULL;
  atc->xy_brdf   = NULL;
  atc->xy_fresnel= NULL;
  atc->xy_dem    = NULL;
  atc->xy_interp = NULL;
  atc->xy_view   = NULL;
  atc->xy_sun    = NULL;
  atc->xy_psi    = NULL;

  atc->xyz_od    = NULL;
  atc->xyz_mod   = NULL;
  atc->xyz_aod   = NULL;
  atc->xyz_Hr    = NULL;
  atc->xyz_Ha    = NULL;
  atc->xyz_rho_p = NULL;
  atc->xyz_Ts    = NULL;
  atc->xyz_tsd   = NULL;
  atc->xyz_tss   = NULL;
  atc->xyz_Tv    = NULL;
  atc->xyz_tvd   = NULL;
  atc->xyz_tvs   = NULL;
  atc->xyz_T     = NULL;
  atc->xyz_s     = NULL;
  atc->xyz_F     = NULL;
  atc->xyz_Tg    = NULL;

  atc->aodmap = false;

  atc->nx = atc->ny = atc->nc = atc->res = (int)atc->nodata;
  atc->cc = atc->wvp = atc->nodata;

  atc->cosszen[0] = atc->cosszen[1] = atc->nodata;
  atc->cosvzen[0] = atc->cosvzen[1] = atc->nodata;
  atc->Hr = atc->Ha = atc->Hp = atc->nodata;

  atc->dem.avg = atc->dem.min =  atc->nodata;
  atc->dem.max = atc->dem.step = atc->nodata;
  atc->dem.cnum = (int)atc->nodata;
  
  atc->view.a = atc->view.b = atc->view.c = atc->view.ab = atc->nodata;
  atc->view.geo_angle_nadir = atc->nodata;
  atc->view.H = atc->view.H2 = atc->nodata;
  
  for (b=0; b<nb; b++){

    atc->od[b] = atc->mod[b] = atc->aod[b] = atc->nodata;
    atc->aod_bands[b] = false;
    
    atc->wvl[b]   = get_stack_wavelength(DN, b);
    atc->lwvl[b]  = log(atc->wvl[b]);
    atc->lwvl2[b] = atc->lwvl[b]*atc->lwvl[b];
    
    atc->E0[b] = E0(meta->cal[b].rsr_band);

  }


  return;
};


/** This function initializes xy parameters used for atmospheric 
+++ correction.
--- pl2:    L2 parameters
--- DN:     Digital Numbers
--- atc:    atmospheric correction factors
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_atc_xy(par_ll_t *pl2, stack_t *DN, atc_t *atc){
int b, nb = get_stack_nbands(DN);
int svgrid = 5000;


  // resolution and number of cells in x- and y-direction
  atc->res = svgrid;
  atc->nx  = ceil(get_stack_width(DN)/(float)atc->res);
  atc->ny  = ceil(get_stack_height(DN)/(float)atc->res);
  atc->nc  = atc->nx*atc->ny;

  #ifdef FORCE_DEBUG
  printf("\nnumber of coarse grid points: %d, ngx: %d, ngy: %d, res: %d\n", atc->nc, atc->nx, atc->ny, atc->res);
  #endif

  if (atc->nc > USHRT_MAX){
    printf("too many coarse grid cells.\n"); exit(1);}

  // copy the DN stack, do not allocate images (need to adjust nx/ny first)
  atc->xy_mod = copy_stack(DN, nb, _DT_NONE_);

  // set metadata
  set_stack_name(atc->xy_mod, "FORCE ATC stack");
  set_stack_product(atc->xy_mod, "ATC");
  set_stack_filename(atc->xy_mod, "ATC-MOD");
  set_stack_ncols(atc->xy_mod, atc->nx);
  set_stack_nrows(atc->xy_mod, atc->ny);
  set_stack_res(atc->xy_mod, atc->res);
  for (b=0; b<nb; b++) set_stack_nodata(atc->xy_mod, b, atc->nodata);
  allocate_stack_bands(atc->xy_mod, nb, atc->nc, _DT_FLOAT_);

  // other stacks can simply be copied with allocation
  atc->xy_aod = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_aod, "ATC-AOD");
  atc->xy_Tvw = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_Tvw, "ATC-Tvw");
  atc->xy_Tsw = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_Tsw, "ATC-Tsw");
  atc->xy_Tvo = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_Tvo, "ATC-Tvo");
  atc->xy_Tso = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_Tso, "ATC-Tso");
  atc->xy_Tg = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_Tg, "ATC-Tg");
  atc->xy_brdf = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
  set_stack_filename(atc->xy_brdf, "ATC-BRDF");

  atc->xy_fresnel = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
  set_stack_filename(atc->xy_fresnel, "ATC-FRESNEL");

  atc->xy_Pr = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
  set_stack_filename(atc->xy_Pr, "ATC-PHASE-MOD");
  atc->xy_Pa = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
  set_stack_filename(atc->xy_Pa, "ATC-PHASE-AOD");
  atc->xy_Hr = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
  set_stack_filename(atc->xy_Hr, "ATC-ZCOR-MOD");
  atc->xy_Ha = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
  set_stack_filename(atc->xy_Ha, "ATC-ZCOR-AOD");

  atc->xy_interp = copy_stack(atc->xy_mod, 1, _DT_SMALL_);
  set_stack_filename(atc->xy_interp, "ATC-INTERPOL-AOD");
  
  atc->xy_dem = copy_stack(atc->xy_mod, 1, _DT_INT_);
  set_stack_filename(atc->xy_dem, "ATC-DEM-BINNED");

  atc->xy_view = copy_stack(atc->xy_mod, 8, _DT_FLOAT_);
  set_stack_filename(atc->xy_view, "ATC-VIEW");
  atc->xy_sun = copy_stack(atc->xy_mod, 8, _DT_FLOAT_);
  set_stack_filename(atc->xy_sun, "ATC-SUN");
  atc->xy_psi = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
  set_stack_filename(atc->xy_psi, "ATC-PSI");


  return;
}


/** This function initializes xyz parameters used for atmospheric 
+++ correction.
--- atc:    atmospheric correction factors
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_atc_xyz(atc_t *atc){
int nb = get_stack_nbands(atc->xy_mod);
int z, nz = NPOW_08;
char fname[NPOW_10];
int nchar;


  alloc((void**)&atc->xyz_od,    nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_mod,   nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_aod,   nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_Hr,    nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_Ha,    nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_rho_p, nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_Ts,    nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_tsd,   nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_tss,   nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_Tv,    nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_tvd,   nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_tvs,   nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_T,     nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_s,     nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_F,     nz, sizeof(stack_t*));
  alloc((void**)&atc->xyz_Tg,    nz, sizeof(stack_t*));

  for (z=0; z<NPOW_08; z++){

    nchar = snprintf(fname, NPOW_10, "ATC-OD-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_od[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_od[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-MOD-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_mod[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_mod[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-AOD-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_aod[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_aod[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-ZCOR-MOD-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_Hr[z] = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
    set_stack_filename(atc->xyz_Hr[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-ZCOR-AOD-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_Ha[z] = copy_stack(atc->xy_mod, 1, _DT_FLOAT_);
    set_stack_filename(atc->xyz_Ha[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-PATH-REF-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_rho_p[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_rho_p[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-Ts-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_Ts[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_Ts[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-tsd-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_tsd[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_tsd[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-tss-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_tss[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_tss[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-Tv-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_Tv[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_Tv[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-tvd-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_tvd[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_tvd[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-tvs-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_tvs[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_tvs[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-T-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_T[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_T[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-s-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_s[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_s[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-F-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_F[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_F[z], fname);

    nchar = snprintf(fname, NPOW_10, "ATC-Tg-Z%03d", z);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    atc->xyz_Tg[z] = copy_stack(atc->xy_mod, nb, _DT_FLOAT_);
    set_stack_filename(atc->xyz_Tg[z], fname);

  }


  return;
}


/** This function deallocates xy atmospheric parameters for coarse grid 
+++ cells
--- atc:    atmospheric correction factors
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_atc_xy(atc_t *atc){

  if (atc->xy_mod     != NULL){ free_stack(atc->xy_mod);     atc->xy_mod     = NULL;}
  if (atc->xy_aod     != NULL){ free_stack(atc->xy_aod);     atc->xy_aod     = NULL;}
  if (atc->xy_Pr      != NULL){ free_stack(atc->xy_Pr);      atc->xy_Pr      = NULL;}
  if (atc->xy_Pa      != NULL){ free_stack(atc->xy_Pa);      atc->xy_Pa      = NULL;}
  if (atc->xy_Hr      != NULL){ free_stack(atc->xy_Hr);      atc->xy_Hr      = NULL;}
  if (atc->xy_Ha      != NULL){ free_stack(atc->xy_Ha);      atc->xy_Ha      = NULL;}
  if (atc->xy_Tvw     != NULL){ free_stack(atc->xy_Tvw);     atc->xy_Tvw     = NULL;}
  if (atc->xy_Tsw     != NULL){ free_stack(atc->xy_Tsw);     atc->xy_Tsw     = NULL;}
  if (atc->xy_Tvo     != NULL){ free_stack(atc->xy_Tvo);     atc->xy_Tvo     = NULL;}
  if (atc->xy_Tso     != NULL){ free_stack(atc->xy_Tso);     atc->xy_Tso     = NULL;}
  if (atc->xy_Tg      != NULL){ free_stack(atc->xy_Tg);      atc->xy_Tg      = NULL;}
  if (atc->xy_brdf    != NULL){ free_stack(atc->xy_brdf);    atc->xy_brdf    = NULL;}
  if (atc->xy_fresnel != NULL){ free_stack(atc->xy_fresnel); atc->xy_fresnel = NULL;}
  if (atc->xy_dem     != NULL){ free_stack(atc->xy_dem);     atc->xy_dem     = NULL;}
  if (atc->xy_interp  != NULL){ free_stack(atc->xy_interp);  atc->xy_interp  = NULL;}
  if (atc->xy_view    != NULL){ free_stack(atc->xy_view);    atc->xy_view    = NULL;}
  if (atc->xy_sun     != NULL){ free_stack(atc->xy_sun);     atc->xy_sun     = NULL;}
  if (atc->xy_psi     != NULL){ free_stack(atc->xy_psi);     atc->xy_psi     = NULL;}

  return;
}


/** This function deallocates xyz atmospheric parameters for coarse grid 
+++ cells
--- atc:    atmospheric correction factors
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_atc_xyz(atc_t *atc){
int z, nz = NPOW_08;

  if (atc->xyz_od != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_od[z] != NULL){ free_stack(atc->xyz_od[z]); atc->xyz_od[z] = NULL;}}
    free((void*)atc->xyz_od); atc->xyz_od = NULL;}

  if (atc->xyz_mod != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_mod[z] != NULL){ free_stack(atc->xyz_mod[z]); atc->xyz_mod[z] = NULL;}}
    free((void*)atc->xyz_mod); atc->xyz_mod = NULL;}

  if (atc->xyz_aod != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_aod[z] != NULL){ free_stack(atc->xyz_aod[z]); atc->xyz_aod[z] = NULL;}}
    free((void*)atc->xyz_aod); atc->xyz_aod = NULL;}

  if (atc->xyz_Hr != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_Hr[z] != NULL){ free_stack(atc->xyz_Hr[z]); atc->xyz_Hr[z] = NULL;}}
    free((void*)atc->xyz_Hr); atc->xyz_Hr = NULL;}

  if (atc->xyz_Ha != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_Ha[z] != NULL){ free_stack(atc->xyz_Ha[z]); atc->xyz_Ha[z] = NULL;}}
    free((void*)atc->xyz_Ha); atc->xyz_Ha = NULL;}

  if (atc->xyz_rho_p != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_rho_p[z] != NULL){ free_stack(atc->xyz_rho_p[z]); atc->xyz_rho_p[z] = NULL;}}
    free((void*)atc->xyz_rho_p); atc->xyz_rho_p = NULL;}

  if (atc->xyz_Ts != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_Ts[z] != NULL){ free_stack(atc->xyz_Ts[z]); atc->xyz_Ts[z] = NULL;}}
    free((void*)atc->xyz_Ts); atc->xyz_Ts = NULL;}

  if (atc->xyz_tsd != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_tsd[z] != NULL){ free_stack(atc->xyz_tsd[z]); atc->xyz_tsd[z] = NULL;}}
    free((void*)atc->xyz_tsd); atc->xyz_tsd = NULL;}

  if (atc->xyz_tss != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_tss[z] != NULL){ free_stack(atc->xyz_tss[z]); atc->xyz_tss[z] = NULL;}}
    free((void*)atc->xyz_tss); atc->xyz_tss = NULL;}

  if (atc->xyz_Tv != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_Tv[z] != NULL){ free_stack(atc->xyz_Tv[z]); atc->xyz_Tv[z] = NULL;}}
    free((void*)atc->xyz_Tv); atc->xyz_Tv = NULL;}

  if (atc->xyz_tvd != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_tvd[z] != NULL){ free_stack(atc->xyz_tvd[z]); atc->xyz_tvd[z] = NULL;}}
    free((void*)atc->xyz_tvd); atc->xyz_tvd = NULL;}

  if (atc->xyz_tvs != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_tvs[z] != NULL){ free_stack(atc->xyz_tvs[z]); atc->xyz_tvs[z] = NULL;}}
    free((void*)atc->xyz_tvs); atc->xyz_tvs = NULL;}

  if (atc->xyz_T != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_T[z] != NULL){ free_stack(atc->xyz_T[z]); atc->xyz_T[z] = NULL;}}
    free((void*)atc->xyz_T); atc->xyz_T = NULL;}

  if (atc->xyz_s != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_s[z] != NULL){ free_stack(atc->xyz_s[z]); atc->xyz_s[z] = NULL;}}
    free((void*)atc->xyz_s); atc->xyz_s = NULL;}

  if (atc->xyz_F != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_F[z] != NULL){ free_stack(atc->xyz_F[z]); atc->xyz_F[z] = NULL;}}
    free((void*)atc->xyz_F); atc->xyz_F = NULL;}

  if (atc->xyz_Tg != NULL){
    for (z=0; z<nz; z++){
      if (atc->xyz_Tg[z] != NULL){ free_stack(atc->xyz_Tg[z]); atc->xyz_Tg[z] = NULL;}}
    free((void*)atc->xyz_Tg); atc->xyz_Tg = NULL;}


  return;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function allocates the atmospheric parameters
+++ Return: atmospheric parameters (must be freed with free_atc)
--- pl2:    L2 parameters
--- meta:   metadata
--- DN:     Digital Numbers
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
atc_t *allocate_atc(par_ll_t *pl2, meta_t *meta, stack_t *DN){
atc_t *atc = NULL;


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif

  alloc((void**)&atc, 1, sizeof(atc_t));
  init_atc(meta, DN, atc);
  init_atc_xy(pl2, DN, atc);
  init_atc_xyz(atc);

  
  #ifdef FORCE_CLOCK
  proctime_print("allocate atc struct", TIME);
  #endif

  return atc;
}


/** This function deallocates parameters used for atmospheric correction.
--- atc:    atmospheric correction factors
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_atc(atc_t *atc){


  #ifdef FORCE_CLOCK
  time_t TIME; time(&TIME);
  #endif


  if (atc == NULL) return;

  if (atc->wvl       != NULL){ free((void*)atc->wvl);       atc->wvl       = NULL;}
  if (atc->lwvl      != NULL){ free((void*)atc->lwvl);      atc->lwvl      = NULL;}
  if (atc->lwvl2     != NULL){ free((void*)atc->lwvl2);     atc->lwvl2     = NULL;}
  if (atc->E0        != NULL){ free((void*)atc->E0);        atc->E0        = NULL;}
  if (atc->od        != NULL){ free((void*)atc->od);        atc->od        = NULL;}
  if (atc->mod       != NULL){ free((void*)atc->mod);       atc->mod       = NULL;}
  if (atc->aod       != NULL){ free((void*)atc->aod);       atc->aod       = NULL;}
  if (atc->aod_bands != NULL){ free((void*)atc->aod_bands); atc->aod_bands = NULL;}

  free_atc_xy(atc);
  free_atc_xyz(atc);

  free((void*)atc); atc = NULL;
  

  #ifdef FORCE_CLOCK
  proctime_print("free atc struct", TIME);
  #endif

  return;  
}


/** This function reshapes xyz parameters used for atmospheric correction.
--- xyz:    xyz atmospheric correction factors
--- b:      band
+++ Return: reshaped parameters
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
float **atc_get_band_reshaped(stack_t **xyz, int b){
int z, nz = NPOW_08;
float **reshape = NULL;

  alloc((void**)&reshape, nz, sizeof(float*));

  for (z=0; z<nz; z++){
    if ((reshape[z] =  get_band_float(xyz[z], b)) == NULL) return NULL;
  }

  return reshape;
}

