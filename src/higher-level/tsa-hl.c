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
This file contains functions for Level 3 processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "tsa-hl.h"


brick_t *compile_tsa_brick(brick_t *ard, int nb, int idx, int write, char *prodname, par_hl_t *phl);
brick_t **compile_tsa(ard_t *ard, tsa_t *tsa, par_hl_t *phl, cube_t *cube, int nt, int ni, int idx, int *nproduct);

typedef struct {
  int  prodlen;
  char prodname[NPOW_03];
  int  prodtype;
  int  enable;
  int  write;
  short ***ptr;
} brick_compile_info_t;

enum { _full_, _stats_, _inter_, _year_, _quarter_, _month_, _week_, _day_, _lsp_, _pol_, _trd_, _cat_, _pyp_ };


int info_tss(brick_compile_info_t *info, int o, int nt, tsa_t *ts, par_hl_t *phl){

  info[o].prodlen  = nt;
  strncpy(info[o].prodname, "TSS", 3); info[o].prodname[3] = '\0';
  info[o].prodtype = _full_;
  info[o].enable   = true;
  info[o].write    = phl->tsa.otss;
  info[o].ptr      = &ts->tss_;

  return o+1;
}

int info_tsi(brick_compile_info_t *info, int o, int ni, tsa_t *ts, par_hl_t *phl){


  info[o].prodlen  = ni;
  strncpy(info[o].prodname, "TSI", 3); info[o].prodname[3] = '\0';
  info[o].prodtype = _inter_;
  info[o].enable   = true;
  info[o].write    = phl->tsa.tsi.otsi;
  info[o].ptr      = &ts->tsi_;

  return o+1;
}

int info_stm(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){


  info[o].prodlen  = phl->tsa.stm.sta.nmetrics;
  strncpy(info[o].prodname, "STM", 3); info[o].prodname[3] = '\0';
  info[o].prodtype = _stats_;
  info[o].enable   = phl->tsa.stm.ostm;
  info[o].write    = phl->tsa.stm.ostm;
  info[o].ptr      = &ts->stm_;

  return o+1;
}

int info_rms(brick_compile_info_t *info, int o, int nt, tsa_t *ts, par_hl_t *phl){


  info[o].prodlen  = nt;
  strncpy(info[o].prodname, "RMS", 3); info[o].prodname[3] = '\0';
  info[o].prodtype = _full_;
  info[o].enable   = phl->tsa.sma.orms;
  info[o].write    = phl->tsa.sma.orms;
  info[o].ptr      = &ts->rms_;

  return o+1;
}

int info_spl(brick_compile_info_t *info, int o, int ni, tsa_t *ts, par_hl_t *phl){


  info[o].prodlen  = ni;
  strncpy(info[o].prodname, "SPL", 3); info[o].prodname[3] = '\0';
  info[o].prodtype = _inter_;
  info[o].enable   = phl->tsa.lsp.ospl;
  info[o].write    = phl->tsa.lsp.ospl;
  info[o].ptr      = &ts->spl_;

  return o+1;
}

int info_fby(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  info[p].prodlen  = phl->ny;
  strncpy(info[p].prodname, "FBY", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _year_;
  info[p].enable   = phl->tsa.fld.ofby+phl->tsa.fld.otry;
  info[p].write    = phl->tsa.fld.ofby;
  info[p++].ptr    = &ts->fby_;

  info[p].prodlen  = _TRD_LENGTH_;
  strncpy(info[p].prodname, "TRY", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otry;
  info[p].write    = phl->tsa.fld.otry;
  info[p++].ptr    = &ts->try_;

  info[p].prodlen  = _CAT_LENGTH_;
  strncpy(info[p].prodname, "CAY", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocay;
  info[p].write    = phl->tsa.fld.ocay;
  info[p++].ptr    = &ts->cay_;

  return p;
}

int info_fbq(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  info[p].prodlen  = phl->nq;
  strncpy(info[p].prodname, "FBQ", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _quarter_;
  info[p].enable   = phl->tsa.fld.ofbq+phl->tsa.fld.otrq;
  info[p].write    = phl->tsa.fld.ofbq;
  info[p++].ptr    = &ts->fbq_;

  info[p].prodlen  = _TRD_LENGTH_;
  strncpy(info[p].prodname, "TRQ", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrq;
  info[p].write    = phl->tsa.fld.otrq;
  info[p++].ptr    = &ts->trq_;

  info[p].prodlen  = _CAT_LENGTH_;
  strncpy(info[p].prodname, "CAQ", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocaq;
  info[p].write    = phl->tsa.fld.ocaq;
  info[p++].ptr    = &ts->caq_;

  return p;
}

int info_fbm(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  info[p].prodlen  = phl->nm;
  strncpy(info[p].prodname, "FBM", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _month_;
  info[p].enable   = phl->tsa.fld.ofbm+phl->tsa.fld.otrm;
  info[p].write    = phl->tsa.fld.ofbm;
  info[p++].ptr    = &ts->fbm_;

  info[p].prodlen  = _TRD_LENGTH_;
  strncpy(info[p].prodname, "TRM", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrm;
  info[p].write    = phl->tsa.fld.otrm;
  info[p++].ptr    = &ts->trm_;

  info[p].prodlen  = _CAT_LENGTH_;
  strncpy(info[p].prodname, "CAM", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocam;
  info[p].write    = phl->tsa.fld.ocam;
  info[p++].ptr    = &ts->cam_;

  return p;
}

int info_fbw(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  info[p].prodlen  = phl->nw;
  strncpy(info[p].prodname, "FBW", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _week_;
  info[p].enable   = phl->tsa.fld.ofbw+phl->tsa.fld.otrw;
  info[p].write    = phl->tsa.fld.ofbw;
  info[p++].ptr    = &ts->fbw_;

  info[p].prodlen  = _TRD_LENGTH_;
  strncpy(info[p].prodname, "TRW", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrw;
  info[p].write    = phl->tsa.fld.otrw;
  info[p++].ptr    = &ts->trw_;

  info[p].prodlen  = _CAT_LENGTH_;
  strncpy(info[p].prodname, "CAW", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocaw;
  info[p].write    = phl->tsa.fld.ocaw;
  info[p++].ptr    = &ts->caw_;

  return p;
}

int info_fbd(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  info[p].prodlen  = phl->nd;
  strncpy(info[p].prodname, "FBD", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _day_;
  info[p].enable   = phl->tsa.fld.ofbd+phl->tsa.fld.otrd;
  info[p].write    = phl->tsa.fld.ofbd;
  info[p++].ptr    = &ts->fbd_;

  info[p].prodlen  = _TRD_LENGTH_;
  strncpy(info[p].prodname, "TRD", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrd;
  info[p].write    = phl->tsa.fld.otrd;
  info[p++].ptr    = &ts->trd_;

  info[p].prodlen  = _CAT_LENGTH_;
  strncpy(info[p].prodname, "CAD", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocad;
  info[p].write    = phl->tsa.fld.ocad;
  info[p++].ptr    = &ts->cad_;

  return p;
}

int info_lsp(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int l, p = o;
int nchar;


  for (l=0; l<_LSP_LENGTH_; l++, p++){
    info[p].prodlen  = phl->tsa.lsp.ny;
    nchar = snprintf(info[p].prodname, NPOW_03, "%s-LSP", _TAGGED_ENUM_LSP_[l].tag);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    info[p].prodtype = _lsp_;
    info[p].enable   = phl->tsa.lsp.use[l]*(phl->tsa.lsp.olsp+phl->tsa.lsp.otrd+phl->tsa.lsp.ocat);
    info[p].write    = phl->tsa.lsp.use[l]*phl->tsa.lsp.olsp;
    info[p].ptr      = &ts->lsp_[l];
  }

  for (l=0; l<_LSP_LENGTH_; l++, p++){
    info[p].prodlen  =_TRD_LENGTH_;
    nchar = snprintf(info[p].prodname, NPOW_03, "%s-TRP", _TAGGED_ENUM_LSP_[l].tag);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    info[p].prodtype = _trd_;
    info[p].enable   = phl->tsa.lsp.use[l]*phl->tsa.lsp.otrd;
    info[p].write    = phl->tsa.lsp.use[l]*phl->tsa.lsp.otrd;
    info[p].ptr      = &ts->trp_[l];
  }

  for (l=0; l<_LSP_LENGTH_; l++, p++){
    info[p].prodlen  = _CAT_LENGTH_;
    nchar = snprintf(info[p].prodname, NPOW_03, "%s-CAP", _TAGGED_ENUM_LSP_[l].tag);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    info[p].prodtype = _cat_;
    info[p].enable   = phl->tsa.lsp.use[l]*phl->tsa.lsp.ocat;
    info[p].write    = phl->tsa.lsp.use[l]*phl->tsa.lsp.ocat;
    info[p].ptr      = &ts->cap_[l];
  }

  return p;
}

int info_pol(brick_compile_info_t *info, int o, int ni, tsa_t *ts, par_hl_t *phl){
int l, p = o;
int nchar;

  info[p].prodlen  = ni;
  strncpy(info[p].prodname, "PCX", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _inter_;
  info[p].enable   = phl->tsa.pol.opct;
  info[p].write    = phl->tsa.pol.opct;
  info[p++].ptr    = &ts->pcx_;

  info[p].prodlen  = ni;
  strncpy(info[p].prodname, "PCY", 3); info[p].prodname[3] = '\0';
  info[p].prodtype = _inter_;
  info[p].enable   = phl->tsa.pol.opct;
  info[p].write    = phl->tsa.pol.opct;
  info[p++].ptr    = &ts->pcy_;

  for (l=0; l<_POL_LENGTH_; l++, p++){
    info[p].prodlen  = phl->tsa.pol.ny;
    nchar = snprintf(info[p].prodname, NPOW_03, "%s-POL", _TAGGED_ENUM_POL_[l].tag);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    info[p].prodtype = _pol_;
    info[p].enable   = phl->tsa.pol.use[l]*(phl->tsa.pol.opol+phl->tsa.pol.otrd+phl->tsa.pol.ocat);
    info[p].write    = phl->tsa.pol.use[l]*phl->tsa.pol.opol;
    info[p].ptr      = &ts->pol_[l];
  }

  for (l=0; l<_POL_LENGTH_; l++, p++){
    info[p].prodlen  =_TRD_LENGTH_;
    nchar = snprintf(info[p].prodname, NPOW_03, "%s-TRO", _TAGGED_ENUM_POL_[l].tag);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    info[p].prodtype = _trd_;
    info[p].enable   = phl->tsa.pol.use[l]*phl->tsa.pol.otrd;
    info[p].write    = phl->tsa.pol.use[l]*phl->tsa.pol.otrd;
    info[p].ptr      = &ts->tro_[l];
  }

  for (l=0; l<_POL_LENGTH_; l++, p++){
    info[p].prodlen  = _CAT_LENGTH_;
    nchar = snprintf(info[p].prodname, NPOW_03, "%s-CAO", _TAGGED_ENUM_POL_[l].tag);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    info[p].prodtype = _cat_;
    info[p].enable   = phl->tsa.pol.use[l]*phl->tsa.pol.ocat;
    info[p].write    = phl->tsa.pol.use[l]*phl->tsa.pol.ocat;
    info[p].ptr      = &ts->cao_[l];
  }

  return p;
}


int info_pyp(stack_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){


  info[o].prodlen  = phl->tsa.pyp.nb;
  strncpy(info[o].prodname, "PYP", 3); info[o].prodname[3] = '\0';
  info[o].prodtype = _pyp_;
  info[o].enable   = phl->tsa.pyp.opyp;
  info[o].write    = phl->tsa.pyp.opyp;
  info[o].ptr      = &ts->pyp_;

  return o+1;
}



/** This function compiles the bricks, in which TSA results are stored. 
+++ It also sets metadata and sets pointers to instantly useable image 
+++ arrays.
--- ard:      ARD
--- ts:       pointer to instantly useable TSA image arrays
--- phl:      HL parameters
--- cube:     datacube definition
--- nt:       number of ARD products over time
--- ni:       number of interpolated products over time
--- idx:      spectral index
--- nproduct: number of output bricks (returned)
+++ Return:   bricks for TSA results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **compile_tsa(ard_t *ard, tsa_t *ts, par_hl_t *phl, cube_t *cube, int nt, int ni, int idx, int *nproduct){
brick_t **TSA = NULL;
int t, k;
date_t date;
char fdate[NPOW_10];
char sensor[NPOW_04];
char domain[NPOW_10];
int nchar;
int o = 0, nprod;
int error = 0;

brick_compile_info_t *info = NULL;


  nprod = 5 +              // TSS, RMS, TSI, STM, SPL,
          5 +              // folds
          5 +              // trend on folds
          5 +              // cat on folds
          _LSP_LENGTH_ +   // phenometrics
          _LSP_LENGTH_ +   // trend on phenometrics
          _LSP_LENGTH_ +   // cat on phenometrics
          2 +              // polar-transformed coordinates
          _POL_LENGTH_ +   // polarmetrics
          _POL_LENGTH_ +   // trend on polarmetrics
          _POL_LENGTH_ +   // cat on polarmetrics
          phl->tsa.pyp.nb; // python plugin metrics

  //printf("%d potential products.\n", nprod);

  alloc((void**)&info, nprod, sizeof(brick_compile_info_t));

  o = info_tss(info, o, nt, ts, phl);
  o = info_rms(info, o, nt, ts, phl);
  o = info_tsi(info, o, ni, ts, phl);
  o = info_stm(info, o,     ts, phl);
  o = info_spl(info, o, ni, ts, phl);
  o = info_fby(info, o,     ts, phl);
  o = info_fbq(info, o,     ts, phl);
  o = info_fbm(info, o,     ts, phl);
  o = info_fbw(info, o,     ts, phl);
  o = info_fbd(info, o,     ts, phl);
  o = info_lsp(info, o,     ts, phl);
  o = info_pol(info, o, ni, ts, phl);
  o = info_pyp(info, o,     ts, phl);


  alloc((void**)&TSA, nprod, sizeof(brick_t*));


//printf("about to allocate %d %d %d %d %d %d %d %d dates\n", 
//nt,ni,phl->ny,phl->nq,phl->nm,phl->nw,phl->nd,phl->tsa.lsp.ny);

  if (nt              > 0) alloc((void**)&ts->d_tss, nt, sizeof(date_t));              else ts->d_tss = NULL;
  if (ni              > 0) alloc((void**)&ts->d_tsi, ni, sizeof(date_t));              else ts->d_tsi = NULL;
  if (phl->ny         > 0) alloc((void**)&ts->d_fby, phl->ny, sizeof(date_t));         else ts->d_fby = NULL;
  if (phl->nq         > 0) alloc((void**)&ts->d_fbq, phl->nq, sizeof(date_t));         else ts->d_fbq = NULL;
  if (phl->nm         > 0) alloc((void**)&ts->d_fbm, phl->nm, sizeof(date_t));         else ts->d_fbm = NULL;
  if (phl->nw         > 0) alloc((void**)&ts->d_fbw, phl->nw, sizeof(date_t));         else ts->d_fbw = NULL;
  if (phl->nd         > 0) alloc((void**)&ts->d_fbd, phl->nd, sizeof(date_t));         else ts->d_fbd = NULL;
  if (phl->tsa.lsp.ny > 0) alloc((void**)&ts->d_lsp, phl->tsa.lsp.ny, sizeof(date_t)); else ts->d_lsp = NULL;
  if (phl->tsa.pol.ny > 0) alloc((void**)&ts->d_pol, phl->tsa.pol.ny, sizeof(date_t)); else ts->d_pol = NULL;

  //printf("scale, date, ts, bandnames, and sensor ID must be set in compile_tsa!!!\n");

  
  for (o=0; o<nprod; o++){

    //printf("%03d: compiling %s product? ", o, info[o].prodname);
    
    if (info[o].enable){
      
      //printf("Yes\n");
      
      if ((TSA[o] = compile_tsa_brick(ard[0].DAT, info[o].prodlen, idx, info[o].write, info[o].prodname, phl)) == NULL || (  *info[o].ptr = get_bands_short(TSA[o])) == NULL){
        printf("Error compiling %s product. ", info[o].prodname); error++;
      } else {

        init_date(&date);
        set_date(&date, 2000, 1, 1);

        for (t=0, k=1; t<info[o].prodlen; t++){

          switch (info[o].prodtype){
            case _full_:
              date = get_brick_date(ard[t].DAT, 0);
              get_brick_sensor(ard[t].DAT, 0, sensor, NPOW_04);
              set_brick_sensor(TSA[o], t, sensor);
              copy_date(&date, &ts->d_tss[t]);
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_brick_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_brick_unit(TSA[o], t, "decimal year");
              nchar = snprintf(domain, NPOW_10, "%s_%s", fdate, sensor);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_bandname(TSA[o], t, domain);
              break;
            case _stats_:
              set_stack_sensor(TSA[o], t, "BLEND");
              set_stack_bandname(TSA[o], t, _TAGGED_ENUM_STA_[phl->tsa.stm.sta.metrics[t]].tag);
              break;
            case _inter_:
              if (phl->tsa.tsi.method == _INT_NONE_){
                date = get_brick_date(ard[t].DAT, 0);
              } else {
                set_date_ce(&date, phl->date_range[_MIN_].ce + t*phl->tsa.tsi.step);
              }
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_tsi[t]);
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_stack_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_stack_unit(TSA[o], t, "decimal year");
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case _year_:
              set_date_year(&date, phl->date_range[_MIN_].year+t);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fby[t]);
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, date.year);
              set_stack_unit(TSA[o], t, "year");
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case  _quarter_:
              while (k < 5 && !phl->date_quarters[k]) k++;
              set_date_quarter(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbq[t]);
              nchar = snprintf(fdate, NPOW_10, "QUARTER-%01d", date.quarter);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "quarter");
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _month_: 
              while (k < 13 && !phl->date_months[k]) k++;
              set_date_month(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbm[t]);
              nchar = snprintf(fdate, NPOW_10, "MONTH-%02d", date.month);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "month");
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _week_: 
              while (k < 53 && !phl->date_weeks[k]) k++;
              set_date_week(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbw[t]);
              nchar = snprintf(fdate, NPOW_10, "WEEK-%02d", date.week);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "week");
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _day_: 
              while (k < 366 && !phl->date_doys[k]) k++;
              set_date_doy(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_fbd[t]);
              nchar = snprintf(fdate, NPOW_10, "DOY-%03d", date.doy);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, k);
              set_stack_unit(TSA[o], t, "day of year");
              set_stack_bandname(TSA[o], t, fdate);
              k++;
              break;
            case _lsp_: 
              set_date_year(&date, phl->date_range[_MIN_].year+t+1);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_lsp[t]);
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, date.year);
              set_stack_unit(TSA[o], t, "year");
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case _pol_: 
              set_date_year(&date, phl->date_range[_MIN_].year+t);
              set_brick_sensor(TSA[o], t, "BLEND");
              copy_date(&date, &ts->d_pol[t]);
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_stack_wavelength(TSA[o], t, date.year);
              set_stack_unit(TSA[o], t, "year");
              set_stack_bandname(TSA[o], t, fdate);
              break;
            case _trd_:
              set_stack_sensor(TSA[o], t, "BLEND");
              set_stack_bandname(TSA[o], t, _TAGGED_ENUM_TRD_[t].tag);
              break;
            case _cat_:
              set_stack_sensor(TSA[o], t, "BLEND");
              set_stack_bandname(TSA[o], t, _TAGGED_ENUM_CAT_[t].tag);
              break;
            case _pyp_:
              set_stack_sensor(TSA[o], t, "BLEND");
              set_stack_bandname(TSA[o], t, "unknown");  // needs to become flexible
              break;
            default:
              printf("unknown tsa type.\n"); error++;
              break;
          }
          
          set_brick_date(TSA[o], t, date);

        }

      }

    } else {
      //printf("No\n");
      TSA[o]  = NULL;
      *info[o].ptr = NULL;
    }
    
    //printf(" ptr: %p\n", *info[o].ptr);
    
  }


  //printf("%02d: ts ptr tss: %p\n", 0, ts->tss_);
  //printf("%02d: ts ptr rms: %p\n", 0, ts->rms_);
  //printf("%02d: ts ptr tsi: %p\n", 0, ts->tsi_);
  //printf("%02d: ts ptr stm: %p\n", 0, ts->stm_);
  //printf("%02d: ts ptr spl: %p\n", 0, ts->spl_);
  //printf("%02d: ts ptr fby: %p\n", 0, ts->fby_);
  //printf("%02d: ts ptr fbq: %p\n", 0, ts->fbq_);
  //printf("%02d: ts ptr fbm: %p\n", 0, ts->fbm_);
  //printf("%02d: ts ptr fbw: %p\n", 0, ts->fbw_);
  //printf("%02d: ts ptr fbd: %p\n", 0, ts->fbd_);
  //printf("%02d: ts ptr try: %p\n", 0, ts->try_);
  //printf("%02d: ts ptr trq: %p\n", 0, ts->trq_);
  //printf("%02d: ts ptr trm: %p\n", 0, ts->trm_);
  //printf("%02d: ts ptr trw: %p\n", 0, ts->trw_);
  //printf("%02d: ts ptr trd: %p\n", 0, ts->trd_);
  //printf("%02d: ts ptr cay: %p\n", 0, ts->cay_);
  //printf("%02d: ts ptr caq: %p\n", 0, ts->caq_);
  //printf("%02d: ts ptr cam: %p\n", 0, ts->cam_);
  //printf("%02d: ts ptr caw: %p\n", 0, ts->caw_);
  //printf("%02d: ts ptr cad: %p\n", 0, ts->cad_);
  //for (o=0; o<_LSP_LENGTH_; o++)  printf("%02d: ts ptr lsp: %p\n", o, ts->lsp_[o]);
  //for (o=0; o<_LSP_LENGTH_; o++)  printf("%02d: ts ptr trp: %p\n", o, ts->trp_[o]);
  //for (o=0; o<_LSP_LENGTH_; o++)  printf("%02d: ts ptr cap: %p\n", o, ts->cap_[o]);
  //for (o=0; o<_POL_LENGTH_; o++)  printf("%02d: ts ptr pol: %p\n", o, ts->pol_[o]);
  //for (o=0; o<_POL_LENGTH_; o++)  printf("%02d: ts ptr tro: %p\n", o, ts->tro_[o]);
  //for (o=0; o<_POL_LENGTH_; o++)  printf("%02d: ts ptr cao: %p\n", o, ts->cao_[o]);


  if (error > 0){
    printf("%d compiling TSA product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(TSA[o]);
    free((void*)TSA);
    if (ts->d_tss != NULL){ free((void*)ts->d_tss); ts->d_tss = NULL;}
    if (ts->d_tsi != NULL){ free((void*)ts->d_tsi); ts->d_tsi = NULL;}
    if (ts->d_fby != NULL){ free((void*)ts->d_fby); ts->d_fby = NULL;}
    if (ts->d_fbq != NULL){ free((void*)ts->d_fbq); ts->d_fbq = NULL;}
    if (ts->d_fbm != NULL){ free((void*)ts->d_fbm); ts->d_fbm = NULL;}
    if (ts->d_fbw != NULL){ free((void*)ts->d_fbw); ts->d_fbw = NULL;}
    if (ts->d_fbd != NULL){ free((void*)ts->d_fbd); ts->d_fbd = NULL;}
    if (ts->d_lsp != NULL){ free((void*)ts->d_lsp); ts->d_lsp = NULL;}
    if (ts->d_pol != NULL){ free((void*)ts->d_pol); ts->d_pol = NULL;}
    return NULL;
  }

  free((void*)info);

  *nproduct = nprod;
  return TSA;
}


/** This function compiles a TSA brick
--- from:      brick from which most attributes are copied
--- nb:        number of bands in brick
--- idx:       spectral index
--- write:     should this brick be written, or only used internally?
--- prodname:  product name
--- phl:       HL parameters
+++ Return:    brick for TSA result
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t *compile_tsa_brick(brick_t *from, int nb, int idx, int write, char *prodname, par_hl_t *phl){
int b;
brick_t *brick = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
char domain[NPOW_10];
int nchar;


  if ((brick = copy_brick(from, nb, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Time Series Analysis");
  set_brick_product(brick, prodname);

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick));
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);

  nchar = snprintf(fname, NPOW_10, "%04d-%04d_%03d-%03d_HL_TSA_%s_%s_%s", 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, 
    phl->doy_range[_MIN_], phl->doy_range[_MAX_], 
    phl->sen.target, phl->tsa.index_name[idx], prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);
  
  if (write){
    set_brick_open(brick, OPEN_BLOCK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, phl->format);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  sprintf(domain, "%s_%s", phl->tsa.index_name[idx], prodname);

  for (b=0; b<nb; b++){
    set_brick_save(brick, b, true);
    set_brick_date(brick, b, date);
    set_brick_domain(brick, b, domain);
  }

  return brick;
}


/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function is the entry point to the time series analysis module
--- ard:       ARD
--- mask:      mask image
--- nt:        number of ARD products over time
--- phl:       HL parameters
--- endmember: endmember (if SMA was selected)
--- cube:      datacube definition
--- nproduct:  number of output bricks (returned)
+++ Return:    bricks with TSA results
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
brick_t **time_series_analysis(ard_t *ard, brick_t *mask, int nt, par_hl_t *phl, aux_emb_t *endmember, cube_t *cube, int *nproduct){
tsa_t ts;
brick_t ***TSA;
brick_t **PTR;
small *mask_ = NULL;
int idx;
int o, k, nprod = 0;
int nc;
int ni;
short nodata;


  // import bricks
  nc = get_brick_chunkncells(ard[0].DAT);
  nodata = get_brick_nodata(ard[0].DAT, 0);

  // import mask (if available)
  if (mask != NULL){
    if ((mask_ = get_band_small(mask, 0)) == NULL){
      printf("Error getting processing mask."); return NULL;}
  }

//  printf("allocating %d indices\n", phl->tsa.n);
  alloc((void**)&TSA, phl->tsa.n, sizeof(brick_t**));

  // number of steps for interpolation
  if (phl->tsa.tsi.method == _INT_NONE_){
    ni = nt;
  } else {
    ni = ceil((phl->date_range[_MAX_].ce-phl->date_range[_MIN_].ce+1)/phl->tsa.tsi.step);
  }


  
  for (idx=0; idx<phl->tsa.n; idx++){

    // compile products + bricks
    if ((TSA[idx] = compile_tsa(ard, &ts, phl, cube, nt, ni, idx, &nprod)) == NULL || nprod == 0){
      printf("Unable to compile TSA products!\n"); 
      free((void*)TSA);
      *nproduct = 0;
      return NULL;
    }

    
    tsa_spectral_index(ard, &ts, mask_, nc, nt, idx, nodata, &phl->tsa, &phl->sen, endmember);
    
    tsa_interpolation(&ts, mask_, nc, nt, ni, nodata, &phl->tsa.tsi);
    
    tsa_python_plugin(&ts, mask_, nc, ni, nodata, &phl->tsa.pyp);
    
    tsa_stm(&ts, mask_, nc, ni, nodata, &phl->tsa.stm);
    
    tsa_fold(&ts, mask_, nc, ni, nodata, phl);
    
    tsa_polar(&ts, mask_, nc, ni, nodata, phl);
    
    tsa_pheno(&ts, mask_, nc, ni, nodata, phl);
    
    tsa_trend(&ts, mask_, nc, nodata, phl);
    
    tsa_cat(&ts, mask_, nc, nodata, phl);
    
    tsa_standardize(&ts, mask_, nc, nt, ni, nodata, phl);


    // clean temporal information
    if (ts.d_tss != NULL){ free((void*)ts.d_tss); ts.d_tss = NULL;}
    if (ts.d_tsi != NULL){ free((void*)ts.d_tsi); ts.d_tsi = NULL;}
    if (ts.d_fby != NULL){ free((void*)ts.d_fby); ts.d_fby = NULL;}
    if (ts.d_fbq != NULL){ free((void*)ts.d_fbq); ts.d_fbq = NULL;}
    if (ts.d_fbm != NULL){ free((void*)ts.d_fbm); ts.d_fbm = NULL;}
    if (ts.d_fbw != NULL){ free((void*)ts.d_fbw); ts.d_fbw = NULL;}
    if (ts.d_fbd != NULL){ free((void*)ts.d_fbd); ts.d_fbd = NULL;}
    if (ts.d_lsp != NULL){ free((void*)ts.d_lsp); ts.d_lsp = NULL;}
    if (ts.d_pol != NULL){ free((void*)ts.d_pol); ts.d_pol = NULL;}

  }
  

  // flatten out TSA bricks for returning to main
  alloc((void**)&PTR, phl->tsa.n*nprod, sizeof(brick_t*));
  
  for (idx=0, k=0; idx<phl->tsa.n; idx++){
    for (o=0; o<nprod; o++, k++) PTR[k] = TSA[idx][o];
  }
  
  for (idx=0; idx<phl->tsa.n; idx++) free((void*)TSA[idx]);
  free((void*)TSA);


  *nproduct = nprod*phl->tsa.n;
  return PTR;
}

