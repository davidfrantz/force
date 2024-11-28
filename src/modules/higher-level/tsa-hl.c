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
This file contains functions for Level 3 processing
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "tsa-hl.h"


typedef struct {
  int      prodlen;
  char   **bandname;
  date_t  *date;
  char     prodname[NPOW_03];
  int      prodtype;
  int      enable;
  int      write;
  short ***ptr;
} brick_compile_info_t;

enum { _full_, _stats_, _inter_, _nrt_, _year_, _quarter_, _month_, _week_, _day_, _pol_, _trd_, _cat_, _udf_};


void alloc_ts_metadata(tsa_t *ts, par_hl_t *phl, int nt, int nr, int ni);
void free_ts_metadata(tsa_t *ts, int ni);
void compile_ts_metadata(ard_t *ard, tsa_t *ts, par_hl_t *phl, int nt, int nr, int ni);
brick_t *compile_tsa_brick(brick_t *ard, int idx, brick_compile_info_t *info, par_hl_t *phl);
brick_t **compile_tsa(ard_t *ard, tsa_t *tsa, par_hl_t *phl, cube_t *cube, int nt, int nr, int ni, int idx, int *nproduct);


int info_tss(brick_compile_info_t *info, int o, int nt, tsa_t *ts, par_hl_t *phl){

  copy_string(info[o].prodname, NPOW_02, "TSS");
  info[o].prodlen  = nt;
  info[o].bandname = NULL;
  info[o].date     = NULL;
  info[o].prodtype = _full_;
  info[o].enable   = true;
  info[o].write    = phl->tsa.otss;
  info[o].ptr      = &ts->tss_;

  return o+1;
}

int info_nrt(brick_compile_info_t *info, int o, int nr, tsa_t *ts, par_hl_t *phl){

  copy_string(info[o].prodname, NPOW_02, "NRT");
  info[o].prodlen  = nr;
  info[o].bandname = NULL;
  info[o].date     = NULL;
  info[o].prodtype = _nrt_;
  info[o].enable   = phl->tsa.tsi.onrt;
  info[o].write    = phl->tsa.tsi.onrt;
  info[o].ptr      = &ts->nrt_;

  return o+1;
}

int info_tsi(brick_compile_info_t *info, int o, int ni, tsa_t *ts, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "TSI");
  info[o].prodlen  = ni;
  info[o].bandname = NULL;
  info[o].date     = NULL;
  info[o].prodtype = _inter_;
  info[o].enable   = true;
  info[o].write    = phl->tsa.tsi.otsi;
  info[o].ptr      = &ts->tsi_;

  return o+1;
}

int info_stm(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "STM");
  info[o].prodlen  = phl->tsa.stm.sta.nmetrics;
  info[o].bandname = NULL;
  info[o].date     = NULL;
  info[o].prodtype = _stats_;
  info[o].enable   = phl->tsa.stm.ostm;
  info[o].write    = phl->tsa.stm.ostm;
  info[o].ptr      = &ts->stm_;

  return o+1;
}

int info_rms(brick_compile_info_t *info, int o, int nt, tsa_t *ts, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "RMS");
  info[o].prodlen  = nt;
  info[o].bandname = NULL;
  info[o].date     = NULL;
  info[o].prodtype = _full_;
  info[o].enable   = phl->tsa.sma.orms;
  info[o].write    = phl->tsa.sma.orms;
  info[o].ptr      = &ts->rms_;

  return o+1;
}

int info_fby(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  copy_string(info[p].prodname, NPOW_02, "FBY");
  info[p].prodlen  = phl->ny;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _year_;
  info[p].enable   = phl->tsa.fld.ofby+phl->tsa.fld.otry+phl->tsa.fld.ocay;
  info[p].write    = phl->tsa.fld.ofby;
  info[p++].ptr    = &ts->fby_;

  copy_string(info[p].prodname, NPOW_02, "TRY");
  info[p].prodlen  = _TRD_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otry;
  info[p].write    = phl->tsa.fld.otry;
  info[p++].ptr    = &ts->try_;

  copy_string(info[p].prodname, NPOW_02, "CAY");
  info[p].prodlen  = _CAT_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocay;
  info[p].write    = phl->tsa.fld.ocay;
  info[p++].ptr    = &ts->cay_;

  return p;
}

int info_fbq(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  copy_string(info[p].prodname, NPOW_02, "FBQ");
  info[p].prodlen  = phl->nq;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _quarter_;
  info[p].enable   = phl->tsa.fld.ofbq+phl->tsa.fld.otrq+phl->tsa.fld.ocaq;
  info[p].write    = phl->tsa.fld.ofbq;
  info[p++].ptr    = &ts->fbq_;

  copy_string(info[p].prodname, NPOW_02, "TRQ");
  info[p].prodlen  = _TRD_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrq;
  info[p].write    = phl->tsa.fld.otrq;
  info[p++].ptr    = &ts->trq_;

  copy_string(info[p].prodname, NPOW_02, "CAQ");
  info[p].prodlen  = _CAT_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocaq;
  info[p].write    = phl->tsa.fld.ocaq;
  info[p++].ptr    = &ts->caq_;

  return p;
}

int info_fbm(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  copy_string(info[p].prodname, NPOW_02, "FBM");
  info[p].prodlen  = phl->nm;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _month_;
  info[p].enable   = phl->tsa.fld.ofbm+phl->tsa.fld.otrm+phl->tsa.fld.ocam;
  info[p].write    = phl->tsa.fld.ofbm;
  info[p++].ptr    = &ts->fbm_;

  copy_string(info[p].prodname, NPOW_02, "TRM");
  info[p].prodlen  = _TRD_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrm;
  info[p].write    = phl->tsa.fld.otrm;
  info[p++].ptr    = &ts->trm_;

  copy_string(info[p].prodname, NPOW_02, "CAM");
  info[p].prodlen  = _CAT_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocam;
  info[p].write    = phl->tsa.fld.ocam;
  info[p++].ptr    = &ts->cam_;

  return p;
}

int info_fbw(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  copy_string(info[p].prodname, NPOW_02, "FBW");
  info[p].prodlen  = phl->nw;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _week_;
  info[p].enable   = phl->tsa.fld.ofbw+phl->tsa.fld.otrw+phl->tsa.fld.ocaw;
  info[p].write    = phl->tsa.fld.ofbw;
  info[p++].ptr    = &ts->fbw_;

  copy_string(info[p].prodname, NPOW_02, "TRW");
  info[p].prodlen  = _TRD_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrw;
  info[p].write    = phl->tsa.fld.otrw;
  info[p++].ptr    = &ts->trw_;

  copy_string(info[p].prodname, NPOW_02, "CAW");
  info[p].prodlen  = _CAT_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocaw;
  info[p].write    = phl->tsa.fld.ocaw;
  info[p++].ptr    = &ts->caw_;

  return p;
}

int info_fbd(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){
int p = o;


  copy_string(info[p].prodname, NPOW_02, "FBD");
  info[p].prodlen  = phl->nd;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _day_;
  info[p].enable   = phl->tsa.fld.ofbd+phl->tsa.fld.otrd+phl->tsa.fld.ocad;
  info[p].write    = phl->tsa.fld.ofbd;
  info[p++].ptr    = &ts->fbd_;

  copy_string(info[p].prodname, NPOW_02, "TRD");
  info[p].prodlen  = _TRD_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _trd_;
  info[p].enable   = phl->tsa.fld.otrd;
  info[p].write    = phl->tsa.fld.otrd;
  info[p++].ptr    = &ts->trd_;

  copy_string(info[p].prodname, NPOW_02, "CAD");
  info[p].prodlen  = _CAT_LENGTH_;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _cat_;
  info[p].enable   = phl->tsa.fld.ocad;
  info[p].write    = phl->tsa.fld.ocad;
  info[p++].ptr    = &ts->cad_;

  return p;
}

int info_pol(brick_compile_info_t *info, int o, int ni, tsa_t *ts, par_hl_t *phl){
int l, p = o;
int nchar;

  copy_string(info[p].prodname, NPOW_02, "PCX");
  info[p].prodlen  = ni;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _inter_;
  info[p].enable   = phl->tsa.pol.opct;
  info[p].write    = phl->tsa.pol.opct;
  info[p++].ptr    = &ts->pcx_;

  copy_string(info[p].prodname, NPOW_02, "PCY");
  info[p].prodlen  = ni;
  info[p].bandname = NULL;
  info[o].date     = NULL;
  info[p].prodtype = _inter_;
  info[p].enable   = phl->tsa.pol.opct;
  info[p].write    = phl->tsa.pol.opct;
  info[p++].ptr    = &ts->pcy_;

  for (l=0; l<_POL_LENGTH_; l++, p++){
    info[p].prodlen  = phl->tsa.pol.ny;
    info[p].bandname = NULL;
    info[o].date     = NULL;
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
    info[p].bandname = NULL;
    info[o].date     = NULL;
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
    info[p].bandname = NULL;
    info[o].date     = NULL;
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


int info_pyp(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "PYP");
  info[o].prodlen  = phl->tsa.pyp.nb;
  info[o].bandname = phl->tsa.pyp.bandname;
  info[o].date     = phl->tsa.pyp.date;
  info[o].prodtype = _udf_;
  info[o].enable   = phl->tsa.pyp.out;
  info[o].write    = phl->tsa.pyp.out;
  info[o].ptr      = &ts->pyp_;

  return o+1;
}


int info_rsp(brick_compile_info_t *info, int o, tsa_t *ts, par_hl_t *phl){


  copy_string(info[o].prodname, NPOW_02, "RSP");
  info[o].prodlen  = phl->tsa.rsp.nb;
  info[o].bandname = phl->tsa.rsp.bandname;
  info[o].date     = phl->tsa.rsp.date;
  info[o].prodtype = _udf_;
  info[o].enable   = phl->tsa.rsp.out;
  info[o].write    = phl->tsa.rsp.out;
  info[o].ptr      = &ts->rsp_;

  return o+1;
}


/** This function allocates the date & bandname arrays
+++ Free the arrays using free_ts_metadata
--- ts:       pointer to instantly useable TSA image arrays
--- phl:      HL parameters
--- nt:       number of ARD products over time
--- nr:       number of NRT ARD products over time
--- ni:       number of interpolated products over time
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_ts_metadata(tsa_t *ts, par_hl_t *phl, int nt, int nr, int ni){


  #ifdef FORCE_DEBUG
  printf("about to allocate %d %d %d %d %d %d %d %d dates\n", 
    nt,nr,ni,phl->ny,phl->nq,phl->nm,phl->nw,phl->nd);
  #endif

  if (nt              > 0) alloc((void**)&ts->d_tss, nt, sizeof(date_t));              else ts->d_tss = NULL;
  if (nr              > 0) alloc((void**)&ts->d_nrt, nr, sizeof(date_t));              else ts->d_nrt = NULL;
  if (ni              > 0) alloc((void**)&ts->d_tsi, ni, sizeof(date_t));              else ts->d_tsi = NULL;
  if (phl->ny         > 0) alloc((void**)&ts->d_fby, phl->ny, sizeof(date_t));         else ts->d_fby = NULL;
  if (phl->nq         > 0) alloc((void**)&ts->d_fbq, phl->nq, sizeof(date_t));         else ts->d_fbq = NULL;
  if (phl->nm         > 0) alloc((void**)&ts->d_fbm, phl->nm, sizeof(date_t));         else ts->d_fbm = NULL;
  if (phl->nw         > 0) alloc((void**)&ts->d_fbw, phl->nw, sizeof(date_t));         else ts->d_fbw = NULL;
  if (phl->nd         > 0) alloc((void**)&ts->d_fbd, phl->nd, sizeof(date_t));         else ts->d_fbd = NULL;
  if (phl->tsa.pol.ny > 0) alloc((void**)&ts->d_pol, phl->tsa.pol.ny, sizeof(date_t)); else ts->d_pol = NULL;

  if (ni > 0) alloc_2D((void***)&ts->bandnames_tsi, ni, NPOW_04, sizeof(char)); else ts->bandnames_tsi = NULL;

  return;
}


/** This function frees the date & bandname arrays
--- ts:       pointer to instantly useable TSA image arrays
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_ts_metadata(tsa_t *ts, int ni){


  if (ts->d_tss != NULL){ free((void*)ts->d_tss); ts->d_tss = NULL;}
  if (ts->d_nrt != NULL){ free((void*)ts->d_nrt); ts->d_nrt = NULL;}
  if (ts->d_tsi != NULL){ free((void*)ts->d_tsi); ts->d_tsi = NULL;}
  if (ts->d_fby != NULL){ free((void*)ts->d_fby); ts->d_fby = NULL;}
  if (ts->d_fbq != NULL){ free((void*)ts->d_fbq); ts->d_fbq = NULL;}
  if (ts->d_fbm != NULL){ free((void*)ts->d_fbm); ts->d_fbm = NULL;}
  if (ts->d_fbw != NULL){ free((void*)ts->d_fbw); ts->d_fbw = NULL;}
  if (ts->d_fbd != NULL){ free((void*)ts->d_fbd); ts->d_fbd = NULL;}
  if (ts->d_pol != NULL){ free((void*)ts->d_pol); ts->d_pol = NULL;}

  if (ts->bandnames_tsi != NULL){ free_2D((void**)ts->bandnames_tsi, ni); ts->bandnames_tsi = NULL;}

  return;
}


/** This function compiles the date & bandname arrays
--- ard:      ARD
--- ts:       pointer to instantly useable TSA image arrays
--- phl:      HL parameters
--- nt:       number of ARD products over time
--- ni:       number of interpolated products over time
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void compile_ts_metadata(ard_t *ard, tsa_t *ts, par_hl_t *phl, int nt, int nr, int ni){
int t, k;
date_t date;
char sensor[NPOW_04];


  init_date(&date);
  set_date(&date, 2000, 1, 1);

  alloc_ts_metadata(ts, phl, nt, nr, ni);

  if (nt > 0){
    for (t=0; t<nt; t++){
      date = get_brick_date(ard[t].DAT, 0);
      copy_date(&date, &ts->d_tss[t]);
    }
  }

  if (nr > 0){
    for (t=0, k=0; t<nt; t++){
      if (get_brick_ce(ard[t].DAT, 0) > phl->tsa.tsi.harm_fit_range[_MAX_].ce){
        date = get_brick_date(ard[t].DAT, 0);
        copy_date(&date, &ts->d_nrt[k++]);
      }
    }
  }

  if (ni > 0){
    for (t=0; t<ni; t++){
      if (phl->tsa.tsi.method == _INT_NONE_){
        date = get_brick_date(ard[t].DAT, 0);
        get_brick_sensor(ard[t].DAT, 0, sensor, NPOW_04);
      } else {
        set_date_ce(&date, phl->date_range[_MIN_].ce + t*phl->tsa.tsi.step);
        copy_string(sensor, NPOW_04, "BLEND");
      }
      copy_date(&date, &ts->d_tsi[t]);
      copy_string(ts->bandnames_tsi[t], NPOW_04, sensor);
    }
  }

  if (phl->ny > 0){
    for (t=0; t<phl->ny; t++){
      set_date_year(&date, phl->date_range[_MIN_].year+t);
      copy_date(&date, &ts->d_fby[t]);
    }
  }

  if (phl->nq > 0){
    for (t=0, k=1; t<phl->nq; t++){
      while (k < 5 && !phl->date_quarters[k]) k++;
      set_date_quarter(&date, k);
      copy_date(&date, &ts->d_fbq[t]);
      k++;
    }
  }

  if (phl->nm > 0){
    for (t=0, k=1; t<phl->nm; t++){
      while (k < 13 && !phl->date_months[k]) k++;
      set_date_month(&date, k);
      copy_date(&date, &ts->d_fbm[t]);
      k++;
    }
  }

  if (phl->nw > 0){
    for (t=0, k=1; t<phl->nw; t++){
      while (k < 53 && !phl->date_weeks[k]) k++;
      set_date_week(&date, k);
      copy_date(&date, &ts->d_fbw[t]);
      k++;
    }
  }

  if (phl->nd > 0){
    for (t=0, k=1; t<phl->nd; t++){
      while (k < 366 && !phl->date_doys[k]) k++;
      set_date_doy(&date, k);
      copy_date(&date, &ts->d_fbd[t]);
      k++;
    }
  }

  if (phl->tsa.pol.ny > 0){
    for (t=0; t<phl->tsa.pol.ny; t++){
      set_date_year(&date, phl->date_range[_MIN_].year+t);
      copy_date(&date, &ts->d_pol[t]);
    }
  }

  return;
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
brick_t **compile_tsa(ard_t *ard, tsa_t *ts, par_hl_t *phl, cube_t *cube, int nt, int nr, int ni, int idx, int *nproduct){
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


  nprod = 5 +              // TSS, RMS, TSI, STM, NRT
          5 +              // folds
          5 +              // trend on folds
          5 +              // cat on folds
          2 +              // polar-transformed coordinates
          _POL_LENGTH_ +   // polarmetrics
          _POL_LENGTH_ +   // trend on polarmetrics
          _POL_LENGTH_ +   // cat on polarmetrics
          1 +              // python UDF metrics
          1;               // R UDF metrics

  //printf("%d potential products.\n", nprod);

  alloc((void**)&info, nprod, sizeof(brick_compile_info_t));

  o = info_tss(info, o, nt, ts, phl);
  o = info_nrt(info, o, nr, ts, phl);
  o = info_rms(info, o, nt, ts, phl);
  o = info_tsi(info, o, ni, ts, phl);
  o = info_stm(info, o,     ts, phl);
  o = info_fby(info, o,     ts, phl);
  o = info_fbq(info, o,     ts, phl);
  o = info_fbm(info, o,     ts, phl);
  o = info_fbw(info, o,     ts, phl);
  o = info_fbd(info, o,     ts, phl);
  o = info_pol(info, o, ni, ts, phl);
  o = info_pyp(info, o,     ts, phl);
  o = info_rsp(info, o,     ts, phl);


  alloc((void**)&TSA, nprod, sizeof(brick_t*));

  //printf("scale, date, ts, bandnames, and sensor ID must be set in compile_tsa!!!\n");

  
  for (o=0; o<nprod; o++){

    //printf("%03d: compiling %s product? ", o, info[o].prodname);
    
    if (info[o].enable > 0){
      
      //printf("Yes\n");
      
      if ((TSA[o] = compile_tsa_brick(ard[0].DAT, idx, &info[o], phl)) == NULL || (  *info[o].ptr = get_bands_short(TSA[o])) == NULL){
        printf("Error compiling %s product. ", info[o].prodname); error++;
      } else {

        init_date(&date);
        set_date(&date, 2000, 1, 1);

        // dates should be set within compile_tsa_brick !!!
        // as done in compile_pyp_brick, and as done with the pyp product
        // This here can be much simpler

        for (t=0, k=1; t<info[o].prodlen; t++){

          switch (info[o].prodtype){
            case _full_:
              date = get_brick_date(ard[t].DAT, 0);
              get_brick_sensor(ard[t].DAT, 0, sensor, NPOW_04);
              set_brick_sensor(TSA[o], t, sensor);
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_brick_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_brick_unit(TSA[o], t, "decimal year");
              nchar = snprintf(domain, NPOW_10, "%s_%s", fdate, sensor);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_bandname(TSA[o], t, domain);
              set_brick_date(TSA[o], t, date);
              break;
            case _nrt_:
              date = get_brick_date(ard[(nt-nr)+t].DAT, 0);
              get_brick_sensor(ard[(nt-nr)+t].DAT, 0, sensor, NPOW_04);
              set_brick_sensor(TSA[o], t, sensor);
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_brick_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_brick_unit(TSA[o], t, "decimal year");
              nchar = snprintf(domain, NPOW_10, "%s_%s", fdate, sensor);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_bandname(TSA[o], t, domain);
              set_brick_date(TSA[o], t, date);
              break;
            case _stats_:
              set_brick_sensor(TSA[o], t, "BLEND");
              set_brick_bandname(TSA[o], t, _TAGGED_ENUM_STA_[phl->tsa.stm.sta.metrics[t]].tag);
              set_brick_date(TSA[o], t, date);
              break;
            case _inter_:
              if (phl->tsa.tsi.method == _INT_NONE_){
                date = get_brick_date(ard[t].DAT, 0);
              } else {
                set_date_ce(&date, phl->date_range[_MIN_].ce + t*phl->tsa.tsi.step);
              }
              set_brick_sensor(TSA[o], t, "BLEND");
              compact_date(date.year, date.month, date.day, fdate, NPOW_10);
              set_brick_wavelength(TSA[o], t, date.year + (date.doy-1)/365.0);
              set_brick_unit(TSA[o], t, "decimal year");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              break;
            case _year_:
              set_date_year(&date, phl->date_range[_MIN_].year+t);
              set_brick_sensor(TSA[o], t, "BLEND");
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_wavelength(TSA[o], t, date.year);
              set_brick_unit(TSA[o], t, "year");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              break;
            case  _quarter_:
              while (k < 5 && !phl->date_quarters[k]) k++;
              set_date_quarter(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              nchar = snprintf(fdate, NPOW_10, "QUARTER-%01d", date.quarter);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_wavelength(TSA[o], t, k);
              set_brick_unit(TSA[o], t, "quarter");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              k++;
              break;
            case _month_: 
              while (k < 13 && !phl->date_months[k]) k++;
              set_date_month(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              nchar = snprintf(fdate, NPOW_10, "MONTH-%02d", date.month);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_wavelength(TSA[o], t, k);
              set_brick_unit(TSA[o], t, "month");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              k++;
              break;
            case _week_: 
              while (k < 53 && !phl->date_weeks[k]) k++;
              set_date_week(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              nchar = snprintf(fdate, NPOW_10, "WEEK-%02d", date.week);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_wavelength(TSA[o], t, k);
              set_brick_unit(TSA[o], t, "week");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              k++;
              break;
            case _day_: 
              while (k < 366 && !phl->date_doys[k]) k++;
              set_date_doy(&date, k);
              set_brick_sensor(TSA[o], t, "BLEND");
              nchar = snprintf(fdate, NPOW_10, "DOY-%03d", date.doy);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_wavelength(TSA[o], t, k);
              set_brick_unit(TSA[o], t, "day of year");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              k++;
              break;
            case _pol_: 
              set_date_year(&date, phl->date_range[_MIN_].year+t);
              set_brick_sensor(TSA[o], t, "BLEND");
              nchar = snprintf(fdate, NPOW_10, "YEAR-%04d", date.year);
              if (nchar < 0 || nchar >= NPOW_10){ 
                printf("Buffer Overflow in assembling domain\n"); error++;}
              set_brick_wavelength(TSA[o], t, date.year);
              set_brick_unit(TSA[o], t, "year");
              set_brick_bandname(TSA[o], t, fdate);
              set_brick_date(TSA[o], t, date);
              break;
            case _trd_:
              set_brick_sensor(TSA[o], t, "BLEND");
              set_brick_bandname(TSA[o], t, _TAGGED_ENUM_TRD_[t].tag);
              set_brick_date(TSA[o], t, date);
              break;
            case _cat_:
              set_brick_sensor(TSA[o], t, "BLEND");
              set_brick_bandname(TSA[o], t, _TAGGED_ENUM_CAT_[t].tag);
              set_brick_date(TSA[o], t, date);
              break;
            case _udf_:
              set_brick_sensor(TSA[o], t, "BLEND");
              break;
            default:
              printf("unknown tsa type.\n"); error++;
              break;
          }
          
        }

        //print_brick_info(TSA[o]);

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
  //for (o=0; o<_POL_LENGTH_; o++)  printf("%02d: ts ptr pol: %p\n", o, ts->pol_[o]);
  //for (o=0; o<_POL_LENGTH_; o++)  printf("%02d: ts ptr tro: %p\n", o, ts->tro_[o]);
  //for (o=0; o<_POL_LENGTH_; o++)  printf("%02d: ts ptr cao: %p\n", o, ts->cao_[o]);


  if (error > 0){
    printf("%d compiling TSA product errors.\n", error);
    for (o=0; o<nprod; o++) free_brick(TSA[o]);
    free((void*)TSA);
    free_ts_metadata(ts, ni);
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
brick_t *compile_tsa_brick(brick_t *from, int idx, brick_compile_info_t *info, par_hl_t *phl){
int b;
brick_t *brick = NULL;
date_t date;
char fname[NPOW_10];
char dname[NPOW_10];
char domain[NPOW_10];
char subname[NPOW_03];
int nchar;


  if ((brick = copy_brick(from, info->prodlen, _DT_SHORT_)) == NULL) return NULL;

  set_brick_name(brick, "FORCE Time Series Analysis");
  set_brick_product(brick, info->prodname);

  if (phl->subfolders){
    copy_string(subname, NPOW_03, info->prodname);
  } else {
    subname[0] = '\0';
  }

  //printf("dirname should be assemlbed in write_brick, check with L2\n");
  nchar = snprintf(dname, NPOW_10, "%s/X%04d_Y%04d/%s", phl->d_higher, 
    get_brick_tilex(brick), get_brick_tiley(brick), subname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling dirname\n"); return NULL;}
  set_brick_dirname(brick, dname);
  set_brick_provdir(brick, phl->d_prov);

  nchar = snprintf(fname, NPOW_10, "%04d-%04d_%03d-%03d_HL_TSA_%s_%s_%s", 
    phl->date_range[_MIN_].year, phl->date_range[_MAX_].year, 
    phl->doy_range[_MIN_], phl->doy_range[_MAX_], 
    phl->sen.target, phl->tsa.index_name[idx], info->prodname);
  if (nchar < 0 || nchar >= NPOW_10){ 
    printf("Buffer Overflow in assembling filename\n"); return NULL;}
  set_brick_filename(brick, fname);
  
  if (info->write){
    set_brick_open(brick, OPEN_BLOCK);
  } else {
    set_brick_open(brick, OPEN_FALSE);
  }
  set_brick_format(brick, &phl->gdalopt);
  set_brick_explode(brick, phl->explode);
  set_brick_par(brick, phl->params->log);

  sprintf(domain, "%s_%s", phl->tsa.index_name[idx], info->prodname);

  for (b=0; b<info->prodlen; b++){
    set_brick_save(brick, b, true);
    set_brick_date(brick, b, date);
    if (info->bandname != NULL) set_brick_bandname(brick, b, info->bandname[b]);
    if (info->date     != NULL) set_brick_date(brick, b, info->date[b]);
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
int t, idx;
int o, k, nprod = 0;
int nx, ny, nc;
int ni, nr;
short nodata;


  // import bricks
  nx = get_brick_chunkncols(ard[0].DAT);
  ny = get_brick_chunknrows(ard[0].DAT);
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

  // number of steps for NRT monitoring
  if (phl->tsa.tsi.onrt){
    for (t=0, nr=0; t<nt; t++){
      if (get_brick_ce(ard[t].DAT, 0) > phl->tsa.tsi.harm_fit_range[_MAX_].ce) nr++;
    }
  } else {
    nr = 0;
  }


  
  for (idx=0; idx<phl->tsa.n; idx++){

    // generate data arrays
    compile_ts_metadata(ard, &ts, phl, nt, nr, ni);

    // initialize UDFs
    init_pyp(NULL, &ts, _HL_TSA_, phl->tsa.index_name[idx], 1, ni, &phl->tsa.pyp);
    init_rsp(NULL, &ts, _HL_TSA_, phl->tsa.index_name[idx], 1, ni, &phl->tsa.rsp);

    // compile products + bricks
    if ((TSA[idx] = compile_tsa(ard, &ts, phl, cube, nt, nr, ni, idx, &nprod)) == NULL || nprod == 0){
      printf("Unable to compile TSA products!\n"); 
      free((void*)TSA);
      *nproduct = 0;
      return NULL;
    }

    
    tsa_spectral_index(ard, &ts, mask_, nc, nt, idx, nodata, &phl->tsa, &phl->sen, endmember);
    
    tsa_interpolation(&ts, mask_, nc, nt, nr, ni, nodata, &phl->tsa.tsi);

    python_udf(NULL, NULL, &ts, mask_, _HL_TSA_, phl->tsa.index_name[idx], 
      nx, ny, nc, 1, ni, nodata, &phl->tsa.pyp, phl->cthread);

    rstats_udf(NULL, NULL, &ts, mask_, _HL_TSA_, phl->tsa.index_name[idx], 
      nx, ny, nc, 1, ni, nodata, &phl->tsa.rsp, phl->cthread);

    tsa_stm(&ts, mask_, nc, ni, nodata, &phl->tsa.stm);
    
    tsa_fold(&ts, mask_, nc, ni, nodata, phl);
    
    tsa_polar(&ts, mask_, nc, ni, nodata, phl);
    
    tsa_trend(&ts, mask_, nc, nodata, phl);
    
    tsa_cat(&ts, mask_, nc, nodata, phl);
    
    tsa_standardize(&ts, mask_, nc, nt, ni, nodata, phl);


    // clean date arrays
    free_ts_metadata(&ts, ni);

    // terminate UDFs
    term_pyp(&phl->tsa.pyp);
    term_rsp(&phl->tsa.rsp);

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

