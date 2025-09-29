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
This file contains functions for parsing parameter files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "param-hl.h"


void register_higher(params_t *params, par_hl_t *phl);
void register_ard1(params_t *params, par_hl_t *phl);
void register_ard2(params_t *params, par_hl_t *phl);
void register_bap(params_t *params, par_hl_t *phl);
void register_tsa(params_t *params, par_hl_t *phl);
void register_cso(params_t *params, par_hl_t *phl);
void register_imp(params_t *params, par_hl_t *phl);
void register_cfi(params_t *params, par_hl_t *phl);
void register_l2i(params_t *params, par_hl_t *phl);
void register_mcl(params_t *params, par_hl_t *phl);
void register_ftr(params_t *params, par_hl_t *phl);
void register_smp(params_t *params, par_hl_t *phl);
void register_txt(params_t *params, par_hl_t *phl);
void register_lsm(params_t *params, par_hl_t *phl);
void register_lib(params_t *params, par_hl_t *phl);
void alloc_ftr(par_ftr_t *ftr);
void free_ftr(par_ftr_t *ftr);
void alloc_mcl(par_mcl_t *mcl);
void free_mcl(par_mcl_t *mcl);
int parse_ftr(par_ftr_t *ftr);
int parse_sta(par_sta_t *sta);
int parse_pol(par_pol_t *pol);
int parse_txt(par_txt_t *txt);
int parse_lsm(par_lsm_t *lsm);
int parse_quality(par_qai_t *qai);


/** This function registers common higher level parameters that are parsed
+++ from the parameter file.
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_higher(params_t *params, par_hl_t *phl){


  register_char_par(params,    "DIR_LOWER",  _CHAR_TEST_EXIST_,         &phl->d_lower);
  register_char_par(params,    "DIR_HIGHER", _CHAR_TEST_EXIST_,         &phl->d_higher);
  register_char_par(params,    "DIR_PROVENANCE", _CHAR_TEST_EXIST_,     &phl->d_prov);
  register_char_par(params,    "DIR_MASK",   _CHAR_TEST_NULL_OR_EXIST_, &phl->d_mask);
  register_char_par(params,    "BASE_MASK",  _CHAR_TEST_NULL_OR_BASE_,  &phl->b_mask);
  register_char_par(params,    "FILE_TILE",  _CHAR_TEST_NULL_OR_EXIST_, &phl->f_tile);
  register_intvec_par(params,  "X_TILE_RANGE", -999, 9999, 2, &phl->tx, &phl->ntx);
  register_intvec_par(params,  "Y_TILE_RANGE", -999, 9999, 2, &phl->ty, &phl->nty);
  register_double_par(params,  "RESOLUTION", 0, FLT_MAX, &phl->res);
  register_doublevec_par(params, "CHUNK_SIZE", 0, FLT_MAX, 2, &phl->chunk_size, &phl->n_chunk_size);
  register_char_par(params,    "FILE_OUTPUT_OPTIONS",   _CHAR_TEST_NULL_OR_EXIST_, &phl->f_gdalopt);
  register_enum_par(params,    "OUTPUT_FORMAT",  _TAGGED_ENUM_FMT_, _FMT_LENGTH_, &phl->format);
  register_bool_par(params,    "OUTPUT_EXPLODE", &phl->explode);
  register_bool_par(params,    "OUTPUT_SUBFOLDERS", &phl->subfolders);
  register_bool_par(params,    "FAIL_IF_EMPTY", &phl->fail_if_empty);
  //register_bool_par(params,    "OUTPUT_OVERWRITE", &phl->owr);
  register_int_par(params,     "NTHREAD_READ",    1, INT_MAX, &phl->ithread);
  register_int_par(params,     "NTHREAD_WRITE",   1, INT_MAX, &phl->othread);
  register_int_par(params,     "NTHREAD_COMPUTE", 1, INT_MAX, &phl->cthread);
  register_bool_par(params,    "STREAMING", &phl->stream);
  register_bool_par(params,    "PRETTY_PROGRESS", &phl->pretty_progress);

  return;
}


/** This function registers ARD parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_ard1(params_t *params, par_hl_t *phl){


  register_charvec_par(params, "SENSORS", _CHAR_TEST_NONE_, -1, &phl->sen.sensor, &phl->sen.n);
  register_char_par(params,    "PRODUCT_TYPE_MAIN",    _CHAR_TEST_NONE_, &phl->sen.main_product);
  register_char_par(params,    "PRODUCT_TYPE_QUALITY", _CHAR_TEST_NONE_, &phl->sen.quality_product);
  register_bool_par(params,    "SPECTRAL_ADJUST", &phl->sen.spec_adjust);
  register_enumvec_par(params, "SCREEN_QAI", _TAGGED_ENUM_QAI_, _QAI_LENGTH_, -1, &phl->qai.flags, &phl->qai.nflags);
  register_datevec_par(params, "DATE_RANGE", "1900-01-01", "2099-12-31", 2, &phl->date_range, &phl->ndate);
  register_intvec_par(params,  "DOY_RANGE", 1, 365, 2, &phl->doy_range, &phl->ndoy);
  register_date_par(params,    "DATE_IGNORE_LANDSAT_7", "1900-01-01", "2099-12-31", &phl->date_ignore_lnd07);

  return;
}


/** This function registers ARD parameters that are only needed if reflec-
+++ tance is used
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_ard2(params_t *params, par_hl_t *phl){


  register_bool_par(params,    "REDUCE_PSF",      &phl->psf);
  register_bool_par(params,    "USE_L2_IMPROPHE", &phl->prd.imp);
  register_float_par(params,   "ABOVE_NOISE", 0, FLT_MAX, &phl->qai.above_noise);
  register_float_par(params,   "BELOW_NOISE", 0, FLT_MAX, &phl->qai.below_noise);

  return;
}


/** This function registers BAP parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_bap(params_t *params, par_hl_t *phl){


  register_int_par(params,   "YEAR_TARGET", 1900, 2100, &phl->bap.Yt);
  register_int_par(params,   "YEAR_NUM", 0, 100, &phl->bap.Yr);
  register_float_par(params, "Y_FACTOR", 0, FLT_MAX, &phl->bap.Yf);
  register_bool_par(params,  "SELECT", &phl->bap.select);
  register_enum_par(params,  "COMBINE_SCORES",  _TAGGED_ENUM_SCR_COMB_, _SCR_COMB_LENGTH_, &phl->bap.combine);

  // additional aux products
  register_charvec_par(params, "REQUIRE_AUX_PRODUCTS", 
    _CHAR_TEST_NONE_, -1, &phl->sen.aux_products, &phl->sen.n_aux_products);

  register_floatvec_par(params, "DOY_SCORE", 0, 1, 3, &phl->bap.Ds, &phl->bap.nDs);
  register_intvec_par(params,   "DOY_STATIC", 1, 365, 3, &phl->bap.Dt, &phl->bap.nDt);

  register_bool_par(params,  "OFF_SEASON", &phl->bap.offsea);
  register_bool_par(params,  "USE_CLOUDY", &phl->bap.use_cloudy);
  register_bool_par(params,  "USE_HAZY", &phl->bap.use_hazy);
  register_float_par(params, "DREQ", 1, FLT_MAX, &phl->bap.dreq);
  register_float_par(params, "VREQ", 1, 90, &phl->bap.vreq);

  register_bool_par(params,   "OUTPUT_BAP", &phl->bap.obap);
  register_bool_par(params,   "OUTPUT_INF", &phl->bap.oinf);
  register_bool_par(params,   "OUTPUT_SCR", &phl->bap.oscr);
  register_bool_par(params,   "OUTPUT_OVV", &phl->bap.oovv);

  register_double_par(params, "SCORE_DOY_WEIGHT",    0, 1, &phl->bap.w.d);
  register_double_par(params, "SCORE_YEAR_WEIGHT",   0, 1, &phl->bap.w.y);
  register_double_par(params, "SCORE_CLOUD_WEIGHT",  0, 1, &phl->bap.w.c);
  register_double_par(params, "SCORE_HAZE_WEIGHT",   0, 1, &phl->bap.w.h);
  register_double_par(params, "SCORE_CORREL_WEIGHT", 0, 1, &phl->bap.w.r);
  register_double_par(params, "SCORE_VZEN_WEIGHT",   0, 1, &phl->bap.w.v);

  register_char_par(params,    "DIR_LSP", _CHAR_TEST_NULL_OR_EXIST_, &phl->con.dname);
  register_charvec_par(params, "BASE_LSP", _CHAR_TEST_NULL_OR_BASE_, 3, &phl->con.fname, &phl->con.n);
  register_int_par(params,     "LSP_NODATA", SHRT_MIN, SHRT_MAX, &phl->con.nodata);
  register_bool_par(params,    "LSP_DO", &phl->bap.pac.lsp);
  register_int_par(params,     "LSP_1ST_YEAR", 1900, 2100, &phl->bap.pac.y0);
  register_int_par(params,     "LSP_START", 1, 2100*365, &phl->bap.pac.start);
  register_float_par(params,   "LSP_THRESHOLD", 0, 365, &phl->bap.pac.rmse);

  return;
}


/** This function registers TSA parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_tsa(params_t *params, par_hl_t *phl){


  // TS parameters
  register_charvec_par(params, "INDEX", _CHAR_TEST_NONE_, -1, &phl->tsa.index.names, &phl->tsa.index.n);
  register_enum_par(params,    "STANDARDIZE_TSS", _TAGGED_ENUM_STD_, _STD_LENGTH_, &phl->tsa.standard);
  register_bool_par(params,    "OUTPUT_TSS", &phl->tsa.otss);

  // SMA parameters
  register_char_par(params, "FILE_ENDMEM",    _CHAR_TEST_NULL_OR_EXIST_, &phl->tsa.sma.f_emb);
  register_bool_par(params, "SMA_SUM_TO_ONE", &phl->tsa.sma.sto);
  register_bool_par(params, "SMA_NON_NEG",    &phl->tsa.sma.pos);
  register_bool_par(params, "SMA_SHD_NORM",   &phl->tsa.sma.shn);
  register_int_par(params,  "SMA_ENDMEMBER",  0, INT_MAX, &phl->tsa.sma.emb);
  register_bool_par(params, "OUTPUT_RMS",     &phl->tsa.sma.orms);

  // interpolation parameters
  register_enum_par(params,    "INTERPOLATE", _TAGGED_ENUM_INT_, _INT_LENGTH_, &phl->tsa.tsi.method);
  register_int_par(params,     "MOVING_MAX",  1, 365, &phl->tsa.tsi.mov_max);
  register_intvec_par(params,  "RBF_SIGMA",   1, 365, -1, &phl->tsa.tsi.rbf_sigma, &phl->tsa.tsi.rbf_nk);
  register_float_par(params,   "RBF_CUTOFF",  0, 1, &phl->tsa.tsi.rbf_cutoff);
  register_int_par(params,     "HARMONIC_MODES",  1, 3, &phl->tsa.tsi.harm_nmodes);
  register_bool_par(params,    "HARMONIC_TREND",  &phl->tsa.tsi.harm_trend);
  register_datevec_par(params, "HARMONIC_FIT_RANGE", "1900-01-01", "2099-12-31", 2, &phl->tsa.tsi.harm_fit_range, &phl->tsa.tsi.harm_fit_nrange);
  register_int_par(params,     "INT_DAY",     1, INT_MAX, &phl->tsa.tsi.step);
  register_enum_par(params,    "STANDARDIZE_TSI", _TAGGED_ENUM_STD_, _STD_LENGTH_, &phl->tsa.tsi.standard);
  register_bool_par(params,    "OUTPUT_TSI",  &phl->tsa.tsi.otsi);
  register_bool_par(params,    "OUTPUT_NRT",  &phl->tsa.tsi.onrt);

  // STM parameters
  register_enumvec_par(params, "STM", _TAGGED_ENUM_STA_, _STA_LENGTH_, -1, &phl->tsa.stm.sta.metrics, &phl->tsa.stm.sta.nmetrics);
  register_bool_par(params,    "OUTPUT_STM", &phl->tsa.stm.ostm);

  // folding parameters
  register_enum_par(params, "FOLD_TYPE", _TAGGED_ENUM_STA_, _STA_LENGTH_, &phl->tsa.fld.type);
  register_enum_par(params, "STANDARDIZE_FOLD", _TAGGED_ENUM_STD_, _STD_LENGTH_, &phl->tsa.fld.standard);
  register_bool_par(params, "OUTPUT_FBY", &phl->tsa.fld.ofby);
  register_bool_par(params, "OUTPUT_FBQ", &phl->tsa.fld.ofbq);
  register_bool_par(params, "OUTPUT_FBM", &phl->tsa.fld.ofbm);
  register_bool_par(params, "OUTPUT_FBW", &phl->tsa.fld.ofbw);
  register_bool_par(params, "OUTPUT_FBD", &phl->tsa.fld.ofbd);
  register_bool_par(params, "OUTPUT_TRY", &phl->tsa.fld.otry);
  register_bool_par(params, "OUTPUT_TRQ", &phl->tsa.fld.otrq);
  register_bool_par(params, "OUTPUT_TRM", &phl->tsa.fld.otrm);
  register_bool_par(params, "OUTPUT_TRW", &phl->tsa.fld.otrw);
  register_bool_par(params, "OUTPUT_TRD", &phl->tsa.fld.otrd);
  register_bool_par(params, "OUTPUT_CAY", &phl->tsa.fld.ocay);
  register_bool_par(params, "OUTPUT_CAQ", &phl->tsa.fld.ocaq);
  register_bool_par(params, "OUTPUT_CAM", &phl->tsa.fld.ocam);
  register_bool_par(params, "OUTPUT_CAW", &phl->tsa.fld.ocaw);
  register_bool_par(params, "OUTPUT_CAD", &phl->tsa.fld.ocad);

  // polar parameters
  register_float_par(params,   "POL_START_THRESHOLD", 0.01, 0.99, &phl->tsa.pol.start);
  register_float_par(params,   "POL_MID_THRESHOLD",   0.01, 0.99, &phl->tsa.pol.mid);
  register_float_par(params,   "POL_END_THRESHOLD",   0.01, 0.99, &phl->tsa.pol.end);
  register_bool_par(params,    "POL_ADAPTIVE",        &phl->tsa.pol.adaptive);
  register_enumvec_par(params, "POL", _TAGGED_ENUM_POL_, _POL_LENGTH_, -1, &phl->tsa.pol.metrics, &phl->tsa.pol.nmetrics);
  register_enum_par(params,    "STANDARDIZE_POL", _TAGGED_ENUM_STD_, _STD_LENGTH_, &phl->tsa.pol.standard);
  register_bool_par(params,    "OUTPUT_PCT",        &phl->tsa.pol.opct);
  register_bool_par(params,    "OUTPUT_POL",        &phl->tsa.pol.opol);
  register_bool_par(params,    "OUTPUT_TRO",        &phl->tsa.pol.otrd);
  register_bool_par(params,    "OUTPUT_CAO",        &phl->tsa.pol.ocat);

  // trend parameters
  register_enum_par(params,  "TREND_TAIL", _TAGGED_ENUM_TAIL_, _TAIL_LENGTH_, &phl->tsa.trd.tail);
  register_float_par(params, "TREND_CONF", 0, 1, &phl->tsa.trd.conf);
  register_bool_par(params,  "CHANGE_PENALTY", &phl->tsa.trd.penalty);

  // python UDF plug-in parameters
  register_char_par(params,    "FILE_PYTHON",  _CHAR_TEST_NULL_OR_EXIST_, &phl->tsa.pyp.f_code);
  register_enum_par(params,    "PYTHON_TYPE",  _TAGGED_ENUM_UDF_, _UDF_LENGTH_, &phl->tsa.pyp.type);
  register_bool_par(params,    "OUTPUT_PYP",    &phl->tsa.pyp.out);

  // R UDF plug-in parameters
  register_char_par(params,    "FILE_RSTATS",  _CHAR_TEST_NULL_OR_EXIST_, &phl->tsa.rsp.f_code);
  register_enum_par(params,    "RSTATS_TYPE",  _TAGGED_ENUM_UDF_, _UDF_LENGTH_, &phl->tsa.rsp.type);
  register_bool_par(params,    "OUTPUT_RSP",   &phl->tsa.rsp.out);

  return;
}


/** This function registers CSO parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_cso(params_t *params, par_hl_t *phl){


  register_int_par(params,     "MONTH_STEP", 1, 12, &phl->cso.step);
  register_enumvec_par(params, "CSO", _TAGGED_ENUM_STA_, _STA_LENGTH_, -1, &phl->cso.sta.metrics, &phl->cso.sta.nmetrics);

  return;
}


/** This function registers ImproPhe parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_imp(params_t *params, par_hl_t *phl){


  register_intvec_par(params,  "SEASONAL_WINDOW",      1, 365, -1, &phl->imp.dwin, &phl->imp.bwin);
  register_double_par(params,  "KERNEL_SIZE",          0, 1e6, &phl->imp.pred_radius);
  register_double_par(params,  "KERNEL_TEXT",          0, 1e6, &phl->imp.text_radius);

  return;
}


/** This function registers CF ImproPhe parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_cfi(params_t *params, par_hl_t *phl){


  register_imp(params, phl);
  register_char_par(params,    "DIR_COARSE",  _CHAR_TEST_NULL_OR_EXIST_, &phl->con.dname);
  register_charvec_par(params, "BASE_COARSE", _CHAR_TEST_BASE_, -1, &phl->con.fname, &phl->con.n);
  register_int_par(params,     "COARSE_NODATA", SHRT_MIN, SHRT_MAX, &phl->con.nodata);
  register_int_par(params,     "COARSE_1ST_YEAR",      1900, 2100, &phl->cfi.y0);
  register_intvec_par(params,  "COARSE_PREDICT_YEARS", 1900, 2100, -1, &phl->cfi.years, &phl->cfi.nyears);

  return;
}


/** This function registers L2 ImproPhe parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_l2i(params_t *params, par_hl_t *phl){

  
  register_imp(params, phl);
  register_charvec_par(params, "SENSORS_LOWRES", _CHAR_TEST_NONE_, -1, &phl->sen2.sensor, &phl->sen2.n);

  return;
}


/** This function registers machine learning parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_mcl(params_t *params, par_hl_t *phl){
int i;


  register_char_par(params,  "DIR_MODEL", _CHAR_TEST_EXIST_, &phl->mcl.d_model);
  register_enum_par(params,  "ML_METHOD", _TAGGED_ENUM_ML_, _ML_LENGTH_, &phl->mcl.method);
  register_float_par(params, "ML_CONVERGENCE", 0, INT_MAX, &phl->mcl.converge);
  register_float_par(params, "ML_SCALE", 0, 1e6, &phl->mcl.scale);
  register_char_par(params,  "ML_BASE",  _CHAR_TEST_NONE_, &phl->mcl.base);
  register_bool_par(params,  "OUTPUT_MLP", &phl->mcl.omlp);
  register_bool_par(params,  "OUTPUT_MLI", &phl->mcl.omli);
  register_bool_par(params,  "OUTPUT_MLU", &phl->mcl.omlu);
  register_bool_par(params,  "OUTPUT_RFP", &phl->mcl.orfp);
  register_bool_par(params,  "OUTPUT_RFM", &phl->mcl.orfm);

  for (i=0; i<phl->mcl.nmodelset; i++) register_charvec_par(params,  "FILE_MODEL",
    _CHAR_TEST_BASE_, -1, &phl->mcl.f_model[i], &phl->mcl.nmodel[i]);

  return;
}


/** This function registers feature parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_ftr(params_t *params, par_hl_t *phl){
int i;


  for (i=0; i<phl->ftr.ntags; i++) register_charvec_par(params,  "INPUT_FEATURE",
    _CHAR_TEST_NONE_, -1, &phl->ftr.cfeature[i], &phl->ftr.ifeature[i]);

  register_int_par(params,  "FEATURE_NODATA", SHRT_MIN, SHRT_MAX, &phl->ftr.nodata);
  register_bool_par(params, "FEATURE_EXCLUDE", &phl->ftr.exclude);

  return;
}


/** This function registers sample parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_smp(params_t *params, par_hl_t *phl){


  register_char_par(params, "FILE_POINTS",      _CHAR_TEST_EXIST_,     &phl->smp.f_coord);
  register_char_par(params, "FILE_SAMPLE",      _CHAR_TEST_NOT_EXIST_, &phl->smp.f_sample);
  register_char_par(params, "FILE_RESPONSE",    _CHAR_TEST_NOT_EXIST_, &phl->smp.f_response);
  register_char_par(params, "FILE_COORDINATES", _CHAR_TEST_NOT_EXIST_, &phl->smp.f_coords);
  register_bool_par(params, "PROJECTED",        &phl->smp.projected);

  return;
}


/** This function registers texture parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_txt(params_t *params, par_hl_t *phl){


  register_double_par(params,  "TXT_RADIUS",    0, 1e6,  &phl->txt.radius);
  register_int_par(params,     "TXT_ITERATION", 1, 1000, &phl->txt.iter);
  register_enumvec_par(params, "TXT", _TAGGED_ENUM_TXT_, _TXT_LENGTH_, -1, &phl->txt.metrics, &phl->txt.nmetrics);
  register_char_par(params,    "TXT_BASE",  _CHAR_TEST_NONE_, &phl->txt.base);

  return;
}


/** This function registers landscape metrics parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_lsm(params_t *params, par_hl_t *phl){


  register_double_par(params,  "LSM_RADIUS",    0, 1e6,  &phl->lsm.radius);
  register_int_par(params,     "LSM_MIN_PATCHSIZE",    0, 1e6,  &phl->lsm.minpatchsize);
  register_enumvec_par(params, "LSM_THRESHOLD_TYPE", _TAGGED_ENUM_QUERY_, _QUERY_LENGTH_, -1, &phl->lsm.query, &phl->lsm.nquery);
  register_intvec_par(params,  "LSM_THRESHOLD", SHRT_MIN, SHRT_MAX, -1, &phl->lsm.threshold, &phl->lsm.nthreshold);
  register_bool_par(params,    "LSM_ALL_PIXELS", &phl->lsm.allpx);
  register_enumvec_par(params, "LSM", _TAGGED_ENUM_LSM_, _LSM_LENGTH_, -1, &phl->lsm.metrics, &phl->lsm.nmetrics);
  register_char_par(params,    "LSM_BASE",  _CHAR_TEST_NONE_, &phl->lsm.base);
  register_enum_par(params,    "LSM_KERNEL_SHAPE", _TAGGED_ENUM_KERNEL_, _KERNEL_LENGTH_, &phl->lsm.kernel);

  return;
}


/** This function registers library completeness parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_lib(params_t *params, par_hl_t *phl){


  register_char_par(params,    "DIR_LIBRARY",  _CHAR_TEST_EXIST_, &phl->lib.d_lib);
  register_charvec_par(params, "FILE_LIBRARY", _CHAR_TEST_BASE_,  -1, &phl->lib.f_lib, &phl->lib.n_lib);
  register_bool_par(params,    "LIB_RESCALE",  &phl->lib.rescale);
  register_char_par(params,    "LIB_BASE",     _CHAR_TEST_NONE_,  &phl->lib.base);

  return;
}


/** This function registers UDF plug-in parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_udf(params_t *params, par_hl_t *phl){

  // python UDF plug-in parameters
  register_char_par(params,    "FILE_PYTHON",  _CHAR_TEST_NULL_OR_EXIST_, &phl->udf.pyp.f_code);
  register_enum_par(params,    "PYTHON_TYPE",  _TAGGED_ENUM_UDF_, _UDF_LENGTH_, &phl->udf.pyp.type);
  register_bool_par(params,    "OUTPUT_PYP",    &phl->udf.pyp.out);
  
  // R UDF plug-in parameters
  register_char_par(params,    "FILE_RSTATS",  _CHAR_TEST_NULL_OR_EXIST_, &phl->udf.rsp.f_code);
  register_enum_par(params,    "RSTATS_TYPE",  _TAGGED_ENUM_UDF_, _UDF_LENGTH_, &phl->udf.rsp.type);
  register_bool_par(params,    "OUTPUT_RSP",   &phl->udf.rsp.out);

  // additional aux products
  register_charvec_par(params, "REQUIRE_AUX_PRODUCTS", 
    _CHAR_TEST_NONE_, -1, &phl->sen.aux_products, &phl->sen.n_aux_products);

  return;
}


/** This function allocates the feature parameters
--- ftr:    feature parameters (must be freed with free_ftr)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_ftr(par_ftr_t *ftr){

  alloc((void**)&ftr->cfeature, ftr->ntags, sizeof(char**));
  alloc((void**)&ftr->ifeature, ftr->ntags, sizeof(int));

  return;
}


/** This function frees the feature parameters
--- ftr:    feature parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_ftr(par_ftr_t *ftr){

  free((void*)ftr->cfeature); ftr->cfeature = NULL;
  free((void*)ftr->ifeature); ftr->ifeature = NULL;
  free((void*)ftr->band); ftr->band = NULL;
  free_2D((void**)ftr->bname, ftr->nfeature); ftr->bname = NULL;

  return;
}


/** This function allocates the machine learning parameters
--- ftr:    machine learning parameters (must be freed with free_mcl)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_mcl(par_mcl_t *mcl){

  alloc((void**)&mcl->f_model, mcl->nmodelset, sizeof(char**));
  alloc((void**)&mcl->nmodel,  mcl->nmodelset, sizeof(int));
  alloc((void**)&mcl->nclass,  mcl->nmodelset, sizeof(int));

  return;
}


/** This function frees the machine learning parameters
--- mcl:    machine learning parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_mcl(par_mcl_t *mcl){

  free((void*)mcl->f_model); mcl->f_model = NULL;
  free((void*)mcl->nmodel);  mcl->nmodel  = NULL;
  free((void*)mcl->nclass);  mcl->nclass  = NULL;

  return;
}


/** This function reparses feature parameters (special para-
+++ meter that cannot be parsed with the general parser).
--- ftr:    feature parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_ftr(par_ftr_t *ftr){
int i, j, k;


  ftr->nfeature = 0;

  for (i=0; i<ftr->ntags; i++) ftr->nfeature += ftr->ifeature[i]-1;
  
  #ifdef FORCE_DEBUG
  printf("number of tag / features %d / %d\n", ftr->ntags, ftr->nfeature);
  #endif

  alloc_2D((void***)&ftr->bname, ftr->nfeature, NPOW_10, sizeof(char));
  alloc((void**)&ftr->band,  ftr->nfeature, sizeof(int));

  for (i=0, k=0; i<ftr->ntags; i++){
    for (j=1; j<ftr->ifeature[i]; j++, k++){

      copy_string(ftr->bname[k], NPOW_10, ftr->cfeature[i][0]);
      ftr->band[k] = atoi(ftr->cfeature[i][j]);

      #ifdef FORCE_DEBUG
      printf("Feature # %04d: %s, band %d\n", k, ftr->bname[k], ftr->band[k]);
      #endif
    }
  }

  for (i=0; i<ftr->nfeature; i++){
    if (strstr(ftr->bname[i], "/") != NULL){
      printf("parameter %s does not have a basename. \"/\" detected.\n", "INPUT_FEATURE");
      return FAILURE;}
    if (strstr(ftr->bname[i], ".") == NULL){
      printf("parameter %s does not have a basename. No file extension detected.\n", "INPUT_FEATURE");
      return FAILURE;}
    if (ftr->band[i] < 0){ printf("parameter %s includes a negative band number.\n", "INPUT_FEATURE");
      return FAILURE;}
  }


  return SUCCESS;
}


/** This function reparses aggregation statistic parameters (special para-
+++ meter that cannot be parsed with the general parser).
--- sta:    aggregation statistic parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_sta(par_sta_t *sta){
int i;


  sta->nquantiles = 0;
  sta->quantiles = false;

  sta->min = sta->max = sta->rng = sta->iqr = -1;
  sta->avg = sta->std = sta->skw = sta->krt = -1;
  for (i=0; i<100; i++) sta->qxx[i] = -1;
    
  for (i=0; i<sta->nmetrics; i++){
    if (sta->metrics[i] == _STA_NUM_){
      sta->num = i;
    } else if (sta->metrics[i] == _STA_MIN_){
      sta->min = i;
    } else if (sta->metrics[i] == _STA_MAX_){
      sta->max = i;
    } else if (sta->metrics[i] == _STA_RNG_){
      sta->rng = i;
    } else if (sta->metrics[i] == _STA_IQR_){
      sta->iqr = i;
    } else if (sta->metrics[i] == _STA_AVG_){
      sta->avg = i;
    } else if (sta->metrics[i] == _STA_STD_){
      sta->std = i;
    } else if (sta->metrics[i] == _STA_SKW_){
      sta->skw = i;
    } else if (sta->metrics[i] == _STA_KRT_){
      sta->krt = i;
    } else if (sta->metrics[i] >= _STA_Q01_ &&
               sta->metrics[i] <= _STA_Q99_){
      sta->qxx[sta->nquantiles] = i;
      sta->q[sta->nquantiles]   = (sta->metrics[i]-_STA_Q01_+1)/100.0;
      sta->nquantiles++;
      sta->quantiles = true;
    } else {
      printf("warning: unknown sta.\n");
    }
  }

  return SUCCESS;
}


/** This function reparses polarmetrics parameters (special para-
+++ meter that cannot be parsed with the general parser).
--- pol:    phenometrics parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_pol(par_pol_t *pol){
int i;


  for (i=0; i<pol->nmetrics; i++) pol->use[pol->metrics[i]] = true;

  return SUCCESS;
}


/** This function reparses texture parameters (special para-
+++ meter that cannot be parsed with the general parser).
--- txt:    texture parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_txt(par_txt_t *txt){
int i;


  for (i=0; i<txt->nmetrics; i++){
    if (txt->metrics[i] == _TXT_ERO_){
      txt->oero = true;
    } else if (txt->metrics[i] == _TXT_DIL_){
      txt->odil = true;
    } else if (txt->metrics[i] == _TXT_OPN_){
      txt->oopn = true;
    } else if (txt->metrics[i] == _TXT_CLS_){
      txt->ocls = true;
    } else if (txt->metrics[i] == _TXT_GRD_){
      txt->ogrd = true;
    } else if (txt->metrics[i] == _TXT_THT_){
      txt->otht = true;
    } else if (txt->metrics[i] == _TXT_BHT_){
      txt->obht = true;
    } else {
      printf("warning: unknown txt.\n");
    }
  }

  return SUCCESS;
}


/** This function reparses landscape metrics parameters (special para-
+++ meter that cannot be parsed with the general parser).
--- lsm:    texture parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_lsm(par_lsm_t *lsm){
int i;


  for (i=0; i<lsm->nmetrics; i++){
    if (lsm->metrics[i] == _LSM_MPA_){
      lsm->ompa = true;
    } else if (lsm->metrics[i] == _LSM_UCI_){
      lsm->ouci = true;
    } else if (lsm->metrics[i] == _LSM_FDI_){
      lsm->ofdi = true;
    } else if (lsm->metrics[i] == _LSM_EDD_){
      lsm->oedd = true;
    } else if (lsm->metrics[i] == _LSM_NBR_){
      lsm->onbr = true;
    } else if (lsm->metrics[i] == _LSM_EMS_){
      lsm->oems = true;
    } else if (lsm->metrics[i] == _LSM_AVG_){
      lsm->oavg = true;
    } else if (lsm->metrics[i] == _LSM_STD_){
      lsm->ostd = true;
    } else if (lsm->metrics[i] == _LSM_GEO_){
      lsm->ogeo = true;
    } else if (lsm->metrics[i] == _LSM_MAX_){
      lsm->omax = true;
    } else if (lsm->metrics[i] == _LSM_ARE_){
      lsm->oare = true;
    } else {
      printf("warning: unknown lsm.\n");
    }
  }

  return SUCCESS;
}


/** This function reparses Quality Assurance Information rules (special para-
+++ meter that cannot be parsed with the general parser).
--- qai:    QAI parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_quality(par_qai_t *qai){
int i;


  for (i=0; i<qai->nflags; i++){
    if (qai->flags[i] == _QAI_OFF_)      qai->off = true;
    if (qai->flags[i] == _QAI_CLD_UNC_)  qai->cld_unc = true;
    if (qai->flags[i] == _QAI_CLD_OPQ_)  qai->cld_opq = true;
    if (qai->flags[i] == _QAI_CLD_CIR_)  qai->cld_cir = true;
    if (qai->flags[i] == _QAI_SHD_)      qai->shd = true;
    if (qai->flags[i] == _QAI_SNW_)      qai->snw = true;
    if (qai->flags[i] == _QAI_WTR_)      qai->wtr = true;
    if (qai->flags[i] == _QAI_AOD_INT_)  qai->aod_int = true;
    if (qai->flags[i] == _QAI_AOD_HIGH_) qai->aod_high = true;
    if (qai->flags[i] == _QAI_AOD_FILL_) qai->aod_fill = true;
    if (qai->flags[i] == _QAI_SUB_)      qai->sub = true;
    if (qai->flags[i] == _QAI_SAT_)      qai->sat = true;
    if (qai->flags[i] == _QAI_SUN_)      qai->sun = true;
    if (qai->flags[i] == _QAI_ILL_LOW_ ) qai->ill_low = true;
    if (qai->flags[i] == _QAI_ILL_POOR_) qai->ill_poor = true;
    if (qai->flags[i] == _QAI_ILL_SHD_)  qai->ill_shd = true;
    if (qai->flags[i] == _QAI_SLP_)      qai->slp = true;
    if (qai->flags[i] == _QAI_WVP_)      qai->wvp = true;
  }

  #ifdef FORCE_DEBUG
  if (qai->off) printf("filter NODATA\n");
  if (qai->cld_opq) printf("filter CLOUD_OPAQUE\n");
  if (qai->cld_unc) printf("filter CLOUD_BUFFER\n");
  if (qai->cld_cir) printf("filter CLOUD_CIRRUS\n");
  if (qai->shd) printf("filter CLOUD_SHADOW\n");
  if (qai->snw) printf("filter SNOW\n");
  if (qai->wtr) printf("filter WATER\n");
  if (qai->aod_fill) printf("filter AOD_FILL\n");
  if (qai->aod_high) printf("filter AOD_HIGH\n");
  if (qai->aod_int) printf("filter AOD_INT\n");
  if (qai->sub) printf("filter SUBZERO\n");
  if (qai->sat) printf("filter SATURATION\n");
  if (qai->sun) printf("filter SUN_LOW\n");
  if (qai->ill_shd) printf("filter ILLUMIN_NONE\n");
  if (qai->ill_poor) printf("filter ILLUMIN_POOR\n");
  if (qai->ill_low) printf("filter ILLUMIN_LOW\n");
  if (qai->slp) printf("filter SLOPED\n");
  if (qai->wvp) printf("filter WVP_NONE\n");
  #endif

  return SUCCESS;
}



/** public functions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


/** This function allocates the higher level parameters
+++ Return: HL parameters (must be freed with free_param_higher)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
par_hl_t *allocate_param_higher(){
par_hl_t *phl = NULL;


  alloc((void**)&phl, 1, sizeof(par_hl_t));

  return phl;
}


/** This function frees the higher level parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_param_higher(par_hl_t *phl){

  if (phl == NULL) return;

  free_params(phl->params);

  if (phl->input_level1 == _INP_QAI_ ||
      phl->input_level1 == _INP_ARD_){
    free_2D((void**)phl->sen.sensor, phl->sen.n);
    free_2D((void**)phl->sen.band_number, phl->sen.n);
    free_2D((void**)phl->sen.band_names, phl->sen.n_bands);
  }

  if (phl->input_level2 == _INP_QAI_ ||
      phl->input_level2 == _INP_ARD_){
    free_2D((void**)phl->sen2.sensor, phl->sen2.n);
    free_2D((void**)phl->sen2.band_number, phl->sen2.n);
    free_2D((void**)phl->sen2.band_names, phl->sen2.n_bands);
  }

  if (phl->type == _HL_TSA_) free_2D((void**)phl->tsa.index.names, phl->tsa.index.n); 

  if (phl->type == _HL_ML_) free_mcl(&phl->mcl);


  if (phl->input_level1 == _INP_FTR_) free_ftr(&phl->ftr);

  free((void*)phl); phl = NULL;

  return;
}


/** This function parses the higher level parameters
--- phl:    HL parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_param_higher(par_hl_t *phl){
FILE *fpar;
char  buffer[NPOW_16] = "\0";
int d, w, m, q, y, tmp;
double tol = 5e-3;
  

  phl->params = allocate_params();


  // open parameter file
  if ((fpar = fopen(phl->f_par, "r")) == NULL){
    printf("Unable to open parameter file!\n"); return FAILURE;}

  if (fscanf(fpar, "%s", buffer) < 0){
    printf("No valid parameter file!\n"); return FAILURE;}


   // detect module

  if (strcmp(buffer, "++PARAM_LEVEL3_START++") == 0){
    phl->type = _HL_BAP_;
    phl->input_level1 = _INP_ARD_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_TSA_START++") == 0){
    phl->type = _HL_TSA_;
    phl->input_level1 = _INP_ARD_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_CSO_START++") == 0){
    phl->type = _HL_CSO_;
    phl->input_level1 = _INP_QAI_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_CFIMP_START++") == 0){
    phl->type = _HL_CFI_;
    phl->input_level1 = _INP_ARD_;
    phl->input_level2 = _INP_CON_;
  } else if (strcmp(buffer, "++PARAM_L2IMP_START++") == 0){
    phl->type = _HL_L2I_;
    phl->input_level1 = _INP_ARD_;
    phl->input_level2 = _INP_ARD_;
  } else if (strcmp(buffer, "++PARAM_ML_START++") == 0){
    phl->type = _HL_ML_;
    phl->input_level1  = _INP_FTR_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_SMP_START++") == 0){
    phl->type = _HL_SMP_;
    phl->input_level1 = _INP_FTR_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_TXT_START++") == 0){
    phl->type = _HL_TXT_;
    phl->input_level1 = _INP_FTR_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_LSM_START++") == 0){
    phl->type = _HL_LSM_;
    phl->input_level1 = _INP_FTR_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_LIB_START++") == 0){
    phl->type = _HL_LIB_;
    phl->input_level1 = _INP_FTR_;
    phl->input_level2 = _INP_NONE_;
  } else if (strcmp(buffer, "++PARAM_UDF_START++") == 0){
    phl->type = _HL_UDF_;
    phl->input_level1 = _INP_ARD_;
    phl->input_level2 = _INP_NONE_;
  } else {
    printf("No valid parameter file!\n"); return FAILURE;
  }



   register_higher(phl->params, phl);

  // register input-specific parameters

  if (phl->input_level1 == _INP_QAI_ ||
      phl->input_level1 == _INP_ARD_){
     register_ard1(phl->params, phl);
  }

  if (phl->input_level1 == _INP_ARD_){
     register_ard2(phl->params, phl);
  }

  if (phl->input_level1 == _INP_FTR_){
    if (prescreen_par(fpar, "INPUT_FEATURE", &phl->ftr.ntags) == FAILURE) return FAILURE;
    alloc_ftr(&phl->ftr);
    register_ftr(phl->params, phl);
  }


  // register module-specific parameters

  switch (phl->type){
    case _HL_BAP_:
      register_bap(phl->params, phl);
      break;
    case _HL_TSA_:
      register_tsa(phl->params, phl);
      break;
    case _HL_CSO_:
      register_cso(phl->params, phl);
      break;
    case _HL_CFI_:
      register_cfi(phl->params, phl);
      break;
    case _HL_L2I_:
      register_l2i(phl->params, phl);
      break;
    case _HL_ML_:
      if (prescreen_par(fpar, "FILE_MODEL", &phl->mcl.nmodelset) == FAILURE) return FAILURE;
      alloc_mcl(&phl->mcl);
      register_mcl(phl->params, phl);
      break;
    case _HL_SMP_:
      register_smp(phl->params, phl);
      break;
    case _HL_TXT_:
      register_txt(phl->params, phl);
      break;
    case _HL_LSM_:
      register_lsm(phl->params, phl);
      break;
    case _HL_LIB_:
      register_lib(phl->params, phl);
      break;
    case _HL_UDF_:
      register_udf(phl->params, phl);
      break;
    default:
      printf("Unknown module!\n"); return FAILURE;
  }


  // process line by line
  while (fgets(buffer, NPOW_16, fpar) != NULL) parse_parameter(phl->params, buffer);
  fclose(fpar);


  #ifdef FORCE_DEBUG
  print_parameter(phl->params);
  #endif

  if (check_parameter(phl->params) == FAILURE) return FAILURE;

  log_parameter(phl->params);


  // re-parse special cases

  if ((phl->input_level1 == _INP_QAI_ ||
       phl->input_level1 == _INP_ARD_) &&
    retrieve_sensor(&phl->sen) != SUCCESS){
    printf("Compiling sensors failed.\n"); return FAILURE;}
    
  if ((phl->input_level2 == _INP_QAI_ ||
       phl->input_level2 == _INP_ARD_) &&
    retrieve_sensor(&phl->sen2) != SUCCESS){
    printf("Compiling secondary sensors failed.\n"); return FAILURE;}

  if (phl->type == _HL_TSA_ && retrieve_indices(&phl->tsa.index, &phl->sen) == FAILURE){
    printf("sth wrong with bandlist."); return FAILURE;}

  if (phl->type == _HL_TSA_) parse_sta(&phl->tsa.stm.sta);
  if (phl->type == _HL_CSO_) parse_sta(&phl->cso.sta);
  
  if (phl->type == _HL_TSA_) parse_pol(&phl->tsa.pol);
  
  if (phl->type == _HL_TXT_) parse_txt(&phl->txt);
  
  if (phl->type == _HL_LSM_) parse_lsm(&phl->lsm);

  if (phl->input_level1 == _INP_FTR_) parse_ftr(&phl->ftr);

  if (phl->input_level1 == _INP_ARD_ ||
      phl->input_level1 == _INP_QAI_) parse_quality(&phl->qai);

  if (phl->input_level1 != _INP_QAI_) phl->prd.ref = true;
  phl->prd.qai = true;

  if (phl->sen.n_aux_products > 0) phl->prd.aux = true;

  if (phl->sen.n_aux_products != 0 && strcmp(phl->sen.aux_products[0], "NULL") == 0) phl->prd.aux = false;

  if (phl->prd.aux && !phl->prd.ref){
    printf("aux products cannot be used when reflectance product is not used.\n");
    return FAILURE;
  }



  // check chunk size
  if (fmod(phl->chunk_size[_X_], phl->res) > tol ||
      fmod(phl->chunk_size[_Y_], phl->res) > tol){
    printf("CHUNK_SIZE must be a multiple of RESOLUTION.\n");
    return FAILURE;
  }

  // check chunk size against 0
  if (phl->chunk_size[_X_] <= 0.0 ||
      phl->chunk_size[_Y_] <= 0.0){
    printf("CHUNK_SIZE must be positive.\n");
    return FAILURE;
  }

    // check resolution against 0
  if (phl->res <= 0.0){
    printf("RESOLUTION must be positive.\n");
    return FAILURE;
  }

  
  // compile temporal window
  if ((phl->input_level1 == _INP_QAI_ ||
       phl->input_level1 == _INP_ARD_) &&
       phl->doy_range[_MIN_] > 0 && 
       phl->doy_range[_MAX_] > 0){

    if (phl->doy_range[_MIN_] < phl->doy_range[_MAX_]){
      for (d=phl->doy_range[_MIN_]; d<=phl->doy_range[_MAX_]; d++) phl->date_doys[d] = true;
    } else {
      for (d=1; d<=phl->doy_range[_MAX_];   d++) phl->date_doys[d] = true;
      for (d=phl->doy_range[_MIN_]; d<=365; d++) phl->date_doys[d] = true;
    }

    for (d=1; d<=365; d++){

      if (!phl->date_doys[d]) continue;

      w = doy2week(d);    phl->date_weeks[w] = true;
      m = doy2m(d);       phl->date_months[m] = true;
      q = doy2quarter(d); phl->date_quarters[q] = true;

    }

    for (d=1; d<=365; d++) phl->nd += phl->date_doys[d];
    for (w=1; w<=52;  w++) phl->nw += phl->date_weeks[w];
    for (m=1; m<=12;  m++) phl->nm += phl->date_months[m];
    for (q=1; q<=4;   q++) phl->nq += phl->date_quarters[q];
    phl->ny = phl->date_range[_MAX_].year - phl->date_range[_MIN_].year + 1;

    // polarmetrics not possible for one year
    phl->tsa.pol.ny = phl->ny;
    phl->tsa.pol.ns = phl->ny-1;
    
    #ifdef FORCE_DEBUG
    printf("ny: %d, nq: %d, nm: %d, nw: %d, nd: %d\n",
      phl->ny, phl->nq, phl->nm, phl->nw, phl->nd);
    for (d=1; d<=365; d++) printf("temporal filter day %d: %d\n", d, phl->date_doys[d]);
    for (w=1; w<=52;  w++) printf("temporal filter week %d: %d\n", w, phl->date_weeks[w]);
    for (m=1; m<=12;  m++) printf("temporal filter month %d: %d\n", m, phl->date_months[m]);
    for (q=1; q<=4;   q++) printf("temporal filter quarter %d: %d\n", q, phl->date_quarters[q]);
    printf("day: %03d - %03d\n", phl->doy_range[_MIN_], phl->doy_range[_MAX_]);
    #endif

  }


  if (phl->type == _HL_TXT_){
    phl->radius = phl->txt.radius*phl->txt.iter;
    phl->txt.radius = (int)(phl->txt.radius/phl->res);
  } else if (phl->type == _HL_LSM_){
    phl->radius = phl->lsm.radius;
    phl->lsm.radius = (int)(phl->lsm.radius/phl->res);
  } else if (phl->type == _HL_L2I_ || phl->type == _HL_CFI_){
    if (phl->imp.pred_radius > phl->imp.text_radius){
      phl->radius = phl->imp.pred_radius;
    } else {
      phl->radius = phl->imp.text_radius;
    }
    phl->imp.ksize = (int)(phl->imp.pred_radius/phl->res);
    phl->imp.ksd   = (int)(phl->imp.text_radius/phl->res);
  } else {
    phl->radius = 0;
  }

  if (phl->radius != 0 && fmod(phl->radius, phl->res) > tol){
    printf("requested RADIUS %f must be a multiple of RESOLUTION %f\n", phl->radius, phl->res);
    return FAILURE;
  }

  if (phl->type == _HL_L2I_ || phl->type == _HL_CFI_){
    if ((phl->imp.nwin = phl->imp.bwin-1) <= 0){
      printf("at least two breakpoints need to be given for SEASONAL_WINDOW\n");
      return FAILURE;}
  }

  if (phl->type == _HL_BAP_){

    // total scoring weight
    if (phl->bap.combine == _SCR_COMB_ADD_){
      phl->bap.w.t = 0.0;
      if (phl->bap.w.d > 0) phl->bap.w.t += phl->bap.w.d;
      if (phl->bap.w.y > 0) phl->bap.w.t += phl->bap.w.y;
      if (phl->bap.w.c > 0) phl->bap.w.t += phl->bap.w.c;
      if (phl->bap.w.h > 0) phl->bap.w.t += phl->bap.w.h;
      if (phl->bap.w.r > 0) phl->bap.w.t += phl->bap.w.r;
      if (phl->bap.w.v > 0) phl->bap.w.t += phl->bap.w.v;
    }

    if (dequal(phl->bap.w.d, 0) && dequal(phl->bap.w.y, 0) &&
        dequal(phl->bap.w.c, 0) && dequal(phl->bap.w.h, 0) &&
        dequal(phl->bap.w.r, 0) && dequal(phl->bap.w.v, 0)){
      printf("ALL scoring weights are zero. This is not allowed. "
             "At least, the seasonal score should be > 0.\n"); return FAILURE;}

    // number of years
    phl->bap.Yn = (phl->bap.Yr*2)+1;

    // choose type of scoring function
    if (phl->bap.Ds[1] > phl->bap.Ds[0] &&
        phl->bap.Ds[1] > phl->bap.Ds[2]){
      phl->bap.score_type = _SCR_TYPE_GAUSS_; // gaussian
    } else if (phl->bap.Ds[0] > phl->bap.Ds[2]){
      phl->bap.score_type = _SCR_TYPE_SIG_DES_; // descending sigmoid
    } else if (phl->bap.Ds[2] > phl->bap.Ds[0]){
      phl->bap.score_type = _SCR_TYPE_SIG_ASC_; // ascending sigmoid
    }

    // check whether products and weights are consistent
    if (phl->bap.w.c > 0 && 
      !vector_contains((const char**)phl->sen.aux_products, phl->sen.n_aux_products, "DST")){
      printf("A cloud distance weight is given, but DST auxiliary product is not specified.\n");
      return FAILURE;
    }
    if (phl->bap.w.h > 0 && 
      !vector_contains((const char**)phl->sen.aux_products, phl->sen.n_aux_products, "HOT")){
      printf("A haze weight is given, but HOT auxiliary product is not specified.\n");
      return FAILURE;
    }
    if (phl->bap.w.v > 0 && 
      !vector_contains((const char**)phl->sen.aux_products, phl->sen.n_aux_products, "VZN")){
      printf("A view zenith weight is given, but VZN auxiliary product is specified.\n");
      return FAILURE;
    }

    if (fequal(phl->bap.w.c, 0) && 
    vector_contains((const char**)phl->sen.aux_products, phl->sen.n_aux_products, "DST")){
      printf("A cloud distance weight is not given, but DST auxiliary product is specified.\n");
      return FAILURE;
    }
    if (fequal(phl->bap.w.h, 0) && 
    vector_contains((const char**)phl->sen.aux_products, phl->sen.n_aux_products, "HOT")){
      printf("A haze weight is not given, but HOT auxiliary product is specified.\n");
      return FAILURE;
    }
    if (fequal(phl->bap.w.v, 0) && 
    vector_contains((const char**)phl->sen.aux_products, phl->sen.n_aux_products, "VZN")){
      printf("A view zenith weight is not given, but VZN auxiliary product is specified.\n");
      return FAILURE;
    }


    if (phl->bap.pac.lsp) phl->input_level2 = _INP_CON_;

  }


  // some more checks

  if (phl->ntx != 2){
    printf("X_TILE_RANGE needs to be a list of 2 integers.\n"); return FAILURE;}
  if (phl->nty != 2){
    printf("Y_TILE_RANGE needs to be a list of 2 integers.\n"); return FAILURE;}
  if (phl->tx[_MIN_] > phl->tx[_MAX_]){
    tmp = phl->tx[_MIN_]; phl->tx[_MIN_] = phl->tx[_MAX_]; phl->tx[_MAX_] = tmp;}
  if (phl->ty[_MIN_] > phl->ty[_MAX_]){
    tmp = phl->ty[_MIN_]; phl->ty[_MIN_] = phl->ty[_MAX_]; phl->ty[_MAX_] = tmp;}

  if (strcmp(phl->d_mask, "NULL") != 0 && strcmp(phl->b_mask, "NULL") == 0){
    printf("BASE_MASK cannot be NULL if DIR_MASK is given.\n"); return FAILURE;}

  if (phl->input_level1 == _INP_QAI_ ||
      phl->input_level1 == _INP_ARD_){
    if (phl->ndate != 2){
      printf("DATE_RANGE needs to be a list of 2 dates.\n"); return FAILURE;}
    if (phl->ndoy != 2){
      printf("DOY_RANGE needs to be a list of 2 DOYs.\n"); return FAILURE;}
  }
  
  if (phl->type == _HL_LSM_){
    if (phl->lsm.nthreshold != phl->ftr.nfeature){
      printf("LSM_THRESHOLD needs as many values as there are FEATURES.\n"); return FAILURE;}
    if (phl->lsm.nquery != phl->ftr.nfeature){
      printf("LSM_THRESHOLD_TYPE needs as many values as there are FEATURES.\n"); return FAILURE;}
  }

  if (phl->type == _HL_TSA_){

    if (phl->tsa.sma.orms && !phl->tsa.sma.v){
      phl->tsa.sma.orms = false;
      printf("Warning: will not output SMA RMSE as INDEX does not contain SMA. Proceed.\n");}
    if (phl->tsa.sma.v && !phl->tsa.sma.pos && !phl->tsa.sma.sto){
       printf("SMA_NON_NEG and SMA_SUM_TO_ONE cannot both be negative.\n"); return FAILURE;}

    if ((strcmp(phl->tsa.sma.f_emb, "NULL") == 0) && phl->tsa.sma.v){
     printf("FILE_ENDMEM cannot be NULL if INDEX = SMA."); return FAILURE;}


    if (phl->tsa.tsi.onrt && phl->tsa.tsi.method != _INT_HARMONIC_){
      phl->tsa.tsi.onrt = false;
      printf("Warning: will not output NRT product as interpolation method is not harmonic. Proceed.\n");
    }

    if (phl->tsa.tsi.onrt && 
        phl->tsa.tsi.harm_fit_range[_MIN_].ce == phl->tsa.tsi.harm_fit_range[_MAX_].ce){
      phl->tsa.tsi.onrt = false;
      printf("Warning: will not output NRT product as harmonic does not include a forecast period. Proceed.\n");
    }

    if (phl->tsa.tsi.harm_fit_range[_MIN_].ce == phl->tsa.tsi.harm_fit_range[_MAX_].ce){
      set_date(&phl->tsa.tsi.harm_fit_range[_MIN_], 1900,  1,  1);
      set_date(&phl->tsa.tsi.harm_fit_range[_MAX_], 2100, 12, 31);
    }
 
    if (phl->tsa.pol.opct || phl->tsa.pol.opol || phl->tsa.pol.otrd || phl->tsa.pol.ocat){
          
      if (phl->tsa.pol.ns < 1){
        printf("POL cannot be estimated for one year.\n");
        printf("Time window is too short.\n");
        return FAILURE;
      }

      if (phl->tsa.tsi.method == _INT_NONE_){
        printf("Polarmetrics require INTERPOLATE != NONE\n"); return FAILURE;}

    }


    if (phl->tsa.pyp.out && strcmp(phl->tsa.pyp.f_code, "NULL") == 0){
      phl->tsa.pyp.out = false;
      printf("Warning: no python code provided. OUTPUT_PYP ignored. Proceed.\n");}

    if (!phl->tsa.pyp.out && strcmp(phl->tsa.pyp.f_code, "NULL") != 0){
      copy_string(phl->tsa.pyp.f_code, NPOW_10, "NULL");
      printf("Warning: python code provided, but OUTPUT_PYP = FALSE. Ignore Python UDF plug-in. Proceed.\n");}

    if (phl->tsa.rsp.out && strcmp(phl->tsa.rsp.f_code, "NULL") == 0){
      phl->tsa.rsp.out = false;
      printf("Warning: no R code provided. OUTPUT_RSP ignored. Proceed.\n");}

    if (!phl->tsa.rsp.out && strcmp(phl->tsa.rsp.f_code, "NULL") != 0){
      copy_string(phl->tsa.rsp.f_code, NPOW_10, "NULL");
      printf("Warning: R code provided, but OUTPUT_RSP = FALSE. Ignore R UDF plug-in. Proceed.\n");}

    if (phl->tsa.rsp.out && phl->stream){
      phl->stream = false;
      printf("Warning: R code provided. STREAMING is forced to FALSE. Proceed.\n");
    }

  }

  if (phl->type == _HL_UDF_){

    if (phl->udf.pyp.out && strcmp(phl->udf.pyp.f_code, "NULL") == 0){
      phl->udf.pyp.out = false;
      printf("Warning: no python code provided. OUTPUT_PYP ignored. Proceed.\n");}

    if (!phl->udf.pyp.out && strcmp(phl->udf.pyp.f_code, "NULL") != 0){
      copy_string(phl->udf.pyp.f_code, NPOW_10, "NULL");
      printf("Warning: python code provided, but OUTPUT_PYP = FALSE. Ignore Python UDF plug-in. Proceed.\n");}

    if (phl->udf.rsp.out && strcmp(phl->udf.rsp.f_code, "NULL") == 0){
      phl->udf.rsp.out = false;
      printf("Warning: no R code provided. OUTPUT_RSP ignored. Proceed.\n");}

    if (!phl->udf.rsp.out && strcmp(phl->udf.rsp.f_code, "NULL") != 0){
      copy_string(phl->udf.rsp.f_code, NPOW_10, "NULL");
      printf("Warning: R code provided, but OUTPUT_RSP = FALSE. Ignore R UDF plug-in. Proceed.\n");}

    if (phl->udf.rsp.out && phl->stream){
      phl->stream = false;
      printf("Warning: R code provided. STREAMING is forced to FALSE. Proceed.\n");
    }

  }

  if (phl->type == _HL_CFI_){
    for (y=0; y<phl->cfi.nyears; y++){
      if (phl->cfi.years[y] < phl->date_range[_MIN_].year || 
          phl->cfi.years[y] > phl->date_range[_MAX_].year){
        printf("The prediction year %d (COARSE_PREDICT_YEARS) is outside of DATE_RANGE. This won't work.\n", phl->cfi.years[y]); return FAILURE;}
    }
  }

  if (phl->type == _HL_ML_){
    if (phl->mcl.orfp && phl->mcl.method != _ML_RFC_){
      phl->mcl.orfp = false;
      printf("Random Forest Class Probabilities cannot be computed. Ignored and continue.\n");
    }
    if (phl->mcl.orfm && phl->mcl.method != _ML_RFC_){
      phl->mcl.orfm = false;
      printf("Random Forest Classifcation Margin cannot be computed. Ignored and continue.\n");
    }
  }

  if (phl->format != _FMT_CUSTOM_){
    default_gdaloptions(phl->format, &phl->gdalopt);
  } else {
    if (strcmp(phl->f_gdalopt, "NULL") == 0 || !fileexist(phl->f_gdalopt)){
      printf("If OUTPUT_FORMAT = CUSTOM, FILE_OUTPUT_OPTIONS needs to be given. "); 
      return FAILURE;
    } else {
      parse_gdaloptions(phl->f_gdalopt, &phl->gdalopt);
    }
  }


  return SUCCESS;
}

