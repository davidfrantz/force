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
int check_bandlist(par_tsa_t *tsa, par_sen_t *sen);
void alloc_ftr(par_ftr_t *ftr);
void free_ftr(par_ftr_t *ftr);
void alloc_mcl(par_mcl_t *mcl);
void free_mcl(par_mcl_t *mcl);
int parse_ftr(par_ftr_t *ftr);
int parse_sta(par_sta_t *sta);
int parse_lsp(par_lsp_t *lsp);
int parse_txt(par_txt_t *txt);
int parse_lsm(par_lsm_t *lsm);
int parse_quality(par_qai_t *qai);
int parse_sensor(par_sen_t *sen);


/** This function registers common higher level parameters that are parsed
+++ from the parameter file.
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_higher(params_t *params, par_hl_t *phl){
par_enum_t format[_FMT_LENGTH_] = {
  { _FMT_ENVI_, "ENVI" }, { _FMT_GTIFF_, "GTiff" }};


  register_char_par(params,    "DIR_LOWER",  _CHAR_TEST_EXIST_,         &phl->d_lower);
  register_char_par(params,    "DIR_HIGHER", _CHAR_TEST_EXIST_,         &phl->d_higher);
  register_char_par(params,    "DIR_MASK",   _CHAR_TEST_NULL_OR_EXIST_, &phl->d_mask);
  register_char_par(params,    "BASE_MASK",  _CHAR_TEST_NULL_OR_BASE_,  &phl->b_mask);
  register_char_par(params,    "FILE_TILE",  _CHAR_TEST_NULL_OR_EXIST_, &phl->f_tile);
  register_intvec_par(params,  "X_TILE_RANGE", -999, 9999, &phl->tx, &phl->ntx);
  register_intvec_par(params,  "Y_TILE_RANGE", -999, 9999, &phl->ty, &phl->nty);
  register_double_par(params,  "RESOLUTION", 0, FLT_MAX, &phl->res);
  register_double_par(params,  "BLOCK_SIZE", 0, FLT_MAX, &phl->blocksize);
  register_enum_par(params,    "OUTPUT_FORMAT", format, _FMT_LENGTH_, &phl->format);
  //register_bool_par(params,    "OUTPUT_OVERWRITE", &phl->owr);
  register_int_par(params,     "NTHREAD_READ",    1, INT_MAX, &phl->ithread);
  register_int_par(params,     "NTHREAD_WRITE",   1, INT_MAX, &phl->othread);
  register_int_par(params,     "NTHREAD_COMPUTE", 1, INT_MAX, &phl->cthread);

  return;
}


/** This function registers ARD parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_ard1(params_t *params, par_hl_t *phl){
par_enum_t sensor[_SEN_LENGTH_] = {
  { _SEN_LND04_, "LND04" }, { _SEN_LND05_, "LND05" },
  { _SEN_LND07_, "LND07" }, { _SEN_LND08_, "LND08" },
  { _SEN_SEN2A_, "SEN2A" }, { _SEN_SEN2B_, "SEN2B" },
  { _SEN_sen2a_, "sen2a" }, { _SEN_sen2b_, "sen2b" },
  { _SEN_LNDLG_, "LNDLG" }, { _SEN_SEN2L_, "SEN2L" },
  { _SEN_SEN2H_, "SEN2H" }, { _SEN_RGB_,   "R-G-B" },
  { _SEN_S1AIA_, "S1AIA" }, { _SEN_S1AID_, "S1AID" },
  { _SEN_S1BIA_, "S1BIA" }, { _SEN_S1BID_, "S1BID" }};
par_enum_t qai[_QAI_LENGTH_] = {
  { _QAI_OFF_,      "NODATA"       }, { _QAI_CLD_OPQ_,  "CLOUD_OPAQUE" },
  { _QAI_CLD_UNC_,  "CLOUD_BUFFER" }, { _QAI_CLD_CIR_,  "CLOUD_CIRRUS" },
  { _QAI_SHD_,      "CLOUD_SHADOW" }, { _QAI_SNW_,      "SNOW"         },
  { _QAI_WTR_,      "WATER"        }, { _QAI_AOD_FILL_, "AOD_FILL"     },
  { _QAI_AOD_HIGH_, "AOD_HIGH"     }, { _QAI_AOD_INT_,  "AOD_INT"      },
  { _QAI_SUB_,      "SUBZERO"      }, { _QAI_SAT_,      "SATURATION"   },
  { _QAI_SUN_,      "SUN_LOW"      }, { _QAI_ILL_SHD_,  "ILLUMIN_NONE" },
  { _QAI_ILL_POOR_, "ILLUMIN_POOR" }, { _QAI_ILL_LOW_,  "ILLUMIN_LOW"  },
  { _QAI_SLP_,      "SLOPED"       }, { _QAI_WVP_,      "WVP_NONE"     }};

  
  register_enumvec_par(params, "SENSORS", sensor, _SEN_LENGTH_, &phl->sen.senid, &phl->sen.n);
  register_enumvec_par(params, "SCREEN_QAI", qai, _QAI_LENGTH_, &phl->qai.flags, &phl->qai.nflags);
  register_datevec_par(params, "DATE_RANGE", "1900-01-01", "2099-12-31", &phl->date_range, &phl->ndate);
  register_intvec_par(params,  "DOY_RANGE", 1, 365, &phl->doy_range, &phl->ndoy);

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

  register_floatvec_par(params, "DOY_SCORE", 0, 1, &phl->bap.Ds, &phl->bap.nDs);
  register_intvec_par(params,   "DOY_STATIC", 1, 365, &phl->bap.Dt, &phl->bap.nDt);

  register_bool_par(params,  "OFF_SEASON", &phl->bap.offsea);
  register_float_par(params, "DREQ", 1, FLT_MAX, &phl->bap.dreq);
  register_float_par(params, "VREQ", 1, 90, &phl->bap.vreq);

  register_bool_par(params,   "OUTPUT_BAP", &phl->bap.obap);
  register_bool_par(params,   "OUTPUT_INF", &phl->bap.oinf);
  register_bool_par(params,   "OUTPUT_SCR", &phl->bap.oscr);
  register_bool_par(params,   "OUTPUT_OVV", &phl->bap.oovv);

  register_float_par(params, "SCORE_DOY_WEIGHT",    0, 1, &phl->bap.w.d);
  register_float_par(params, "SCORE_YEAR_WEIGHT",   0, 1, &phl->bap.w.y);
  register_float_par(params, "SCORE_CLOUD_WEIGHT",  0, 1, &phl->bap.w.c);
  register_float_par(params, "SCORE_HAZE_WEIGHT",   0, 1, &phl->bap.w.h);
  register_float_par(params, "SCORE_CORREL_WEIGHT", 0, 1, &phl->bap.w.r);
  register_float_par(params, "SCORE_VZEN_WEIGHT",   0, 1, &phl->bap.w.v);

  register_char_par(params,    "DIR_LSP", _CHAR_TEST_NULL_OR_EXIST_, &phl->con.dname);
  register_charvec_par(params, "BASE_LSP", _CHAR_TEST_NULL_OR_BASE_, &phl->con.fname, &phl->con.n);
  register_int_par(params,     "LSP_NODATA", INT_MIN, INT_MAX, &phl->con.nodata);
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
par_enum_t index[_IDX_LENGTH_] = {
  { _IDX_BLU_, "BLUE"   }, { _IDX_GRN_, "GREEN"     }, { _IDX_RED_, "RED"      },
  { _IDX_NIR_, "NIR"    }, { _IDX_SW1_, "SWIR1"     }, { _IDX_SW2_, "SWIR2"    },
  { _IDX_RE1_, "RE1"    }, { _IDX_RE2_, "RE2"       }, { _IDX_RE3_, "RE3"      },
  { _IDX_BNR_, "BNIR"   }, { _IDX_NDV_, "NDVI"      }, { _IDX_EVI_, "EVI"      },
  { _IDX_NBR_, "NBR"    }, { _IDX_ARV_, "ARVI"      }, { _IDX_SAV_, "SAVI"     },
  { _IDX_SRV_, "SARVI"  }, { _IDX_TCB_, "TC-BRIGHT" }, { _IDX_TCG_, "TC-GREEN" },
  { _IDX_TCW_, "TC-WET" }, { _IDX_TCD_, "TC-DI"     }, { _IDX_NDB_, "NDBI"     },
  { _IDX_NDW_, "NDWI"   }, { _IDX_MNW_, "MNDWI"     }, { _IDX_NDS_, "NDSI"     },
  { _IDX_SMA_, "SMA"    }, { _IDX_BVV_, "VV"        }, { _IDX_BVH_, "VH"       }};
par_enum_t interpol[_INT_LENGTH_] = {
  { _INT_NONE_,   "NONE"   }, { _INT_LINEAR_, "LINEAR" },
  { _INT_MOVING_, "MOVING" }, { _INT_RBF_,    "RBF"    }};
par_enum_t stats[_STA_LENGTH_] = {
  { _STA_MIN_, "MIN" }, { _STA_MAX_, "MAX" }, { _STA_RNG_, "RNG" }, { _STA_IQR_, "IQR" },
  { _STA_AVG_, "AVG" }, { _STA_STD_, "STD" }, { _STA_SKW_, "SKW" }, { _STA_KRT_, "KRT" },
  { _STA_Q01_, "Q01" }, { _STA_Q02_, "Q02" }, { _STA_Q03_, "Q03" }, { _STA_Q04_, "Q04" },
  { _STA_Q05_, "Q05" }, { _STA_Q06_, "Q06" }, { _STA_Q07_, "Q07" }, { _STA_Q08_, "Q08" },
  { _STA_Q09_, "Q09" }, { _STA_Q10_, "Q10" }, { _STA_Q11_, "Q11" }, { _STA_Q12_, "Q12" },
  { _STA_Q13_, "Q13" }, { _STA_Q14_, "Q14" }, { _STA_Q15_, "Q15" }, { _STA_Q16_, "Q16" },
  { _STA_Q17_, "Q17" }, { _STA_Q18_, "Q18" }, { _STA_Q19_, "Q19" }, { _STA_Q20_, "Q20" },
  { _STA_Q21_, "Q21" }, { _STA_Q22_, "Q22" }, { _STA_Q23_, "Q23" }, { _STA_Q24_, "Q24" },
  { _STA_Q25_, "Q25" }, { _STA_Q26_, "Q26" }, { _STA_Q27_, "Q27" }, { _STA_Q28_, "Q28" },
  { _STA_Q29_, "Q29" }, { _STA_Q30_, "Q30" }, { _STA_Q31_, "Q31" }, { _STA_Q32_, "Q32" },
  { _STA_Q33_, "Q33" }, { _STA_Q34_, "Q34" }, { _STA_Q35_, "Q35" }, { _STA_Q36_, "Q36" },
  { _STA_Q37_, "Q37" }, { _STA_Q38_, "Q38" }, { _STA_Q39_, "Q39" }, { _STA_Q40_, "Q40" },
  { _STA_Q41_, "Q41" }, { _STA_Q42_, "Q42" }, { _STA_Q43_, "Q43" }, { _STA_Q44_, "Q44" },
  { _STA_Q45_, "Q45" }, { _STA_Q46_, "Q46" }, { _STA_Q47_, "Q47" }, { _STA_Q48_, "Q48" },
  { _STA_Q49_, "Q49" }, { _STA_Q50_, "Q50" }, { _STA_Q51_, "Q51" }, { _STA_Q52_, "Q52" },
  { _STA_Q53_, "Q53" }, { _STA_Q54_, "Q54" }, { _STA_Q55_, "Q55" }, { _STA_Q56_, "Q56" },
  { _STA_Q57_, "Q57" }, { _STA_Q58_, "Q58" }, { _STA_Q59_, "Q59" }, { _STA_Q60_, "Q60" },
  { _STA_Q61_, "Q61" }, { _STA_Q62_, "Q62" }, { _STA_Q63_, "Q63" }, { _STA_Q64_, "Q64" },
  { _STA_Q65_, "Q65" }, { _STA_Q66_, "Q66" }, { _STA_Q67_, "Q67" }, { _STA_Q68_, "Q68" },
  { _STA_Q69_, "Q69" }, { _STA_Q70_, "Q70" }, { _STA_Q71_, "Q71" }, { _STA_Q72_, "Q72" },
  { _STA_Q73_, "Q73" }, { _STA_Q74_, "Q74" }, { _STA_Q75_, "Q75" }, { _STA_Q76_, "Q76" },
  { _STA_Q77_, "Q77" }, { _STA_Q78_, "Q78" }, { _STA_Q79_, "Q79" }, { _STA_Q80_, "Q80" },
  { _STA_Q81_, "Q81" }, { _STA_Q82_, "Q82" }, { _STA_Q83_, "Q83" }, { _STA_Q84_, "Q84" },
  { _STA_Q85_, "Q85" }, { _STA_Q86_, "Q86" }, { _STA_Q87_, "Q87" }, { _STA_Q88_, "Q88" },
  { _STA_Q89_, "Q89" }, { _STA_Q90_, "Q90" }, { _STA_Q91_, "Q91" }, { _STA_Q92_, "Q92" },
  { _STA_Q93_, "Q93" }, { _STA_Q94_, "Q94" }, { _STA_Q95_, "Q95" }, { _STA_Q96_, "Q96" },
  { _STA_Q97_, "Q97" }, { _STA_Q98_, "Q98" }, { _STA_Q99_, "Q99" }, { _STA_NUM_, "NUM" }};
par_enum_t hemi[_HEMI_LENGTH_] = {
  { _HEMI_NORTH_, "NORTH" }, { _HEMI_SOUTH_, "SOUTH" }, { _HEMI_MIXED_, "MIXED" }};
par_enum_t lsp[_LSP_LENGTH_] = {
  {_LSP_DEM_, "DEM" }, {_LSP_DSS_, "DSS" }, {_LSP_DRI_, "DRI" }, {_LSP_DPS_, "DPS" },
  {_LSP_DFI_, "DFI" }, {_LSP_DES_, "DES" }, {_LSP_DLM_, "DLM" }, {_LSP_LTS_, "LTS" },
  {_LSP_LGS_, "LGS" }, {_LSP_VEM_, "VEM" }, {_LSP_VSS_, "VSS" }, {_LSP_VRI_, "VRI" },
  {_LSP_VPS_, "VPS" }, {_LSP_VFI_, "VFI" }, {_LSP_VES_, "VES" }, {_LSP_VLM_, "VLM" },
  {_LSP_VBL_, "VBL" }, {_LSP_VSA_, "VSA" }, {_LSP_IST_, "IST" }, {_LSP_IBL_, "IBL" },
  {_LSP_IBT_, "IBT" }, {_LSP_IGS_, "IGS" }, {_LSP_RAR_, "RAR" }, {_LSP_RAF_, "RAF" },
  {_LSP_RMR_, "RMR" }, {_LSP_RMF_, "RMF" }};
par_enum_t tail[_TAIL_LENGTH_] = {
  { _TAIL_LEFT_, "LEFT" }, { _TAIL_TWO_, "TWO" }, { _TAIL_RIGHT_, "RIGHT" }};
par_enum_t standard[_STD_LENGTH_] = {
  { _STD_NONE_, "NONE" }, {_STD_NORMAL_, "NORMALIZE"}, {_STD_CENTER_, "CENTER" }};

  // TS parameters
  register_enumvec_par(params, "INDEX",      index, _IDX_LENGTH_, &phl->tsa.index, &phl->tsa.n);
  register_enum_par(params,    "STANDARDIZE_TSS", standard, _STD_LENGTH_, &phl->tsa.standard);
  register_bool_par(params,    "OUTPUT_TSS", &phl->tsa.otss);

  // SMA parameters
  register_char_par(params, "FILE_ENDMEM",    _CHAR_TEST_NULL_OR_EXIST_, &phl->tsa.sma.f_emb);
  register_bool_par(params, "SMA_SUM_TO_ONE", &phl->tsa.sma.sto);
  register_bool_par(params, "SMA_NON_NEG",    &phl->tsa.sma.pos);
  register_bool_par(params, "SMA_SHD_NORM",   &phl->tsa.sma.shn);
  register_int_par(params,  "SMA_ENDMEMBER",  0, INT_MAX, &phl->tsa.sma.emb);
  register_bool_par(params, "OUTPUT_RMS",     &phl->tsa.sma.orms);

  // interpolation parameters
  register_enum_par(params,   "INTERPOLATE", interpol, _INT_LENGTH_, &phl->tsa.tsi.method);
  register_int_par(params,    "MOVING_MAX",  1, 365, &phl->tsa.tsi.mov_max);
  register_intvec_par(params, "RBF_SIGMA",   1, 365, &phl->tsa.tsi.rbf_sigma, &phl->tsa.tsi.rbf_nk);
  register_float_par(params,  "RBF_CUTOFF",  0, 1, &phl->tsa.tsi.rbf_cutoff);
  register_int_par(params,    "INT_DAY",     1, INT_MAX, &phl->tsa.tsi.step);
  register_enum_par(params,   "STANDARDIZE_TSI", standard, _STD_LENGTH_, &phl->tsa.tsi.standard);
  register_bool_par(params,   "OUTPUT_TSI",  &phl->tsa.tsi.otsi);

  // STM parameters
  register_enumvec_par(params, "STM",        stats, _STA_LENGTH_, &phl->tsa.stm.sta.metrics, &phl->tsa.stm.sta.nmetrics);
  register_bool_par(params,    "OUTPUT_STM", &phl->tsa.stm.ostm);

  // folding parameters
  register_enum_par(params, "FOLD_TYPE",  stats, _STA_LENGTH_, &phl->tsa.fld.type);
  register_enum_par(params, "STANDARDIZE_FOLD", standard, _STD_LENGTH_, &phl->tsa.fld.standard);
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

  // phenology parameters
  register_int_par(params,     "LSP_DOY_PREV_YEAR", 1, 365, &phl->tsa.lsp.dprev);
  register_int_par(params,     "LSP_DOY_NEXT_YEAR", 1, 365, &phl->tsa.lsp.dnext);
  register_enum_par(params,    "LSP_HEMISPHERE",    hemi, _HEMI_LENGTH_, &phl->tsa.lsp.hemi);
  register_int_par(params,     "LSP_N_SEGMENT",     1, INT_MAX, &phl->tsa.lsp.nseg);
  register_float_par(params,   "LSP_AMP_THRESHOLD", 0.01, 0.99, &phl->tsa.lsp.start);
  register_float_par(params,   "LSP_MIN_VALUE",     0, 10000, &phl->tsa.lsp.minval);
  register_float_par(params,   "LSP_MIN_AMPLITUDE", 0, 10000, &phl->tsa.lsp.minamp);
  register_enumvec_par(params, "LSP",               lsp, _LSP_LENGTH_, &phl->tsa.lsp.metrics, &phl->tsa.lsp.nmetrics);
  register_enum_par(params,    "STANDARDIZE_LSP",   standard, _STD_LENGTH_, &phl->tsa.lsp.standard);
  register_bool_par(params,    "OUTPUT_SPL",        &phl->tsa.lsp.ospl);
  register_bool_par(params,    "OUTPUT_LSP",        &phl->tsa.lsp.olsp);
  register_bool_par(params,    "OUTPUT_TRP",        &phl->tsa.lsp.otrd);
  register_bool_par(params,    "OUTPUT_CAP",        &phl->tsa.lsp.ocat);

  // trend parameters
  register_enum_par(params,  "TREND_TAIL", tail, _TAIL_LENGTH_, &phl->tsa.trd.tail);
  register_float_par(params, "TREND_CONF", 0, 1, &phl->tsa.trd.conf);


  return;
}


/** This function registers CSO parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_cso(params_t *params, par_hl_t *phl){
par_enum_t stats[_STA_LENGTH_] = {
  { _STA_MIN_, "MIN" }, { _STA_MAX_, "MAX" }, { _STA_RNG_, "RNG" }, { _STA_IQR_, "IQR" },
  { _STA_AVG_, "AVG" }, { _STA_STD_, "STD" }, { _STA_SKW_, "SKW" }, { _STA_KRT_, "KRT" },
  { _STA_Q01_, "Q01" }, { _STA_Q02_, "Q02" }, { _STA_Q03_, "Q03" }, { _STA_Q04_, "Q04" },
  { _STA_Q05_, "Q05" }, { _STA_Q06_, "Q06" }, { _STA_Q07_, "Q07" }, { _STA_Q08_, "Q08" },
  { _STA_Q09_, "Q09" }, { _STA_Q10_, "Q10" }, { _STA_Q11_, "Q11" }, { _STA_Q12_, "Q12" },
  { _STA_Q13_, "Q13" }, { _STA_Q14_, "Q14" }, { _STA_Q15_, "Q15" }, { _STA_Q16_, "Q16" },
  { _STA_Q17_, "Q17" }, { _STA_Q18_, "Q18" }, { _STA_Q19_, "Q19" }, { _STA_Q20_, "Q20" },
  { _STA_Q21_, "Q21" }, { _STA_Q22_, "Q22" }, { _STA_Q23_, "Q23" }, { _STA_Q24_, "Q24" },
  { _STA_Q25_, "Q25" }, { _STA_Q26_, "Q26" }, { _STA_Q27_, "Q27" }, { _STA_Q28_, "Q28" },
  { _STA_Q29_, "Q29" }, { _STA_Q30_, "Q30" }, { _STA_Q31_, "Q31" }, { _STA_Q32_, "Q32" },
  { _STA_Q33_, "Q33" }, { _STA_Q34_, "Q34" }, { _STA_Q35_, "Q35" }, { _STA_Q36_, "Q36" },
  { _STA_Q37_, "Q37" }, { _STA_Q38_, "Q38" }, { _STA_Q39_, "Q39" }, { _STA_Q40_, "Q40" },
  { _STA_Q41_, "Q41" }, { _STA_Q42_, "Q42" }, { _STA_Q43_, "Q43" }, { _STA_Q44_, "Q44" },
  { _STA_Q45_, "Q45" }, { _STA_Q46_, "Q46" }, { _STA_Q47_, "Q47" }, { _STA_Q48_, "Q48" },
  { _STA_Q49_, "Q49" }, { _STA_Q50_, "Q50" }, { _STA_Q51_, "Q51" }, { _STA_Q52_, "Q52" },
  { _STA_Q53_, "Q53" }, { _STA_Q54_, "Q54" }, { _STA_Q55_, "Q55" }, { _STA_Q56_, "Q56" },
  { _STA_Q57_, "Q57" }, { _STA_Q58_, "Q58" }, { _STA_Q59_, "Q59" }, { _STA_Q60_, "Q60" },
  { _STA_Q61_, "Q61" }, { _STA_Q62_, "Q62" }, { _STA_Q63_, "Q63" }, { _STA_Q64_, "Q64" },
  { _STA_Q65_, "Q65" }, { _STA_Q66_, "Q66" }, { _STA_Q67_, "Q67" }, { _STA_Q68_, "Q68" },
  { _STA_Q69_, "Q69" }, { _STA_Q70_, "Q70" }, { _STA_Q71_, "Q71" }, { _STA_Q72_, "Q72" },
  { _STA_Q73_, "Q73" }, { _STA_Q74_, "Q74" }, { _STA_Q75_, "Q75" }, { _STA_Q76_, "Q76" },
  { _STA_Q77_, "Q77" }, { _STA_Q78_, "Q78" }, { _STA_Q79_, "Q79" }, { _STA_Q80_, "Q80" },
  { _STA_Q81_, "Q81" }, { _STA_Q82_, "Q82" }, { _STA_Q83_, "Q83" }, { _STA_Q84_, "Q84" },
  { _STA_Q85_, "Q85" }, { _STA_Q86_, "Q86" }, { _STA_Q87_, "Q87" }, { _STA_Q88_, "Q88" },
  { _STA_Q89_, "Q89" }, { _STA_Q90_, "Q90" }, { _STA_Q91_, "Q91" }, { _STA_Q92_, "Q92" },
  { _STA_Q93_, "Q93" }, { _STA_Q94_, "Q94" }, { _STA_Q95_, "Q95" }, { _STA_Q96_, "Q96" },
  { _STA_Q97_, "Q97" }, { _STA_Q98_, "Q98" }, { _STA_Q99_, "Q99" }, { _STA_NUM_, "NUM" }};

  
  register_int_par(params,     "MONTH_STEP", 1, 12, &phl->cso.step);
  register_enumvec_par(params, "CSO",        stats, _STA_LENGTH_, &phl->cso.sta.metrics, &phl->cso.sta.nmetrics);

  return;
}


/** This function registers ImproPhe parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_imp(params_t *params, par_hl_t *phl){


  register_intvec_par(params,  "SEASONAL_WINDOW",      1, 365, &phl->imp.dwin, &phl->imp.bwin);
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
  register_charvec_par(params, "BASE_COARSE", _CHAR_TEST_BASE_, &phl->con.fname, &phl->con.n);
  register_int_par(params,     "COARSE_NODATA", INT_MIN, INT_MAX, &phl->con.nodata);
  register_int_par(params,     "COARSE_1ST_YEAR",      1900, 2100, &phl->cfi.y0);
  register_intvec_par(params,  "COARSE_PREDICT_YEARS", 1900, 2100, &phl->cfi.years, &phl->cfi.nyears);

  return;
}


/** This function registers L2 ImproPhe parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_l2i(params_t *params, par_hl_t *phl){
par_enum_t sensor[_SEN_LENGTH_] = {
  { _SEN_LND04_, "LND04" }, { _SEN_LND05_, "LND05" },
  { _SEN_LND07_, "LND07" }, { _SEN_LND08_, "LND08" },
  { _SEN_SEN2A_, "SEN2A" }, { _SEN_SEN2B_, "SEN2B" },
  { _SEN_sen2a_, "sen2a" }, { _SEN_sen2b_, "sen2b" },
  { _SEN_LNDLG_, "LNDLG" }, { _SEN_SEN2L_, "SEN2L" },
  { _SEN_SEN2H_, "SEN2H" }, { _SEN_RGB_,   "R-G-B" },
  { _SEN_S1AIA_, "S1AIA" }, { _SEN_S1AID_, "S1AID" },
  { _SEN_S1BIA_, "S1BIA" }, { _SEN_S1BID_, "S1BID" }};

  
  register_imp(params, phl);
  register_enumvec_par(params, "SENSORS_LOWRES", sensor, _SEN_LENGTH_, &phl->sen2.senid, &phl->sen2.n);

  return;
}


/** This function registers machine learning parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_mcl(params_t *params, par_hl_t *phl){
int i;
par_enum_t method[_ML_LENGTH_] = {
  { _ML_SVR_, "SVR" }, { _ML_SVC_, "SVC" }, { _ML_RFR_, "RFR" }, { _ML_RFC_, "RFC" }};


  register_char_par(params,  "DIR_MODEL", _CHAR_TEST_EXIST_, &phl->mcl.d_model);
  register_enum_par(params,  "ML_METHOD", method, _ML_LENGTH_, &phl->mcl.method);
  register_float_par(params, "ML_CONVERGENCE", 0, INT_MAX, &phl->mcl.converge);
  register_float_par(params, "ML_SCALE", 0, 1e6, &phl->mcl.scale);
  register_char_par(params,  "ML_BASE",  _CHAR_TEST_NONE_, &phl->mcl.base);
  register_bool_par(params,  "OUTPUT_MLP", &phl->mcl.omlp);
  register_bool_par(params,  "OUTPUT_MLI", &phl->mcl.omli);
  register_bool_par(params,  "OUTPUT_MLU", &phl->mcl.omlu);

  for (i=0; i<phl->mcl.nmodelset; i++) register_charvec_par(params,  "FILE_MODEL",
    _CHAR_TEST_BASE_, &phl->mcl.f_model[i], &phl->mcl.nmodel[i]);

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
    _CHAR_TEST_NONE_, &phl->ftr.cfeature[i], &phl->ftr.ifeature[i]);

  register_int_par(params, "FEATURE_NODATA", -32767, 32767, &phl->ftr.nodata);
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
par_enum_t text[_TXT_LENGTH_] = {
  { _TXT_ERO_, "ERO" }, { _TXT_DIL_, "DIL" }, { _TXT_OPN_, "OPN" }, { _TXT_CLS_, "CLS" }, 
  { _TXT_GRD_, "GRD" }, { _TXT_THT_, "THT" }, { _TXT_BHT_, "BHT" }};


  register_double_par(params,  "TXT_RADIUS",    0, 1e6,  &phl->txt.radius);
  register_int_par(params,     "TXT_ITERATION", 1, 1000, &phl->txt.iter);
  register_enumvec_par(params, "TXT", text, _TXT_LENGTH_, &phl->txt.metrics, &phl->txt.nmetrics);
  register_char_par(params,    "TXT_BASE",  _CHAR_TEST_NONE_, &phl->txt.base);

  return;
}


/** This function registers landscape metrics parameters
--- params: registered parameters
--- phl:    HL parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void register_lsm(params_t *params, par_hl_t *phl){
par_enum_t query[_QUERY_LENGTH_] = {
  { _QUERY_EQ_, "EQ" }, { _QUERY_GT_, "GT" }, { _QUERY_LT_, "LT" }};
par_enum_t lsm[_LSM_LENGTH_] = {
  { _LSM_MPA_, "MPA" }, { _LSM_UCI_, "UCI" }, { _LSM_FDI_, "FDI" }, 
  { _LSM_WED_, "WED" }, { _LSM_NBR_, "NBR" }, { _LSM_EMS_, "EMS" }, 
  { _LSM_AVG_, "AVG" }, { _LSM_STD_, "STD" }, { _LSM_GEO_, "GEO" }, 
  { _LSM_MAX_, "MAX" }};
par_enum_t kernel[_KERNEL_LENGTH_] = {
  { _KERNEL_SQUARE_, "SQUARE" }, { _KERNEL_CIRCLE_, "CIRCLE" }};

  register_double_par(params,  "LSM_RADIUS",    0, 1e6,  &phl->lsm.radius);
  register_enumvec_par(params, "LSM_THRESHOLD_TYPE", query, _QUERY_LENGTH_, &phl->lsm.query, &phl->lsm.nquery);
  register_intvec_par(params,  "LSM_THRESHOLD", -32767, 32767, &phl->lsm.threshold, &phl->lsm.nthreshold);
  register_bool_par(params,    "LSM_ALL_PIXELS", &phl->lsm.allpx);
  register_enumvec_par(params, "LSM", lsm, _LSM_LENGTH_, &phl->lsm.metrics, &phl->lsm.nmetrics);
  register_char_par(params,    "LSM_BASE",  _CHAR_TEST_NONE_, &phl->lsm.base);
  register_enum_par(params,    "LSM_KERNEL_SHAPE", kernel, _KERNEL_LENGTH_, &phl->lsm.kernel);

  return;
}


/** This function checks that each index can be computed with the given
+++ set of sensors. It also kicks out unused bands to remove I/O
--- tsa:    TSA parameters
--- sen:    sensor parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int check_bandlist(par_tsa_t *tsa, par_sen_t *sen){
int idx, b, nb = _WVL_LENGTH_, k, s;
bool v[_WVL_LENGTH_] = { 
  false, false, false, false, false,
  false, false, false, false, false,
  false, false };
int *band_ptr[_WVL_LENGTH_] = { 
  &sen->blue, &sen->green, &sen->red,
  &sen->rededge1, &sen->rededge2, &sen->rededge3,
  &sen->bnir, &sen->nir, &sen->swir1, &sen->swir2,
  &sen->vv, &sen->vh };


  alloc_2D((void***)&tsa->index_name, tsa->n, NPOW_02, sizeof(char));

  // for each requested index, flag required wavelength, 
  // set short index name for filename
  for (idx=0; idx<tsa->n; idx++){

    switch (tsa->index[idx]){
      case _IDX_BLU_:
        v[_WVL_BLUE_] = true;
        strncpy(tsa->index_name[idx] , "BLU", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_GRN_:
        v[_WVL_GREEN_] = true;
        strncpy(tsa->index_name[idx] , "GRN", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_RED_:
        v[_WVL_RED_] = true;
        strncpy(tsa->index_name[idx] , "RED", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_NIR_:
        v[_WVL_NIR_] = true;
        strncpy(tsa->index_name[idx] , "NIR", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_SW1_:
        v[_WVL_SWIR1_] = true;
        strncpy(tsa->index_name[idx] , "SW1", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_SW2_:
        v[_WVL_SWIR2_] = true;
        strncpy(tsa->index_name[idx] , "SW2", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_RE1_:
        v[_WVL_REDEDGE1_] = true;
        strncpy(tsa->index_name[idx] , "RE1", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_RE2_:
        v[_WVL_REDEDGE2_] = true;
        strncpy(tsa->index_name[idx] , "RE2", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_RE3_:
        v[_WVL_REDEDGE3_] = true;
        strncpy(tsa->index_name[idx] , "RE3", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_BNR_:
        v[_WVL_BNIR_] = true;
        strncpy(tsa->index_name[idx] , "BNR", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_NDV_:
        v[_WVL_NIR_] = v[_WVL_RED_] = true;
        strncpy(tsa->index_name[idx] , "NDV", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_EVI_:
        v[_WVL_NIR_] = v[_WVL_RED_] = v[_WVL_BLUE_] = true;
        strncpy(tsa->index_name[idx] , "EVI", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_NBR_:
        v[_WVL_NIR_] = v[_WVL_SWIR2_] = true;
        strncpy(tsa->index_name[idx] , "NBR", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_ARV_:
        v[_WVL_RED_] = v[_WVL_BLUE_] = v[_WVL_NIR_] = true;
        strncpy(tsa->index_name[idx] , "ARV", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_SAV_:
        v[_WVL_NIR_] = v[_WVL_RED_] = true;
        strncpy(tsa->index_name[idx] , "SAV", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_SRV_:
        v[_WVL_RED_] = v[_WVL_BLUE_] = v[_WVL_NIR_] = true;
        strncpy(tsa->index_name[idx] , "SRV", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_TCB_:
        v[_WVL_BLUE_] = v[_WVL_GREEN_] = v[_WVL_RED_]   = true;
        v[_WVL_NIR_]  = v[_WVL_SWIR1_] = v[_WVL_SWIR2_] = true;
        strncpy(tsa->index_name[idx] , "TCB", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_TCG_:
        v[_WVL_BLUE_] = v[_WVL_GREEN_] = v[_WVL_RED_]   = true;
        v[_WVL_NIR_]  = v[_WVL_SWIR1_] = v[_WVL_SWIR2_] = true;
        strncpy(tsa->index_name[idx] , "TCG", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_TCW_:
        v[_WVL_BLUE_] = v[_WVL_GREEN_] = v[_WVL_RED_]   = true;
        v[_WVL_NIR_]  = v[_WVL_SWIR1_] = v[_WVL_SWIR2_] = true;
        strncpy(tsa->index_name[idx] , "TCW", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_TCD_:
        v[_WVL_BLUE_] = v[_WVL_GREEN_] = v[_WVL_RED_]   = true;
        v[_WVL_NIR_]  = v[_WVL_SWIR1_] = v[_WVL_SWIR2_] = true;
        strncpy(tsa->index_name[idx] , "TCD", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_NDB_:
        v[_WVL_SWIR1_] = v[_WVL_NIR_] = true;
        strncpy(tsa->index_name[idx] , "NDB", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_NDW_:
        v[_WVL_GREEN_] = v[_WVL_NIR_] = true;
        strncpy(tsa->index_name[idx] , "NDW", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_MNW_:
        v[_WVL_GREEN_] = v[_WVL_SWIR1_] = true;
        strncpy(tsa->index_name[idx] , "MNW", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_NDS_:
        v[_WVL_GREEN_] = v[_WVL_SWIR1_] = true;
        strncpy(tsa->index_name[idx] , "NDS", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_SMA_:
        for (b=0; b<nb; b++) v[b] = (*band_ptr[b] >= 0);
        strncpy(tsa->index_name[idx] , "SMA", 3); tsa->index_name[idx][3] = '\0';
        tsa->sma.v = true;
        break;
      case _IDX_BVV_:
        v[_WVL_VV_] = true;
        strncpy(tsa->index_name[idx] , "BVV", 3); tsa->index_name[idx][3] = '\0';
        break;
      case _IDX_BVH_:
        v[_WVL_VH_] = true;
        strncpy(tsa->index_name[idx] , "BVH", 3); tsa->index_name[idx][3] = '\0';
        break;
      default:
        printf("unknown INDEX\n");
        break;
    }
  }


  // check if index can be computed with the given sensor combination
  // remove unused bands from each sensor to reduce I/O
  for (b=0; b<nb; b++){
    if (v[b]  && *band_ptr[b] <  0){
      printf("cannot compute index, band is missing (check SENSORS). ");
      return FAILURE;
    }
    if (!v[b] && *band_ptr[b] >= 0){
      for (s=0; s<sen->n; s++){ sen->band[s][*band_ptr[b]] = -1;}; *band_ptr[b] = -1;
    }
  }

  for (b=0, k=0; b<nb; b++){
    if (v[b]) *band_ptr[b] = k++;
  }

  #ifdef FORCE_DEBUG
  printf("filtered bandlist with requested indices:\n");
  for (s=0; s<sen->n; s++){
    printf("%s: ", sen->sensor[s]);
    for (b=0; b<sen->nb; b++) printf("%2d ", sen->band[s][b]); printf("\n");
  }
  #endif

  #ifdef FORCE_DEBUG
  printf("blue  %02d, green %02d, red   %02d\n", sen->blue, sen->green, sen->red);
  printf("re_1  %02d, re_2  %02d, re_3  %02d\n", sen->rededge1, sen->rededge2, sen->rededge3);
  printf("bnir  %02d, nir   %02d, swir1 %02d\n", sen->bnir, sen->nir, sen->swir1);
  printf("swir2 %02d  vv    %02d, vh    %02d\n", sen->swir2, sen->vv, sen->vh);
  #endif

  return SUCCESS;
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

  return;
}


/** This function frees the machine learning parameters
--- mcl:    machine learning parameters
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_mcl(par_mcl_t *mcl){

  free((void*)mcl->f_model); mcl->f_model = NULL;
  free((void*)mcl->nmodel);  mcl->nmodel  = NULL;

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
      if (strlen(ftr->cfeature[i][0]) > NPOW_10-1){
        printf("cannot copy, string too long.\n"); return FAILURE;
      } else {
        strncpy(ftr->bname[k], ftr->cfeature[i][0], strlen(ftr->cfeature[i][0])); 
        ftr->bname[k][strlen(ftr->cfeature[i][0])] = '\0';
      }
      
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


/** This function reparses phenometrics parameters (special para-
+++ meter that cannot be parsed with the general parser).
--- lsp:    phenometrics parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_lsp(par_lsp_t *lsp){
int i;


  for (i=0; i<lsp->nmetrics; i++){
    if (lsp->metrics[i] == _LSP_DEM_){
      lsp->odem = true;
    } else if (lsp->metrics[i] == _LSP_DSS_){
      lsp->odss = true;
    } else if (lsp->metrics[i] == _LSP_DRI_){
      lsp->odri = true;
    } else if (lsp->metrics[i] == _LSP_DPS_){
      lsp->odps = true;
    } else if (lsp->metrics[i] == _LSP_DFI_){
      lsp->odfi = true;
    } else if (lsp->metrics[i] == _LSP_DES_){
      lsp->odes = true;
    } else if (lsp->metrics[i] == _LSP_DLM_){
      lsp->odlm = true;
    } else if (lsp->metrics[i] == _LSP_LTS_){
      lsp->olts = true;
    } else if (lsp->metrics[i] == _LSP_LGS_){
      lsp->olgs = true;
    } else if (lsp->metrics[i] == _LSP_VEM_){
      lsp->ovem = true;
    } else if (lsp->metrics[i] == _LSP_VSS_){
      lsp->ovss = true;
    } else if (lsp->metrics[i] == _LSP_VRI_){
      lsp->ovri = true;
    } else if (lsp->metrics[i] == _LSP_VPS_){
      lsp->ovps = true;
    } else if (lsp->metrics[i] == _LSP_VFI_){
      lsp->ovfi = true;
    } else if (lsp->metrics[i] == _LSP_VES_){
      lsp->oves = true;
    } else if (lsp->metrics[i] == _LSP_VLM_){
      lsp->ovlm = true;
    } else if (lsp->metrics[i] == _LSP_VBL_){
      lsp->ovbl = true;
    } else if (lsp->metrics[i] == _LSP_VSA_){
      lsp->ovsa = true;
    } else if (lsp->metrics[i] == _LSP_IST_){
      lsp->oist = true;
    } else if (lsp->metrics[i] == _LSP_IBL_){
      lsp->oibl = true;
    } else if (lsp->metrics[i] == _LSP_IBT_){
      lsp->oibt = true;
    } else if (lsp->metrics[i] == _LSP_IGS_){
      lsp->oigs = true;
    } else if (lsp->metrics[i] == _LSP_RAR_){
      lsp->orar = true;
    } else if (lsp->metrics[i] == _LSP_RAF_){
      lsp->oraf = true;
    } else if (lsp->metrics[i] == _LSP_RMR_){
      lsp->ormr = true;
    } else if (lsp->metrics[i] == _LSP_RMF_){
      lsp->ormf = true;
    } else {
      printf("warning: unknown lsp.\n");
    }
  }


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
    } else if (lsm->metrics[i] == _LSM_WED_){
      lsm->owed = true;
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


/** This function reparses sensor parameters (special para-
+++ meter that cannot be parsed with the general parser).
+++ This function builds a Level 2 sensor dictionary, which is needed to
+++ generate multi-sensor products. It computes the most restrictive over-
+++ lap between matching bands, determine rules how to read these, and de-
+++ fines commonly used wavelengths.
--- sen:    sensor parameters
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int parse_sensor(par_sen_t *sen){
int s, ss, b, bb, c;
char upper[NPOW_10] = "UPPER";
const int  ns = _SEN_LENGTH_, nb = _WVL_LENGTH_;
const char sensor[_SEN_LENGTH_][NPOW_10] = {
  "LND04", "LND05", "LND07",
  "LND08", "SEN2A", "SEN2B",
  "sen2a", "sen2b", "LNDLG",
  "SEN2L", "SEN2H", "R-G-B",
  "S1AIA", "S1AID", "S1BIA",
  "S1BID", "VVVHP" };
const int  band[_SEN_LENGTH_][_WVL_LENGTH_] = {
  { 1, 2, 3, 0, 0, 0, 0, 4, 5,  6, 0, 0 },  // Landsat 4 TM   (legacy bands)
  { 1, 2, 3, 0, 0, 0, 0, 4, 5,  6, 0, 0 },  // Landsat 5 TM   (legacy bands)
  { 1, 2, 3, 0, 0, 0, 0, 4, 5,  6, 0, 0 },  // Landsat 7 ETM+ (legacy bands)
  { 1, 2, 3, 0, 0, 0, 0, 4, 5,  6, 0, 0 },  // Landsat 8 OLI  (legacy bands)
  { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0 },  // Sentinel-2A MSI (land surface bands)
  { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0 },  // Sentinel-2B MSI (land surface bands)
  { 1, 2, 3, 0, 0, 0, 7, 0, 0,  0, 0, 0 },  // Sentinel-2A MSI (high-res bands)
  { 1, 2, 3, 0, 0, 0, 7, 0, 0,  0, 0, 0 },  // Sentinel-2B MSI (high-res bands)
  { 1, 2, 3, 0, 0, 0, 0, 4, 5,  6, 0, 0 },  // Landsat legacy bands
  { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0 },  // Sentinel-2 land surface bands
  { 1, 2, 3, 0, 0, 0, 0, 4, 5,  6, 0, 0 },  // Sentinel-2 high-res bands
  { 1, 2, 3, 0, 0, 0, 0, 0, 0,  0, 0, 0 },  // VIS bands
  { 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 2 },  // Sentinel-1A IW Ascending
  { 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 2 },  // Sentinel-1A IW Descending
  { 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 2 },  // Sentinel-1B IW Ascending
  { 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 2 },  // Sentinel-1B IW Descending
  { 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 1, 2 }}; // VV/VH polarized
char domains[_WVL_LENGTH_][NPOW_04] = {
  "BLUE", "GREEN", "RED", "REDEDGE1", "REDEDGE2",
  "REDEDGE3", "BROADNIR", "NIR", "SWIR1", "SWIR2",
  "VV", "VH" };
bool vs[_SEN_LENGTH_], vb[_WVL_LENGTH_];
int *band_ptr[_WVL_LENGTH_] = {
  &sen->blue, &sen->green, &sen->red, &sen->rededge1, &sen->rededge2,
  &sen->rededge3, &sen->bnir, &sen->nir, &sen->swir1, &sen->swir2,
  &sen->vv, &sen->vh };


  // match available sensors with requested sensors
  for (s=0; s<ns; s++) vs[s] = false;
  for (s=0; s<sen->n; s++) vs[sen->senid[s]] = true;


  // kick out bands that are incomplete
  for (b=0, bb=0; b<nb; b++){

    for (s=0, vb[b]=true; s<ns; s++){
      if (vs[s] && band[s][b] == 0) vb[b] = false;
    }

    if (vb[b]){
      *band_ptr[b] = bb++;
    } else {
      *band_ptr[b] = -1;
    }

  }

  if ((sen->nb = bb) == 0){ printf("no band overlap for requested sensors.\n"); return FAILURE;}


  // set target sensor
  if (sen->nb == 6){
    strncpy(sen->target, "LNDLG", 5); sen->target[5] = '\0';
  } else if (sen->nb == 10){
    strncpy(sen->target, "SEN2L", 5); sen->target[5] = '\0';
  } else if (sen->nb == 4){
    strncpy(sen->target, "SEN2H", 5); sen->target[5] = '\0';
  } else if (sen->nb == 3){
    strncpy(sen->target, "R-G-B", 5); sen->target[5] = '\0';
  } else if (sen->nb == 2){
    strncpy(sen->target, "VVVHP", 5); sen->target[5] = '\0';
  } else {
    printf("unknown sensors.\n"); return FAILURE;
  }


  // build sensor struct
  alloc_2D((void***)&sen->sensor, sen->n,  NPOW_10, sizeof(char));
  alloc_2D((void***)&sen->band,   sen->n,  sen->nb, sizeof(int));
  alloc_2D((void***)&sen->domain, sen->nb, NPOW_04, sizeof(char));

  for (b=0, bb=0; b<nb; b++){
    if (!vb[b]) continue;
    if (strlen(domains[b]) > NPOW_04-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { strncpy(sen->domain[bb], domains[b], strlen(domains[b])); sen->domain[bb][strlen(domains[b])] = '\0';}
    for (s=0, ss=0; s<ns; s++){
      if (!vs[s]) continue;
      for (c=0; c<5; c++) upper[c] = toupper(sensor[s][c]);
      if (strlen(upper) > NPOW_10-1){
        printf("cannot copy, string too long.\n"); exit(1);
      } else { strncpy(sen->sensor[ss], upper, strlen(upper)); sen->sensor[ss][strlen(upper)] = '\0';}
      sen->band[ss][bb] = band[s][b];
      ss++;
    }
    bb++;
  }

  #ifdef FORCE_DEBUG
  printf("blue  %02d, green %02d, red   %02d\n", sen->blue, sen->green, sen->red);
  printf("re_1  %02d, re_2  %02d, re_3  %02d\n", sen->rededge1, sen->rededge2, sen->rededge3);
  printf("bnir  %02d, nir   %02d, swir1 %02d\n", sen->bnir, sen->nir, sen->swir1);
  printf("swir2 %02d, vv    %02d, vh    %02d\n", sen->swir2, sen->vv, sen->vh);
  #endif

  #ifdef FORCE_DEBUG
  printf("waveband mapping:\n");
  for (b=0; b<nb; b++) printf("%s %d\n", domains[b], vb[b]); printf("\n");
  printf("%d bands, target sensor: %s\n", sen->nb, sen->target);
  #endif

  #ifdef FORCE_DEBUG
  printf("processing with %d sensors and %d bands\n", sen->n, sen->nb);
  for (s=0; s<sen->n; s++){
    printf("%s: ", sen->sensor[s]);
    for (b=0; b<sen->nb; b++) printf("%2d ", sen->band[s][b]); printf("\n");
  }
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
    free_2D((void**)phl->sen.band,   phl->sen.n);
    free_2D((void**)phl->sen.domain, phl->sen.nb);
  }

  if (phl->input_level2 == _INP_QAI_ ||
      phl->input_level2 == _INP_ARD_){
    free_2D((void**)phl->sen2.sensor, phl->sen2.n);
    free_2D((void**)phl->sen2.band,   phl->sen2.n);
    free_2D((void**)phl->sen2.domain, phl->sen2.nb);
  }

  if (phl->type == _HL_TSA_) free_2D((void**)phl->tsa.index_name, phl->tsa.n); 

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
char  buffer[NPOW_10] = "\0";
int d, w, m, q, tmp;
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
    default:
      printf("Unknown module!\n"); return FAILURE;
  }


  // process line by line
  while (fgets(buffer, NPOW_10, fpar) != NULL) parse_parameter(phl->params, buffer);
  fclose(fpar);


  #ifdef FORCE_DEBUG
  print_parameter(phl->params);
  #endif

  if (check_parameter(phl->params) == FAILURE) return FAILURE;

  log_parameter(phl->params);


  // re-parse special cases

  if ((phl->input_level1 == _INP_QAI_ ||
       phl->input_level1 == _INP_ARD_) &&
    parse_sensor(&phl->sen) != SUCCESS){
    printf("Compiling sensors failed.\n"); return FAILURE;}
    
  if ((phl->input_level2 == _INP_QAI_ ||
       phl->input_level2 == _INP_ARD_) &&
    parse_sensor(&phl->sen2) != SUCCESS){
    printf("Compiling secondary sensors failed.\n"); return FAILURE;}
    
  if (phl->type == _HL_TSA_ && check_bandlist(&phl->tsa, &phl->sen) == FAILURE){
    printf("sth wrong with bandlist."); return FAILURE;}

  if (phl->type == _HL_TSA_) parse_sta(&phl->tsa.stm.sta);
  if (phl->type == _HL_CSO_) parse_sta(&phl->cso.sta);
  
  if (phl->type == _HL_TSA_) parse_lsp(&phl->tsa.lsp);
  
  if (phl->type == _HL_TXT_) parse_txt(&phl->txt);
  
  if (phl->type == _HL_LSM_) parse_lsm(&phl->lsm);

  if (phl->input_level1 == _INP_FTR_) parse_ftr(&phl->ftr);

  if (phl->input_level1 == _INP_ARD_ ||
      phl->input_level1 == _INP_QAI_) parse_quality(&phl->qai);

  if (phl->input_level1 != _INP_QAI_) phl->prd.ref = true;
  phl->prd.qai = true;


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

    // phenology not possible for first and last year
    phl->tsa.lsp.ny = phl->ny-2;

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
    phl->bap.w.t = phl->bap.w.d + phl->bap.w.y + phl->bap.w.c  + phl->bap.w.h +
             phl->bap.w.r + phl->bap.w.v;

    // number of years
    phl->bap.Yn = (phl->bap.Yr*2)+1;

    // choose type of scoring function
    if (phl->bap.Ds[1] > phl->bap.Ds[0] &&
        phl->bap.Ds[1] > phl->bap.Ds[2]){
      phl->bap.score_type = _SCR_GAUSS_; // gaussian
    } else if (phl->bap.Ds[0] > phl->bap.Ds[2]){
      phl->bap.score_type = _SCR_SIG_DES_; // descending sigmoid
    } else if (phl->bap.Ds[2] > phl->bap.Ds[0]){
      phl->bap.score_type = _SCR_SIG_ASC_; // ascending sigmoid
    }

    // choose products
    if (phl->bap.w.c > 0) phl->prd.dst = true;
    if (phl->bap.w.h > 0) phl->prd.hot = true;
    if (phl->bap.w.v > 0) phl->prd.vzn = true;
    
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


    if (phl->tsa.lsp.ny < 1 && (phl->tsa.lsp.ospl || phl->tsa.lsp.olsp || phl->tsa.lsp.otrd || phl->tsa.lsp.ocat)){
      printf("LSP cannot be estimated for first and last year.\n");
      printf("Time window is too short.\n");
      return FAILURE;
    }

    if ((phl->tsa.lsp.ospl || phl->tsa.lsp.olsp || phl->tsa.lsp.otrd || phl->tsa.lsp.ocat) && phl->tsa.tsi.method == _INT_NONE_){
      printf("Phenology options require INTERPOLATE != NONE\n"); return FAILURE;}

    #ifndef SPLITS
    if (phl->tsa.lsp.ospl || phl->tsa.lsp.olsp || phl->tsa.lsp.otrd || phl->tsa.lsp.ocat){
      printf("Phenology options require to have FORCE compiled with SPLITS\n");
      printf("Install SPLITS and re-compile (see user guide)\n");
      printf(" OR disable phenology options\n");
      return FAILURE;
    }
    #endif
    
  }


  return SUCCESS;
}

