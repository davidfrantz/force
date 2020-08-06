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
This file contains tagged enum definitions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "enum-cl.h"


const tagged_enum_t _TAGGED_ENUM_RESAMPLE_[_RESAMPLE_LENGTH_] = {
  { _RESAMPLE_NN_, "NN" }, { _RESAMPLE_BL_, "BL" }, { _RESAMPLE_CC_, "CC" }};

const tagged_enum_t _TAGGED_ENUM_RES_MERGE_[_RES_MERGE_LENGTH_] = {
  { _RES_MERGE_NONE_,     "NONE" },     { _RES_MERGE_REGRESSION_, "REGRESSION" }, 
  { _RES_MERGE_IMPROPHE_, "IMPROPHE" }, { _RES_MERGE_STARFM_,     "STARFM" }};

  const tagged_enum_t _TAGGED_ENUM_FMT_[_FMT_LENGTH_] = {
  { _FMT_ENVI_, "ENVI" }, { _FMT_GTIFF_, "GTiff" }};

const tagged_enum_t _TAGGED_ENUM_SEN_[_SEN_LENGTH_] = {
  { _SEN_LND04_, "LND04" }, { _SEN_LND05_, "LND05" },
  { _SEN_LND07_, "LND07" }, { _SEN_LND08_, "LND08" },
  { _SEN_SEN2A_, "SEN2A" }, { _SEN_SEN2B_, "SEN2B" },
  { _SEN_sen2a_, "sen2a" }, { _SEN_sen2b_, "sen2b" },
  { _SEN_LNDLG_, "LNDLG" }, { _SEN_SEN2L_, "SEN2L" },
  { _SEN_SEN2H_, "SEN2H" }, { _SEN_RGB_,   "R-G-B" },
  { _SEN_S1AIA_, "S1AIA" }, { _SEN_S1AID_, "S1AID" },
  { _SEN_S1BIA_, "S1BIA" }, { _SEN_S1BID_, "S1BID" }};

const tagged_enum_t _TAGGED_ENUM_QAI_[_QAI_LENGTH_] = {
  { _QAI_OFF_,      "NODATA"       }, { _QAI_CLD_OPQ_,  "CLOUD_OPAQUE" },
  { _QAI_CLD_UNC_,  "CLOUD_BUFFER" }, { _QAI_CLD_CIR_,  "CLOUD_CIRRUS" },
  { _QAI_SHD_,      "CLOUD_SHADOW" }, { _QAI_SNW_,      "SNOW"         },
  { _QAI_WTR_,      "WATER"        }, { _QAI_AOD_FILL_, "AOD_FILL"     },
  { _QAI_AOD_HIGH_, "AOD_HIGH"     }, { _QAI_AOD_INT_,  "AOD_INT"      },
  { _QAI_SUB_,      "SUBZERO"      }, { _QAI_SAT_,      "SATURATION"   },
  { _QAI_SUN_,      "SUN_LOW"      }, { _QAI_ILL_SHD_,  "ILLUMIN_NONE" },
  { _QAI_ILL_POOR_, "ILLUMIN_POOR" }, { _QAI_ILL_LOW_,  "ILLUMIN_LOW"  },
  { _QAI_SLP_,      "SLOPED"       }, { _QAI_WVP_,      "WVP_NONE"     }};
  
const tagged_enum_t _TAGGED_ENUM_IDX_[_IDX_LENGTH_] = {
  { _IDX_BLU_, "BLUE"   }, { _IDX_GRN_, "GREEN"     }, { _IDX_RED_, "RED"      },
  { _IDX_NIR_, "NIR"    }, { _IDX_SW1_, "SWIR1"     }, { _IDX_SW2_, "SWIR2"    },
  { _IDX_RE1_, "RE1"    }, { _IDX_RE2_, "RE2"       }, { _IDX_RE3_, "RE3"      },
  { _IDX_BNR_, "BNIR"   }, { _IDX_NDV_, "NDVI"      }, { _IDX_EVI_, "EVI"      },
  { _IDX_NBR_, "NBR"    }, { _IDX_ARV_, "ARVI"      }, { _IDX_SAV_, "SAVI"     },
  { _IDX_SRV_, "SARVI"  }, { _IDX_TCB_, "TC-BRIGHT" }, { _IDX_TCG_, "TC-GREEN" },
  { _IDX_TCW_, "TC-WET" }, { _IDX_TCD_, "TC-DI"     }, { _IDX_NDB_, "NDBI"     },
  { _IDX_NDW_, "NDWI"   }, { _IDX_MNW_, "MNDWI"     }, { _IDX_NDS_, "NDSI"     },
  { _IDX_SMA_, "SMA"    }, { _IDX_BVV_, "VV"        }, { _IDX_BVH_, "VH"       },
  { _IDX_NDT_, "NDTI"   }, { _IDX_NDM_, "NDMI"      }};

const tagged_enum_t _TAGGED_ENUM_INT_[_INT_LENGTH_] = {
  { _INT_NONE_,   "NONE"   }, { _INT_LINEAR_, "LINEAR" },
  { _INT_MOVING_, "MOVING" }, { _INT_RBF_,    "RBF"    }};

const tagged_enum_t _TAGGED_ENUM_STA_[_STA_LENGTH_] = {
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

const tagged_enum_t _TAGGED_ENUM_HEMI_[_HEMI_LENGTH_] = {
  { _HEMI_NORTH_, "NORTH" }, { _HEMI_SOUTH_, "SOUTH" }, { _HEMI_MIXED_, "MIXED" }};

const tagged_enum_t _TAGGED_ENUM_LSP_[_LSP_LENGTH_] = {
  {_LSP_DEM_, "DEM" }, {_LSP_DSS_, "DSS" }, {_LSP_DRI_, "DRI" }, {_LSP_DPS_, "DPS" },
  {_LSP_DFI_, "DFI" }, {_LSP_DES_, "DES" }, {_LSP_DLM_, "DLM" }, {_LSP_LTS_, "LTS" },
  {_LSP_LGS_, "LGS" }, {_LSP_VEM_, "VEM" }, {_LSP_VSS_, "VSS" }, {_LSP_VRI_, "VRI" },
  {_LSP_VPS_, "VPS" }, {_LSP_VFI_, "VFI" }, {_LSP_VES_, "VES" }, {_LSP_VLM_, "VLM" },
  {_LSP_VBL_, "VBL" }, {_LSP_VSA_, "VSA" }, {_LSP_IST_, "IST" }, {_LSP_IBL_, "IBL" },
  {_LSP_IBT_, "IBT" }, {_LSP_IGS_, "IGS" }, {_LSP_RAR_, "RAR" }, {_LSP_RAF_, "RAF" },
  {_LSP_RMR_, "RMR" }, {_LSP_RMF_, "RMF" }};

const tagged_enum_t _TAGGED_ENUM_POL_[_POL_LENGTH_] = {
  { _POL_DSS_, "DSS" }, { _POL_DMS_, "DMS" }, { _POL_DES_, "DES" }, { _POL_DEV_, "DEV" }, 
  { _POL_DAV_, "DAV" }, { _POL_DLV_, "DLV" }, { _POL_LGS_, "LGS" }, { _POL_LBV_, "LBV" }, 
  { _POL_VSS_, "VSS" }, { _POL_VMS_, "VMS" }, { _POL_VES_, "VES" }, { _POL_VEV_, "VEV" }, 
  { _POL_VAV_, "VAV" }, { _POL_VLV_, "VLV" }, { _POL_VGA_, "VGA" }, { _POL_VGV_, "VGV" }, 
  { _POL_DPY_, "DPY" }};

const tagged_enum_t _TAGGED_ENUM_TAIL_[_TAIL_LENGTH_] = {
  { _TAIL_LEFT_, "LEFT" }, { _TAIL_TWO_, "TWO" }, { _TAIL_RIGHT_, "RIGHT" }};

const tagged_enum_t _TAGGED_ENUM_TRD_[_TRD_LENGTH_] = {
  { _TRD_MEAN_, "MEAN"}, { _TRD_OFFSET_, "OFFSET"}, { _TRD_SLOPE_, "SLOPE"}, 
  { _TRD_RSQ_,  "RSQ"},  { _TRD_SIG_,    "SIG"},    { _TRD_RMSE_,  "RMSE"},  
  { _TRD_MAE_,  "MAE"},  { _TRD_MAXE_,   "MAXRES"}, { _TRD_NUM_,   "NUM"}};
       
const tagged_enum_t _TAGGED_ENUM_CAT_[_CAT_LENGTH_] = {
  { _CAT_CHANGE_, "CHANGE"},               { _CAT_YEAR_, "YEAR-OF-CHANGE"},
  { _CAT_TOTAL_MEAN_, "TOTAL-MEAN"},       { _CAT_TOTAL_OFFSET_, "TOTAL-OFFSET"}, 
  { _CAT_TOTAL_SLOPE_, "TOTAL-SLOPE"},     { _CAT_TOTAL_RSQ_, "TOTAL-RSQ"}, 
  { _CAT_TOTAL_SIG_, "TOTAL-SIG"},         { _CAT_TOTAL_RMSE_, "TOTAL-RMSE"},   
  { _CAT_TOTAL_MAE_, "TOTAL-MAE"},         { _CAT_TOTAL_MAXE_, "TOTAL-MAXRES"}, 
  { _CAT_TOTAL_NUM_, "TOTAL-NUM"},         { _CAT_BEFORE_MEAN_, "BEFORE-MEAN"}, 
  { _CAT_BEFORE_OFFSET_, "BEFORE-OFFSET"}, { _CAT_BEFORE_SLOPE_, "BEFORE-SLOPE"}, 
  { _CAT_BEFORE_RSQ_, "BEFORE-RSQ"},       { _CAT_BEFORE_SIG_, "BEFORE-SIG"}, 
  { _CAT_BEFORE_RMSE_, "BEFORE-RMSE"},     { _CAT_BEFORE_MAE_, "BEFORE-MAE"},  
  { _CAT_BEFORE_MAXE_, "BEFORE-MAXRES"},   { _CAT_BEFORE_NUM_, "BEFORE-NUM"},
  { _CAT_AFTER_MEAN_, "AFTER-MEAN"},       { _CAT_AFTER_OFFSET_, "AFTER-OFFSET"}, 
  { _CAT_AFTER_SLOPE_, "AFTER-SLOPE"},     { _CAT_AFTER_RSQ_, "AFTER-RSQ"}, 
  { _CAT_AFTER_SIG_, "AFTER-SIG"},         { _CAT_AFTER_RMSE_, "AFTER-RMSE"}, 
  { _CAT_AFTER_MAE_, "AFTER-MAE"},         { _CAT_AFTER_MAXE_, "AFTER-MAXRES"}, 
  { _CAT_AFTER_NUM_, "AFTER-NUM"}};
  
const tagged_enum_t _TAGGED_ENUM_STD_[_STD_LENGTH_] = {
  { _STD_NONE_, "NONE" }, {_STD_NORMAL_, "NORMALIZE"}, {_STD_CENTER_, "CENTER" }};

const tagged_enum_t _TAGGED_ENUM_ML_[_ML_LENGTH_] = {
  { _ML_SVR_, "SVR" }, { _ML_SVC_, "SVC" }, { _ML_RFR_, "RFR" }, { _ML_RFC_, "RFC" }};

const tagged_enum_t _TAGGED_ENUM_TXT_[_TXT_LENGTH_] = {
  { _TXT_ERO_, "ERO" }, { _TXT_DIL_, "DIL" }, { _TXT_OPN_, "OPN" }, { _TXT_CLS_, "CLS" }, 
  { _TXT_GRD_, "GRD" }, { _TXT_THT_, "THT" }, { _TXT_BHT_, "BHT" }};

const tagged_enum_t _TAGGED_ENUM_QUERY_[_QUERY_LENGTH_] = {
  { _QUERY_EQ_, "EQ" }, { _QUERY_GT_, "GT" }, { _QUERY_LT_, "LT" }};

const tagged_enum_t _TAGGED_ENUM_LSM_[_LSM_LENGTH_] = {
  { _LSM_MPA_, "MPA" }, { _LSM_UCI_, "UCI" }, { _LSM_FDI_, "FDI" }, 
  { _LSM_EDD_, "EDD" }, { _LSM_NBR_, "NBR" }, { _LSM_EMS_, "EMS" }, 
  { _LSM_AVG_, "AVG" }, { _LSM_STD_, "STD" }, { _LSM_GEO_, "GEO" }, 
  { _LSM_MAX_, "MAX" }};

const tagged_enum_t _TAGGED_ENUM_KERNEL_[_KERNEL_LENGTH_] = {
  { _KERNEL_SQUARE_, "SQUARE" }, { _KERNEL_CIRCLE_, "CIRCLE" }};

const tagged_enum_t _TAGGED_ENUM_SCR_[_SCR_LENGTH_] = {
  { _SCR_TOTAL_, "TOTAL" },    { _SCR_DOY_,  "DOY" },  { _SCR_YEAR_,   "YEAR" }, 
  { _SCR_DST_,   "DISTANCE" }, { _SCR_HAZE_, "HAZE" }, { _SCR_CORREL_, "CORREL" }, 
  { _SCR_VZEN_,  "VZEN" }};
       
const tagged_enum_t _TAGGED_ENUM_INF_[_INF_LENGTH_] = {
  { _INF_QAI_,  "QAI" },  { _INF_NUM_,  "NUM" },  { _INF_DOY_, "DOY" }, 
  { _INF_YEAR_, "YEAR" }, { _INF_DIFF_, "dDOY" }, { _INF_SEN_, "SENSOR" }};

const tagged_enum_t _TAGGED_ENUM_RGB_[_RGB_LENGTH_] = {
  { _RGB_R_,  "RED" }, { _RGB_G_,  "GREEN" }, { _RGB_B_,  "BLUE" }};
  
  