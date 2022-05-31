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
Enum definitions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef ENUM_CL_H
#define ENUM_CL_H

//#include "../cross-level/const-cl.h"

#ifdef __cplusplus
extern "C" {
#endif

// common numbers
enum { NPOW_00 = 1,    NPOW_01 = 2,     NPOW_02 = 4,     NPOW_03 = 8, 
       NPOW_04 = 16,   NPOW_05 = 32,    NPOW_06 = 64,    NPOW_07 = 128,   
       NPOW_08 = 256,  NPOW_09 = 512,   NPOW_10 = 1024,  NPOW_11 = 2048, 
       NPOW_12 = 4096, NPOW_13 = 8192,  NPOW_14 = 16384, NPOW_15 = 32768, 
       NPOW_16 = 65536 };

typedef struct{
  int en;
  char tag[NPOW_04];
} tagged_enum_t;

// function return codes
enum { SUCCESS = 0, FAILURE = 1, CANCEL = 10 };


// dimensions
enum { _X_, _Y_, _Z_ };

// range
enum { _MIN_, _MAX_ };

// modules
enum { _LL_LEVEL2_, _HL_BAP_, _HL_TSA_, _HL_CSO_, _HL_CFI_, 
       _HL_L2I_,    _HL_ML_,  _HL_SMP_, _HL_TXT_, _HL_LSM_, 
       _HL_LIB_,    _HL_UDF_, _AUX_TRAIN_, _AUX_SYNTHMIX_ };

// level of module
enum { _LOWER_LEVEL_, _HIGHER_LEVEL_, _AUX_LEVEL_ };

// input type
enum { _INP_RAW_, _INP_ARD_, _INP_QAI_, _INP_FTR_, 
       _INP_AUX_, _INP_CON_, _INP_NONE_, _INP_LENGTH_ };

// parameter datatype
enum { _PAR_INT_,  _PAR_ENUM_, _PAR_FLOAT_, _PAR_DOUBLE_, 
       _PAR_BOOL_, _PAR_DATE_, _PAR_CHAR_ };

// string parameter test type
enum { _CHAR_TEST_NULL_OR_EXIST_, _CHAR_TEST_EXIST_, _CHAR_TEST_NOT_EXIST_, 
       _CHAR_TEST_NULL_OR_BASE_,  _CHAR_TEST_BASE_,  _CHAR_TEST_NONE_ };

// image datatypes
enum { _DT_NONE_,  _DT_SHORT_, _DT_SMALL_, 
       _DT_FLOAT_, _DT_INT_,   _DT_USHORT_ };

// output formats
enum { _FMT_ENVI_, _FMT_GTIFF_, _FMT_COG_, _FMT_JPEG_, _FMT_CUSTOM_, _FMT_LENGTH_ };

// t-test tailtype
enum { _TAIL_LEFT_, _TAIL_TWO_, _TAIL_RIGHT_, _TAIL_LENGTH_ };

// RGB
enum { _RGB_R_, _RGB_G_, _RGB_B_, _RGB_LENGTH_ };

// bit positions for QAI
enum { _QAI_BIT_OFF_ =  0, _QAI_BIT_CLD_ =  1, _QAI_BIT_SHD_ =  3, 
       _QAI_BIT_SNW_ =  4, _QAI_BIT_WTR_ =  5, _QAI_BIT_AOD_ =  6, 
       _QAI_BIT_SUB_ =  8, _QAI_BIT_SAT_ =  9, _QAI_BIT_SUN_ = 10, 
       _QAI_BIT_ILL_ = 11, _QAI_BIT_SLP_ = 13, _QAI_BIT_WVP_ = 14};

// QAI flag ordering
enum { _QAI_FLAG_OFF_, _QAI_FLAG_CLD_, _QAI_FLAG_SHD_, 
       _QAI_FLAG_SNW_, _QAI_FLAG_WTR_, _QAI_FLAG_AOD_, 
       _QAI_FLAG_SUB_, _QAI_FLAG_SAT_, _QAI_FLAG_SUN_, 
       _QAI_FLAG_ILL_, _QAI_FLAG_SLP_, _QAI_FLAG_WVP_,
       _QAI_FLAG_LENGTH_ };

// QAI flags/states
enum { _QAI_OFF_,      _QAI_CLD_OPQ_, _QAI_CLD_UNC_,  _QAI_CLD_CIR_, 
       _QAI_SHD_,      _QAI_SNW_,     _QAI_WTR_,      _QAI_AOD_FILL_, 
       _QAI_AOD_HIGH_, _QAI_AOD_INT_, _QAI_SUB_,      _QAI_SAT_,
       _QAI_SUN_,      _QAI_ILL_SHD_, _QAI_ILL_POOR_, _QAI_ILL_LOW_,
       _QAI_SLP_,      _QAI_WVP_,     _QAI_LENGTH_ };

// output option
enum { OPEN_FALSE,   // do not write
       OPEN_CREATE,  // create file from scratch
       OPEN_UPDATE,  // read file, and update values
       OPEN_MERGE,   // read file, and merge values
       OPEN_BLOCK }; // write block into file

// sensors
enum { _SEN_LND04_, _SEN_LND05_, _SEN_LND07_, _SEN_LND08_, _SEN_LND09_, 
       _SEN_SEN2A_, _SEN_SEN2B_, _SEN_sen2a_, _SEN_sen2b_, _SEN_LNDLG_, 
       _SEN_SEN2L_, _SEN_SEN2H_, _SEN_RGB_,   _SEN_S1AIA_, _SEN_S1AID_, 
       _SEN_S1BIA_, _SEN_S1BID_, _SEN_VVVHP_, _SEN_MOD01_, _SEN_MOD02_, 
       _SEN_MODIS_, _SEN_LENGTH_ };

// wavelength domains
enum { _WVL_BLUE_,     _WVL_GREEN_,    _WVL_RED_,   _WVL_REDEDGE1_, 
       _WVL_REDEDGE2_, _WVL_REDEDGE3_, _WVL_BNIR_,  _WVL_NIR_, 
       _WVL_SWIR0_,    _WVL_SWIR1_,    _WVL_SWIR2_, _WVL_VV_, 
       _WVL_VH_,       _WVL_LENGTH_ };

// AOD target types
enum { _AOD_WAT_, _AOD_SHD_, _AOD_VEG_ };

// ARD input type
enum { _ARD_REF_, _ARD_AUX_, _ARD_MSK_, _ARD_FTR_ };
       
// scoring method
enum { _SCR_TYPE_GAUSS_, _SCR_TYPE_SIG_DES_, _SCR_TYPE_SIG_ASC_, _SCR_TYPE_LENGTH_ };

// compositing scores
enum { _SCR_TOTAL_,  _SCR_DOY_,  _SCR_YEAR_, _SCR_DST_, _SCR_HAZE_, 
       _SCR_CORREL_, _SCR_VZEN_, _SCR_LENGTH_ };
       
// compositing information
enum { _INF_QAI_,  _INF_NUM_, _INF_DOY_, _INF_YEAR_, 
       _INF_DIFF_, _INF_SEN_, _INF_LENGTH_ };

// interpolation method
enum { _INT_NONE_, _INT_LINEAR_, _INT_MOVING_, _INT_RBF_, _INT_HARMONIC_, _INT_LENGTH_ };

// spectral indices
enum { _IDX_BLU_, _IDX_GRN_, _IDX_RED_, _IDX_NIR_, _IDX_SW1_, _IDX_SW2_, 
       _IDX_RE1_, _IDX_RE2_, _IDX_RE3_, _IDX_BNR_, _IDX_NDV_, _IDX_EVI_, 
       _IDX_NBR_, _IDX_ARV_, _IDX_SAV_, _IDX_SRV_, _IDX_TCB_, _IDX_TCG_, 
       _IDX_TCW_, _IDX_TCD_, _IDX_NDB_, _IDX_NDW_, _IDX_MNW_, _IDX_NDS_,
       _IDX_SMA_, _IDX_BVV_, _IDX_BVH_, _IDX_NDT_, _IDX_NDM_, _IDX_SW0_,
       _IDX_KNV_, _IDX_ND1_, _IDX_ND2_, _IDX_CRE_, _IDX_NR1_, _IDX_NR2_,
       _IDX_NR3_, _IDX_N1n_, _IDX_N2n_, _IDX_N3n_, _IDX_Mre_, _IDX_Mrn_,
       _IDX_CCI_, _IDX_LENGTH_};

// standardization
enum { _STD_NONE_, _STD_NORMAL_, _STD_CENTER_, _STD_LENGTH_ };

// machine learning methods
enum { _ML_SVR_, _ML_SVC_, _ML_RFR_, _ML_RFC_, _ML_LENGTH_ };

// statistics
enum { _STA_MIN_, _STA_MAX_, _STA_RNG_, _STA_IQR_, _STA_AVG_, _STA_STD_,
       _STA_SKW_, _STA_KRT_, _STA_Q01_, _STA_Q02_, _STA_Q03_, _STA_Q04_,
       _STA_Q05_, _STA_Q06_, _STA_Q07_, _STA_Q08_, _STA_Q09_, _STA_Q10_,
       _STA_Q11_, _STA_Q12_, _STA_Q13_, _STA_Q14_, _STA_Q15_, _STA_Q16_,
       _STA_Q17_, _STA_Q18_, _STA_Q19_, _STA_Q20_, _STA_Q21_, _STA_Q22_,
       _STA_Q23_, _STA_Q24_, _STA_Q25_, _STA_Q26_, _STA_Q27_, _STA_Q28_,
       _STA_Q29_, _STA_Q30_, _STA_Q31_, _STA_Q32_, _STA_Q33_, _STA_Q34_,
       _STA_Q35_, _STA_Q36_, _STA_Q37_, _STA_Q38_, _STA_Q39_, _STA_Q40_,
       _STA_Q41_, _STA_Q42_, _STA_Q43_, _STA_Q44_, _STA_Q45_, _STA_Q46_,
       _STA_Q47_, _STA_Q48_, _STA_Q49_, _STA_Q50_, _STA_Q51_, _STA_Q52_,
       _STA_Q53_, _STA_Q54_, _STA_Q55_, _STA_Q56_, _STA_Q57_, _STA_Q58_,
       _STA_Q59_, _STA_Q60_, _STA_Q61_, _STA_Q62_, _STA_Q63_, _STA_Q64_,
       _STA_Q65_, _STA_Q66_, _STA_Q67_, _STA_Q68_, _STA_Q69_, _STA_Q70_,
       _STA_Q71_, _STA_Q72_, _STA_Q73_, _STA_Q74_, _STA_Q75_, _STA_Q76_,
       _STA_Q77_, _STA_Q78_, _STA_Q79_, _STA_Q80_, _STA_Q81_, _STA_Q82_,
       _STA_Q83_, _STA_Q84_, _STA_Q85_, _STA_Q86_, _STA_Q87_, _STA_Q88_,
       _STA_Q89_, _STA_Q90_, _STA_Q91_, _STA_Q92_, _STA_Q93_, _STA_Q94_,
       _STA_Q95_, _STA_Q96_, _STA_Q97_, _STA_Q98_, _STA_Q99_, _STA_NUM_, _STA_LENGTH_ };

// hemisphere
enum { _HEMI_NORTH_, _HEMI_SOUTH_, _HEMI_MIXED_, _HEMI_LENGTH_ };

// phenometrics
enum { _LSP_DEM_, _LSP_DSS_, _LSP_DRI_, _LSP_DPS_, _LSP_DFI_, _LSP_DES_, 
       _LSP_DLM_, _LSP_LTS_, _LSP_LGS_, _LSP_VEM_, _LSP_VSS_, _LSP_VRI_, 
       _LSP_VPS_, _LSP_VFI_, _LSP_VES_, _LSP_VLM_, _LSP_VBL_, _LSP_VSA_, 
       _LSP_IST_, _LSP_IBL_, _LSP_IBT_, _LSP_IGS_, _LSP_RAR_, _LSP_RAF_, 
       _LSP_RMR_, _LSP_RMF_, _LSP_LENGTH_ };

// polar metrics
enum { _POL_DEM_, _POL_DLM_, _POL_DPS_, _POL_DSS_, _POL_DMS_, _POL_DES_, 
       _POL_DEV_, _POL_DAV_, _POL_DLV_, _POL_LTS_, _POL_LGS_, _POL_LGV_, 
       _POL_VEM_, _POL_VLM_, _POL_VPS_, _POL_VSS_, _POL_VMS_, _POL_VES_, 
       _POL_VEV_, _POL_VAV_, _POL_VLV_, _POL_VBL_, _POL_VGA_, _POL_VSA_, 
       _POL_VPA_, _POL_VGM_, _POL_VGV_, _POL_DPY_, _POL_DPV_, _POL_IST_, 
       _POL_IBL_, _POL_IBT_, _POL_IGS_, _POL_IRR_, _POL_IFR_, _POL_RAR_, 
       _POL_RAF_, _POL_RMR_, _POL_RMF_, _POL_LENGTH_ };

// folding
enum { _FLD_YEAR_, _FLD_QUARTER_, _FLD_MONTH_, _FLD_WEEK_, _FLD_DOY_, _FLD_LENGTH_ };

// time series parts
enum { _PART_TOTAL_, _PART_BEFORE_, _PART_AFTER_, _PART_LENGTH_ };

// trend
enum { _TRD_MEAN_,  _TRD_OFFSET_, 
       _TRD_SLOPE_, _TRD_PRC_GAIN_, _TRD_ABS_GAIN_, 
       _TRD_RSQ_,   _TRD_SIG_,
       _TRD_RMSE_,  _TRD_MAE_,      _TRD_MAXE_,
       _TRD_NUM_,   _TRD_LEN_,      _TRD_LENGTH_ };

// change, aftereffect, trend
enum { _CAT_CHANGE_,          _CAT_LOSS_, 
       _CAT_YEAR_,
       _CAT_TOTAL_MEAN_,      _CAT_TOTAL_OFFSET_,
       _CAT_TOTAL_SLOPE_,     _CAT_TOTAL_PCT_GAIN_,
       _CAT_TOTAL_ABS_GAIN_,  _CAT_TOTAL_RSQ_, 
       _CAT_TOTAL_SIG_,       _CAT_TOTAL_RMSE_,   
       _CAT_TOTAL_MAE_,       _CAT_TOTAL_MAXE_, 
       _CAT_TOTAL_NUM_,       _CAT_TOTAL_LEN_,
       _CAT_BEFORE_MEAN_,     _CAT_BEFORE_OFFSET_,
       _CAT_BEFORE_SLOPE_,    _CAT_BEFORE_PCT_GAIN_,
       _CAT_BEFORE_ABS_GAIN_, _CAT_BEFORE_RSQ_,
       _CAT_BEFORE_SIG_,      _CAT_BEFORE_RMSE_, 
       _CAT_BEFORE_MAE_,      _CAT_BEFORE_MAXE_, 
       _CAT_BEFORE_NUM_,      _CAT_BEFORE_LEN_,
       _CAT_AFTER_MEAN_,      _CAT_AFTER_OFFSET_,
       _CAT_AFTER_SLOPE_,     _CAT_AFTER_PCT_GAIN_,
       _CAT_AFTER_ABS_GAIN_,  _CAT_AFTER_RSQ_,  
       _CAT_AFTER_SIG_,       _CAT_AFTER_RMSE_, 
       _CAT_AFTER_MAE_,       _CAT_AFTER_MAXE_,  
       _CAT_AFTER_NUM_,       _CAT_AFTER_LEN_,
       _CAT_LENGTH_ };

// texture metrics
enum { _TXT_ERO_, _TXT_DIL_, _TXT_OPN_, _TXT_CLS_, 
       _TXT_GRD_, _TXT_THT_, _TXT_BHT_, _TXT_LENGTH_ };
       
// landscape metrics
enum { _LSM_MPA_, _LSM_UCI_, _LSM_FDI_, _LSM_EDD_, _LSM_NBR_, 
       _LSM_EMS_, _LSM_AVG_, _LSM_STD_, _LSM_GEO_, _LSM_MAX_, _LSM_ARE_, _LSM_LENGTH_ };

// satellite mission
enum { LANDSAT, SENTINEL2, _UNKNOWN_, _MISSION_LENGTH_ };

// sun/view angles
enum { ZEN, AZI, cZEN, cAZI, sZEN, sAZI, tZEN, tAZI };

// resampling method
enum { _RESAMPLE_NN_, _RESAMPLE_BL_, _RESAMPLE_CC_, _RESAMPLE_LENGTH_ };

// resolution merge method
enum { _RES_MERGE_NONE_, _RES_MERGE_REGRESSION_, _RES_MERGE_IMPROPHE_, _RES_MERGE_STARFM_, _RES_MERGE_LENGTH_ };

// query type
enum  { _QUERY_EQ_, _QUERY_GT_, _QUERY_GE_, _QUERY_LT_, _QUERY_LE_, _QUERY_LENGTH_ };

// kernel type
enum { _KERNEL_SQUARE_, _KERNEL_CIRCLE_, _KERNEL_LENGTH_ };

// task type
enum { _TASK_INPUT_, _TASK_COMPUTE_, _TASK_OUTPUT_, 
       _TASK_ALL_,   _TASK_RUNTIME_, _TASK_LENGTH_};

// clock type
enum { _CLOCK_NULL_, _CLOCK_TICK_, _CLOCK_TOCK_, _CLOCK_LENGTH_ };

// user-defined function type
enum { _UDF_PIXEL_, _UDF_BLOCK_, _UDF_LENGTH_ };

// tag and value
enum { _TV_TAG_, _TV_VAL_, _TV_LENGTH_ };

// projection
enum { _PROJ_CUSTOM_, _PROJ_EQUI7_, _PROJ_GLANCE7_, _PROJ_LENGTH_ };

// continent
enum { _CONTINENT_AF_, _CONTINENT_AN_, _CONTINENT_AS_, _CONTINENT_EU_, 
       _CONTINENT_NA_, _CONTINENT_OC_, _CONTINENT_SA_, _CONTINENT_LENGTH_ };

// tagged enums
extern const tagged_enum_t _TAGGED_ENUM_RESAMPLE_[_RESAMPLE_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_RES_MERGE_[_RES_MERGE_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_FMT_[_FMT_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_SEN_[_SEN_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_QAI_[_QAI_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_IDX_[_IDX_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_INT_[_INT_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_STA_[_STA_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_HEMI_[_HEMI_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_LSP_[_LSP_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_POL_[_POL_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_TAIL_[_TAIL_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_STD_[_STD_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_ML_[_ML_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_TXT_[_TXT_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_QUERY_[_QUERY_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_LSM_[_LSM_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_KERNEL_[_KERNEL_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_TRD_[_TRD_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_CAT_[_CAT_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_SCR_[_SCR_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_INF_[_INF_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_RGB_[_RGB_LENGTH_];
extern const tagged_enum_t _TAGGED_ENUM_UDF_[_UDF_LENGTH_];

#ifdef __cplusplus
}
#endif

#endif

