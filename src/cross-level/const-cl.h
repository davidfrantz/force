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
Named constant definitions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef CONSTANT_CL_H
#define CONSTANT_CL_H

#ifdef __cplusplus
extern "C" {
#endif

// current version
#define _VERSION_ "3.0"

// abbreviated datatypes
typedef unsigned short int ushort;
typedef unsigned char small;

// coordinate struct
typedef struct {
  double x, y;
} coord_t;

// function return codes
enum { SUCCESS = 0, FAILURE = -1, CANCEL = 1 };

// common numbers
enum { NPOW_00 = 1,    NPOW_01 = 2,     NPOW_02 = 4,     NPOW_03 = 8, 
       NPOW_04 = 16,   NPOW_05 = 32,    NPOW_06 = 64,    NPOW_07 = 128,   
       NPOW_08 = 256,  NPOW_09 = 512,   NPOW_10 = 1024,  NPOW_11 = 2048, 
       NPOW_12 = 4096, NPOW_13 = 8192,  NPOW_14 = 16384, NPOW_15 = 32768, 
       NPOW_16 = 65536 };

// dimensions
enum { _X_, _Y_, _Z_ };

// range
enum { _MIN_, _MAX_ };

// modules
enum { _LL_LEVEL2_, _HL_BAP_, _HL_TSA_, _HL_CSO_, _HL_CFI_, 
       _HL_L2I_,    _HL_ML_,  _HL_SMP_, _HL_TXT_, _HL_LSM_, 
       _AUX_TRAIN_ };

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
enum { _FMT_ENVI_, _FMT_GTIFF_, _FMT_JPEG_, _FMT_LENGTH_ };

// t-test tailtype
enum { _TAIL_LEFT_, _TAIL_TWO_, _TAIL_RIGHT_, _TAIL_LENGTH_ };

// RGB
enum { _RGB_R_, _RGB_G_, _RGB_B_ };

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
enum { _SEN_LND04_, _SEN_LND05_, _SEN_LND07_, _SEN_LND08_, _SEN_SEN2A_, 
       _SEN_SEN2B_, _SEN_sen2a_, _SEN_sen2b_, _SEN_LNDLG_, _SEN_SEN2L_, 
       _SEN_SEN2H_, _SEN_RGB_,   _SEN_S1AIA_, _SEN_S1AID_, _SEN_S1BIA_,
       _SEN_S1BID_, _SEN_VVVHP_, _SEN_LENGTH_ };

// wavelength domains
enum { _WVL_BLUE_, _WVL_GREEN_, _WVL_RED_,  _WVL_REDEDGE1_, _WVL_REDEDGE2_, _WVL_REDEDGE3_, 
       _WVL_BNIR_, _WVL_NIR_,   _WVL_SWIR1_, _WVL_SWIR2_,   _WVL_VV_,       _WVL_VH_, 
       _WVL_LENGTH_ };

// AOD target types
enum { _AOD_WAT_, _AOD_SHD_, _AOD_VEG_ };

// ARD input type
enum { _ARD_REF_, _ARD_AUX_, _ARD_MSK_, _ARD_FTR_ };
       
// scoring method
enum { _SCR_GAUSS_, _SCR_SIG_DES_, _SCR_SIG_ASC_, _SCR_LENGTH_ };

// interpolation method
enum { _INT_NONE_, _INT_LINEAR_, _INT_MOVING_, _INT_RBF_, _INT_LENGTH_ };

// spectral indices
enum { _IDX_BLU_, _IDX_GRN_, _IDX_RED_, _IDX_NIR_, _IDX_SW1_, _IDX_SW2_, 
       _IDX_RE1_, _IDX_RE2_, _IDX_RE3_, _IDX_BNR_, _IDX_NDV_, _IDX_EVI_, 
       _IDX_NBR_, _IDX_ARV_, _IDX_SAV_, _IDX_SRV_, _IDX_TCB_, _IDX_TCG_, 
       _IDX_TCW_, _IDX_TCD_, _IDX_NDB_, _IDX_NDW_, _IDX_MNW_, _IDX_NDS_,
       _IDX_SMA_, _IDX_BVV_, _IDX_BVH_, _IDX_LENGTH_};
       
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

// texture metrics
enum { _TXT_ERO_, _TXT_DIL_, _TXT_OPN_, _TXT_CLS_, 
       _TXT_GRD_, _TXT_THT_, _TXT_BHT_, _TXT_LENGTH_ };
       
// landscape metrics
enum { _LSM_MPA_, _LSM_UCI_, _LSM_FDI_, _LSM_EDD_, _LSM_NBR_, 
       _LSM_EMS_, _LSM_AVG_, _LSM_STD_, _LSM_GEO_, _LSM_MAX_, _LSM_LENGTH_ };

// satellite mission
enum { LANDSAT, SENTINEL2 };

// sun/view angles
enum { ZEN, AZI, cZEN, cAZI, sZEN, sAZI, tZEN, tAZI };

// resampling method
enum { _RESAMPLE_NN_, _RESAMPLE_BL_, _RESAMPLE_CC_, _RESAMPLE_LENGTH_ };

// resolution merge method
enum { _RES_MERGE_NONE_, _RES_MERGE_REGRESSION_, _RES_MERGE_IMPROPHE_, _RES_MERGE_STARFM_, _RES_MERGE_LENGTH_ };

// query type
enum  { _QUERY_EQ_, _QUERY_GT_, _QUERY_LT_, _QUERY_LENGTH_ };

// kernel type
enum { _KERNEL_SQUARE_, _KERNEL_CIRCLE_, _KERNEL_LENGTH_ };

// task type
enum { _TASK_INPUT_, _TASK_COMPUTE_, _TASK_OUTPUT_, 
       _TASK_ALL_,   _TASK_RUNTIME_, _TASK_LENGTH_};

// clock type
enum { _CLOCK_NULL_, _CLOCK_TICK_, _CLOCK_TOCK_, _CLOCK_LENGTH_ };


// pi
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// radians to degree conversion
#define _R2D_CONV_  57.29577951308232286465
#define _D2R_CONV_   0.01745329251994329547

// compiler options
//#define FORCE_CLOCK
//#define FORCE_DEBUG
//#define FORCE_DEV

//#define SPLITS

//#define ACIX
//#define ACIX2
//#define CMIX_FAS
//#define CMIX_FAS_2

#ifdef __cplusplus
}
#endif

#endif

