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
Higher Level Processing paramater header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef PARAM_HL_H
#define PARAM_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <ctype.h>   // transform individual characters
#include <float.h>   // macro constants of the floating-point library

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"
#include "../cross-level/param-cl.h"
#include "../cross-level/utils-cl.h"
#include "../cross-level/gdalopt-cl.h"
#include "../higher-level/sensor-hl.h"
#include "../higher-level/index-parse-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

// QAI filter rules
typedef struct {
  int *flags, nflags;
  int off;      // filter out nodata
  int cld_opq;  // filter out opaque, confident clouds
  int cld_unc;  // filter out opaque, less confident clouds
  int cld_cir;  // filter out cirrus clouds
  int shd;      // filter out cloud shadow
  int snw;      // filter out snow
  int wtr;      // filter out water
  int aod_fill; // filter out filled AOD
  int aod_high; // filter out high AOD
  int aod_int;  // filter out interpolated AOD
  int sub;      // filter out subzero
  int sat;      // filter out saturated
  int sun;      // filter out high sun zenith
  int ill_shd;  // filter out no illumination
  int ill_poor; // filter out poor illumination
  int ill_low;  // filter out low illumination
  int slp;      // filter out sloped
  int wvp;      // filter out no water vapor estimation
  float above_noise;
  float below_noise;
} par_qai_t;

// BAP scoring
typedef struct {
  double t;
  double d;
  double y;
  double c;
  double h;
  double r;
  double v;
} par_scr_t;

// phenology-adaptive compositing
typedef struct {
  int lsp;       // flag: phenology-adaptive compositing
  int  y0;       // 1st year of phenology
  int  start;    // start of LSP values
  float rmse;    // cutoff threshold for phenology
} par_pac_t;

// BAP compositing
typedef struct {
  int Yt;               // target year
  int Yr;               // number of bracketing years
  int Yn;               // number of years
  float Yf;             // Y-factos
  int *Dt;            // target DOYs
  int nDt, nDs;
  float *Ds;           // function values for target DOYs
  int offsea; // use off-season data?
  int use_cloudy; // use ultra-cloudy data?
  int use_hazy; // use ultra-hazy data?
  int select; // select or weight?
  int combine; // how to combine scores?

  int band_dst;
  int band_hot;
  int band_vzn;

  int score_type;
  par_scr_t w;          // scoring struct for weigting parameters
  float dreq, vreq, hreq;     // cloud / view zenith / haze scoring  

  par_pac_t pac;

  int obap;         // flag: output best available pixel composite
  int oinf;           // flag: output compositing information
  int oscr;           // flag: output compositing scores
  int oovv;           // flag: output overview images
} par_bap_t;

// folding
typedef struct {
  int type;                 // folding statistic
  float q;                  // quantile;
  int ofby;           // flag: output fold-by-year time series
  int ofbq;           // flag: output fold-by-year time series
  int ofbm;           // flag: output fold-by-month time series
  int ofbw;           // flag: output fold-by-week time series
  int ofbd;           // flag: output fold-by-doy time series
  int otry;           // flag: output fold-by-year time series
  int otrq;           // flag: output fold-by-year time series
  int otrm;           // flag: output fold-by-month time series
  int otrw;           // flag: output fold-by-week time series
  int otrd;           // flag: output fold-by-doy time series
  int ocay;           // flag: output fold-by-year time series
  int ocaq;           // flag: output fold-by-year time series
  int ocam;           // flag: output fold-by-month time series
  int ocaw;           // flag: output fold-by-week time series
  int ocad;           // flag: output fold-by-doy time series
  int standard;
} par_fld_t;

// interpolation
typedef struct {
  int method;            // interpolation method
  int step;              // interpolate each n days
  int mov_max;                // max temp. dist for moving stats filter
  int rbf_nk;                 // number of kernels for RBF fit
  int *rbf_sigma; // sigmas for RBF fit
  float rbf_cutoff;           // cutoff for RBF fit
  int harm_trend; // trend in harmonic model?
  int harm_nmodes; // number of modes for harmonic
  date_t *harm_fit_range; // date range for fitting harmonic
  int harm_fit_nrange; // number of dates for fitting harmonic
  int otsi;           // flag: output time series interpolation
  int onrt;           // flag: output near-real time product
  int standard;
} par_tsi_t;

// SMA
typedef struct {
  int v;              // do SMA?
  char *f_emb; // endmember file
  int sto;      // sum-to-one constrained
  int pos;      // non-negative constrained
  int shn;      // shade normalization
  int emb;             // endmember number
  int orms;    // flag: output model residuals
} par_sma_t;

// polar metrics
typedef struct {
  int ny;
  int ns;
  float start;
  float mid;
  float end;
  int opct;       // flag: output polar coordinate transformed TS
  int opol;       // flag: output polar metrics
  int *metrics, nmetrics;
  int use[_POL_LENGTH_];
  //int odem, odss, odev, odms, odav, odlv;
  //int odes, odlm, olts, olgs, olgv, ovem;
  //int ovss, ovev, ovms, ovav, ovlv, oves;
  //int ovlm, ovbl, ovsa, ovga, ovgv, odpy;
  //int odpv;
  int otrd;       // flag: output POL trends
  int ocat;       // flag: output POL cats
  int standard;
  int adaptive;
} par_pol_t;

// trend
typedef struct {
  int tail;    // tail type
  float conf;  // confidence level
  int penalty; // penalty for non-permanent change (CAT)
} par_trd_t;

// user-defined function
typedef struct {
  char   *f_code;
  int     out;
  int     nb;
  char  **bandname;
  date_t *date;
  int     type;
} par_udf_t;

// aggregation statistics
typedef struct {
  int *metrics, nmetrics;
  int num;
  int min;
  int max;
  int rng;
  int iqr;
  int avg;
  int std;
  int skw;
  int krt;
  int quantiles;
  int nquantiles;
  int qxx[100];
  float q[100];
} par_sta_t;

// CSO
typedef struct {
  int step;
  par_sta_t sta;
} par_cso_t;

// STM
typedef struct {
  int ostm;  // flag: output spectral temporal metrics
  par_sta_t sta;
} par_stm_t;


// general TSA
typedef struct {
  index_t index; // index definitions
  int otss;           // flag: output time series brick
  int standard;

  par_stm_t stm;
  par_fld_t fld;
  par_tsi_t tsi;
  par_sma_t sma;
  par_pol_t pol;
  par_trd_t trd;
  par_udf_t pyp;
  par_udf_t rsp;
} par_tsa_t;

// features
typedef struct {
  char  **bname;
  int   *band;
  char  ***cfeature;
  int   *ifeature;
  int   nfeature;
  int   ntags;
  int   nodata;
  int  exclude;
} par_ftr_t;

// continuous fields
typedef struct {
  char  *dname;  // directory
  char **fname;  // file names
  int    n;      // n
  int    nodata; // nodata
} par_con_t;

// machine learning
typedef struct {
  char *d_model;
  char ***f_model;
  int nmodelset;
  int *nmodel;
  int *nclass;
  int nclass_all_sets;
  float converge;
  float scale;
  int method;
  int omlp; 
  int omli; 
  int omlu; 
  int orfp; 
  int orfm; 
  char *base;
} par_mcl_t;

// sample
typedef struct {
  char *f_coord;
  char *f_sample;
  char *f_response;
  char *f_coords;
  int  projected;
} par_smp_t;

// texture
typedef struct {
  double radius;
  int iter;
  int *metrics, nmetrics;
  int oero;
  int odil;
  int oopn;
  int ocls;
  int ogrd;
  int otht;
  int obht;
  char *base;
} par_txt_t;

// landscape metrics
typedef struct {
  double radius;
  int minpatchsize;
  int *query;
  int nquery;
  int *threshold;
  int nthreshold;
  int *metrics, nmetrics;
  int allpx;
  int ompa;
  int ouci;
  int ofdi;
  int oedd;
  int onbr;
  int oems;
  int oavg;
  int ostd;
  int ogeo;
  int omax;
  int oare;
  char *base;
  int kernel;
} par_lsm_t;

// library completeness testing
typedef struct {
  char  *d_lib;
  char **f_lib;
  int    n_lib;
  int    n_sample;
  int    rescale;
  char  *base;
} par_lib_t;

// UDF plug-in
typedef struct {
  par_udf_t pyp;
  par_udf_t rsp;
} par_udp_t;

// improphe core
typedef struct {
  int *dwin;
  int bwin, nwin;
  double pred_radius;
  double text_radius;
  int ksize;
  int ksd;
} par_imp_t;

// CF improphe
typedef struct {
  int *years, nyears, y0;
} par_cfi_t;

// L2 products usage
typedef struct {
  int ref;
  int qai;
  int aux;
  int imp;
} par_prd_t;

// higher level parameters
typedef struct {

  params_t *params;
  int type;

  // directory variables
  char  f_par[NPOW_10];    // parameter file
  char *d_lower;  // Lower  Level directory
  char *d_higher; // Higher Level directory
  char *d_prov;   // provenance directory
  char *d_mask;   // mask directory
  char *b_mask;   // mask basename
  char *f_tile;   // tile allow-list

  // spatial variables
  int *tx;
  int *ty;
  int ntx, nty;
  double radius;
  double res;
  double *chunk_size; // size of chunks to read and process
  int n_chunk_size;   // number of sizes that were written (should be 2)
  int psf;             // flag: point spread function

  // sensors
  sen_t sen;     // Level-2 sensor dictionary
  sen_t sen2;    // secondary Level-2 sensor dictionary

  // features
  par_ftr_t ftr;
  
  // continuous fields multi-band
  par_con_t con;

  // QAI screening
  par_qai_t qai;
  
  // level of input data
  int input_level1;
  int input_level2;

  // temporal parameters
  date_t *date_range; // date range for the analysis (continous time period)
  date_t date_ignore_lnd07; // date after which LND07 should be ignored (deorbiting)
  int ndate, ndoy;
  int *doy_range;
  int date_doys[366];        // doys   that should be used (modulates date_from, date_to)
  int date_weeks[53];
  int date_months[13];
  int date_quarters[5];
  int nd, nw, nm, nq, ny;

  // miscellaneous
  char *f_gdalopt;   // file for GDAL options
  gdalopt_t gdalopt; // GDAL output options
  int format;        // output format
  int explode;
  int subfolders;
  int owr;             // flag: overwrite output
  int ithread;
  int othread;
  int cthread;
  int stream;
  int pretty_progress;
  int fail_if_empty;    // warn (false) or error (true) without IO

  // products
  par_prd_t prd;

  // algorithms
  par_bap_t bap;
  par_tsa_t tsa;
  par_cso_t cso;
  par_imp_t imp;
  par_cfi_t cfi;
  par_mcl_t mcl;
  par_smp_t smp;
  par_txt_t txt;
  par_lsm_t lsm;
  par_lib_t lib;
  par_udp_t udf;

} par_hl_t;

par_hl_t *allocate_param_higher();
void free_param_higher(par_hl_t *phl);
int parse_param_higher(par_hl_t *phl);

#ifdef __cplusplus
}
#endif

#endif

