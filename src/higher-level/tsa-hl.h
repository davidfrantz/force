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
TSA Processing header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef TSA_HL_H
#define TSA_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/brick-cl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/read-ard-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  double **tab; // table
  int nb;       // number of bands
  int ne;       // number of endmembers
} aux_emb_t;

typedef struct {
  short **tss_, **rms_, **stm_, **tsi_, **spl_;
  short **fby_, **fbq_, **fbm_, **fbw_, **fbd_;
  short **try_, **trq_, **trm_, **trw_, **trd_;
  short **cay_, **caq_, **cam_, **caw_, **cad_;
  short **lsp_[_LSP_LENGTH_];
  short **trp_[_LSP_LENGTH_];
  short **cap_[_LSP_LENGTH_];
  short **pcx_, **pcy_;
  short **pol_[_POL_LENGTH_];
  short **tro_[_POL_LENGTH_];
  short **cao_[_POL_LENGTH_];
  short **pyp_, **rsp_, **nrt_;
  date_t *d_tss, *d_nrt, *d_tsi;
  date_t *d_fby, *d_fbq, *d_fbm, *d_fbw, *d_fbd;
  date_t *d_lsp, *d_pol;
} tsa_t;

#include "../higher-level/index-hl.h"
#include "../higher-level/interpolate-hl.h"
#include "../higher-level/stm-hl.h"
#include "../higher-level/fold-hl.h"
#include "../higher-level/trend-hl.h"
#include "../higher-level/pheno-hl.h"
#include "../higher-level/polar-hl.h"
#include "../higher-level/standardize-hl.h"
#include "../higher-level/py-udf-hl.h"
#include "../higher-level/r-udf-hl.h"

brick_t **time_series_analysis(ard_t *ard, brick_t *mask, int nt, par_hl_t *phl, aux_emb_t *endmember, cube_t *cube, int *nproduct);


#ifdef __cplusplus
}
#endif

#endif

