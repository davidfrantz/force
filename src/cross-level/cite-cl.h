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
Citation header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef CITE_CL_H
#define CITE_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h>  // boolean data type
#include <time.h>    // date and time handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/dir-cl.h"
#include "../cross-level/lock-cl.h"
#include "../cross-level/utils-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

enum { _CITE_FORCE_,       _CITE_L2PS_,       _CITE_ATMVAL_,
       _CITE_AODFALLBACK_, _CITE_CLOUD_,      _CITE_IMPROPHE_,
       _CITE_STARFM_,      _CITE_REGRESSION_, _CITE_BAP_,
       _CITE_WVDB_,        _CITE_BRDF_,       _CITE_COREG_,
       _CITE_RADTRAN_,     _CITE_ADJACENCY_,  _CITE_AODEST_,
       _CITE_RADCOR_,      _CITE_TOPCOR_,     _CITE_CSO_,
       _CITE_STM_,         _CITE_RBF_,        _CITE_SPLITS_,
       _CITE_CAT_,         _CITE_NDVI_,       _CITE_EVI_,
       _CITE_NBR_,         _CITE_SARVI_,      _CITE_TCAP_,
       _CITE_DISTURBANCE_, _CITE_NDBI_,       _CITE_NDWI_,
       _CITE_MNDWI_,       _CITE_NDSI_,       _CITE_SMA_,
       _CITE_EQUI7_,       _CITE_RESMERGE_,   _CITE_LSM_,
       _CITE_NDTI_,        _CITE_NDMI_,       _CITE_POL_,
       _CITE_SPECADJUST_,  _CITE_KNDVI_,      _CITE_NDRE1_,
       _CITE_NDRE2_,       _CITE_CIre_,       _CITE_NDVIre1_,
       _CITE_NDVIre2_,     _CITE_NDVIre3_,    _CITE_NDVIre1n_,
       _CITE_NDVIre2n_,    _CITE_NDVIre3n_,   _CITE_MSRre_,
       _CITE_MSRren_,      _CITE_CCI_,        _CITE_EV2_,
       _CITE_HARMONIC_,    _CITE_RVI_,        _CITE_LENGTH_ };

typedef struct {
  char description[NPOW_10];
  char reference[NPOW_12];
  bool cited;
} cite_t;

extern cite_t _cite_me_[_CITE_LENGTH_];
extern FILE *_cite_fp_;

void cite_me(int i);
void cite_push(char *dname);

#ifdef __cplusplus
}
#endif

#endif

