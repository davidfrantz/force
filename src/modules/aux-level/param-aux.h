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
Aux parameter header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef PARAM_AUX_H
#define PARAM_AUX_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h> // boolean data type


#ifdef __cplusplus
extern "C" {
#endif

void write_par_ll_dirs(FILE *fp, bool verbose);
void write_par_ll_aoi(FILE *fp, bool verbose);
void write_par_ll_dem(FILE *fp, bool verbose);
void write_par_ll_cube(FILE *fp, bool verbose);
void write_par_ll_atcor(FILE *fp, bool verbose);
void write_par_ll_wvp(FILE *fp, bool verbose);
void write_par_ll_aod(FILE *fp, bool verbose);
void write_par_ll_cloud(FILE *fp, bool verbose);
void write_par_ll_resmerge(FILE *fp, bool verbose);
void write_par_ll_coreg(FILE *fp, bool verbose);
void write_par_ll_misc(FILE *fp, bool verbose);
void write_par_ll_tier(FILE *fp, bool verbose);
void write_par_ll_thread(FILE *fp, bool verbose);
void write_par_ll_output(FILE *fp, bool verbose);
void write_par_hl_dirs(FILE *fp, bool verbose);
void write_par_hl_mask(FILE *fp, bool verbose);
void write_par_hl_extent(FILE *fp, bool verbose);
void write_par_hl_psf(FILE *fp, bool verbose);
void write_par_hl_improphed(FILE *fp, bool verbose);
void write_par_hl_sensor(FILE *fp, bool verbose);
void write_par_hl_qai(FILE *fp, bool verbose);
void write_par_hl_noise(FILE *fp, bool verbose);
void write_par_hl_time(FILE *fp, bool verbose);
void write_par_hl_output(FILE *fp, bool verbose);
void write_par_hl_thread(FILE *fp, bool verbose);
void write_par_hl_bap(FILE *fp, bool verbose);
void write_par_hl_pac(FILE *fp, bool verbose);
void write_par_hl_index(FILE *fp, bool verbose);
void write_par_hl_sma(FILE *fp, bool verbose);
void write_par_hl_tsi(FILE *fp, bool verbose);
void write_par_hl_pyp(FILE *fp, bool verbose);
void write_par_hl_rsp(FILE *fp, bool verbose);
void write_par_hl_stm(FILE *fp, bool verbose);
void write_par_hl_fold(FILE *fp, bool verbose);
void write_par_hl_pol(FILE *fp, bool verbose);
void write_par_hl_trend(FILE *fp, bool verbose);
void write_par_hl_cso(FILE *fp, bool verbose);
void write_par_hl_imp(FILE *fp, bool verbose);
void write_par_hl_cfi(FILE *fp, bool verbose);
void write_par_hl_l2i(FILE *fp, bool verbose);
void write_par_hl_feature(FILE *fp, bool verbose);
void write_par_hl_txt(FILE *fp, bool verbose);
void write_par_hl_lsm(FILE *fp, bool verbose);
void write_par_hl_lib(FILE *fp, bool verbose);
void write_par_hl_smp(FILE *fp, bool verbose);
void write_par_hl_ml(FILE *fp, bool verbose);
void write_par_aux_train(FILE *fp, bool verbose);
void write_par_aux_synthmix(FILE *fp, bool verbose);

#ifdef __cplusplus
}
#endif

#endif

