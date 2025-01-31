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
R UDF plug-in header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef RSP_HL_H
#define RSP_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../higher-level/tsa-hl.h"
#include "../higher-level/udf-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

void register_rstats(par_hl_t *phl);
void deregister_rstats(par_hl_t *phl);
void init_rsp(ard_t *ard, tsa_t *ts, int submodule, char *idx_name, int nb, int nt, par_udf_t *udf);
void term_rsp(par_udf_t *udf);
int rstats_udf(ard_t *ard, udf_t *udf_, tsa_t *ts, small *mask_, int submodule, char *idx_name, int nx, int ny, int nc, int nb, int nt, short nodata, par_udf_t *udf, int cthread);

#ifdef __cplusplus
}
#endif

#endif

