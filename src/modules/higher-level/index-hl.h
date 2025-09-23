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
Spectral index header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef INDEX_HL_H
#define INDEX_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library

#include "../cross-level/cite-cl.h"
#include "../cross-level/table-cl.h"
#include "../higher-level/read-ard-hl.h"
#include "../higher-level/param-hl.h"
#include "../higher-level/tsa-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

int tsa_spectral_index(ard_t *ard, tsa_t *ts, small *mask_, int nc, int nt, int idx, short nodata, par_tsa_t *tsa, par_sen_t *sen, table_t *endmember);

#ifdef __cplusplus
}
#endif

#endif

