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
Enhance spatial resolution header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef RESMERGE_LL_H
#define RESMERGE_LL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h> // boolean data type

#include "../cross-level/const-cl.h"
#include "../cross-level/quality-cl.h"
#include "../cross-level/brick_base-cl.h"
#include "../cross-level/brick_io-cl.h"
#include "../cross-level/imagefuns-cl.h"
#include "../cross-level/cite-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

int resolution_merge(int mission, int resmerge, brick_t *TOA, brick_t *QAI);

#ifdef __cplusplus
}
#endif

#endif

