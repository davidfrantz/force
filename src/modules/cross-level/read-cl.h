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
Reading all-purpose files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef READ_CL_H
#define READ_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/alloc-cl.h"
#include "../cross-level/string-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

char ***read_tagvalue(char *fname, int *nrows);

#ifdef __cplusplus
}
#endif

#endif


