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
Image methods header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef IMAGEFUNS_CL_H
#define IMAGEFUNS_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>   // standard general utilities library

#include "../cross-level/const-cl.h"
#include "../cross-level/dir-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

int download_file(char *f_remote, char *f_local, char *header);
int download_pattern(char *d_local, char *pattern, char *header);

#ifdef __cplusplus
}
#endif

#endif

