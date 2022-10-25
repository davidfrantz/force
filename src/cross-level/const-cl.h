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
Named constant definitions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef CONSTANT_CL_H
#define CONSTANT_CL_H

#include "../cross-level/_version-cl.h"
#include "../cross-level/enum-cl.h"

#ifdef __cplusplus
extern "C" {
#endif

// abbreviated datatypes
typedef unsigned short int ushort;
typedef unsigned char small;

// coordinate struct
typedef struct {
  double x, y;
} coord_t;


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

