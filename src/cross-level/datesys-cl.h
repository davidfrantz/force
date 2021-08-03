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
System date header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef DATESYS_CL_H
#define DATESYS_CL_H

#include <stdio.h>   // core input and output functions
#include <math.h>    // common mathematical functions
#include <time.h>    // date and time handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/date-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

void current_date(date_t *d);
void date_plus(date_t *d);
void date_minus(date_t *d);
bool date_overshoot(date_t *d);
int date_order(date_t *d_early, date_t *d_late);

#ifdef __cplusplus
}
#endif

#endif

