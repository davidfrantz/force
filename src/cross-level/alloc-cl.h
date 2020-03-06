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
Allocation header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef ALLOC_CL_H
#define ALLOC_CL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <string.h>  // string handling functions


#ifdef __cplusplus
extern "C" {
#endif

void alloc(void **ptr, size_t n, size_t size);
void alloc_2D(void ***ptr, size_t n1, size_t n2, size_t size);
void alloc_3D(void ****ptr, size_t n1, size_t n2, size_t n3, size_t size);
void alloc_2DC(void ***ptr, size_t n1, size_t n2, size_t size);
void re_alloc(void **ptr, size_t n_now, size_t n, size_t size);
void re_alloc_2D(void ***ptr, size_t n1_now, size_t n2_now, size_t n1, size_t n2, size_t size);
void re_alloc_3D(void ****ptr, size_t n1_now, size_t n2_now, size_t n3_now, size_t n1, size_t n2, size_t n3, size_t size);
void re_alloc_2DC(void ***ptr, size_t n1_now, size_t n2_now, size_t n1, size_t n2, size_t size);
void free_2D(void **ptr, size_t n);
void free_3D(void ***ptr, size_t n1, size_t n2);
void free_2DC(void **ptr);

#ifdef __cplusplus
}
#endif

#endif

