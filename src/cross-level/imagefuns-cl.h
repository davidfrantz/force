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
#include <string.h>  // string handling functions
#include <limits.h>  // macro constants of the integer types

#include "../cross-level/const-cl.h"
#include "../cross-level/stack-cl.h"
#include "../cross-level/queue-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

float find_sigma(float r);
int gauss_kernel(int nk, float sigma, float ***kernel);
int distance_kernel(int nk, float ***kernel);
int buffer(stack_t *stack, int b, int r);
int buffer_(small *image, int nx, int ny, int r);
int majorfill(stack_t *stack, int b);
int majorfill_(small *image, int nx, int ny);
ushort *dist_transform(stack_t *stack, int b);
ushort *dist_transform_(small *image, int nx, int ny);
int dt_dfun(int nx, int x, int i, int y, ushort *G);
int dt_Sep(int nx, int i, int u, int y, ushort *G);
int connectedcomponents(stack_t *stack, int b_stack, stack_t *segmentation, int b_segmentation);
int connectedcomponents_(small *image, int *CCL, int nx, int ny);
void ccl_tracer(int *cy, int *cx, int *dir, small *image, int *CCL, int nx, int ny);
void ccl_contourtrace(int cy, int cx, int label, int dir, small *image, int *CCL, int nx, int ny);
int binary_to_objects(small *image, int nx, int ny, int nmin, int **OBJ, int **SIZE, int *nobj);
int greyscale_reconstruction(stack_t *mask, int b_mask, stack_t *marker, int b_marker);
int greyscale_reconstruction_(short *MASK, short *MARKER, int nx, int ny);

#ifdef __cplusplus
}
#endif

#endif

