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
Progress header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef PROGRESS_HL_H
#define PROGRESS_HL_H

#include <stdio.h>   // core input and output functions
#include <stdlib.h>  // standard general utilities library
#include <stdbool.h>  // boolean data type
#include <time.h>    // date and time handling functions

#include "../cross-level/const-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/cube-cl.h"
#include "../cross-level/utils-cl.h"
#include "../higher-level/param-hl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int thread[_TASK_LENGTH_];
  int pu, pu_next, pu_prev, npu;
  int tile, tile_next, tile_prev;
  int chunk, chunk_next, chunk_prev, nchunk;
  int tx, tx_next, tx_prev;
  int ty, ty_next, ty_prev;
  int *tiles_x, *tiles_y;
  float done;
  time_t TIME[_TASK_LENGTH_];
  double secs[_TASK_LENGTH_];
  double secs_total[_TASK_LENGTH_];
  double secs_bound[_TASK_LENGTH_];
  date_t eta, runtime, saved;
  date_t bound[_TASK_LENGTH_];
  date_t sequential[_TASK_LENGTH_];
} progress_t;

void measure_progress(progress_t *pro, int task, int clock);
void init_progess(progress_t *pro, cube_t *cube, par_hl_t *phl);
bool read_this_chunk(progress_t *pro);
bool compute_this_chunk(progress_t *pro);
bool write_this_chunk(progress_t *pro, int *nprod);
bool progress(progress_t *pro);

#ifdef __cplusplus
}
#endif

#endif

