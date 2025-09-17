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
  int tile[2];
  int chunk[2];
  int tile_number;
  int chunk_number;
  int processing_unit;
} team_t;

typedef struct {
  time_t TIME[_TASK_LENGTH_];
  double secs[_TASK_LENGTH_];
  double secs_total[_TASK_LENGTH_];
  double secs_bound[_TASK_LENGTH_];
  date_t eta, runtime, saved;
  date_t bound[_TASK_LENGTH_];
  date_t sequential[_TASK_LENGTH_];
} stopwatch_t;

typedef struct {
  team_t now, next, last;
  stopwatch_t stopwatch;
  int thread[_TASK_LENGTH_];
  int **tiles;
  int n_tiles; // tile layout is irregular
  dim_t dim_chunks; // chunk layout is rectangular
  int n_processing_units; // # of tiles x chunks
  float done;
  int pretty_progress;
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

