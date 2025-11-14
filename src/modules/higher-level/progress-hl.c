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
This file contains functions for progress monitoring
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "progress-hl.h"


#define MAX_(a,b) (((a)>(b))? (a) : (b))

void rewind_stdout(progress_t *progress);
void processing_unit(progress_t *progress);
void reset_time(progress_t *progress);
void bound_time(progress_t *progress);
void add_time(progress_t *progress);
void eta(progress_t *progress);
void percent_done(progress_t *progress);
void print_progress_summary(progress_t *progress);
void print_progress_details(progress_t *progress);
void print_progress_runtime(progress_t *progress);


/** This function measures the progress
--- progress:      progress handle
--- task:     what task to measure
--- clock:    start or take measurement?
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void measure_progress(progress_t *progress, int task, int clock){
  
  
  if (clock == _CLOCK_NULL_) {
    progress->stopwatch.secs[task] = 0;
  } else if (clock == _CLOCK_TICK_) {
    time(&progress->stopwatch.TIME[task]);
  } else if (clock == _CLOCK_TOCK_) {
    progress->stopwatch.secs[task] = proctime(progress->stopwatch.TIME[task]);
  }

  return;
}


/** This function computes the chunk layout
--- cube:     datacube definition
--- chunk_size: requested chunk size
--- dim_chunks: dimensions of chunk layout
+++ Return:     void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void chunk_layout(cube_t *cube, double chunk_size[], dim_t *dim_chunks){
double tol = 5e-3;


  dim_chunks->cols = (int)(cube->tile_size[_X_]/chunk_size[_X_]);
  dim_chunks->rows = (int)(cube->tile_size[_Y_]/chunk_size[_Y_]);
  dim_chunks->cells = dim_chunks->cols*dim_chunks->rows;

  if (dim_chunks->cols < 1){
    printf("CHUNK_SIZE (X) cannot be larger than TILE_SIZE (X).\n");
    exit(FAILURE);
  }

  if (dim_chunks->rows < 1){
    printf("CHUNK_SIZE (Y) cannot be larger than TILE_SIZE (Y).\n");
    exit(FAILURE);
  }

  if (fmod(cube->tile_size[_X_], chunk_size[_X_]) > tol){
    printf("TILE_SIZE (X) must be a multiple of CHUNK_SIZE (X).\n");
    exit(FAILURE);
  }

  if (fmod(cube->tile_size[_Y_], chunk_size[_Y_]) > tol){
    printf("TILE_SIZE (Y) must be a multiple of CHUNK_SIZE (Y).\n");
    exit(FAILURE);
  }

  return;
}


/** This function initializes the progress handle
--- progress:      progress handle
--- cube:     datacube definition
--- phl:      HL parameters
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_progess(progress_t *progress, cube_t *cube, par_hl_t *phl){

  // initialize progress handle to zero
  memset(progress, 0, sizeof(progress_t));
  
  // 1st measurement of runtime
  measure_progress(progress, _TASK_RUNTIME_, _CLOCK_TICK_);
  
  // set non-zero initial values
  progress->pretty_progress = phl->pretty_progress;
  
  progress->thread[_TASK_INPUT_]   = phl->ithread;
  progress->thread[_TASK_COMPUTE_] = phl->cthread;
  progress->thread[_TASK_OUTPUT_]  = phl->othread;
  progress->thread[_TASK_ALL_]     = phl->ithread+phl->cthread+phl->othread;
  progress->thread[_TASK_RUNTIME_] = phl->ithread+phl->cthread+phl->othread;

  
  alloc_2D((void***)&progress->tiles, 2, cube->n_allowed_tiles, sizeof(int));
  memmove(progress->tiles[_X_], cube->allowed_tiles[_X_], cube->n_allowed_tiles*sizeof(int));
  memmove(progress->tiles[_Y_], cube->allowed_tiles[_Y_], cube->n_allowed_tiles*sizeof(int));
  progress->n_tiles = cube->n_allowed_tiles;

  chunk_layout(cube, phl->chunk_size, &progress->dim_chunks);

  progress->last.processing_unit = -3; // init processing unit at -3!
  progress->now.processing_unit  = -2; // init processing unit at -2!
  progress->next.processing_unit = -1; // init processing unit at -1!
  progress->n_processing_units = progress->n_tiles * progress->dim_chunks.cells;

  printf("number of processing units: %d\n", progress->n_processing_units);
  printf(" (active tiles: %d, chunks per tile: %d)\n", progress->n_tiles, progress->dim_chunks.cells);
  printf("\n");

  return;
}


/** This function rewinds stdout
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void rewind_stdout(progress_t *progress){
  
  
  if (!progress->pretty_progress) return;

#ifndef FORCE_DEBUG

int line, nline = 13;

  if (progress->now.processing_unit >= 0){

    for (line=0; line<=nline; line++) printf("\033[A\r");
    printf("\n");

  }

#endif
  
  return;
}


void forward_processing_unit(progress_t *progress, team_t *subset){


  subset->processing_unit++;

  if (subset->processing_unit < 0 ||
      subset->processing_unit >= progress->n_processing_units) {
    subset->tile_number = -1;
    subset->chunk_number = -1;
    subset->tile[_X_] = -1;
    subset->tile[_Y_] = -1;
    subset->chunk[_X_] = -1;
    subset->chunk[_Y_] = -1;
    return;
  }

  subset->tile_number = 
    floor(subset->processing_unit / 
    (float)progress->dim_chunks.cells);

  subset->tile[_X_] = progress->tiles[_X_][subset->tile_number]; 
  subset->tile[_Y_] = progress->tiles[_Y_][subset->tile_number];

  subset->chunk_number = 
    subset->processing_unit - 
    (subset->tile_number * progress->dim_chunks.cells);

  subset->chunk[_Y_] = 
    floor(subset->chunk_number / progress->dim_chunks.cols);
  subset->chunk[_X_] = 
    subset->chunk_number - 
    subset->chunk[_Y_] * progress->dim_chunks.cols;

  return;
}

/** This function sets the current processing units
--- progress: progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void processing_unit(progress_t *progress){


  forward_processing_unit(progress, &progress->now);
  forward_processing_unit(progress, &progress->next);
  forward_processing_unit(progress, &progress->last);

  return;
}


/** This function resets the clock
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void reset_time(progress_t *progress){


  measure_progress(progress, _TASK_INPUT_,   _CLOCK_NULL_);
  measure_progress(progress, _TASK_COMPUTE_, _CLOCK_NULL_);
  measure_progress(progress, _TASK_OUTPUT_,  _CLOCK_NULL_);
  
  return;
}


/** This function computes task-bound time
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void bound_time(progress_t *progress){


  progress->stopwatch.secs_bound[_TASK_INPUT_] += 
    MAX_(
      0, 
      progress->stopwatch.secs[_TASK_INPUT_] - 
      MAX_(
        progress->stopwatch.secs[_TASK_COMPUTE_], 
        progress->stopwatch.secs[_TASK_OUTPUT_]
      )
    );

  progress->stopwatch.secs_bound[_TASK_COMPUTE_] += 
    MAX_(
      0, 
      progress->stopwatch.secs[_TASK_COMPUTE_] - 
      MAX_(
        progress->stopwatch.secs[_TASK_INPUT_], 
        progress->stopwatch.secs[_TASK_OUTPUT_]
      )
    );

  progress->stopwatch.secs_bound[_TASK_OUTPUT_] += 
    MAX_(
      0, 
      progress->stopwatch.secs[_TASK_OUTPUT_] - 
      MAX_(
        progress->stopwatch.secs[_TASK_INPUT_], 
        progress->stopwatch.secs[_TASK_COMPUTE_]
      )
    );

  return;
}


/** This function adds the task-time
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void add_time(progress_t *progress){


  progress->stopwatch.secs_total[_TASK_INPUT_] += 
    progress->stopwatch.secs[_TASK_INPUT_];

  progress->stopwatch.secs_total[_TASK_COMPUTE_] += 
    progress->stopwatch.secs[_TASK_COMPUTE_];

  progress->stopwatch.secs_total[_TASK_OUTPUT_] += 
    progress->stopwatch.secs[_TASK_OUTPUT_];

  progress->stopwatch.secs_total[_TASK_ALL_] = 
    progress->stopwatch.secs_total[_TASK_INPUT_]   + 
    progress->stopwatch.secs_total[_TASK_COMPUTE_] + 
    progress->stopwatch.secs_total[_TASK_OUTPUT_];

  return;
}


/** This function computes the ETA
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void eta(progress_t *progress){


  if (progress->now.processing_unit > 0 && 
      progress->stopwatch.secs_total[_TASK_ALL_] > 0){

    set_secs(&progress->stopwatch.eta, 
      (progress->n_processing_units - progress->now.processing_unit) * 
      (progress->stopwatch.secs_total[_TASK_ALL_]/progress->now.processing_unit)
    );

  } else {

    set_secs(&progress->stopwatch.eta, 0);

  }

  return;
}


/** This function computes the percent progress
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void percent_done(progress_t *progress){


  if (progress->now.processing_unit >= 0){

    progress->done = 
      (float)progress->now.processing_unit / 
      progress->n_processing_units * 
      100.0;

  } else {

    progress->done = 0;

  }

  return;
}


/** This function prints progress summary
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_progress_summary(progress_t *progress){


  printf("________________________________________\n");
  printf("Progress:                        %6.2f%%\n", 
    progress->done);

  if (progress->stopwatch.secs_total[_TASK_ALL_] > 0){
    printf("Time for I/C/O:           %03.0f%%/%03.0f%%/%03.0f%%\n", 

      round(progress->stopwatch.secs_total[_TASK_INPUT_]   / 
            progress->stopwatch.secs_total[_TASK_ALL_]*100), 
      round(progress->stopwatch.secs_total[_TASK_COMPUTE_] / 
            progress->stopwatch.secs_total[_TASK_ALL_]*100), 
      round(progress->stopwatch.secs_total[_TASK_OUTPUT_]  / 
            progress->stopwatch.secs_total[_TASK_ALL_]*100));

  } else {

    printf("Time for I/C/O:        not available yet\n");

  }

  if (progress->now.processing_unit > 0 && 
      progress->stopwatch.secs_total[_TASK_ALL_] > 0){

    printf("ETA:             %02dy %02dm %02dd %02dh %02dm %02ds\n", 
      progress->stopwatch.eta.year, progress->stopwatch.eta.month, 
      progress->stopwatch.eta.day,  progress->stopwatch.eta.hh, 
      progress->stopwatch.eta.mm,   progress->stopwatch.eta.ss);

  } else {

    printf("ETA:                   not available yet\n"); 

  }

  return;
}



void print_progress_item_string(char *string, int width, bool align_left){

  if (align_left) {
    printf("%-*s", width, string);
  } else {
    printf("%*s", width, string);
  }

  return;
}


void print_progress_item_int(int val, int width){
 
  if (val < 0){
    printf("%*s", width, "NA");
  } else {
    printf("%*d", width, val);
  }

  return;
}

/** This function prints progress details
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_progress_details(progress_t *progress){


  printf("________________________________________\n");
  printf("                   input compute  output\n");

  print_progress_item_string("Processing unit:", 16, true);
  print_progress_item_int(progress->next.processing_unit, 8);
  print_progress_item_int(progress->now.processing_unit, 8);
  print_progress_item_int(progress->last.processing_unit, 8);
  printf("\n");

  print_progress_item_string("Tile X-ID:", 16, true);
  print_progress_item_int(progress->next.tile[_X_], 8);
  print_progress_item_int(progress->now.tile[_X_], 8);
  print_progress_item_int(progress->last.tile[_X_], 8);
  printf("\n");

  print_progress_item_string("Tile Y-ID:", 16, true);
  print_progress_item_int(progress->next.tile[_Y_], 8);
  print_progress_item_int(progress->now.tile[_Y_], 8);
  print_progress_item_int(progress->last.tile[_Y_], 8);
  printf("\n");

  print_progress_item_string("Chunk X-ID:", 16, true);
  print_progress_item_int(progress->next.chunk[_X_], 8);
  print_progress_item_int(progress->now.chunk[_X_], 8);
  print_progress_item_int(progress->last.chunk[_X_], 8);
  printf("\n");

  print_progress_item_string("Chunk Y-ID:", 16, true);
  print_progress_item_int(progress->next.chunk[_Y_], 8);
  print_progress_item_int(progress->now.chunk[_Y_], 8);
  print_progress_item_int(progress->last.chunk[_Y_], 8);
  printf("\n");

  print_progress_item_string("Threads:", 16, true);
  print_progress_item_int(progress->thread[_TASK_INPUT_], 8);
  print_progress_item_int(progress->thread[_TASK_COMPUTE_], 8);
  print_progress_item_int(progress->thread[_TASK_OUTPUT_], 8);
  printf("\n");

  print_progress_item_string("Time (sec):", 16, true);
  print_progress_item_int(progress->stopwatch.secs[_TASK_INPUT_], 8);
  print_progress_item_int(progress->stopwatch.secs[_TASK_COMPUTE_], 8);
  print_progress_item_int(progress->stopwatch.secs[_TASK_OUTPUT_], 8);
  printf("\n");

  return;
}


/** This function prints progress information at the end of runtime
--- progress:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_progress_runtime(progress_t *progress){


  progress->stopwatch.secs_total[_TASK_RUNTIME_] = progress->stopwatch.secs[_TASK_RUNTIME_];

  // total runtime
  set_secs(
    &progress->stopwatch.runtime, 
    progress->stopwatch.secs_total[_TASK_RUNTIME_]
  ); 

  // total time spent for I/C/O when streaming wouldn't take place
  set_secs(
    &progress->stopwatch.sequential[_TASK_ALL_], 
    progress->stopwatch.secs_total[_TASK_ALL_]
  );

  // time saved through streaming relative to when streaming wouldn't take place
  set_secs(
    &progress->stopwatch.saved,
    progress->stopwatch.secs_total[_TASK_ALL_] - 
    progress->stopwatch.secs_total[_TASK_RUNTIME_]
  ); 


  set_secs(
    &progress->stopwatch.sequential[_TASK_INPUT_],   
    progress->stopwatch.secs_total[_TASK_INPUT_]
  );

  set_secs(
    &progress->stopwatch.sequential[_TASK_COMPUTE_], 
    progress->stopwatch.secs_total[_TASK_COMPUTE_]
  );

  set_secs(
    &progress->stopwatch.sequential[_TASK_OUTPUT_],  
    progress->stopwatch.secs_total[_TASK_OUTPUT_]
  );


  set_secs(
    &progress->stopwatch.bound[_TASK_INPUT_],   progress->stopwatch.secs_bound[_TASK_INPUT_]
  );

  set_secs(
    &progress->stopwatch.bound[_TASK_COMPUTE_], progress->stopwatch.secs_bound[_TASK_COMPUTE_]
  );

  set_secs(
    &progress->stopwatch.bound[_TASK_OUTPUT_],  progress->stopwatch.secs_bound[_TASK_OUTPUT_]
  );

  printf("\n________________________________________\n");
  printf("Real time:       %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.runtime.year, 
    progress->stopwatch.runtime.month,
    progress->stopwatch.runtime.day,  
    progress->stopwatch.runtime.hh, 
    progress->stopwatch.runtime.mm,   
    progress->stopwatch.runtime.ss
  );

  printf("Virtual time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.sequential[_TASK_ALL_].year, 
    progress->stopwatch.sequential[_TASK_ALL_].month,
    progress->stopwatch.sequential[_TASK_ALL_].day,  
    progress->stopwatch.sequential[_TASK_ALL_].hh, 
    progress->stopwatch.sequential[_TASK_ALL_].mm,   
    progress->stopwatch.sequential[_TASK_ALL_].ss
  );

  printf("Saved time:      %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.saved.year, 
    progress->stopwatch.saved.month,
    progress->stopwatch.saved.day,  
    progress->stopwatch.saved.hh, 
    progress->stopwatch.saved.mm,   
    progress->stopwatch.saved.ss
  );

  printf("\n________________________________________\n");
  printf("Virtual I-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.sequential[_TASK_INPUT_].year, 
    progress->stopwatch.sequential[_TASK_INPUT_].month, 
    progress->stopwatch.sequential[_TASK_INPUT_].day,  
    progress->stopwatch.sequential[_TASK_INPUT_].hh,  
    progress->stopwatch.sequential[_TASK_INPUT_].mm,   
    progress->stopwatch.sequential[_TASK_INPUT_].ss
  );

  printf("Virtual C-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.sequential[_TASK_COMPUTE_].year, 
    progress->stopwatch.sequential[_TASK_COMPUTE_].month,
    progress->stopwatch.sequential[_TASK_COMPUTE_].day,  
    progress->stopwatch.sequential[_TASK_COMPUTE_].hh, 
    progress->stopwatch.sequential[_TASK_COMPUTE_].mm,   
    progress->stopwatch.sequential[_TASK_COMPUTE_].ss
  );

  printf("Virtual O-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.sequential[_TASK_OUTPUT_].year, 
    progress->stopwatch.sequential[_TASK_OUTPUT_].month,
    progress->stopwatch.sequential[_TASK_OUTPUT_].day,  
    progress->stopwatch.sequential[_TASK_OUTPUT_].hh, 
    progress->stopwatch.sequential[_TASK_OUTPUT_].mm,   
    progress->stopwatch.sequential[_TASK_OUTPUT_].ss
  );

  printf("\n________________________________________\n");
  printf("I-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.bound[_TASK_INPUT_].year, 
    progress->stopwatch.bound[_TASK_INPUT_].month,
    progress->stopwatch.bound[_TASK_INPUT_].day,  
    progress->stopwatch.bound[_TASK_INPUT_].hh, 
    progress->stopwatch.bound[_TASK_INPUT_].mm,   
    progress->stopwatch.bound[_TASK_INPUT_].ss
  );

  printf("C-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.bound[_TASK_COMPUTE_].year, 
    progress->stopwatch.bound[_TASK_COMPUTE_].month,
    progress->stopwatch.bound[_TASK_COMPUTE_].day,  
    progress->stopwatch.bound[_TASK_COMPUTE_].hh, 
    progress->stopwatch.bound[_TASK_COMPUTE_].mm,   
    progress->stopwatch.bound[_TASK_COMPUTE_].ss
  );

  printf("O-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    progress->stopwatch.bound[_TASK_OUTPUT_].year, 
    progress->stopwatch.bound[_TASK_OUTPUT_].month, 
    progress->stopwatch.bound[_TASK_OUTPUT_].day,  
    progress->stopwatch.bound[_TASK_OUTPUT_].hh, 
    progress->stopwatch.bound[_TASK_OUTPUT_].mm,   
    progress->stopwatch.bound[_TASK_OUTPUT_].ss
  );

  return;
}


/** This function tells whether this chunk should be input
--- progress:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool read_this_chunk(progress_t *progress){
  
  return (progress->next.processing_unit < progress->n_processing_units);
}


/** This function tells whether this chunk should be computed
--- progress:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool compute_this_chunk(progress_t *progress){

  return (progress->now.processing_unit >= 0 && 
    progress->now.processing_unit < progress->n_processing_units);
}


/** This function tells whether this chunk should be output
--- progress:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool write_this_chunk(progress_t *progress, int *nprod){

  return (progress->last.processing_unit >= 0 && 
          nprod[progress->last.processing_unit] > 0);
}


/** This function handles and prints progress
--- progress:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool progress(progress_t *progress){


  processing_unit(progress);

  rewind_stdout(progress);

  bound_time(progress);
  add_time(progress);


  if (progress->now.processing_unit > progress->n_processing_units){

    measure_progress(progress, _TASK_RUNTIME_, _CLOCK_TOCK_);

    print_progress_summary(progress);
    print_progress_runtime(progress);

    free_2D((void*)progress->tiles, 2);
    
    return false;
  
  } else {

    eta(progress);
    percent_done(progress);

    print_progress_summary(progress);
    print_progress_details(progress);

    reset_time(progress);
    
    return true;

  }

}


