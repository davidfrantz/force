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
This file contains functions for progress monitoring
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "progress-hl.h"


#define MAX_(a,b) (((a)>(b))? (a) : (b))

void rewind_stdout(progress_t *pro);
void processing_unit(progress_t *pro);
void reset_time(progress_t *pro);
void bound_time(progress_t *pro);
void add_time(progress_t *pro);
void eta(progress_t *pro);
void percent_done(progress_t *pro);
void print_progress_summary(progress_t *pro);
void print_progress_details(progress_t *pro);
void print_progress_runtime(progress_t *pro);


/** This function measures the progress
--- pro:      progress handle
--- task:     what task to measure
--- clock:    start or take measurement?
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void measure_progress(progress_t *pro, int task, int clock){
  
  
  if (clock == _CLOCK_NULL_) pro->secs[task] = 0;
  if (clock == _CLOCK_TICK_) time(&pro->TIME[task]);
  if (clock == _CLOCK_TOCK_) pro->secs[task] = proctime(pro->TIME[task]);
  
  return;
}


/** This function initializes the progress handle
--- pro:      progress handle
--- cube:     datacube definition
--- phl:      HL parameters
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void init_progess(progress_t *pro, cube_t *cube, par_hl_t *phl){


  measure_progress(pro, _TASK_RUNTIME_, _CLOCK_TICK_);
  
  pro->thread[_TASK_INPUT_]   = phl->ithread;
  pro->thread[_TASK_COMPUTE_] = phl->cthread;
  pro->thread[_TASK_OUTPUT_]  = phl->othread;
  pro->thread[_TASK_ALL_]     = phl->ithread+phl->cthread+phl->othread;
  pro->thread[_TASK_RUNTIME_] = phl->ithread+phl->cthread+phl->othread;

  pro->npu = cube->tn*cube->cn;

  alloc((void**)&pro->tiles_x, cube->tn, sizeof(int));
  alloc((void**)&pro->tiles_y, cube->tn, sizeof(int));
  memmove(pro->tiles_x, cube->tx, cube->tn*sizeof(int));
  memmove(pro->tiles_y, cube->ty, cube->tn*sizeof(int));


  pro->pu       = -2; // init processing unit at -2!
  pro->pu_prev  =  0;
  pro->pu_next  =  0;
  pro->done     =  0;

  pro->tile      = 0;
  pro->tile_prev = 0;
  pro->tile_next = 0;
  
  pro->tx      = pro->tiles_x[0];
  pro->tx_prev = pro->tiles_x[0];
  pro->tx_next = pro->tiles_x[0];
  
  pro->ty      = pro->tiles_y[0];
  pro->ty_prev = pro->tiles_y[0];
  pro->ty_next = pro->tiles_y[0];

  pro->chunk      = -1;
  pro->chunk_prev = -2;
  pro->chunk_next =  0;
  pro->nchunk     = cube->cn;


  pro->secs[_TASK_INPUT_]   = 0; 
  pro->secs[_TASK_COMPUTE_] = 0; 
  pro->secs[_TASK_OUTPUT_]  = 0;
  pro->secs[_TASK_ALL_]     = 0;
  pro->secs[_TASK_RUNTIME_] = 0;

  pro->secs_total[_TASK_INPUT_]   = 0; 
  pro->secs_total[_TASK_COMPUTE_] = 0; 
  pro->secs_total[_TASK_OUTPUT_]  = 0;
  pro->secs_total[_TASK_ALL_]     = 0;
  pro->secs_total[_TASK_RUNTIME_] = 0;

  pro->secs_bound[_TASK_INPUT_]   = 0; 
  pro->secs_bound[_TASK_COMPUTE_] = 0; 
  pro->secs_bound[_TASK_OUTPUT_]  = 0;
  pro->secs_bound[_TASK_ALL_]     = 0;
  pro->secs_bound[_TASK_RUNTIME_] = 0;

  init_date(&pro->eta);
  init_date(&pro->runtime);
  init_date(&pro->saved);

  init_date(&pro->bound[_TASK_INPUT_]);
  init_date(&pro->bound[_TASK_COMPUTE_]);
  init_date(&pro->bound[_TASK_OUTPUT_]);
  init_date(&pro->bound[_TASK_ALL_]);
  init_date(&pro->bound[_TASK_RUNTIME_]);
  
  init_date(&pro->sequential[_TASK_INPUT_]);
  init_date(&pro->sequential[_TASK_COMPUTE_]);
  init_date(&pro->sequential[_TASK_OUTPUT_]);
  init_date(&pro->sequential[_TASK_ALL_]); 
  init_date(&pro->sequential[_TASK_RUNTIME_]); 
  
  printf("number of processing units: %d\n", pro->npu);
  printf(" (active tiles: %d, chunks per tile: %d)\n", cube->tn, cube->cn);
  
  return;
}


/** This function rewinds stdout
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void rewind_stdout(progress_t *pro){
  
#ifndef FORCE_DEBUG

int line, nline = 12;

  if (pro->pu >=0){
    for (line=0; line<=nline; line++) printf("\033[A\r");
    printf("\n");
  }

#endif
  
  return;
}


/** This function sets the current processing units
--- pro:      progress handle
--- cube:     datacube definition
--- phl:      HL parameters
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void processing_unit(progress_t *pro){
  
  pro->pu++;
  pro->pu_prev  = pro->pu-1;
  pro->pu_next  = pro->pu+1;

  if (pro->pu >= 0 && pro->pu < pro->npu){
    pro->tile = floor(pro->pu      / (float)pro->nchunk);
  } else pro->tile = -1;

  if (pro->pu_next >= 0 && pro->pu_next < pro->npu){
    pro->tile_next = floor(pro->pu_next / (float)pro->nchunk);
  } else pro->tile_next = -1;
  
  if (pro->pu_prev >= 0 && pro->pu_prev < pro->npu){
    pro->tile_prev = floor(pro->pu_prev / (float)pro->nchunk);
  } else pro->tile_prev = -1;


  if (pro->tile > -1){
    pro->chunk = pro->pu - (pro->tile * pro->nchunk);
  } else pro->chunk  = -1;
  
  if (pro->tile_next > -1){
    pro->chunk_next = pro->pu_next - (pro->tile_next * pro->nchunk);
  } else pro->chunk_next = -1;
  
  if (pro->tile_prev > -1){
    pro->chunk_prev = pro->pu_prev - (pro->tile_prev * pro->nchunk);
  } else pro->chunk_prev = -1;


  if (pro->tile > -1){
    pro->tx = pro->tiles_x[pro->tile]; 
  } else pro->tx = -1;

  if (pro->tile > -1){
    pro->ty = pro->tiles_y[pro->tile];
  } else pro->ty = -1;

  if (pro->tile_next > -1){
    pro->tx_next = pro->tiles_x[pro->tile_next]; 
  } else pro->tx_next = -1;

  if (pro->tile_next > -1){
    pro->ty_next = pro->tiles_y[pro->tile_next];
  } else pro->ty_next = -1;

  if (pro->tile_prev > -1){
    pro->tx_prev = pro->tiles_x[pro->tile_prev]; 
  } else pro->tx_prev = -1;

  if (pro->tile_prev > -1){
    pro->ty_prev = pro->tiles_y[pro->tile_prev];
  } else pro->ty_prev = -1;
  
  return;
}


/** This function resets the clock
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void reset_time(progress_t *pro){


  measure_progress(pro, _TASK_INPUT_,   _CLOCK_NULL_);
  measure_progress(pro, _TASK_COMPUTE_, _CLOCK_NULL_);
  measure_progress(pro, _TASK_OUTPUT_,  _CLOCK_NULL_);
  
  return;
}


/** This function computes task-bound time
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void bound_time(progress_t *pro){


  pro->secs_bound[_TASK_INPUT_] += 
    MAX_(0, pro->secs[_TASK_INPUT_] - 
    MAX_(pro->secs[_TASK_COMPUTE_], pro->secs[_TASK_OUTPUT_]));

  pro->secs_bound[_TASK_COMPUTE_] += 
    MAX_(0, pro->secs[_TASK_COMPUTE_] - 
    MAX_(pro->secs[_TASK_INPUT_], pro->secs[_TASK_OUTPUT_]));

  pro->secs_bound[_TASK_OUTPUT_] += 
    MAX_(0, pro->secs[_TASK_OUTPUT_] - 
    MAX_(pro->secs[_TASK_INPUT_], pro->secs[_TASK_COMPUTE_]));

  return;
}


/** This function adds the task-time
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void add_time(progress_t *pro){


  pro->secs_total[_TASK_INPUT_]   += pro->secs[_TASK_INPUT_];
  pro->secs_total[_TASK_COMPUTE_] += pro->secs[_TASK_COMPUTE_];
  pro->secs_total[_TASK_OUTPUT_]  += pro->secs[_TASK_OUTPUT_];
  pro->secs_total[_TASK_ALL_] = pro->secs_total[_TASK_INPUT_]   + 
                                pro->secs_total[_TASK_COMPUTE_] + 
                                pro->secs_total[_TASK_OUTPUT_];

  return;
}


/** This function computes the ETA
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void eta(progress_t *pro){


  if (pro->pu > 0 && pro->secs_total[_TASK_ALL_] > 0){
    set_secs(&pro->eta, (pro->npu-pro->pu) * 
                        (pro->secs_total[_TASK_ALL_]/pro->pu));
  } else {
    set_secs(&pro->eta, 0);
  }

  return;
}


/** This function computes the percent progress
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void percent_done(progress_t *pro){


  if (pro->pu >= 0){
    pro->done = (float)pro->pu/pro->npu*100.0;
  } else {
    pro->done = 0;
  }

  return;
}


/** This function prints progress summary
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_progress_summary(progress_t *pro){


  printf("________________________________________\n");
  printf("Progress:                        %6.2f%%\n", 
    pro->done);

  if (pro->secs_total[_TASK_ALL_] > 0){
    printf("Time for I/C/O:           %03.0f%%/%03.0f%%/%03.0f%%\n", 
      round(pro->secs_total[_TASK_INPUT_]   / 
            pro->secs_total[_TASK_ALL_]*100), 
      round(pro->secs_total[_TASK_COMPUTE_] / 
            pro->secs_total[_TASK_ALL_]*100), 
      round(pro->secs_total[_TASK_OUTPUT_]  / 
            pro->secs_total[_TASK_ALL_]*100));
  } else {
    printf("Time for I/C/O:        not available yet\n");
  }

  if (pro->pu > 0 && pro->secs_total[_TASK_ALL_] > 0){
    printf("ETA:             %02dy %02dm %02dd %02dh %02dm %02ds\n", 
      pro->eta.year, pro->eta.month, 
      pro->eta.day,  pro->eta.hh, 
      pro->eta.mm,   pro->eta.ss);
  } else {
    printf("ETA:                   not available yet\n"); 
  }

  return;
}


/** This function prints progress details
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_progress_details(progress_t *pro){


  printf("________________________________________\n");
  printf("                   input compute  output\n");

  printf("Processing unit: %7d %7d %7d\n", 
    pro->pu_next, pro->pu, pro->pu_prev);
    
  printf("Tile X-ID:       %7d %7d %7d\n", 
    pro->tx_next, pro->tx, pro->tx_prev);
    
  printf("Tile Y-ID:       %7d %7d %7d\n", 
    pro->ty_next, pro->ty, pro->ty_prev);
    
  printf("Chunk ID:        %7d %7d %7d\n", 
    pro->chunk_next, pro->chunk, pro->chunk_prev);
  
  printf("Threads:         %7d %7d %7d\n", 
    pro->thread[_TASK_INPUT_], pro->thread[_TASK_COMPUTE_], pro->thread[_TASK_OUTPUT_]);

  printf("Time (sec):      %7.0f %7.0f %7.0f\n", 
    pro->secs[_TASK_INPUT_], pro->secs[_TASK_COMPUTE_], pro->secs[_TASK_OUTPUT_]);

  return;
}


/** This function prints progress information at the end of runtime
--- pro:      progress handle
+++ Return:   void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void print_progress_runtime(progress_t *pro){


  pro->secs_total[_TASK_RUNTIME_] = pro->secs[_TASK_RUNTIME_];

  set_secs(&pro->runtime,             // total runtime
    pro->secs_total[_TASK_RUNTIME_]); 
  set_secs(&pro->sequential[_TASK_ALL_], // total time spent for I/C/O 
    pro->secs_total[_TASK_ALL_]);     // when streaming wouldn't take place
  set_secs(&pro->saved,               // time saved through streaming 
    pro->secs_total[_TASK_ALL_] -     // relative to when streaming
    pro->secs_total[_TASK_RUNTIME_]); //  wouldn't take place


  set_secs(&pro->sequential[_TASK_INPUT_],   pro->secs_total[_TASK_INPUT_]);
  set_secs(&pro->sequential[_TASK_COMPUTE_], pro->secs_total[_TASK_COMPUTE_]);
  set_secs(&pro->sequential[_TASK_OUTPUT_],  pro->secs_total[_TASK_OUTPUT_]);
  
  set_secs(&pro->bound[_TASK_INPUT_],   pro->secs_bound[_TASK_INPUT_]);
  set_secs(&pro->bound[_TASK_COMPUTE_], pro->secs_bound[_TASK_COMPUTE_]);
  set_secs(&pro->bound[_TASK_OUTPUT_],  pro->secs_bound[_TASK_OUTPUT_]);

  printf("\n________________________________________\n");
  printf("Real time:       %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->runtime.year, pro->runtime.month,
    pro->runtime.day,  pro->runtime.hh, 
    pro->runtime.mm,   pro->runtime.ss);  

  printf("Virtual time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->sequential[_TASK_ALL_].year, pro->sequential[_TASK_ALL_].month,
    pro->sequential[_TASK_ALL_].day,  pro->sequential[_TASK_ALL_].hh, 
    pro->sequential[_TASK_ALL_].mm,   pro->sequential[_TASK_ALL_].ss);
    
  printf("Saved time:      %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->saved.year, pro->saved.month,
    pro->saved.day,  pro->saved.hh, 
    pro->saved.mm,   pro->saved.ss);

  printf("\n________________________________________\n");
  printf("Virtual I-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->sequential[_TASK_INPUT_].year, pro->sequential[_TASK_INPUT_].month, 
    pro->sequential[_TASK_INPUT_].day,  pro->sequential[_TASK_INPUT_].hh,  
    pro->sequential[_TASK_INPUT_].mm,   pro->sequential[_TASK_INPUT_].ss);
    
  printf("Virtual C-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->sequential[_TASK_COMPUTE_].year, pro-> sequential[_TASK_COMPUTE_].month,
    pro->sequential[_TASK_COMPUTE_].day,  pro->sequential[_TASK_COMPUTE_].hh, 
    pro->sequential[_TASK_COMPUTE_].mm,   pro->sequential[_TASK_COMPUTE_].ss);
    
  printf("Virtual O-time:  %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->sequential[_TASK_OUTPUT_].year, pro->sequential[_TASK_OUTPUT_].month,
    pro->sequential[_TASK_OUTPUT_].day,  pro->sequential[_TASK_OUTPUT_].hh, 
    pro->sequential[_TASK_OUTPUT_].mm,   pro->sequential[_TASK_OUTPUT_].ss);

  printf("\n________________________________________\n");
  printf("I-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->bound[_TASK_INPUT_].year, pro->bound[_TASK_INPUT_].month,
    pro->bound[_TASK_INPUT_].day,  pro->bound[_TASK_INPUT_].hh, 
    pro->bound[_TASK_INPUT_].mm,   pro->bound[_TASK_INPUT_].ss);
    
  printf("C-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->bound[_TASK_COMPUTE_].year, pro->bound[_TASK_COMPUTE_].month,
    pro->bound[_TASK_COMPUTE_].day,  pro->bound[_TASK_COMPUTE_].hh, 
    pro->bound[_TASK_COMPUTE_].mm,   pro->bound[_TASK_COMPUTE_].ss);
    
  printf("O-bound time:    %02dy %02dm %02dd %02dh %02dm %02ds\n", 
    pro->bound[_TASK_OUTPUT_].year, pro->bound[_TASK_OUTPUT_].month, 
    pro->bound[_TASK_OUTPUT_].day,  pro->bound[_TASK_OUTPUT_].hh, 
    pro->bound[_TASK_OUTPUT_].mm,   pro->bound[_TASK_OUTPUT_].ss);

  return;
}


/** This function tells whether this chunk should be input
--- pro:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool read_this_chunk(progress_t *pro){
  
  return (pro->pu_next < pro->npu);
}


/** This function tells whether this chunk should be computed
--- pro:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool compute_this_chunk(progress_t *pro){
  
  return (pro->pu >= 0 && pro->pu < pro->npu);
}


/** This function tells whether this chunk should be output
--- pro:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool write_this_chunk(progress_t *pro, int *nprod){
  
  return (pro->pu_prev >= 0 && nprod[pro->pu_prev] > 0);
}


/** This function handles and prints progress
--- pro:      progress handle
+++ Return:   true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool progress(progress_t *pro){


  processing_unit(pro);

//  rewind_stdout(pro);

  bound_time(pro);
  add_time(pro);


  if (pro->pu > pro->npu){

    measure_progress(pro, _TASK_RUNTIME_, _CLOCK_TOCK_);

    print_progress_summary(pro);
    print_progress_runtime(pro);

    free((void*)pro->tiles_x);
    free((void*)pro->tiles_y);
    
    return false;
  
  } else {

    eta(pro);
    percent_done(pro);

    print_progress_summary(pro);
    print_progress_details(pro);

    reset_time(pro);
    
    return true;

  }

}


