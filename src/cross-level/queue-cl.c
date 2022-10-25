/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2022 David Frantz, Sebastian Mader

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
This file contains an implementation of a First-In-First-Out queue
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "queue-cl.h"


/** This function creates a FIFO queue using a circular buffer. Free with 
+++ destroy_queue.
--- q:      queue
--- size:   size of the buffer
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int create_queue(queue_t *q, int size){

  q->head = 0;
  q->tail = 0;
  q->size = size;

  alloc((void**)&q->buf_x, size, sizeof(short));
  alloc((void**)&q->buf_y, size, sizeof(short));

  return SUCCESS;
}


/** This function frees a FIFO queue
--- q:      queue
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void destroy_queue(queue_t *q){

  q->head = 0;
  q->tail = 0;
  q->size = 0;
  
  free((void*)q->buf_x); q->buf_x = NULL;
  free((void*)q->buf_y); q->buf_y = NULL;

  return;
}


/** This function puts an image coordinate to the FIFO queue
--- q:      queue
--- x:      column
--- y:      row
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int enqueue(queue_t *q, int x, int y){

  if ((q->head+1 == q->tail) ||
      (q->head+1 == q->size && q->tail == 0)){
    return FAILURE; // head runs into tail (buffer is full)...
  } else {
    q->buf_x[q->head] = (short)x;
    q->buf_y[q->head] = (short)y;
    q->head++; // step forward head
    if(q->head == q->size) q->head = 0; // circular buffer
  }

  return SUCCESS;
}


/** This function pulls an image coordinate from the FIFO queue
--- q:      queue
--- x:      column
--- y:      row
+++ Return: SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int dequeue(queue_t *q, int *x, int *y){


  if (q->tail != q->head){ //see if any data is available
    *x = (int)q->buf_x[q->tail];
    *y = (int)q->buf_y[q->tail];
    q->tail++;  // step forward  tail
    if (q->tail == q->size) q->tail = 0; // circular buffer
  } else {
    return FAILURE; // nothing in buffer
  }

  return SUCCESS;
}

