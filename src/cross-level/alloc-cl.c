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
This file contains general functions for memory allocation / deallocation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "alloc-cl.h"


/** Allocate array
+++ This function allocates a block of memory, and initializes it with 0.
--- ptr:    Pointer to the memory block
--- n:      Number of elements to allocate
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc(void **ptr, size_t n, size_t size){
void *arr = NULL;

  arr = (void*) calloc(n, size);
  if (arr == NULL){ printf("unable to allocate memory!\n"); exit(1);}

  *ptr = arr;
  return;
}


/** Allocate 2D-array
+++ This function allocates blocks of memory, and initializes them with 0.
--- ptr:    Pointer to the memory block
--- n1:     Number of elements to allocate (1st dimension)
--- n2:     Number of elements to allocate (2nd dimension)
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_2D(void ***ptr, size_t n1, size_t n2, size_t size){
void **arr = NULL;
int i;

  alloc((void**)&arr, n1, sizeof(void*));
  for (i=0; i<n1; i++) alloc((void**)&arr[i], n2, size);

  *ptr = arr;
  return;
}


/** Allocate contiguous 2D-array
+++ This function allocates a block of memory, and initializes it with 0.
+++ Unlike alloc_2D, one single block of memory is allocated and pointers
+++ are set for the 2nd dimension.
--- ptr:    Pointer to the memory block
--- n1:     Number of elements to allocate (1st dimension)
--- n2:     Number of elements to allocate (2nd dimension)
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_2DC(void ***ptr, size_t n1, size_t n2, size_t size){
void  *arr  = NULL;
void **arr_ = NULL;
int i;

  alloc((void**)&arr, n1*n2, size);
  alloc((void**)&arr_, n1, sizeof(void*));
  for (i=0; i<n1; i++) arr_[i] = (char*)arr + (size_t)i*n2*size;

  *ptr = arr_;
  return;
}


/** Allocate 3D-array
+++ This function allocates blocks of memory, and initializes them with 0.
--- ptr:    Pointer to the memory block
--- n1:     Number of elements to allocate (1st dimension)
--- n2:     Number of elements to allocate (2nd dimension)
--- n3:     Number of elements to allocate (3nd dimension)
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void alloc_3D(void ****ptr, size_t n1, size_t n2, size_t n3, size_t size){
void ***arr = NULL;
int i, j;

  alloc((void**)&arr, n1, sizeof(void**));
  for (i=0; i<n1; i++){
    alloc((void**)&arr[i], n2, sizeof(void*));
    for (j=0; j<n2; j++) alloc((void**)&arr[i][j], n3, size);
  }

  *ptr = arr;
  return;
}


/** Re-Allocate array
+++ This function re-allocates a block of memory. If the block is larger
+++ than before, the new part is initialized with 0.
--- ptr:    Pointer to the memory block
--- n_now:  Number of elements of current block
--- n:      Number of elements that block should have
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void re_alloc(void **ptr, size_t n_now, size_t n, size_t size){
void *arr = NULL;

  if (n_now == n) return;

  arr = (void*) realloc(*ptr, n*size);
  if (arr == NULL){ printf("unable to reallocate memory!\n"); exit(1);}
  
  if (n > n_now) memset((char*)arr + n_now*size, 0, (n-n_now)*size);
  
  #ifdef FORCE_DEBUG
  printf("reallocated from %lu to %lu elements of size %lu\n", n_now, n, size);
  #endif

  *ptr = arr;
  return;
}


/** Re-Allocate 2D-array
+++ This function re-allocates blocks of memory. If the block is larger
+++ than before, the new part is initialized with 0.
--- ptr:    Pointer to the memory block
--- n1_now:  Number of elements of current block (1st dimension)
--- n2_now:  Number of elements of current block (2nd dimension)
--- n1:      Number of elements that block should have (1st dimension)
--- n2:      Number of elements that block should have (2nd dimension)
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void re_alloc_2D(void ***ptr, size_t n1_now, size_t n2_now, size_t n1, size_t n2, size_t size){
void **arr = *ptr;
int i;

  if (n1_now == n1 && n2_now == n2) return;

  if (n1_now > n1){
    for (i=n1; i<n1_now; i++) free((void*)arr[i]);
    #ifdef FORCE_DEBUG
    for (i=n1; i<n1_now; i++) printf("freed element %d\n", i);
    #endif
  }

  re_alloc((void**)&arr, n1_now, n1, sizeof(void*));

  if (n1_now < n1){
    for (i=n1_now; i<n1; i++) alloc((void**)&arr[i], n2, size);
    #ifdef FORCE_DEBUG
    for (i=n1_now; i<n1; i++) printf("allocated element %d of length %lu and size %lu\n", i, n2, size);
    #endif
  }

  for (i=0; i<n1_now; i++) re_alloc((void**)&arr[i], n2_now, n2, size);

  *ptr = arr;
  return;
}


/** Re-Allocate contiguous 2D-array
+++ This function re-allocates blocks of memory. If the block is larger
+++ than before, the new part is initialized with 0.
+++ Unlike alloc_2D, one single block of memory is allocated and pointers
+++ are set for the 2nd dimension.
--- ptr:    Pointer to the memory block
--- n1_now:  Number of elements of current block (1st dimension)
--- n2_now:  Number of elements of current block (2nd dimension)
--- n1:      Number of elements that block should have (1st dimension)
--- n2:      Number of elements that block should have (2nd dimension)
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void re_alloc_2DC(void ***ptr, size_t n1_now, size_t n2_now, size_t n1, size_t n2, size_t size){
void  *arr  = *ptr[0];
void **arr_ = *ptr;
int i;

  if (n1_now == n1 && n2_now == n2) return;

  re_alloc((void**)&arr, n1_now*n2_now, n1*n2, size);  
  re_alloc((void**)&arr_, n1_now, n1, sizeof(void*));
  for (i=0; i<n1; i++) arr_[i] = (char*)arr + (size_t)i*n2*size;

  *ptr = arr_;
  return;
}


/** Re-Allocate 3D-array
+++ This function re-allocates blocks of memory. If the block is larger
+++ than before, the new part is initialized with 0.
--- ptr:    Pointer to the memory block
--- n1_now:  Number of elements of current block (1st dimension)
--- n2_now:  Number of elements of current block (2nd dimension)
--- n3_now:  Number of elements of current block (3rd dimension)
--- n1:      Number of elements that block should have (1st dimension)
--- n2:      Number of elements that block should have (2nd dimension)
--- n3:      Number of elements that block should have (3rd dimension)
--- size:   Size of each element
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void re_alloc_3D(void ****ptr, size_t n1_now, size_t n2_now, size_t n3_now, size_t n1, size_t n2, size_t n3, size_t size){
void ***arr = *ptr;
int i;

  if (n1_now == n1 && n2_now == n2 && n3_now == n3) return;

  if (n2_now != n2 || n3_now != n3){
    for (i=0; i<n1_now; i++) re_alloc_2D((void***)&arr[i], n2_now, n3_now, n2, n3, size);
  }


  if (n1_now > n1){
    for (i=n1; i<n1_now; i++) free_2D((void**)arr[i], n2);
    #ifdef FORCE_DEBUG
    for (i=n1; i<n1_now; i++) printf("freed element %d\n", i);
    #endif
  }

  re_alloc((void**)&arr, n1_now, n1, sizeof(void*));

  if (n1_now < n1){
    for (i=n1_now; i<n1; i++) alloc_2D((void***)&arr[i], n2, n3, size);
    #ifdef FORCE_DEBUG
    for (i=n1_now; i<n1; i++) printf("allocated element %d of length %lu and size %lu\n", i, n2, size);
    #endif
  }

  *ptr = arr;
  return;
}


/** Free 2D-array
+++ This function deallocates an allocated 2D array.
--- ptr:    Pointer to the memory block
--- n1:     Number of elements to deallocate (1st dimension)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_2D(void **ptr, size_t n){
int i;

  for (i=0; i<n; i++) free((void*)ptr[i]);
  free((void*)ptr);

  return;
} 



/** Free contiguos 2D-array
+++ This function deallocates an allocated, contiguous 2D array.
--- ptr:    Pointer to the memory block
--- n1:     Number of elements to deallocate (1st dimension)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_2DC(void **ptr){
  
  free(ptr[0]);
  free(ptr);

  return;
} 


/** Free 3D-array
+++ This function deallocates an allocated 2D array.
--- ptr:    Pointer to the memory block
--- n1:     Number of elements to deallocate (1st dimension)
--- n2:     Number of elements to deallocate (2nd dimension)
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void free_3D(void ***ptr, size_t n1, size_t n2){
int i, j;

  for (i=0; i<n1; i++){
    for (j=0; j<n2; j++) free((void*)ptr[i][j]);
    free((void*)ptr[i]);
  }
  free((void*)ptr);

  return;
} 

