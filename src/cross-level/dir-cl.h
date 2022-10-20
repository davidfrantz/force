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
Directory/file support header
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#ifndef DIR_CL_H
#define DIR_CL_H

#include <stdio.h>    // core input and output functions
#include <stdlib.h>   // standard general utilities library
#include <string.h>   // string handling functions
#include <stdbool.h>  // boolean data type

#include <dirent.h>   // allows the opening and listing of directories
#include <sys/stat.h> // file information
#include <unistd.h>   // essential POSIX functions and constants
#include <errno.h>    // error numbers

#include "../cross-level/const-cl.h"
#include "../cross-level/string-cl.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  char name[NPOW_10];   // directory name
  struct dirent **LIST; // files in directory
  int N;                // number of files in directory
  char **list;          // filtered files
  int n;                // number of filtered files
} dir_t;

bool fileexist(char *fname);
int findfile(char *dir_path, char *pattern, char *filter, char fname[], int size);
int countfile(char *dir_path, char *pattern);
int createdir(char *dir_path);
void extension(char* path, char extension[], int size);
void extension2(char* path, char extension[], int size);
void basename_without_ext(char* path, char basename[], int size);
void basename_with_ext(char* path, char basename[], int size);
void directoryname(char* path, char dirname[], int size);

#ifdef __cplusplus
}
#endif

#endif

