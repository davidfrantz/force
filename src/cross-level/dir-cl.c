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
This file contains functions for reading directories and handling names
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "dir-cl.h"


/** Test for existing file/directory
--- fname:  filename
+++ Return: true/false
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
bool fileexist(char *fname){

  if (access(fname, F_OK) == 0) return true; else return false;
}


/** This function searches the given directory for a file, which contains
+++ the given pattern. The first match will be returned. NULL will be re-
+++ turned if no file was found.
--- dir_path: directory
--- pattern:  pattern
--- filter:   an additional filter, e.g. for file extensions, 
              NULL to disable filtering
--- fname:    buffer that will hold the filename
--- size:     length of the buffer
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int findfile(char *dir_path, char *pattern, char *filter, char fname[], int size){
DIR *dp;
int nchar;
struct dirent *ep;


  dp = opendir(dir_path);

  if (dp != NULL){

    while ((ep = readdir(dp))){

      if (ep->d_name && strstr(ep->d_name, pattern)){

        if (filter != NULL && !strstr(ep->d_name, filter)) continue;

        if (strlen(dir_path)+strlen(ep->d_name)+1 > size){
          printf("array is too short for holding filename.\n");
          fname[0] = '\0';
          (void) closedir(dp);
          return FAILURE;
        } else {

          nchar = snprintf(fname, size, "%s/%s", dir_path, ep->d_name);
          if (nchar < 0 || nchar >= size){ 
            printf("Buffer Overflow in assembling filename\n"); return FAILURE;}
            
          (void) closedir(dp);
          return SUCCESS;
        }

      }

    }

    (void) closedir(dp);

  } else perror("Couldn't open the directory");

  fname[0] = '\0';

  return FAILURE;
}


/** This function searches the given directory for files, which contains
+++ the given pattern. The number of files will be returned.
--- dir_path: directory
--- pattern:  pattern
+++ Return:   number of files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int countfile(char *dir_path, char *pattern){
DIR *dp;
struct dirent *ep;
dp = opendir(dir_path);
int k = 0;

  if (dp != NULL){
    while ((ep = readdir(dp))){
      if (ep->d_name && strstr(ep->d_name, pattern)) k++;
    }
    (void) closedir(dp);
  } else perror("Couldn't open the directory");

  return k;
}


/** This function creates a directory. The function will return SUCCESS if
+++ the directory was successfully created or if it was already existent.
--- dir_path: directory
+++ Return:   SUCCESS/FAILURE
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int createdir(char *dir_path){
int e, errno;
struct stat sb;

  if ((e = stat(dir_path, &sb)) != 0){
    if (errno == ENOENT){
      if ((e = mkdir(dir_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)) != 0){
        printf("mkdir failed\n"); exit(1);}
    }
  }

  return SUCCESS;
}


/** This function extracts the extension from a file (after 1st! dot in 
+++ basename)
--- path:      file path
--- extension: buffer that will hold the extension
--- size:      length of the buffer
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void extension(char* path, char extension[], int size){
char basename[NPOW_10];
char *dot;

  basename_with_ext(path, basename, NPOW_10);

  // Locate the first dot and copy from there
  dot = strchr(basename, '.');
  if (dot != NULL){
    copy_string(extension, size, dot);
  } else {
    extension[0] = '\0';
  }

  return;
}


/** This function extracts the extension from a file (after last! dot in 
+++ basename)
--- path:      file path
--- extension: buffer that will hold the extension
--- size:      length of the buffer
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void extension2(char* path, char extension[], int size){
char basename[NPOW_10];
char *dot;

  basename_with_ext(path, basename, NPOW_10);

  // Locate the first dot and copy from there
  dot = strrchr(basename, '.');
  if (dot != NULL){
    copy_string(extension, size, dot);
  } else {
    extension[0] = '\0';
  }

  return;
}


/** This function extracts the basename from a file and removes the file
+++ extension.
--- path:     file path
--- basename: buffer that will hold the basename
--- size:     length of the buffer
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void basename_without_ext(char* path, char basename[], int size){
char *dot;

  basename_with_ext(path, basename, size);

  // Locate the first dot
  dot = strchr(basename, '.');
  if (dot != NULL) *dot = '\0';

  return;
}


/** Basename of file
+++ This function extracts the basename from a file.
--- path:     file path
--- basename: buffer that will hold the basename
--- size:     length of the buffer
+++ Return:    void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void basename_with_ext(char* path, char basename[], int size){
int plen;
char *start;

  if (path == NULL){
    basename[0] = '\0';
    return;
  }

  // length of path
  plen = strlen(path);

  // Strip trailing slashes if any
  while (path[plen-1] == '/'){
    path[plen-1] = '\0';
    plen--;
  }

  // Locate the last slash, start string from there
  start = strrchr(path, '/');
  if (start == NULL){
    start = path;
  } else start++;


  // copy string from starting point, add terminating 0
  copy_string(basename, size, start);

  return;
}


/** This function extracts the directory name of a file.
--- path:    file path
--- dirname: buffer that will hold the dirname
--- size:    length of the buffer
+++ Return:  void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void directoryname(char* path, char dirname[], int size){
char *slash;

  if (path == NULL){
    dirname[0] = '\0';
    return;
  }


  // copy path to dir
  copy_string(dirname, size, path);


  // Locate the last slash and set terminating 0
  slash = strrchr(dirname, '/');
  if (slash != NULL){
    *slash = '\0';
  } else {
    if (getcwd(dirname, size) == NULL){
     printf("No directoryname detected and getting current directory failed.\n"); 
     exit(1);
   }
  }

  return;
}

