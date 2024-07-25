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
This file contains a public interface for locking files
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "lock-cl.h"

/** Geospatial Data Abstraction Library (GDAL) **/
#include "cpl_multiproc.h" // CPL Multi-Threading


/** Lock a file
--- fname:   filename
--- timeout: try to lock the file for a maximum time of x seconds
+++ Return:  filename of lockfile
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
char *lock_file(char *fname, int timeout){
char *lock = NULL;

  if ((lock = (char*)CPLLockFile(fname, timeout)) == NULL){
    printf("Unable to lock file (timeout: %ds): %s\n", timeout, fname); 
  }

  return lock;
}


/** Unlock a file
--- lock:   filename of lockfile
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void unlock_file(char *lock){

  CPLUnlockFile(lock);
  return;
}


/** Compute a timeout for attempting to create the lockfile in dependency
+++ of the number of bytes that are to be written. The timeout has a mini-
+++ mum of 60secs and a maximum of 600secs, which corresponds to a full S2
+++ image (10*10980*10980*2 bytes)
--- bytes:  number of bytes
+++ Return: suggested timeout
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
  
double lock_timeout(size_t bytes){

return 540.0 / 
  (1.0+exp(-1.0*((float)bytes/241120800.0-5.0))) + 60;
}

