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
This file contains a public interface for getting system info
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "sys-cl.h"

/** C POSIX library **/
#include <sys/utsname.h> // system name structure
#include <unistd.h> // standard symbolic constants and types 


/** This function gathers system info like information on the platform,
+++ OS, and user
--- n:      how many elements were retrieved (returned)
+++ Return: list of all information
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
char **system_info(int *n){
struct utsname sys;
char **stringlist = NULL;
char user[NPOW_10];
int i = 0, num = 6*2;

  alloc_2DC((void***)&stringlist, num, NPOW_10, sizeof(char));

  // system info
  if (uname(&sys) != -1){

    copy_string(stringlist[i++], NPOW_10, "Sys_system_name");
    copy_string(stringlist[i++], NPOW_10, sys.sysname);

    copy_string(stringlist[i++], NPOW_10, "Sys_host_name");
    copy_string(stringlist[i++], NPOW_10, sys.nodename);

    copy_string(stringlist[i++], NPOW_10, "Sys_OS_release");
    copy_string(stringlist[i++], NPOW_10, sys.release);

    copy_string(stringlist[i++], NPOW_10, "Sys_OS_version");
    copy_string(stringlist[i++], NPOW_10, sys.version);

    copy_string(stringlist[i++], NPOW_10, "Sys_machine");
    copy_string(stringlist[i++], NPOW_10, sys.machine);

  } else {

    copy_string(stringlist[i++], NPOW_10, "Sys_system_name");
    copy_string(stringlist[i++], NPOW_10, "unknown");

    copy_string(stringlist[i++], NPOW_10, "Sys_host_name");
    copy_string(stringlist[i++], NPOW_10, "unknown");

    copy_string(stringlist[i++], NPOW_10, "Sys_OS_release");
    copy_string(stringlist[i++], NPOW_10, "unknown");

    copy_string(stringlist[i++], NPOW_10, "Sys_OS_version");
    copy_string(stringlist[i++], NPOW_10, "unknown");

    copy_string(stringlist[i++], NPOW_10, "Sys_machine");
    copy_string(stringlist[i++], NPOW_10, "unknown");

  }


  if (getlogin_r(user, NPOW_10) == 0){

    copy_string(stringlist[i++], NPOW_10, "Sys_operator");
    copy_string(stringlist[i++], NPOW_10, user);

  } else {

    copy_string(stringlist[i++], NPOW_10, "Sys_operator");
    copy_string(stringlist[i++], NPOW_10, "unknown");

  }

  *n = num;
  return stringlist;
}


/** Get installation path of the executable
--- path:   destination buffer holding path
--- size:   length of buffer
+++ Return: void
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
void get_install_path(char *path, size_t size){
ssize_t len;


  len = readlink("/proc/self/exe", path, size-1);
  if (len == -1){
    printf("Could not get installation path.\n");
    exit(FAILURE);
  }

  if (len == size-1){
    printf("Installation path is too long for buffer.\n");
    exit(FAILURE);
  }
  
  path[len] = '\0';

  #ifdef FORCE_DEBUG
  printf("Installation path: %s\n", d_exe);
  #endif

  return;
}

