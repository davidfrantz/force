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
    
    strncpy(stringlist[i], "Sys_system_name", 15); stringlist[i][15] = '\0'; i++;
    if (strlen(sys.sysname) > NPOW_10-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { 
      strncpy(stringlist[i], sys.sysname, strlen(sys.sysname)); 
      stringlist[i][strlen(sys.sysname)] = '\0'; i++;
    }
    
    strncpy(stringlist[i], "Sys_host_name",   13); stringlist[i][13] = '\0'; i++;
    if (strlen(sys.nodename) > NPOW_10-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { 
      strncpy(stringlist[i], sys.nodename, strlen(sys.nodename)); 
      stringlist[i][strlen(sys.nodename)] = '\0'; i++;
    }
    
    strncpy(stringlist[i], "Sys_OS_release",  14); stringlist[i][14] = '\0'; i++;
    if (strlen(sys.release) > NPOW_10-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { 
      strncpy(stringlist[i], sys.release, strlen(sys.release)); 
      stringlist[i][strlen(sys.release)] = '\0'; i++;
    }
    
    strncpy(stringlist[i], "Sys_OS_version",  14); stringlist[i][14] = '\0'; i++;
    if (strlen(sys.version) > NPOW_10-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { 
      strncpy(stringlist[i], sys.version, strlen(sys.version)); 
      stringlist[i][strlen(sys.version)] = '\0'; i++;
    }
    
    strncpy(stringlist[i], "Sys_machine",     11); stringlist[i][11] = '\0'; i++;
    if (strlen(sys.machine) > NPOW_10-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { 
      strncpy(stringlist[i], sys.machine, strlen(sys.machine)); 
      stringlist[i][strlen(sys.machine)] = '\0'; i++;
    }
    
  } else {
    
    strncpy(stringlist[i], "Sys_system_name", 15); stringlist[i][15] = '\0'; i++;
    strncpy(stringlist[i], "unknown",          7); stringlist[i][7]  = '\0'; i++;
    
    strncpy(stringlist[i], "Sys_host_name",   13); stringlist[i][13] = '\0'; i++;
    strncpy(stringlist[i], "unknown",          7); stringlist[i][7]  = '\0'; i++;
    
    strncpy(stringlist[i], "Sys_OS_release",  14); stringlist[i][14] = '\0'; i++;
    strncpy(stringlist[i], "unknown",          7); stringlist[i][7]  = '\0'; i++;
    
    strncpy(stringlist[i], "Sys_OS_version",  14); stringlist[i][14] = '\0'; i++;
    strncpy(stringlist[i], "unknown",          7); stringlist[i][7]  = '\0'; i++;
    
    strncpy(stringlist[i], "Sys_machine",     11); stringlist[i][11] = '\0'; i++;
    strncpy(stringlist[i], "unknown",          7); stringlist[i][7]  = '\0'; i++;
    
  }

  
  if (getlogin_r(user, NPOW_10) == 0){
    
    strncpy(stringlist[i], "Sys_operator", 12);    stringlist[i][12] = '\0'; i++;
    if (strlen(user) > NPOW_10-1){
      printf("cannot copy, string too long.\n"); exit(1);
    } else { 
      strncpy(stringlist[i], user, strlen(user)); 
      stringlist[i][strlen(user)] = '\0'; i++;
    }
    
  } else {
    
    strncpy(stringlist[i], "Sys_operator", 12);    stringlist[i][12] = '\0'; i++;
    strncpy(stringlist[i], "unknown", 7);          stringlist[i][7]  = '\0'; i++;
    
  }

  *n = num;
  return stringlist;
}

