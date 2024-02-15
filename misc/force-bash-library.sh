#!/bin/bash

##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2022 David Frantz
# 
# FORCE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# FORCE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with FORCE.  If not, see <http://www.gnu.org/licenses/>.
# 
##########################################################################

# bash function library

# print version
force_version() {
  cat "$BIN/force-misc/force-version.txt"; 
}
export -f force_version

echoerr(){ 
  echo "$PROG: $@" 1>&2; 
}    # warnings and/or errormessages go to STDERR
export -f echoerr

export DEBUG=false # display debug messages?
debug(){ 
  if [ "$DEBUG" == "true" ]; then 
    echo "DEBUG: $@"; 
  fi 
} # debug message
export -f DEBUG debug

cmd_not_found(){      # check required external commands
  for cmd in "$@"; do
    stat=`which $cmd`
    if [ $? != 0 ] ; then 
      echoerr "\"$cmd\": external command not found, terminating..."; 
      exit 1; 
    fi
  done
}
export -f cmd_not_found

file_not_found() {      # check required files
  for file in "$@"; do
    stat=`which $file`
    if [ ! -r $file ] ; then 
      echoerr "\"$file\": file not found, terminating..."; 
      exit 1; 
    fi
  done
}
export -f file_not_found

issmaller(){
  awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1<n2) ? "true" : "false"}'
}
export -f issmaller

isgreater(){
  awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1>n2) ? "true" : "false"}'
}
export -f isgreater
