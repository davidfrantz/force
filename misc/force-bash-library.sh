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

# warnings and/or error messages go to STDERR
echoerr(){ 
  echo "$PROG: $*" 1>&2
}
export -f echoerr

# print debug messages
export DEBUG=false
debug(){ 
  if is_true "$DEBUG"; then 
    echo "DEBUG: $*"
  fi 
}
export -f debug

# check required external commands
cmd_not_found(){
  for cmd in "$@"; do
    if ! eval which "$cmd" >/dev/null 2>&1; then
      echoerr "external command ($cmd) not found, terminating..."; 
      exit 1; 
    fi
  done
}
export -f cmd_not_found

# check files (exist+read)
file_not_found(){
  for file in "$@"; do
    if [ ! -f "$file" ]; then
      echoerr "file not found ($file), terminating..."; 
      exit 1; 
    fi
    if [ ! -r "$file" ]; then 
      echoerr "file not readable ($file), terminating..."; 
      exit 1; 
    fi
  done
}
export -f file_not_found

# check files (exist+read+write)
file_not_writeable(){
  for file in "$@"; do
    file_not_found "$file"
    if [ ! -w "$file" ]; then 
      echoerr "file not writeable ($file), terminating..."; 
      exit 1; 
    fi
  done
}
export -f file_not_writeable

# check directories (exist+read)
dir_not_found(){
  for dir in "$@"; do
    if [ ! -d "$dir" ]; then
      echoerr "directory not found ($dir), terminating..."; 
      exit 1; 
    fi
    if [ ! -x "$dir" ]; then 
      echoerr "directory not executable ($dir), terminating..."; 
      exit 1; 
    fi
  done
}
export -f dir_not_found

# check directories (exist+read+write)
dir_not_writeable(){
  for dir in "$@"; do
    dir_not_found "$dir"
    if [ ! -w "$dir" ]; then 
      echoerr "directory not writeable ($dir), terminating..."; 
      exit 1; 
    fi
  done
}
export -f dir_not_writeable

# value is true?
is_true(){
  if [ "$1" == true ]; then return 0; else return 1; fi
}
export -f is_true

# number is smaller than other number?
is_lt(){
  res=$(awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1<n2) ? "true" : "false"}')
  if [ "$res" == "true" ]; then return 0; else return 1; fi
}
export -f is_lt

# number is greater than other number?
is_gt(){
  res=$(awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1>n2) ? "true" : "false"}')
  if [ "$res" == "true" ]; then return 0; else return 1; fi
}
export -f is_gt

# number is smaller or equal than other number?
is_le(){
  res=$(awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1<=n2) ? "true" : "false"}')
  if [ "$res" == "true" ]; then return 0; else return 1; fi
}
export -f is_le

# number is greater or equal than other number?
is_ge(){
  res=$(awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1>=n2) ? "true" : "false"}')
  if [ "$res" == "true" ]; then return 0; else return 1; fi
}
export -f is_ge

# number is equal to other number?
is_eq(){
  res=$(awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1==n2) ? "true" : "false"}')
  if [ "$res" == "true" ]; then return 0; else return 1; fi
}
export -f is_eq

# number is an integer?
is_integer(){
  if [[ "$1" =~ ^-?[0-9]+$ ]]; then return 0; else return 1; fi
}
export -f is_integer
