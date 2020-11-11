#!/bin/bash

##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2020 David Frantz
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

# functions/definitions ------------------------------------------------------------------
PROG=`basename $0`;
BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

PARALLEL_EXE="parallel"
PYRAMID_EXE="gdaladdo"

MANDATORY_ARGS=1

echoerr() { echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR

cmd_not_found() {      # check required external commands
  for cmd in "$@"; do
    stat=`which $cmd`
    if [ $? != 0 ] ; then echoerr "\"$cmd\": external command not found, terminating..."; exit 1; fi
  done
}

help () {
cat <<HELP

Usage: $PROG [-h] file [file]*

  -h  = show his help

$PROG:  compute image pyramids
        see https://force-eo.readthedocs.io/en/latest/components/auxilliary/pyramid.html

HELP
exit 1
}

cmd_not_found "$PARALLEL_EXE";    # important, check required commands !!! dies on missing
cmd_not_found "$PYRAMID_EXE";    # important, check required commands !!! dies on missing

# now get the options --------------------------------------------------------------------
ARGS=`getopt -o h: --long help: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

while :; do
  case "$1" in
    -h|--help) help ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
fi

pyramid(){

  FINP=$(readlink -f $1) # absolute file path
  BINP=$(basename $FINP) # basename
  CINP=${BINP%%.*}       # corename (without extension)
  DINP=$(dirname  $FINP) # directory name

  INP=$(readlink -f $i)
  #echo $INP
  # input file exists?
  if [ ! -r $INP ]; then
    echo $INP "ist not readable/existing"
    exit
  fi

  BASE=$(basename $INP)
  DIR=$(dirname $INP)


  # output dir writeable?
  if [ ! -w $DIR ]; then
    echo $DIR "ist not writeable/existing"
    exit
  fi

  echo "computing pyramids for $BASE"
  $PYRAMID_EXE -ro --config COMPRESS_OVERVIEW DEFLATE --config BIGTIFF_OVERVIEW YES -r nearest $INP 2 4 8 16

}

export -f pyramid()

for i in "$@"; do

done
