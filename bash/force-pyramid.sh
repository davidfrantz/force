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
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MANDATORY_ARGS=1

export PARALLEL_EXE="parallel"
export PYRAMID_EXE="gdaladdo"

echoerr() { echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR
export -f echoerr

export DEBUG=false # display debug messages?
debug(){ if [ "$DEBUG" == "true" ]; then echo "DEBUG: $@"; fi } # debug message
export -f debug

cmd_not_found() {      # check required external commands
  for cmd in "$@"; do
    stat=`which $cmd`
    if [ $? != 0 ] ; then echoerr "\"$cmd\": external command not found, terminating..."; exit 1; fi
  done
}
export -f cmd_not_found

help () {
cat <<HELP

Usage: $PROG [-hjrl] image [image]*

  -h  = show his help
  -j  = number of jobs
        defaults to 'as many as possible'
  -r  = resampling option
        default: nearest
  -l  = levels, comma-separated
        default: 2 4 8 16

$PROG:  compute image pyramids
        see https://force-eo.readthedocs.io/en/latest/components/auxilliary/pyramid.html

HELP
exit 1
}
export -f help

cmd_not_found "$PARALLEL_EXE"; # important, check required commands !!! dies on missing
cmd_not_found "$PYRAMID_EXE";  # important, check required commands !!! dies on missing

# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hj:r:l: --long help,jobs:,resample:,levels: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

# default options
NJOB=0
LEVELS="2,4,8,16"
RESAMPLE="nearest"

while :; do
  case "$1" in
    -h|--help) help ;;
    -j|--jobs) NJOB="$2"; shift ;;
    -r|--resample) RESAMPLE="$2"; shift ;;
    -l|--levels) LEVELS="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

export LEVELS=$(echo $LEVELS | sed 's/,/ /g')
export RESAMPLE
debug "jobs: $NJOB"
debug "levels: $LEVELS"
debug "resample: $RESAMPLE"

if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
fi

pyramid(){

  FINP=$(readlink -f $1) # absolute file path
  BINP=$(basename $FINP) # basename
  CINP=${BINP%%.*}       # corename (without extension)
  DINP=$(dirname  $FINP) # directory name

  debug "$FINP"
  debug "$BINP"
  debug "$CINP"
  debug "$DINP"

  # input file exists?
  if [ ! -r $FINP ]; then
    echoerr "$FINP ist not readable/existing"
    exit 1
  fi

  # output dir writeable?
  if [ ! -w $DINP ]; then
    echoerr "$DINP ist not writeable/existing"
    exit 1
  fi

  echo "computing pyramids for $BINP"
  $PYRAMID_EXE -ro --config COMPRESS_OVERVIEW DEFLATE --config BIGTIFF_OVERVIEW YES -r $RESAMPLE $FINP $LEVELS

}
export -f pyramid


# main thing -----------------------------------------------------------------------------

echo "computing pyramids for $# images:"

printf '%s\n' "$@" | $PARALLEL_EXE -j $NJOB pyramid {}

