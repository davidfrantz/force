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

# this script displays the size of your datacube

# functions/definitions ------------------------------------------------------------------
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MANDATORY_ARGS=1

echoerr(){ echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR
export -f echoerr

export DEBUG=false # display debug messages?
debug(){ if [ "$DEBUG" == "true" ]; then echo "DEBUG: $@"; fi } # debug message
export -f debug

cmd_not_found(){      # check required external commands
  for cmd in "$@"; do
    stat=`which $cmd`
    if [ $? != 0 ] ; then echoerr "\"$cmd\": external command not found, terminating..."; exit 1; fi
  done
}
export -f cmd_not_found


help(){
cat <<HELP

Usage: $PROG [-hvit] datacube-dir

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose
  -t = temp directory (default: current directory)

  Positional arguments:
  - 'datacube-dir': directory of existing datacube

  -----
    see https://force-eo.readthedocs.io/en/latest/components/auxilliary/dc-size.html

HELP
exit 1
}
export -f help


function pretty_size(){

  sensor=$1
  size=$2
  unit="B"

  unit="B"
  if [ ${size%%.*} -gt 1024 ]; then
    size=$(echo $size | awk '{printf "%f", $1 / 1024.0}')
    unit="KB"
  fi

  if [ ${size%%.*} -gt 1024 ]; then
    size=$(echo $size | awk '{printf "%f", $1 / 1024.0}')
    unit="MB"
  fi

  if [ ${size%%.*} -gt 1024 ]; then
    size=$(echo $size | awk '{printf "%f", $1 / 1024.0}')
    unit="GB"
  fi

  if [ ${size%%.*} -gt 1024 ]; then
    size=$(echo $size | awk '{printf "%f", $1 / 1024.0}')
    unit="TB"
  fi

  if [ ${size%%.*} -gt 1024 ]; then
    size=$(echo $size | awk '{printf "%f", $1 / 1024.0}')
    unit="PB"
  fi


  echo $sensor $size $unit | awk '{printf "%-5s: %6.2f %2s\n", $1, $2, $3}'

}
export -f pretty_size


# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hvit: --long help,version,info,temp: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

DIR_TEMP=$PWD

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) "$BIN"/force -v; exit 0;;
    -i|--info) echo "Display the size of your datacube"; exit 0;;
    -t|--temp) DIR_TEMP="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
else
  DIR_INPUT=$(readlink -f $1) # absolute file path
fi
debug "datacube-dir: $DIR_INPUT"

# options received, check now ------------------------------------------------------------
if ! [[ -d "$DIR_TEMP" && -w "$DIR_TEMP" ]]; then
  echoerr "$DIR_TEMP is not a writeable directory, exiting."; exit 1;
fi


# further checks and preparations --------------------------------------------------------
DC_DEFINITION="$DIR_INPUT/datacube-definition.prj"
if ! [[ -f "$DC_DEFINITION" && -r "$DC_DEFINITION" ]]; then
  echo "$DC_DEFINITION is missing, not a datacube, exiting."; exit 1;
fi


# main thing -----------------------------------------------------------------------------
FILE_TEMP="$DIR_TEMP/force-datacube-size_temp.txt"

if [[ -f "$FILE_TEMP" ]]; then
  echoerr "$FILE_TEMP exists, clean up an re-run, exiting."; exit 1;
fi

# walk tiles
for d in "$DIR_INPUT"/X*; do 
  echo "walking tile $d" 
  ls -l $d >> "$FILE_TEMP"
done

if ! [[ -f "$FILE_TEMP" ]]; then
  echoerr "no $FILE_TEMP generated, check if $DIR_INPUT is ok, exiting."; exit 1;
fi

echo ""
echo "Sizes per sensor:"

SENSORS="LND04 LND05 LND07 LND08 LND09 SEN2A SEN2B SEN2C SEN2D sen2a sen2b sen2c sen2d SEN2L SEN2H R-G-B S1AIA S1AID S1BIA S1BID S1CIA S1CID S1DIA S1DID VVVHP MOD01 MOD02 MODIS"

ALL_SIZE=0

for s in $SENSORS; do

  SIZE=$(grep $s $FILE_TEMP | tr -s ' ' | cut -d ' ' -f 5 | awk '{ sum += $1 } END { print sum }')

  if ! [[ -z $SIZE ]]; then

    ALL_SIZE=$(echo $ALL_SIZE $SIZE | awk '{ print $1 + $2 }')

    pretty_size $s $SIZE

  fi

done

pretty_size "TOTAL" $ALL_SIZE


# remove temp file
if [[ -f "$FILE_TEMP" ]]; then
  rm $FILE_TEMP
fi

exit 0
