#!/bin/bash

##########################################################################
# 
# This file is part of FORCE - Framework for Operational Radiometric 
# Correction for Environmental monitoring.
# 
# Copyright (C) 2013-2025 David Frantz
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

# this script builds a virtual datacube from a physical datacube

# functions/definitions ------------------------------------------------------------------
PROG=$(basename "$0")
BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MISC="$BIN/force-misc"
export PROG BIN MISC

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
eval ". ${LIB}" >/dev/null 2>&1 ;[[ "$?" -ne "0" ]] && echo "loading bash library failed" && exit 1;
export LIB


MANDATORY_ARGS=2

export TRANSLATE_EXE="gdal_translate"
export PARALLEL_EXE="parallel"

help(){
cat <<HELP

Usage: $PROG [-hvi] [-p pattern] [-w] [-j nJobs] physical-datacube-dir virtual-datacube-dir

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose
  -p = search pattern (default: '*.tif')
  -w = overwrite output file (default: skips existing files)
  -j = number of jobs, defaults to 100%

  Positional arguments:
  - 'physical-datacube-dir': directory of existing, physical datacube (input)
  - 'virtual-datacube-dir': directory of virtual datacube (output)

  -----
    see https://force-eo.readthedocs.io/en/latest/components/auxilliary/virtual-datacube.html

HELP
exit 1
}
export -f help


# important, check required commands !!! dies on missing
cmd_not_found "$TRANSLATE_EXE"
cmd_not_found "$PARALLEL_EXE"


function translate_to_virtual(){

  inputfile="$1"
  outputfile="$2"
  overwrite="$3"

  if [[ "$overwrite" == "TRUE" || ! -f "$outputfile" ]]; then 
    $TRANSLATE_EXE -f VRT "$inputfile" "$outputfile" >/dev/null
  fi

}
export -f translate_to_virtual


# now get the options --------------------------------------------------------------------
ARGS=$(getopt -o hvip:wj: --long help,version,info,pattern:,overwrite,jobs: -n "$0" -- "$@")
if [ "$?" != 0 ] ; then help; fi
eval set -- "$ARGS"

PATTERN='*.tif'
OVERWRITE=FALSE
NJOB="100%"

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Display the size of your datacube"; exit 0;;
    -p|--pattern) PATTERN="$2"; shift ;;
    -w|-overwrite) OVERWRITE=TRUE; shift ;;
    -j|--jobs) NJOB="$2"; shift ;;
    
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done
debug "PATTERN: $PATTERN"
debug "OVERWRITE: $OVERWRITE"

if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument(s) are missing."; help
else
  DIR_INPUT=$(readlink -f "$1") # absolute file path
  DIR_OUTPUT=$(readlink -f "$2") # absolute file path
fi
debug "DIR_INPUT: $DIR_INPUT"
debug "DIR_OUTPUT: $DIR_OUTPUT"


# options received, check now ------------------------------------------------------------
if ! [[ -d "$DIR_OUTPUT" && -w "$DIR_OUTPUT" ]]; then
  echoerr "$DIR_OUTPUT is not a writeable directory, exiting."; exit 1;
fi


# further checks and preparations --------------------------------------------------------
DC_DEFINITION_INPUT="$DIR_INPUT/datacube-definition.prj"
if ! [[ -f "$DC_DEFINITION_INPUT" && -r "$DC_DEFINITION_INPUT" ]]; then
  echo "$DC_DEFINITION_INPUT is missing, not a datacube, exiting."; exit 1;
fi

DC_DEFINITION_OUTPUT="$DIR_OUTPUT/datacube-definition.prj"
if [[ -d "$DIR_OUTPUT" && -f "$DC_DEFINITION_OUTPUT" ]]; then

  echo "$DC_DEFINITION_OUTPUT already exists."

  if diff "$DC_DEFINITION_INPUT" "$DC_DEFINITION_OUTPUT" >/dev/null; then
    echo "It is identical with $DC_DEFINITION_INPUT. All OK. Proceed."
  else
    echoerr "It is not identical with $DC_DEFINITION_INPUT. Abort."
    exit 1
  fi

else
  cp "$DC_DEFINITION_INPUT" "$DC_DEFINITION_OUTPUT" 
fi
debug "DC_DEFINITION_INPUT: $DC_DEFINITION_INPUT"
debug "DC_DEFINITION_OUTPUT: $DC_DEFINITION_OUTPUT"


# main thing -----------------------------------------------------------------------------

# walk tiles
for d in "$DIR_INPUT"/X*; do 

  echo "walking tile $d" 

  mkdir -p "$DIR_OUTPUT"/"${d#"$DIR_INPUT"}"

  # check free RAM
  MEMORY=$(LANG="C"; free --mega | awk '/^Mem/ { printf("%.0fM\n", $2 * 0.05) }')

  # do the thing
  find "$d" -maxdepth 1 -type f -name "$PATTERN" | \
    $PARALLEL_EXE -j "$NJOB" --memsuspend "$MEMORY" \
    translate_to_virtual {} "$DIR_OUTPUT"/"${d#"$DIR_INPUT"}"/{/.}.vrt "$OVERWRITE"

done

exit 0
