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

# functions/definitions ------------------------------------------------------------------
PROG=$(basename "$0")
BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
MISC="$BIN/force-misc"
export PROG BIN MISC

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
if ! eval ". ${LIB}" >/dev/null 2>&1; then 
  echo "loading bash library failed" 1>&2
  exit 1
fi


MANDATORY_ARGS=1

export FORCE_CUBE_EXE="$BIN/force-cube"


help(){
cat <<HELP

Usage: $PROG [-hvi] [-d datacube-dir] [-a allow-list] [-j jobs] input-vector

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose
  
  -d = directory of existing datacube
       defaults to current directory
       'datacube-definition.prj' needs to exist in there
  -a = tile allow-list to restrict the processing extent
       The extent will be computed based on the input AOI
       Defaults to './allow-list.txt'
  -j = number of jobs, defaults to 100%

  Positional arguments:
  - input-vector: a polygon vector file for the area of interest (AOI)

  -----
    see https://force-eo.readthedocs.io/en/latest/components/auxilliary/tile-extent.html

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$FORCE_CUBE_EXE"


# now get the options --------------------------------------------------------------------
ARGS=$(getopt -o hvid:a:j: --long help,version,info,datacube:,allow:,jobs: -n "$0" -- "$@")
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

DIR_CUBE="$PWD"
FILE_ALLOW="$PWD/allow-list.txt"
NJOB="100%"

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Compute processing extent based on AOI"; exit 0;;
    -d|--datacube) DIR_CUBE=$(readlink -f "$2"); shift ;;
    -a|--allow) FILE_ALLOW=$(readlink -f "$2"); shift ;;
    -j|--jobs) NJOB="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -ne $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
else
  FILE_AOI=$(readlink -f "$1")
fi

debug "datacube dir: $DIR_CUBE"
debug "allow-list:   $FILE_ALLOW"
debug "AOI:          $FILE_AOI"
debug "njobs:        $NJOB"

DIR_ALLOW=$(dirname "FILE_ALLOW")
FILE_CUBE="$DIR_CUBE/datacube-definition.prj"

# checks
file_not_found "$FILE_AOI"
dir_not_found "$DIR_CUBE"
file_not_found "$FILE_CUBE"
dir_not_writeable "$DIR_ALLOW"


# make temporary directory
TIME=$(date +"%Y%m%d%H%M%S")
BASE=$(basename "$FILE_AOI")
DIR_TMP="$DIR_ALLOW/${BASE%%.*}_TEMP_$TIME"
debug "DIR_TMP: $DIR_TMP"
mkdir -p "$DIR_TMP"
dir_not_writeable "$DIR_TMP" # safety check
cp "$FILE_CUBE" -t "$DIR_TMP"

# generate masks
$FORCE_CUBE_EXE -o "$DIR_TMP" -b force-extent "$FILE_AOI" &> /dev/null

# go to temporary directory
DIR_NOW="$PWD"
cd "$DIR_TMP" || {
  echoerr "could not change to $DIR_TMP"
  exit 1
}

if [ $? -ne 0 ]; then
  echoerr "force-cube returned an error."
  echoerr "run '$BINDIR"/"force-cube -j 0 -s 10 -o $TMP -b force-extent $INP' to see where the error occurred."
  exit 1
fi

# find all the generated masks
FILES_MASK=$(ls -d X*/*.tif)
TILES_MASK=($(dirname $FILES_MASK))
NTILES=${#TILES_MASK[*]}


# compute range of tiles
XMIN=9999
XMAX=-999
YMIN=9999
YMAX=-999

for i in "${TILES_MASK[@]}"; do

  X=${i:1:4}
  Y=${i:7:4}

  X=$(echo $X | sed 's/^0*//')
  Y=$(echo $Y | sed 's/^0*//')

  is_gt "$X" "$XMAX" && XMAX="$X"
  is_lt "$X" "$XMIN" && XMIN="$X"
  is_gt "$Y" "$YMAX" && YMAX="$Y"
  is_lt "$Y" "$YMIN" && YMIN="$Y"

done

# write allow-list
echo "$NTILES" > "$FILE_ALLOW"
printf "%s\n" "${TILES_MASK[@]}" >> "$FILE_ALLOW"

# suggest parameters
echo ""
echo "Suggested Processing extent parameters:"
echo "X_TILE_RANGE = $XMIN $XMAX"
echo "Y_TILE_RANGE = $YMIN $YMAX"
echo "FILE_TILE = $FILE_ALLOW"
echo ""

# change to previous dir
cd "$DIR_NOW" || {
  echoerr "could not change back to $DIR_NOW"
  exit 1
}

# clean temporary data
rm -r "$DIR_TMP"

exit 0

