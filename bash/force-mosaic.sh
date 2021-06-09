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

export INFO_EXE="gdalinfo"
export VRTBUILD_EXE="gdalbuildvrt"
export MDCOPY_EXE="$BIN/force-mdcp"


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

Usage: $PROG [-h] [-v] [-i] [-j] [-m] datacube-dir

  -h  = show this help
  -v  = show version
  -i  = show program's purpose

  -j  = number of parallel processes (default: all)

  -m  = mosaic directory (default: mosaic)
        This should be a directory relative to the tiles

  Positional arguments:
  - 'datacube-dir': directory of existing datacube

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$INFO_EXE $VRTBUILD_EXE $MDCOPY_EXE"

function mosaic_this(){

  num=$1
  prd=$2
  LIST="force-mosaic_list_"$prd".txt"

  echo "mosaicking" $prd

  ONAME=${prd/.dat/.vrt}
  ONAME=${ONAME/.tif/.vrt}
  debug "output name: $ONAME"

  # file list (relative to $DOUT)
  find -L "$ROUT" -name $prd 1> $LIST 2> /dev/null

  # file list exists?
  if [ ! -f $LIST ]; then
    echo "could not create file listing."
    exit 1
  fi

  # number of chips
  N=$(wc -l $LIST | cut -d " " -f 1)
  debug "$N chips"

  # nodata value
  FIRST=$(head -1 $LIST)
  NODATA=$($INFO_EXE $FIRST | grep 'NoData Value' | head -1 | cut -d '=' -f 2)
  debug "Nodata value: $NODATA"

  # build vrt
  if [ $N -gt 0 ]; then

    echo $N "chips found".

    #build VRT
    $VRTBUILD_EXE -q -srcnodata $NODATA -vrtnodata $NODATA -input_file_list $LIST $ONAME

    # set vrt to relative paths
    sed -i.tmp 's/relativeToVRT="0"/relativeToVRT="1"/g' $ONAME
    chmod --reference $ONAME".tmp" $ONAME
    rm $ONAME".tmp"

    # copy metadata
    $MDCOPY_EXE $FIRST $ONAME

  else
    echo "no chip found."
  fi

  # delete list
  rm $LIST
  if [ -f $LIST ]; then
    echo "deleting file listing failed."
    exit
  fi
  
  echo ""

}
export -f mosaic_this


# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hvij:m: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

# default options
MOSAIC='mosaic'
CPU=0

while :; do
  case "$1" in
    -h) help ;;
    -v) echo "this should print the version. todo"; exit 0 ;;
    -i) echo "Mosaicking of image chips"; exit 0 ;;
    -j) CPU="$2"; shift ;;
    -m) MOSAIC="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -ne $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
else
  FINP=$(readlink -f $1) # absolute file path
  BINP=$(basename $FINP) # basename
  CINP=${BINP%%.*}       # corename (without extension)
  DINP=$(dirname  $FINP) # directory name
fi

# options received, check now ------------------------------------------------------------
# something to check?
debug "mosaic directory: $MOSAIC"
debug "CPUs: $CPU"

# further checks and preparations --------------------------------------------------------
if ! [[ -d "$FINP" && -w "$FINP" ]]; then
  echoerr "$FINP is not a writeable directory, exiting."; exit 1;
fi


# main thing -----------------------------------------------------------------------------

NOW=$PWD

DOUT="$FINP/$MOSAIC"

mkdir -p "$DOUT"

# output dir exists?
if [ ! -d $DOUT ]; then
  echo "creating output directory failed."
  exit 1
fi

cd $DOUT


PRODUCTS="force-mosaic_products.txt"

ROUT=$(perl -e 'use File::Spec; print File::Spec->abs2rel(@ARGV) . "\n"' "$FINP" "$DOUT")
export ROUT
debug "relative output path: $ROUT"

find -L "$ROUT" \( -name '*.dat' -o -name '*.tif' \) -exec basename {} \; | sort | uniq > $PRODUCTS
NPROD=$(wc -l $PRODUCTS | cut -d " " -f 1)

echo "mosaicking $NPROD products:"
parallel -j $CPU -a $PRODUCTS echo        {#} {}
parallel -j $CPU -a $PRODUCTS mosaic_this {#} {}

rm $PRODUCTS

cd $PWD

exit 0
