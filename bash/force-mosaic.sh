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
export PROG=$(basename "$0")
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MISC="$BIN/force-misc"

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
eval ". ${LIB}" >/dev/null 2>&1 ;[[ "$?" -ne "0" ]] && echo "loading bash library failed" && exit 1;
export LIB


MANDATORY_ARGS=1

export INFO_EXE="gdalinfo"
export VRTBUILD_EXE="gdalbuildvrt"
export MDCOPY_EXE="$BIN/force-mdcp"

help(){
cat <<HELP

Usage: $PROG [-h] [-v] [-i] [-j] [-m] [-e] datacube-dir

  -h  = show this help
  -v  = show version
  -i  = show program's purpose

  -j  = number of parallel processes, defaults to 100%
  -m  = mosaic directory (default: mosaic, created in datacube-dir)
  -e  = extension of the chips (default: tif)

  Positional arguments:
  - 'datacube-dir': directory of existing datacube

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$INFO_EXE $VRTBUILD_EXE $MDCOPY_EXE"

function mosaic_this(){

  # silly workaround to make my linter work :/
  if [ "$1" -eq 0 ] && [ "$2" -eq 0 ]; then
    return 0
  fi

  num="$1"
  prd="$2"
  LIST=force-mosaic_list_"$prd".txt

  echo "mosaicking $num: $prd"

  ONAME=${prd/.$EXTENSION/.vrt}
  debug "output name: $ONAME"

  # file list
  grep "$prd" force-mosaic_all-files.txt 1> "$LIST" 2> /dev/null

  # file list exists?
  if [ ! -f "$LIST" ]; then
    echo "could not create file listing."
    exit 1
  fi

  # number of chips
  N=$(wc -l "$LIST" | cut -d " " -f 1)
  debug "$N chips"

  # build vrt
  if [ "$N" -gt 0 ]; then

    echo "$N chips found".

    # nodata value
    FIRST=$(head -1 "$LIST")
    NODATA=$($INFO_EXE "$FIRST" | grep 'NoData Value' | head -1 | cut -d '=' -f 2)
    debug "Nodata value: $NODATA"

    #build VRT
    $VRTBUILD_EXE -q -srcnodata "$NODATA" -vrtnodata "$NODATA" -input_file_list "$LIST" "$ONAME"

    # set vrt to relative paths
    sed -i.tmp 's/relativeToVRT="0"/relativeToVRT="1"/g' "$ONAME"
    chmod --reference "$ONAME".tmp "$ONAME"
    rm "$ONAME".tmp

    # copy metadata
    $MDCOPY_EXE "$FIRST" "$ONAME"

  else
    echo "no chip found."
  fi

  # delete list
  rm "$LIST"
  if [ -f "$LIST" ]; then
    echo "deleting file listing failed."
    exit
  fi

  echo ""

}
export -f mosaic_this


# now get the options --------------------------------------------------------------------
ARGS=$(getopt -o hvij:e:m: -n "$0" -- "$@")
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

# default options
MOSAIC=''
EXTENSION='tif'
CPU="100%"

while :; do
  case "$1" in
    -h) help ;;
    -v) force_version; exit 0 ;;
    -i) echo "Mosaicking of image chips"; exit 0 ;;
    -j) CPU="$2"; shift ;;
    -m) MOSAIC="$2"; shift ;;
    -e) EXTENSION="$2"; shift ;;
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
debug "EXTENSION: $EXTENSION"

# further checks and preparations --------------------------------------------------------

NOW=$PWD

# check if datacube directory exists
dir_not_found "$FINP"

# output directory for mosaics
if [ -z "$MOSAIC" ]; then
  DOUT="$FINP/$MOSAIC"
else 
  DOUT="$MOSAIC"
fi

if [ ! -d "$DOUT" ]; then

  mkdir -p "$DOUT"
  if [ ! -d "$DOUT" ]; then
    echoerr "creating output directory failed ($DOUT). check permissions."
    exit 1
  fi

fi

# check if output directory is writeable
dir_not_writeable "$DOUT"

# change to output directory, use as working directory
cd "$DOUT" || exit 1


# main thing -----------------------------------------------------------------------------

# retrieve relative path from output directory to datacube directory
ROUT=$(perl -e 'use File::Spec; print File::Spec->abs2rel(@ARGV) . "\n"' "$FINP" "$DOUT")
export ROUT

debug "output directory: $DOUT"
debug "working directory: $PWD"
debug "relative path from output to datacube: $ROUT"

# retrieve all tiles in datacube directory
TILES=$(ls -d "$ROUT"/X* 2>/dev/null)
if [ -z "$TILES" ]; then
  echoerr "no tiles found in datacube directory ($FINP). Make sure you run this tool on a datacube!"
  exit 1
fi


# make file list for each tile
export EXTENSION
echo "$TILES" | parallel -j "$CPU" "ls {}/*.$EXTENSION > force-mosaic_files-{#}.txt 2> /dev/null"

# delete all empty files
find . -type f -name 'force-mosaic_files-*.txt' -empty -delete
NFILES=$(ls -1 force-mosaic_files-*.txt 2> /dev/null | wc -l)
debug "number of files found: $NFILES"

if [ "$NFILES" -eq 0 ]; then
  echoerr "no matching files found in datacube directory ($FINP). Make sure you run this tool on a datacube! Check file extensions."
  exit 1
fi


# all files in one list
cat force-mosaic_files-*.txt > force-mosaic_all-files.txt

# available products for each tile
ls force-mosaic_files-*.txt | parallel -j "$CPU" "cat {} | xargs basename -a | sort | uniq > force-mosaic_products-{#}.txt"

# all products in one list
cat force-mosaic_products-*.txt | sort | uniq > force-mosaic_all-products.txt
NPROD=$(wc -l force-mosaic_all-products.txt | cut -d " " -f 1)

echo "mosaicking $NPROD products:"
#parallel -a force-mosaic_all-products.txt -j "$CPU" echo        {#} {}
parallel -a force-mosaic_all-products.txt -j "$CPU" mosaic_this {#} {}

rm force-mosaic_files-*.txt
rm force-mosaic_all-files.txt
rm force-mosaic_products-*.txt
rm force-mosaic_all-products.txt

# silly workaround to make my linter work :/
mosaic_this 0 0

cd "$NOW" || exit 1

exit 0
