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
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MANDATORY_ARGS=2

export CALC_EXE="gdal_calc.py"
export PARALLEL_EXE="parallel"

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

Usage: $PROG [-sldobj] input-basename calc-expr

  optional:
  -s = pixel resolution of cubed data, defaults to 10
  -l = input-layer: band number in case of multi-band input rasters,
       defaults to 1
  -d = input directory: the datacube directory
       defaults to current directory
      'datacube-definition.prj' needs to exist in there
  -o = output directory: the directory where you want to store the cubes
       defaults to current directory
  -b = basename of output file (without extension)
       defaults to the basename of the input-file, 
       appended by '_procmask'
  -j = number of jobs, defaults to 1

  Positional arguments:
  - input-basename: basename of input data
  - calc-expr: Calculation in gdalnumeric syntax, e.g. 'A>2500'"
               The input variable is 'A'
               For details about GDAL expressions, see 
               https://gdal.org/programs/gdal_calc.html

  -----
    see https://force-eo.readthedocs.io/en/latest/components/auxilliary/procmask.html

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$CALC_EXE $PARALLEL_EXE"

function mask(){

  FILE=$1
  TILE=$(dirname $FILE)
 
  mkdir -p $DOUT/$TILE

  gdal_calc.py \
    -A $FILE \
    --A_band=$LAYER \
    --outfile=$DOUT/$TILE/$FOUT \
    --calc=$CALC \
    --NoDataValue=255 \
    --type=Byte \
    --format='GTiff' \
    --creation-option='COMPRESS=LZW' \
    --creation-option='PREDICTOR=2' \
    --creation-option='NUM_THREADS=ALL_CPUS' \
    --creation-option='BIGTIFF=YES' \
    --creation-option="BLOCKXSIZE=$XBLOCK" \
    --creation-option="BLOCKYSIZE=$YBLOCK" \
    --quiet

}
export -f mask


# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hvis:l:d:o:b:j: --long help,version,info,resolution:,layer:,input:,output:,basename:,jobs: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

RES=10
LAYER=1
DINP=$PWD
DOUT=$PWD
OBASE="DEFAULT"
NJOB=1

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) "$BIN"/force -v; exit 0;;
    -i|--info) echo "Processing masks from raster images"; exit 0;;
    -s|--resolution) RES="$2"; shift ;;
    -l|--layer) LAYER="$2"; shift;;
    -d|--input) DINP=$(readlink -f "$2"); shift ;;
    -o|--output) DOUT=$(readlink -f "$2"); shift ;;
    -b|--basename) OBASE="$2"; shift ;;
    -j|--jobs) NJOB="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

if [ $# -ne $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
else
  IBASE="$1"         # basename (with extension)
  CALC="$2"          # GDAL calc expression
fi

debug "input  dir:  $DINP"
debug "output dir:  $DOUT"
debug "input  base: $IBASE"
debug "output base: $OBASE"
debug "input layer: $LAYER"
debug "output res:  $RES"
debug "expression:  $CALC"
debug "njobs:       $NJOB"

if [ ! -r $DINP ]; then
  echoerr "$DINP is not existing/readable"; help
fi

if [ ! -w $DOUT ]; then
  echoerr "$DOUT is not existing/writeable"; help
fi

if [ ! -r $DINP/datacube-definition.prj ]; then
  echoerr "$DINP/datacube-definition.prj is not existing/readable"; help
fi

if [ -r $DOUT/datacube-definition.prj ]; then
  DIFF=$(diff "$DINP/datacube-definition.prj" "$DOUT/datacube-definition.prj")
  NUM=$(echo $DIFF | wc -w)
  if [ $NUM -gt 0 ]; then
    echoerr "input and output datacubes do not match"; help
  fi
else
  echo "copying datacube-definition.prj"
  cp "$DINP/datacube-definition.prj" "$DOUT/datacube-definition.prj"
fi

if [ $OBASE == "DEFAULT" ]; then
  CORE=${IBASE%%.*} # corename from input file
  FOUT="$CORE""_procmask.tif"
else
  CORE=${OBASE%%.*} # corename from user parameters
  FOUT="$CORE.tif"
fi

debug "output file: $FOUT"

# main thing -----------------------------------------------------------------------------

NOW=$PWD
cd $DINP

# list with input images
TEMP="$DOUT/temp-force-procmask.txt"

ls X*/$IBASE > "$TEMP"

if [ $(cat "$TEMP" | wc -l) -lt 1 ]; then
  echoerr "could not find any instance of $IBASE in $DINP"
  rm "$TEMP"
  help
fi

# tile /chunk size
TILESIZE=$(head -n 6 $DOUT/datacube-definition.prj | tail -1 )
CHUNKSIZE=$(head -n 7 $DOUT/datacube-definition.prj | tail -1 )

# block size
XBLOCK=$(echo $TILESIZE  $RES | awk '{print int($1/$2)}')
YBLOCK=$(echo $CHUNKSIZE $RES | awk '{print int($1/$2)}')

export DOUT=$DOUT
export FOUT=$FOUT
export LAYER=$LAYER
export CALC=$CALC
export XBLOCK=$XBLOCK
export YBLOCK=$YBLOCK

$PARALLEL_EXE -j $NJOB -a $TEMP --eta mask {}

rm $TEMP

cd $PWD

exit 0
