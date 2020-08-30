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

export RASTER_INFO_EXE="gdalinfo"
export VECTOR_INFO_EXE="ogrinfo"
export VECTOR_WARP_EXE="ogr2ogr"
export RASTER_WARP_EXE="gdalwarp"
export RASTER_MERGE_EXE="gdal_merge.py"
export RASTERIZE_EXE="gdal_rasterize"
export PARALLEL_EXE="parallel"
#GDAL_CMD="$GDAL_EXE -of 'JPEG' -ot 'Byte' -b 3 -b 2 -b 1 -outsize 256 256 -scale 0 1000 -q "


echoerr(){ echo "$PROG: $@" 1>&2; }    # warnings and/or errormessages go to STDERR
export -f echoerr

export DEBUG=true # display debug messages?
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

Usage: $PROG [-h] [-r] [-s] [-o] input-file

  optional:
  -h = show this help
  -r = resampling method
       any GDAL resampling method for raster data, e.g. cubic (default)
       is ignored for vector data
  -s = pixel resolution of cubed data, defaults to 10
  -o = output directory: the directory where you want to store the cubes
       defaults to current directory
       \"datacube-definition.prj\" needs to exist in there

  mandatory;
  input-file = the file you want to cube

$PROG: cube raster images or vector geometries
  see https://force-eo.readthedocs.io/en/latest/components/auxilliary/cube.html

HELP
exit 1
}
export -f help

# important, check required commands !!! dies on missing
cmd_not_found "$RASTER_INFO_EXE"
cmd_not_found "$VECTOR_INFO_EXE"
cmd_not_found "$VECTOR_WARP_EXE"
cmd_not_found "$RASTER_WARP_EXE"
cmd_not_found "$RASTER_MERGE_EXE"
cmd_not_found "$RASTERIZE_EXE"
cmd_not_found "$PARALLEL_EXE"

issmaller(){
  awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1<n2) ? "true" : "false"}'
}
export -f issmaller

isgreater(){
  awk -v n1="$1" -v n2="$2" 'BEGIN {print (n1>n2) ? "true" : "false"}'
}
export -f isgreater

function cubethis(){

  x=$1
  y=$2

  debug "$x $y $ORIGX $ORIGY $TILESIZE $CHUNKSIZE $RES $FINP $DOUT $CINP"

  ULX=$(echo $ORIGX $x $TILESIZE | awk '{printf "%f",  $1 + $2*$3}')
  ULY=$(echo $ORIGY $y $TILESIZE | awk '{printf "%f",  $1 - $2*$3}')
  LRX=$(echo $ORIGX $x $TILESIZE | awk '{printf "%f",  $1 + ($2+1)*$3}')
  LRY=$(echo $ORIGY $y $TILESIZE | awk '{printf "%f",  $1 - ($2+1)*$3}')
  TILE=$(printf "X%04d_Y%04d" $x $y)
  XBLOCK=$(echo $TILESIZE  $RES | awk '{print int($1/$2)}')
  YBLOCK=$(echo $CHUNKSIZE $RES | awk '{print int($1/$2)}')
  debug "$ULX $ULY $LRX $LRY $TILE $XBLOCK $YBLOCK"

  mkdir -p "$DOUT/$TILE"
  FOUT="$DOUT/$TILE/$CINP.tif"
  debug "$FOUT"

  if [ "$RASTER" == "false" ]; then

    $RASTERIZE_EXE "-burn 1 -a_nodata 255 -ot Byte -of GTiff -co COMPRESS=LZW -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co BLOCKXSIZE=$XBLOCK -co BLOCKYSIZE=$YBLOCK -init 0 -tr $RES $RES -te $ULX $LRY $LRX $ULY $FINP $FOUT"

    MAX=$($RASTER_INFO_EXE -stats "$FOUT" | grep Maximum | head -n 1 | sed 's/[=,]/ /g' | tr -s ' ' | cut -d ' ' -f 5 | sed 's/\..*//' )
    rm "$FOUT.aux.xml"
    debug "max: $MAX"

    if [ $MAX -lt 1 ]; then
      rm "$FOUT"
      exit 1
    fi

  else

    if [ -r "$FOUT" ]; then
      debug "exists"
      EXIST="true"
      FOUT="$DOUT/$TILE/$CINP'_TEMP2.tif'"
    else
      EXIST="false"
    fi

    $RASTER_WARP_EXE "-q -srcnodata $NODATA -dstnodata $NODATA -of GTiff -co INTERLEAVE=BAND -co COMPRESS=LZW -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co BLOCKXSIZE=$XBLOCK -co BLOCKYSIZE=$YBLOCK -t_srs $WKT -te $ULX $LRY $LRX $ULY -tr $RES $RES -r $RESAMPLE $FINP $FOUT"

    if [ "$EXIST" == "true" ]; then
      debug "building mosaic"
      mv "$DOUT/$TILE/$CINP.tif" "$DOUT/$TILE/$CINP'_TEMP1.tif'"
      $RASTER_MERGE_EXE.py "-q -o $DOUT/$TILE/$CINP.tif -n $NODATA -a_nodata $NODATA -init $NODATA -of GTiff -co INTERLEAVE=BAND -co COMPRESS=LZW -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS -co BIGTIFF=YES -co BLOCKXSIZE=$XBLOCK -co BLOCKYSIZE=$YBLOCK $DOUT/$TILE/$CINP'_TEMP1.tif' $DOUT/$TILE/$CINP'_TEMP2.tif'"
      rm "$DOUT/$TILE/$CINP'_TEMP1.tif'" "$DOUT/$TILE/$CINP'_TEMP2.tif'"
    fi

  fi

}
export -f cubethis


# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hr:s:o: --long help,output: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

RESAMPLE='cubic'
RES=10
DOUT=$PWD
while :; do
  case "$1" in
    -h|--help) help ;;
    -r) RESAMPLE="$2"; shift ;;
    -s) RES="$2"; shift ;;
    -o|--output) DOUT="$2"; shift ;;
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
RESOPT=$($RASTER_WARP_EXE 2>/dev/null | grep -A 1 'Available resampling methods:')
TEMP=$(echo $RESOPT | sed 's/[., ]/%/g')
if [[ ! $TEMP =~ "%$RESAMPLE%" ]]; then 
  echoerr "Unknown resampling method."; 
  echo $RESOPT
  help
fi

if [ $(isgreater $RES 0) == "false" ]; then 
  echoerr "Resolution must be > 0"; help
fi

# further checks and preparations --------------------------------------------------------
if ! [[ -f "$FINP" && -r "$FINP" ]]; then
  echoerr "$FINP is not a readable file, exiting."; exit 1;
fi

# raster, vector, or non-such (then die)
if $RASTER_INFO_EXE $FINP >& /dev/null; then 
  RASTER=true
elif $VECTOR_INFO_EXE $FINP >& /dev/null; then 
  RASTER=false
else
  echoerr "$FINP is not recognized as vector or raster file, exiting."; exit 1;
fi

if ! [[ -d "$DOUT" && -w "$DOUT" ]]; then
  echoerr "$DOUT is not a writeable directory, exiting."; exit 1;
fi

DCDEF="$DOUT/datacube-definition.prj"
if ! [[ -f "$DCDEF" && -r "$DCDEF" ]]; then
  echo "$DCDEF is not a readable file, exiting."; exit 1;
fi


# main thing -----------------------------------------------------------------------------

TMPSTRING="_force-cube-temp"
FTMP=$(echo $FINP | sed 's+/+_+g')
FTMP="$FTMP$TMPSTRING"

# datacube parameters
WKT=$(head -n 1 $DCDEF)
ORIGX=$(head -n 4 $DCDEF  | tail -1 )
ORIGY=$(head -n 5 $DCDEF | tail -1 )
TILESIZE=$(head -n 6 $DCDEF | tail -1 )
CHUNKSIZE=$(head -n 7 $DCDEF | tail -1 )
debug "$WKT $ORIGX $ORIGY $TILESIZE $CHUNKSIZE"

# nodata value
if [ "$RASTER" == "true" ]; then
  NODATA=$($RASTER_INFO_EXE $FINP | grep NoData | head -n 1 |  sed 's/ //g' | cut -d '=' -f 2)
  debug "$NODATA"
fi


# bounding box
if [ "$RASTER" == "true" ]; then
  FTMP="$FTMP.vrt"
  debug "$FTMP"
  $RASTER_WARP_EXE -q -of 'VRT' -t_srs "$WKT" -tr $RES $RES -r near $FINP $FTMP &> /dev/null
  XMIN=$($RASTER_INFO_EXE $FTMP | grep 'Upper Left'  | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMAX=$($RASTER_INFO_EXE $FTMP | grep 'Upper Left'  | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
  XMAX=$($RASTER_INFO_EXE $FTMP | grep 'Lower Right' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMIN=$($RASTER_INFO_EXE $FTMP | grep 'Lower Right' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
else
  FTMP="$FTMP.gpkg"
  debug "$FTMP"
  $VECTOR_WARP_EXE -f 'GPKG' -t_srs "$WKT" $FTMP $FINP &> /dev/null
  XMIN=$($VECTOR_INFO_EXE $FTMP -so $CINP | grep 'Extent:' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMIN=$($VECTOR_INFO_EXE $FTMP -so $CINP | grep 'Extent:' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
  XMAX=$($VECTOR_INFO_EXE $FTMP -so $CINP | grep 'Extent:' | cut -d "(" -f 3 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMAX=$($VECTOR_INFO_EXE $FTMP -so $CINP | grep 'Extent:' | cut -d "(" -f 3 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
  FINP=$FTMP
fi
debug "$XMIN $YMAX $XMAX $YMIN"


# 1st tile, and ulx/uly of 1st tile
TXMIN=$(echo $XMIN $ORIGX $TILESIZE | awk '{f=($1-$2)/$3;i=int(f);print(i==f||f>0)?i:i-1}')
TYMIN=$(echo $YMAX $ORIGY $TILESIZE | awk '{f=($2-$1)/$3;i=int(f);print(i==f||f>0)?i:i-1}')
ULX=$(echo $ORIGX $TXMIN $TILESIZE | awk '{printf "%f", $1 + $2*$3}')
ULY=$(echo $ORIGY $TYMIN $TILESIZE | awk '{printf "%f", $1 - $2*$3}')
debug "UL grid $ULX $ULY"

# step a tile to the west, and check if image is in tile, find last tile
TXMAX=$TXMIN
ULX=$(echo $ULX $TILESIZE | awk '{printf "%f",  $1+$2}')
while [ $(issmaller $ULX $XMAX) == "true" ]; do
  TXMAX=$(echo $TXMAX | awk '{print $1+1}')
  ULX=$(echo $ULX $TILESIZE | awk '{printf "%f",  $1+$2}')
done

# step a tile to the south, and check if image is in tile, find last tile
TYMAX=$TYMIN
ULY=$(echo $ULY $TILESIZE | awk '{printf "%f", $1-$2}')
while [ $(issmaller $YMIN $ULY) == "true" ]; do
  TYMAX=$(echo $TYMAX | awk '{print $1+1}')
  ULY=$(echo $ULY $TILESIZE | awk '{printf "%f",  $1-$2}')
done
debug "X_TILE_RANGE = $TXMIN $TXMAX"
debug "Y_TILE_RANGE = $TYMIN $TYMAX"


# cube the file, spawn multiple jobs for each tile
export WKT ORIGX ORIGY TILESIZE CHUNKSIZE RES FINP DOUT CINP NODATA RESAMPLE 
$PARALLEL_EXE cubethis {1} {2} ::: $(seq $TXMIN $TXMAX) ::: $(seq $TYMIN $TYMAX)

# remove the temporary file
rm "$FTMP"

