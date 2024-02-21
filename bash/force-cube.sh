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

# this script ingests external data into a datacube structure as needed by FORCE HLPS

# functions/definitions ------------------------------------------------------------------
export PROG=`basename $0`;
export BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export MISC="$BIN/force-misc"

# source bash "library" file
LIB="$MISC/force-bash-library.sh"
eval ". ${LIB}" >/dev/null 2>&1 ;[[ "$?" -ne "0" ]] && echo "loading bash library failed" && exit 1;
export LIB


MANDATORY_ARGS=1

export RASTER_INFO_EXE="gdalinfo"
export VECTOR_INFO_EXE="ogrinfo"
export VECTOR_WARP_EXE="ogr2ogr"
export RASTER_WARP_EXE="gdalwarp"
export RASTER_MERGE_EXE="gdal_merge.py"
export RASTERIZE_EXE="gdal_rasterize"
export PARALLEL_EXE="parallel"

help(){
cat <<HELP

Usage: $PROG [-hvirsantobj] input-file(s)

  optional:
  -h = show this help
  -v = show version
  -i = show program's purpose
  -r = resampling method
       any GDAL resampling method for raster data, e.g. near (default)
       is ignored for vector data
  -s = pixel resolution of cubed data, defaults to 10
  -a = optional attribute name for vector data. $PROG will burn these values 
       into the output raster. default: no attribute is used; a binary mask 
       with geometry presence (1) or absence (0) is generated
  -l = layer name for vector data (default: basename of input, without extension)
  -n = output nodata value (defaults to 255) 
  -t = output data type (defaults to Byte; see GDAL for datatypes; 
       but note that FORCE HLPS only understands Int16 and Byte types correctly)
  -o = output directory: the directory where you want to store the cubes
       defaults to current directory
       'datacube-definition.prj' needs to exist in there
  -b = basename of output file (without extension)
       defaults to the basename of the input-file
       cannot be used when multiple input files are given
  -j = number of jobs, defaults to 100%

  mandatory:
  input-file(s) = the file(s) you want to cube

  -----
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

function cubethis(){

  x=$1
  y=$2

  debug "$x $y $ORIGX $ORIGY $TILESIZE $CHUNKSIZE $RES $FINP $DOUT $COUT"

  ULX=$(echo $ORIGX $x $TILESIZE | awk '{printf "%f",  $1 + $2*$3}')
  ULY=$(echo $ORIGY $y $TILESIZE | awk '{printf "%f",  $1 - $2*$3}')
  LRX=$(echo $ORIGX $x $TILESIZE | awk '{printf "%f",  $1 + ($2+1)*$3}')
  LRY=$(echo $ORIGY $y $TILESIZE | awk '{printf "%f",  $1 - ($2+1)*$3}')
  TILE=$(printf "X%04d_Y%04d" $x $y)
  XBLOCK=$(echo $TILESIZE  $RES | awk '{print int($1/$2)}')
  YBLOCK=$(echo $CHUNKSIZE $RES | awk '{print int($1/$2)}')
  debug "$ULX $ULY $LRX $LRY $TILE $XBLOCK $YBLOCK"

  mkdir -p "$DOUT/$TILE"
  FOUT="$DOUT/$TILE/$COUT.tif"
  debug "$FOUT"

  if [ "$RASTER" == "false" ]; then

    if [ "$ATTRIBUTE" == "DEFAULT" ]; then
      BURNOPT=( -burn 1 )
    else
      BURNOPT=( -a "$ATTRIBUTE" )
    fi

    $RASTERIZE_EXE "${BURNOPT[@]}" -a_nodata $ONODATA -ot $DATATYPE -of GTiff \
      -co COMPRESS=LZW -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS \
      -co BIGTIFF=YES -co BLOCKXSIZE=$XBLOCK -co BLOCKYSIZE=$YBLOCK \
      -init 0 -tr $RES $RES -te $ULX $LRY $LRX $ULY "$FINP" "$FOUT"
    if [ $? -ne 0 ]; then
      echoerr "rasterizing failed."; exit 1
    fi

    VALID=$($RASTER_INFO_EXE -stats "$FOUT" 2>/dev/null | grep STATISTICS_VALID_PERCENT | sed 's/ //g; s/[=,]/ /g' | cut -d ' ' -f2 | awk '{sum +=$1} END {print sum != 0}' )
    rm "$FOUT.aux.xml"
    debug "valid: $VALID"

    if [ $VALID -eq 0 ]; then
      rm "$FOUT"
      exit 1
    fi

  else

    if [ -r "$FOUT" ]; then
      debug "exists"
      EXIST="true"
      FOUT="$DOUT/$TILE/$COUT'_TEMP2.tif'"
    else
      EXIST="false"
    fi

    $RASTER_WARP_EXE -q -srcnodata $INODATA -dstnodata $ONODATA -ot $DATATYPE -of GTiff \
      -co INTERLEAVE=BAND -co COMPRESS=LZW -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS \
      -co BIGTIFF=YES -co BLOCKXSIZE=$XBLOCK -co BLOCKYSIZE=$YBLOCK \
      -t_srs "$WKT" -te $ULX $LRY $LRX $ULY -tr $RES $RES -r $RESAMPLE "$FINP" "$FOUT"
    if [ $? -ne 0 ]; then
      echoerr "warping failed."; exit 1
    fi

    VALID=$($RASTER_INFO_EXE -stats "$FOUT" 2>/dev/null | grep STATISTICS_VALID_PERCENT | sed 's/ //g; s/[=,]/ /g' | cut -d ' ' -f2 | awk '{sum +=$1} END {print sum != 0}' )
    rm "$FOUT.aux.xml"
    debug "valid: $VALID"

    if [ $VALID -eq 0 ]; then
      rm "$FOUT"
      exit 1
    fi


    if [ "$EXIST" == "true" ]; then
      debug "building mosaic"
      mv "$DOUT/$TILE/$COUT.tif" "$DOUT/$TILE/$COUT'_TEMP1.tif'"
      $RASTER_MERGE_EXE -q -o $DOUT/$TILE/$COUT.tif -ot $DATATYPE -of GTiff \
        -n $ONODATA -a_nodata $ONODATA -init $ONODATA \
        -co INTERLEAVE=BAND -co COMPRESS=LZW -co PREDICTOR=2 -co NUM_THREADS=ALL_CPUS \
        -co BIGTIFF=YES -co BLOCKXSIZE=$XBLOCK -co BLOCKYSIZE=$YBLOCK \
        "$DOUT/$TILE/$COUT'_TEMP1.tif'" "$DOUT/$TILE/$COUT'_TEMP2.tif'"
      if [ $? -ne 0 ]; then
        echoerr "merging failed."; exit 1
      fi
      rm "$DOUT/$TILE/$COUT'_TEMP1.tif'" "$DOUT/$TILE/$COUT'_TEMP2.tif'"
    fi

  fi

}
export -f cubethis


# now get the options --------------------------------------------------------------------
ARGS=`getopt -o hvir:s:o:b:j:a:n:t:l: --long help,version,info,resample:,resolution:,output:,basename:,jobs:,attribute:,nodata:,datatype:,layer: -n "$0" -- "$@"`
if [ $? != 0 ] ; then help; fi
eval set -- "$ARGS"

RESAMPLE="near"
RES=10
DOUT=$PWD
BASE="DEFAULT"
NJOB="100%"
ATTRIBUTE="DEFAULT"
DATATYPE="Byte"
ONODATA=255
LAYER="DEFAULT"

while :; do
  case "$1" in
    -h|--help) help ;;
    -v|--version) force_version; exit 0;;
    -i|--info) echo "Ingestion of auxiliary data into datacube format"; exit 0;;
    -r|--resample) RESAMPLE="$2"; shift ;;
    -s|--resolution) RES="$2"; shift ;;
    -a|attribute) ATTRIBUTE="$2"; shift;;
    -l|layer) LAYER="$2"; shift;;
    -n|nodata) ONODATA="$2"; shift;;
    -t|datatype) DATATYPE="$2"; shift;;
    -o|--output) DOUT="$2"; shift ;;
    -b|--basename) BASE="$2"; shift ;;
    -j|--jobs) NJOB="$2"; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
  shift
done

debug "resample = $RESAMPLE"
debug "resolution = $RES"
debug "attribute = $ATTRIBUTE"
debug "nodata = $ONODATA"
debug "datatype = $DATATYPE"
debug "output = $DOUT"
debug "basename = $BASE"
debug "jobs = $NJOB"

if [ $# -lt $MANDATORY_ARGS ] ; then 
  echoerr "Mandatory argument is missing."; help
fi
debug "$# input files will be cubed"

# options received, check now ------------------------------------------------------------
RESOPT=$($RASTER_WARP_EXE 2>/dev/null | grep -A 1 'Available resampling methods:')
TEMP=$(echo $RESOPT | sed 's/[., ]/%/g')
if [[ ! $TEMP =~ "%$RESAMPLE%" ]]; then 
  echoerr "Unknown resampling method."; 
  echo $RESOPT
  help
fi

if is_le "$RES" 0; then 
  echoerr "Resolution must be > 0"; help
fi

if [ $# -gt 1  ] && [ "$BASE" != "DEFAULT" ]; then 
  echoerr "Multiple input files are given. Do not give a new basename."; help
fi

# further checks and preparations --------------------------------------------------------
if ! [[ -d "$DOUT" && -w "$DOUT" ]]; then
  echoerr "$DOUT is not a writeable directory, exiting."; exit 1;
fi

DCDEF="$DOUT/datacube-definition.prj"
if ! [[ -f "$DCDEF" && -r "$DCDEF" ]]; then
  echo "$DCDEF is missing in $DOUT, exiting."; exit 1;
fi


# main thing -----------------------------------------------------------------------------
for i in "$@"; do

  FINP=$(readlink -f $i) # absolute file path
  DINP=$(dirname  $FINP) # directory name
  BINP=$(basename $FINP) # basename
  CINP=${BINP%%.*}       # corename (without extension)

  if [ $BASE == "DEFAULT" ]; then
    COUT=$CINP  # corename from input file
  else
    COUT=$BASE # corename from user parameters
  fi

  if [ $LAYER == "DEFAULT" ]; then
    LAYER=$CINP  # layername from input file
  fi

  debug "input  file:      $FINP"
  debug "input  directory: $DINP"
  debug "input  layername: $LAYER"
  debug "input  basename:  $BINP"
  debug "input  corename:  $CINP"
  debug "output corename:  $COUT"

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
  debug "raster: $RASTER"

  # temporary file
  TMPSTRING="_force-cube-temp"
  FTMP=$(echo $FINP | sed 's+/+_+g')
  FTMP="$DOUT/$FTMP$TMPSTRING"
  debug "tempfile: $FTMP"

  # datacube parameters
  WKT=$(head -n 1 $DCDEF)
  ORIGX=$(head -n 4 $DCDEF  | tail -1 )
  ORIGY=$(head -n 5 $DCDEF | tail -1 )
  TILESIZE=$(head -n 6 $DCDEF | tail -1 )
  CHUNKSIZE=$(head -n 7 $DCDEF | tail -1 )
  debug "$WKT $ORIGX $ORIGY $TILESIZE $CHUNKSIZE"

  # nodata value
  if [ "$RASTER" == "true" ]; then
    INODATA=$($RASTER_INFO_EXE $FINP | grep NoData | head -n 1 |  sed 's/ //g' | cut -d '=' -f 2)
    debug "nodata: $INODATA"
    if [ -z $INODATA ]; then
      echoerr "NODATA not found in $FINP"; exit 1;
    fi
  fi

  # is given layer name present?
  if [ "$RASTER" == "false" ]; then
    $VECTOR_INFO_EXE $FINP $LAYER &> /dev/null
    if [ $? -ne 0 ]; then
      echoerr "requested layer was not found."; exit 1
    fi
  fi

  # bounding box
  if [ "$RASTER" == "true" ]; then
    FTMP="$FTMP.vrt"
    debug "tempfile: $FTMP"
    $RASTER_WARP_EXE -q -of 'VRT' -t_srs "$WKT" -tr $RES $RES -r near $FINP $FTMP #&> /dev/null
    if [ $? -ne 0 ]; then
      echoerr "quick warping failed."; exit 1
    fi
    XMIN=$($RASTER_INFO_EXE $FTMP | grep 'Upper Left'  | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
    YMAX=$($RASTER_INFO_EXE $FTMP | grep 'Upper Left'  | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
    XMAX=$($RASTER_INFO_EXE $FTMP | grep 'Lower Right' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
    YMIN=$($RASTER_INFO_EXE $FTMP | grep 'Lower Right' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
  else
    FTMP="$FTMP.gpkg"
    debug "tempfile: $FTMP"
    $VECTOR_WARP_EXE -f 'GPKG' -t_srs "$WKT" $FTMP $FINP &> /dev/null
    if [ $? -ne 0 ]; then
      echoerr "warping vector failed."; exit 1
    fi
    XMIN=$($VECTOR_INFO_EXE $FTMP -so $LAYER | grep 'Extent:' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
    YMIN=$($VECTOR_INFO_EXE $FTMP -so $LAYER | grep 'Extent:' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
    XMAX=$($VECTOR_INFO_EXE $FTMP -so $LAYER | grep 'Extent:' | cut -d "(" -f 3 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
    YMAX=$($VECTOR_INFO_EXE $FTMP -so $LAYER | grep 'Extent:' | cut -d "(" -f 3 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
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
  while is_lt "$ULX" "$XMAX"; do
    TXMAX=$(echo $TXMAX | awk '{print $1+1}')
    ULX=$(echo $ULX $TILESIZE | awk '{printf "%f",  $1+$2}')
  done

  # step a tile to the south, and check if image is in tile, find last tile
  TYMAX=$TYMIN
  ULY=$(echo $ULY $TILESIZE | awk '{printf "%f", $1-$2}')
  while is_lt "$YMIN" "$ULY"; do
    TYMAX=$(echo $TYMAX | awk '{print $1+1}')
    ULY=$(echo $ULY $TILESIZE | awk '{printf "%f",  $1-$2}')
  done
  debug "X_TILE_RANGE = $TXMIN $TXMAX"
  debug "Y_TILE_RANGE = $TYMIN $TYMAX"


  # cube the file, spawn multiple jobs for each tile
  export WKT ORIGX ORIGY TILESIZE CHUNKSIZE RES 
  export FINP DOUT COUT INODATA ONODATA RESAMPLE RASTER DATATYPE ATTRIBUTE
  $PARALLEL_EXE -j $NJOB cubethis {1} {2} ::: $(seq $TXMIN $TXMAX) ::: $(seq $TYMIN $TYMAX)

  # remove the temporary file
  rm "$FTMP"
  # remove empty folders in datacube
  find $DOUT -type d -regextype grep -regex ".*X[0-9-]\{4\}_Y[0-9-]\{4\}" -empty -delete

done

exit 0
