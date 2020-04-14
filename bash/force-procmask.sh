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


function mask(){

  FILE=$1
  TILE=$(dirname $FILE)
 
  mkdir -p $OUT/$TILE

  gdal_calc.py -A $FILE --A_band=$IBAND --outfile=$OUT/$TILE/$OBASE --calc=$CALC --NoDataValue=255 --type=Byte --format='GTiff' --creation-option='COMPRESS=LZW' --creation-option='PREDICTOR=2' --creation-option='NUM_THREADS=ALL_CPUS' --creation-option='BIGTIFF=YES' --creation-option="BLOCKXSIZE=$XBLOCK" --creation-option="BLOCKYSIZE=$YBLOCK"

}

export -f mask


EXPECTED_ARGS=7

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` input-cube output-cube"
  echo "                         input-base output-base"
  echo "                         input-band calc-expr resolution"
  echo ""
  echo "       input-cube:   directory of input  cubes"
  echo "       output-cube:  directory of output cubes"
  echo "       input-base:   basename of cubed input raster"
  echo "       output-base:  basename of cubed processing masks"
  echo "       input-base:   band of cubed input raster, "
  echo "                     from which to generate the processing mask"
  echo "       calc-expr:    Calculation in gdalnumeric syntax, e.g. 'A>2500'"
  echo "                     The input variable is 'A'"
  echo "                     For details about GDAL expressions, see "
  echo "                       https://gdal.org/programs/gdal_calc.html"
  echo "       resolution:   the resolution of the cubed data"
  echo ""
  exit
fi


INP=$1
OUT=$2
IBASE=$3
OBASE=$4
IBAND=$5
CALC=$6
RES=$7


if [ ! -r $INP ]; then
  echo "$INP is not existing/readable"
  exit
fi

if [ ! -w $OUT ]; then
  echo "$INP is not existing/writeable"
  exit
fi

if [ ! -r $INP/datacube-definition.prj ]; then
  echo "$OUT/datacube-definition.prj is not existing/readable"
  exit
fi

if [ ! -r $OUT/datacube-definition.prj ]; then
  echo "copying datacube-definition.prj"
  cp $INP/datacube-definition.prj $OUT/datacube-definition.prj
fi

# force tif extension
OBASE=$(basename $OBASE)
OBASE=${OBASE%%.*}
OBASE=$OBASE".tif"


NOW=$PWD
cd $INP

# list with input images
TEMP=temp-force-procmask.txt

ls X*/$IBASE > $TEMP

if [ $(cat $TEMP | wc -l) -lt 1 ]; then
  echo "could not find any instance of $IBASE in $INP"
  rm $TEMP
  exit
fi

# tile /chunk size
TILESIZE=$(head -n 6 $OUT/datacube-definition.prj | tail -1 )
CHUNKSIZE=$(head -n 7 $OUT/datacube-definition.prj | tail -1 )

# block size
XBLOCK=$(echo $TILESIZE  $RES | awk '{print int($1/$2)}')
YBLOCK=$(echo $CHUNKSIZE $RES | awk '{print int($1/$2)}')

export OUT=$OUT
export OBASE=$OBASE
export IBAND=$IBAND
export CALC=$CALC
export XBLOCK=$XBLOCK
export YBLOCK=$YBLOCK

parallel -a $TEMP --eta mask {}

rm $TEMP

cd $PWD

