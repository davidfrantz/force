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


issmaller(){
  awk -v n1="$1" -v n2="$2" 'BEGIN {print n1<n2}'
}

floor(){
  echo $1
  awk -v n1="$1" 'BEGIN{x=int(n1);print(x==n1||n1>0)?x:x-1}'
}


EXPECTED_ARGS=4

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` input-file datacube-dir resample resolution"
  echo ""
  echo "       input-file:   the file you want to cube"
  echo "       datacube-dir: the directory you want to store the cubes;"
  echo "                     datacube-definition.prj needs to exist in there"
  echo "       resample:     resampling method"
  echo "                     (1) any GDAL resampling method for raster data, e.g. cubic"
  echo "                     (2) rasterize for vector data"
  echo "       resolution:   the resolution of the cubed data"
  echo ""
  exit
fi


INP=$1
OUT=$2
RESAMPLE=$3
RES=$4

if [ ! -r $INP ]; then
  echo "$INP is not existing/readable"
  exit
fi

if [ ! -r $OUT/datacube-definition.prj ]; then
  echo "$OUT/datacube-definition.prj is not existing/readable"
  exit
fi

BASE=$(basename $INP)
BASE=${BASE%%.*}
TMP=$(echo $INP | sed 's+/+_+g')


WKT=$(head -n 1 $OUT/datacube-definition.prj)
ORIGX=$(head -n 4 $OUT/datacube-definition.prj  | tail -1 )
ORIGY=$(head -n 5 $OUT/datacube-definition.prj | tail -1 )
TILESIZE=$(head -n 6 $OUT/datacube-definition.prj | tail -1 )
CHUNKSIZE=$(head -n 7 $OUT/datacube-definition.prj | tail -1 )

if [[ ! "$RESAMPLE" =~ "rasterize" ]]; then
  NODATA=$(gdalinfo $INP | grep NoData | head -n 1 |  sed 's/ //g' | cut -d '=' -f 2)
fi


#echo $WKT $ORIGX $ORIGY $TILESIZE $CHUNKSIZE

if [ "$RESAMPLE" == "rasterize" ]; then
  TMP=$TMP"_force-cube-temp.shp"
  ogr2ogr -t_srs "$WKT" $TMP $INP &> /dev/null
  XMIN=$(ogrinfo $TMP -so ${TMP%.*} | grep 'Extent:' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)  
  YMIN=$(ogrinfo $TMP -so ${TMP%.*} | grep 'Extent:' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)  
  XMAX=$(ogrinfo $TMP -so ${TMP%.*} | grep 'Extent:' | cut -d "(" -f 3 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMAX=$(ogrinfo $TMP -so ${TMP%.*} | grep 'Extent:' | cut -d "(" -f 3 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
  INP=$TMP
else
  TMP=$TMP"_force-cube-temp.vrt"
  gdalwarp -q -of 'VRT' -t_srs "$WKT" -tr $RES $RES -r near $INP $TMP
  XMIN=$(gdalinfo $TMP | grep 'Upper Left'  | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMAX=$(gdalinfo $TMP | grep 'Upper Left'  | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
  XMAX=$(gdalinfo $TMP | grep 'Lower Right' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 1)
  YMIN=$(gdalinfo $TMP | grep 'Lower Right' | cut -d "(" -f 2 | cut -d ")" -f 1 | sed 's/[ ]//g' | cut -d ',' -f 2)
fi

#echo $XMIN $YMAX $XMAX $YMIN


TXMIN=$(echo $XMIN $ORIGX $TILESIZE | awk '{f=($1-$2)/$3;i=int(f);print(i==f||f>0)?i:i-1}')
TYMIN=$(echo $YMAX $ORIGY $TILESIZE | awk '{f=($2-$1)/$3;i=int(f);print(i==f||f>0)?i:i-1}')


ULX=$(echo $ORIGX $TXMIN $TILESIZE | awk '{printf "%f", $1 + $2*$3}')
ULY=$(echo $ORIGY $TYMIN $TILESIZE | awk '{printf "%f", $1 - $2*$3}')
#echo $TXMIN $TYMIN $ULX $ULY

TXMAX=$TXMIN
TYMAX=$TYMIN

ULX=$(echo $ULX $TILESIZE | awk '{printf "%f",  $1+$2}')
while [[ $(issmaller $ULX $XMAX) -eq 1 ]]; do
  TXMAX=$(echo $TXMAX | awk '{print $1+1}')
  ULX=$(echo $ULX $TILESIZE | awk '{printf "%f",  $1+$2}')
done

ULY=$(echo $ULY $TILESIZE | awk '{printf "%f", $1-$2}')
while [[ $(issmaller $YMIN $ULY) -eq 1 ]]; do
  TYMAX=$(echo $TYMAX | awk '{print $1+1}')
  ULY=$(echo $ULY $TILESIZE | awk '{printf "%f",  $1-$2}')
done



function cubethis(){

  x=$1
  y=$2

  #echo $x $y $ORIGX $ORIGY $TILESIZE $CHUNKSIZE $RES $INP $OUT $BASE

  ULX=$(echo $ORIGX $x $TILESIZE | awk '{printf "%f",  $1 + $2*$3}')
  ULY=$(echo $ORIGY $y $TILESIZE | awk '{printf "%f",  $1 - $2*$3}')
  LRX=$(echo $ORIGX $x $TILESIZE | awk '{printf "%f",  $1 + ($2+1)*$3}')
  LRY=$(echo $ORIGY $y $TILESIZE | awk '{printf "%f",  $1 - ($2+1)*$3}')
  
  TILE=$(printf "X%04d_Y%04d" $x $y)
  #echo $ULX $ULY $LRX $LRY $TILE

  XBLOCK=$(echo $TILESIZE  $RES | awk '{print int($1/$2)}')
  YBLOCK=$(echo $CHUNKSIZE $RES | awk '{print int($1/$2)}')
  
  mkdir -p $OUT/$TILE

  OUTFILE=$OUT/$TILE/$BASE".tif"
  
  if [ "$RESAMPLE" == "rasterize" ]; then

    gdal_rasterize -burn 1 -a_nodata 255 -ot 'Byte' -of 'GTiff' -co 'COMPRESS=LZW' -co 'PREDICTOR=2' -co 'NUM_THREADS=ALL_CPUS' -co 'BIGTIFF=YES' -co "BLOCKXSIZE=$XBLOCK" -co "BLOCKYSIZE=$YBLOCK" -init 0 -tr $RES $RES -te $ULX $LRY $LRX $ULY $INP $OUTFILE

    MAX=$(gdalinfo -stats $OUTFILE | grep Maximum | head -n 1 | sed 's/[=,]/ /g' | tr -s ' ' | cut -d ' ' -f 5 | sed 's/\..*//' )
    rm $OUTFILE".aux.xml"
    #echo "max: " $MAX

    if [ $MAX -lt 1 ]; then
      rm $OUTFILE
      exit
    fi

  else
  
    if [ -r $OUTFILE ]; then
      #echo "exists"
      EXIST=1
      OUTFILE=$OUT/$TILE/$BASE"_TEMP2.tif"
    else
      EXIST=0
    fi
  
    gdalwarp -q -srcnodata $NODATA -dstnodata $NODATA -of GTiff -co 'INTERLEAVE=BAND' -co 'COMPRESS=LZW' -co 'PREDICTOR=2' -co 'NUM_THREADS=ALL_CPUS' -co 'BIGTIFF=YES' -co "BLOCKXSIZE=$XBLOCK" -co "BLOCKYSIZE=$YBLOCK" -t_srs "$WKT" -te $ULX $LRY $LRX $ULY -tr $RES $RES -r $RESAMPLE $INP $OUTFILE

    if [ $EXIST -eq 1 ]; then
      #echo "building mosaic"
      mv $OUT/$TILE/$BASE".tif" $OUT/$TILE/$BASE"_TEMP1.tif"
      gdal_merge.py -q -o $OUT/$TILE/$BASE".tif" -n $NODATA -a_nodata $NODATA -init $NODATA -of GTiff -co 'INTERLEAVE=BAND' -co 'COMPRESS=LZW' -co 'PREDICTOR=2' -co 'NUM_THREADS=ALL_CPUS' -co 'BIGTIFF=YES' -co "BLOCKXSIZE=$XBLOCK" -co "BLOCKYSIZE=$YBLOCK" $OUT/$TILE/$BASE"_TEMP1.tif" $OUT/$TILE/$BASE"_TEMP2.tif"
      rm $OUT/$TILE/$BASE"_TEMP1.tif" $OUT/$TILE/$BASE"_TEMP2.tif"
    fi

  fi

}

export -f cubethis 
export WKT=$WKT
export ORIGX=$ORIGX
export ORIGY=$ORIGY
export TILESIZE=$TILESIZE
export CHUNKSIZE=$CHUNKSIZE
export RES=$RES
export INP=$INP
export OUT=$OUT
export BASE=$BASE
export NODATA=$NODATA
export RESAMPLE=$RESAMPLE

parallel cubethis {1} {2}  ::: $(seq $TXMIN $TXMAX) ::: $(seq $TYMIN $TYMAX)

rm *_force-cube-temp* 

