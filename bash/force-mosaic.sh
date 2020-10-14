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


EXPECTED_ARGS=1

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: `basename $0` datacube-dir"
  echo ""
  exit
fi

NOW=$PWD
BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

INP=$(readlink -f $1)
OUT=$INP/mosaic



# input dir exists?
if [ ! -d $INP ]; then
  echo $INP "does not exist"
  exit
fi

# output dir exists?
if [ ! -d $OUT ]; then
  mkdir $OUT
  if [ ! -d $OUT ]; then
    echo "creating directory failed."
    exit
  fi
fi

cd $OUT

function mosaic_this(){

  num=$1
  prd=$2
  bin=$3
  LIST="force-mosaic_list_$2.txt"

  #echo $bin

  echo "mosaicking" $prd

  ONAME=${prd/.dat/.vrt}
  ONAME=${ONAME/.tif/.vrt}

  # file list (relative to $OUT)
  find .. -name $prd > $LIST

  # file list exists?
  if [ ! -f $LIST ]; then
    echo "could not create file listing."
    exit
  fi

  # number of chips
  N=$(wc -l $LIST | cut -d " " -f 1)

  # nodata value
  FIRST=$(head -1 $LIST)
  NODATA=$(gdalinfo $FIRST | grep 'NoData Value' | head -1 | cut -d '=' -f 2)


  # build vrt
  if [ $N -gt 0 ]; then

    echo $N "chips found".

    #build VRT
    gdalbuildvrt -q -srcnodata $NODATA -vrtnodata $NODATA -input_file_list $LIST $ONAME

    # set vrt to relative paths
    sed -i.tmp 's/relativeToVRT="0"/relativeToVRT="1"/g' $ONAME
    chmod --reference $ONAME".tmp" $ONAME
    rm $ONAME".tmp"

    # copy metadata
    $bin"/"force-mdcp $FIRST $ONAME

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


PRODUCTS="force-mosaic_products.txt"

find .. \( -name '*.dat' -o -name '*.tif' \) -exec basename {} \; | sort | uniq > $PRODUCTS
NPROD=$(wc -l $PRODUCTS | cut -d " " -f 1)

echo "mosaicking $NPROD products:"
parallel -a $PRODUCTS echo        {#} {}
parallel -a $PRODUCTS mosaic_this {#} {} $BINDIR

rm $PRODUCTS

cd $PWD

exit 0
