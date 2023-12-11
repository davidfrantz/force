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

EXPECTED_ARGS=3

if [ $# -ne $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` input-vector datacube-dir allow-list"
  echo ""
  echo "       input-file:   a polygon vector file"
  echo "       datacube-dir: the directory of a datacube;"
  echo "                     datacube-definition.prj needs to exist in there"
  echo "       allow-list:   a tile allow-list to restrict the processing extent."
  echo "                     This file will be written"
  echo ""
  exit
fi


INP=$1
DIR=$2
LIST=$3

if [ ! -r $INP ]; then
  echo "$INP is not existing/readable"
  exit
fi

if [ ! -r $DIR/datacube-definition.prj ]; then
  echo "$DIR/datacube-definition.prj is not existing/readable"
  exit
fi


BASE=$(basename $INP)
BASE=${BASE%%.*}
TMP=$DIR"/"$BASE
#echo $TMP

mkdir -p $TMP
cp $DIR/datacube-definition.prj $TMP/datacube-definition.prj

# generate masks, use force-cube version relative to this program
BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
$BINDIR"/"force-cube -j 0 -s 10 -o $TMP -b force-extent $INP &> /dev/null


WD=$PWD
cd $TMP

# find all the generated masks
FILES=$(ls -d X*/*.tif)
TILES=($(dirname $FILES))
NTILES=${#TILES[*]}


# compute range of tiles
XMIN=9999
XMAX=-999
YMIN=9999
YMAX=-999

for i in "${TILES[@]}"; do

  X=${i:1:4}
  Y=${i:7:4}

  X=$(echo $X | sed 's/^0*//')
  Y=$(echo $Y | sed 's/^0*//')

  (( X > XMAX )) && XMAX=$X
  (( X < XMIN )) && XMIN=$X
  (( Y > YMAX )) && YMAX=$Y
  (( Y < YMIN )) && YMIN=$Y

done

echo ""
echo "Suggested Processing extent:"
echo "X_TILE_RANGE =" $XMIN $XMAX
echo "Y_TILE_RANGE =" $YMIN $YMAX
echo ""


# check if processing extent is square or not
NSQX=$((XMAX-XMIN+1))
NSQY=$((YMAX-YMIN+1))
NSQ=$((NSQX*NSQY))

if [ "$NTILES" -lt "$NSQ" ]; then

  echo "Processing extent is not square."
  echo "Suggest to use the tile allow-list:"
  echo "FILE_TILE =" $LIST

else

  echo "Processing extent is square."
  echo "Using the tile allow-list is not necessary, but can be used with:"
  echo "FILE_TILE =" $LIST

fi

echo ""

cd $WD

# write allow-list
echo $NTILES > $LIST
printf "%s\n" "${TILES[@]}" >> $LIST


rm -r $TMP

exit 0

