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

# this script is a wrapper, and acts as bridge between a batch interface, and the core FORCE Level-2 processing system


EXPECTED_ARGS=6

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: `basename $0` image parfile bindir logdir tempdir timeout_zip"
  echo ""
  exit 1
fi


IMAGE=$1
PRM=$2
BINDIR=$3
LOGDIR=$4
TEMPDIR=$5
TIMEOUT_ZIP=$6

BASE=$(basename $IMAGE)
LOGFILE=$LOGDIR/$BASE.log

# test if image is ok
if [ ! -r $IMAGE ]; then
  echo "$IMAGE: $IMAGE is not existing/readable" > $LOGFILE
  exit 1
fi

# test if parameterfile is ok
if [ ! -r $PRM ]; then
  echo "$IMAGE: $PRM is not existing/readable" > $LOGFILE
  exit 1
fi

# test if bindir is ok
if [ ! -r $BINDIR ]; then
  echo "$IMAGE: $BINDIR is not existing/readable" > $LOGFILE
  exit 1
fi

# test if logdir is ok
if [ ! -r $LOGDIR ]; then
  echo "$IMAGE: $LOGDIR is not existing/readable" > $LOGFILE
  exit 1
fi

# test if tempdir is ok
if [ ! -r $TEMPDIR ]; then
  echo "$IMAGE: $TEMPDIR is not existing/readable" > $LOGFILE
  exit 1
fi

# test if executable
EXE=$BINDIR/force-l2ps
if [ ! -x $EXE ]; then
  echo "$IMAGE: $EXE is not existing/executable" > $LOGFILE
  exit 1
fi

# test if DEBUG mode is on
$EXE ? ?
if [ $? -eq 1 ]; then
  echo "$IMAGE: $EXE was compiled in DEBUG mode" > $LOGFILE
  exit 1
fi

TODO=$IMAGE

# extract Landsat image
if [[ $IMAGE == *".tar.gz"* ]]; then

  BASE_=$(echo $BASE | sed 's/.tar.gz//')
  TODO=$TEMPDIR/$BASE_

  mkdir $TODO
  if [ ! -d $TODO ]; then
    echo "$IMAGE: creating temp directory failed" > $LOGFILE
    exit 1
  fi

  timeout -k $TIMEOUT_ZIP 10m tar --ignore-command-error -xzf $IMAGE --exclude gap_mask --exclude='*GCP.txt' --exclude='*VER.jpg' --exclude='*VER.txt' --exclude='*BQA.TIF' --exclude='*.GTF' --exclude='LE07*B8.TIF' --exclude='LE07*B6_VCID_2.TIF' --exclude='LC08*B11.TIF' --exclude='LC08*B8.TIF' -C $TODO &> /dev/null
  if [ ! $? -eq 0 ]; then
    echo "$IMAGE: tar.gz container is corrupt, connection stalled or similar." > $LOGFILE
    exit 1
  fi

fi

# extract Sentinel-2 image 
if [[ $IMAGE == *".zip"* ]]; then

   timeout -k $TIMEOUT_ZIP 10m unzip -qq -d $TEMPDIR $IMAGE &>/dev/null
   if [ ! $? -eq 0 ]; then
     echo "$IMAGE: zip container is corrupt, connection stalled or similar." > $LOGFILE
     exit 1
   fi

   #BASE_=$(echo $BASE | sed 's/.zip/.SAFE/')
   #TODO=$TEMPDIR/$BASE_/GRANULE/*
   #if [ ! -d $TODO ]; then
   #  echo "$IMAGE: unzipping image failed" > $LOGFILE
   #  exit 1
   #fi
   BASE_=$(echo $BASE | sed 's/.zip/.SAFE/')
   TODO=$TEMPDIR/$BASE_
   if [ ! -d $TODO ]; then
     echo "$IMAGE: unzipping image failed" > $LOGFILE
     exit 1
   fi

fi

# process the image
printf "$IMAGE: " > $LOGFILE
$EXE $TODO $PRM >> $LOGFILE
#if [ ! $? -eq 0 ]; then
#  echo "FORCE L2PS signaled non-zero exit status" >> $LOGFILE
#  exit 1
#fi

# clean up if Landsat was extracted
if [[ $IMAGE == *".tar.gz"* ]]; then
  rm -r $TODO
fi

# clean up if Sentinel-2 was extracted
if [[ $IMAGE == *".zip"* ]]; then
  rm -r $TEMPDIR/$BASE_
fi

exit 0
