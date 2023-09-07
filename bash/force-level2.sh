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

# this script is an higher-level batch interface for the FORCE Level-2 processing system

function update_queue(){

  EXPECTED_ARGS=2

  # if wrong number of input args, stop
  if [ $# -ne $EXPECTED_ARGS ]; then
    echo "Usage: `basename $0` image queue"
    exit 1
  fi
  
  IMAGE=$1
  QUEUE=$2

  # update queue status; queue must be locked and unlocked
  SEARCH=$(echo $IMAGE | sed 's_/_\\/_g')
  lockfile-create $QUEUE
  sed -i.tmp "s/$SEARCH.*/$SEARCH DONE/g" $QUEUE
  chmod --reference $QUEUE".tmp" $QUEUE
  rm $QUEUE".tmp"
  lockfile-remove $QUEUE

}

export -f update_queue


EXPECTED_ARGS=1

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: `basename $0` parameter-file"
  echo ""
  exit
fi

PRM=$1
BINDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# if parameter file doesn't exist, stop
if [ ! -f $PRM ]; then
  echo $PRM "does not exist"; exit
elif [ ! -r $PRM ]; then
  echo $PRM "is not readable"; exit
else
  dos2unix -q $PRM
fi


# get input file list from parameter file
if [ $(grep '^FILE_QUEUE' $PRM | wc -l) -eq 0 ]; then
  echo "FILE_QUEUE is not given in parameter file"; exit
else
  IFILE=$(grep '^FILE_QUEUE' $PRM | sed 's/.* *= *//g')
  if [ ! -f $IFILE ]; then
    echo $IFILE "does not exist"; exit
  elif [ ! -r $IFILE ]; then
    echo $IFILE "is not readable"; exit
  fi
fi


# get output directory from parameter file
if [ $(grep '^DIR_LEVEL2' $PRM | wc -l) -eq 0 ]; then
  echo "DIR_LEVEL2 is not given in parameter file"; exit
else
  OD=$(grep '^DIR_LEVEL2' $PRM | sed 's/.* *= *//g')
  if [ ! -d $OD ]; then
    echo $OD "does not exist"; exit
  elif [ ! -w $OD ]; then
    echo $OD "is not writeable"; exit
  fi
fi


# get temporary directory from parameter file
if [ $(grep '^DIR_TEMP' $PRM | wc -l) -eq 0 ]; then
  echo "DIR_TEMP is not given in parameter file"; exit
else
  TEMPDIR=$(grep '^DIR_TEMP' $PRM | sed 's/.* *= *//g')
  if [ ! -d $TEMPDIR ]; then
    echo $TEMPDIR "does not exist"; exit
  elif [ ! -w $TEMPDIR ]; then
    echo $TEMPDIR "is not writeable"; exit
  fi
fi


# get logfile directory from parameter file
if [ $(grep '^DIR_LOG' $PRM | wc -l) -eq 0 ]; then
  echo "DIR_LOG is not given in parameter file"; exit
else
  LOGDIR=$(grep '^DIR_LOG' $PRM | sed 's/.* *= *//g')
  if [ ! -d $LOGDIR ]; then
    echo $LOGDIR "does not exist"; exit
  elif [ ! -w $LOGDIR ]; then
    echo $LOGDIR "is not writeable"; exit
  fi
fi


# get number of processes from parameter file
if [ $(grep '^NPROC' $PRM | wc -l) -eq 0 ]; then
  echo "NPROC is not given in parameter file"; exit
else
  CPU=$(grep '^NPROC' $PRM | sed 's/.* *= *//g')
  if [ $CPU -lt 1 ]; then
    echo "NPROC needs to be >= 1"; exit
  fi
fi


# get number of processes from parameter file
if [ $(grep '^DELAY' $PRM | wc -l) -eq 0 ]; then
  echo "DELAY is not given in parameter file"; exit
else
  DELAY=$(grep '^DELAY' $PRM | sed 's/.* *= *//g')
  if [ $DELAY -lt 0 ]; then
    echo "DELAY needs to be >= 0"; exit
  fi
fi


# get number of processes from parameter file
if [ $(grep '^TIMEOUT_ZIP' $PRM | wc -l) -eq 0 ]; then
  echo "TIMEOUT_ZIP is not given in parameter file"; exit
else
  TIMEOUT_ZIP=$(grep '^TIMEOUT_ZIP' $PRM | sed 's/.* *= *//g')
  if [ $TIMEOUT_ZIP -lt 0 ]; then
    echo "TIMEOUT_ZIP needs to be >= 0"; exit
  fi
fi


# cuurent time
TIME=$(date +"%Y%m%d%H%M%S")


# protect against multiple calls
if [ $(ps aux | grep 'L2PS' | wc -l) -gt 1 ]; then
  echo "FORCE L2PS is already running. Exit." > $OD/FORCE-L2PS_$TIME.log
  exit
fi


QUEUE=$(grep 'QUEUED' $IFILE | sed 's/QUEUED//g')

# count input
NUM=$(echo $QUEUE | wc -w)
if [ $NUM -eq 0 ]; then
  echo "No images queued in " $IFILE; exit
else
  echo $NUM "images enqueued. Start processing with" $CPU "CPUs"
fi


# test if full paths are given in $IFILE
for f in $QUEUE; do
  if [[ ! $f == /* ]]; then
    echo "relative pathnames are not allowed in FILE_QUEUE"; exit
  fi
done


# test if force-l2ps was compiled in DEBUG mode
$BINDIR/force-l2ps -d
if [ $? -ne 0 ]; then
  echo "force-l2ps was compiled in DEBUG mode"
  if [ $NUM -gt 1 ] || [ $CPU -gt 1 ]; then
    echo "  cannot use DEBUG mode in force-level2"
    echo "  solutions:"
    echo "    (1) re-compile with disabled DEBUG mode or"
    echo "    (2) reduce ncpu to 1 and reduce queue length in FILE_QUEUE to 1"
    exit
  fi
fi


CPUFILE=$TEMPDIR"/cpu-"$TIME
echo $CPU > $CPUFILE

EXE=$BINDIR/force-l2ps_

echo $QUEUE | parallel -d ' ' -j$CPUFILE --delay $DELAY --eta "$EXE {} $PRM $LOGDIR $TEMPDIR $TIMEOUT_ZIP; update_queue {} $IFILE"

rm $CPUFILE

exit 0
