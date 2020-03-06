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
if [ $# -lt $EXPECTED_ARGS ]; then
  echo "Usage: `basename $0` file"
  echo ""
  exit
fi


for i in "$@"; do

  INP=$(readlink -f $i)
  echo $INP
  # input file exists?
  if [ ! -r $INP ]; then
    echo $INP "ist not readable/existing"
    exit
  fi

  BASE=$(basename $INP)
  DIR=$(dirname $INP)


  # output dir writeable?
  if [ ! -w $DIR ]; then
    echo $DIR "ist not writeable/existing"
    exit
  fi

  echo "computing pyramids for $BASE"
  gdaladdo -ro --config COMPRESS_OVERVIEW DEFLATE --config BIGTIFF_OVERVIEW YES -r nearest $INP 2 4 8 16

done
