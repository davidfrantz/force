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

# this script enables or disables the SPLITS installation in FORCE


EXPECTED_ARGS=1

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: `basename $0` enable/disable"
  echo ""
  exit 1
fi


if [ $1 == enable ]; then
  enable=1
elif [ $1 == disable ]; then
  enable=0
else
  echo "Usage: `basename $0` enable/disable"
  echo ""
  exit 1
fi

MAKEFILE=Makefile
CONST=src/cross-level/const-cl.h

if [ ! -r $MAKEFILE ]; then
  echo "$MAKEFILE is not existing/readable"
  exit 1
fi

if [ ! -r $CONST ]; then
  echo "$CONST is not existing/readable"
  exit 1
fi


  
if [ $enable -eq 1 ]; then

  sed -i -e 's%^[/]*\(#define SPLITS\)%\1%' src/cross-level/const-cl.h
  sed -i -e 's%^[#]*\(SPLITS\)%\1%' Makefile
  sed -i -e 's%^[#]*\(LDSPLITS\)%\1%' Makefile

elif [ $enable -eq 0 ]; then

  sed -i -e 's%^[/]*\(#define SPLITS\)%//\1%' src/cross-level/const-cl.h
  sed -i -e 's%^[#]*\(SPLITS\)%#\1%' Makefile
  sed -i -e 's%^[#]*\(LDSPLITS\)%#\1%' Makefile

fi


exit 0

