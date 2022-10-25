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

# this script installs a fully-featured FORCE
# this will only work if all dependencies are met, and if all paths and environments are properly defined as expected
# admin rights required

EXPECTED_ARGS=0

# if wrong number of input args, stop
if [ $# -ne $EXPECTED_ARGS ]; then
  echo "Usage: $(basename $0)"
  echo ""
  exit 1
fi

BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $BIN

./debug.sh disable
./splits.sh enable

make -j
sudo make install

make clean

./splits.sh disable

exit 0
